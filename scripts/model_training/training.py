"""
Train tabular MLP classifiers on combined feature CSV (numeric features only).
See project plan: stratified holdout + CV, metrics, visualizations.
"""

from __future__ import annotations

import argparse
import copy
import os
import warnings
from datetime import datetime

import numpy as numpy
import pandas as pandas
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from Model import (
    BinaryClassifierMLP,
    LABEL_COL,
    TEXT_COL,
    Model,
    extract_feature_matrix_and_labels,
)
from Visualizer import Visualizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FEATURE_DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "feature_data")
DEFAULT_VISUALIZATIONS_ROOT_DIRECTORY = os.path.join(BASE_DIR, "visualizations")

PERCENT_TEST_DATA = 0.1
DATA_SPLIT_SEED = 42

_cpu_fallback_warning_issued = False


def feature_columns(dataframe: pandas.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if column not in (TEXT_COL, LABEL_COL)]


def import_data(csv_file_path: str) -> pandas.DataFrame:
    return pandas.read_csv(csv_file_path)


def test_split(data: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    labels = data[LABEL_COL]
    return train_test_split(
        data,
        test_size=PERCENT_TEST_DATA,
        stratify=labels,
        random_state=DATA_SPLIT_SEED,
    )


def cross_validation_split(
    num_folds: int,
    training_data: pandas.DataFrame,
) -> list[pandas.DataFrame]:
    """each element is one validation chunk."""
    labels = training_data[LABEL_COL].to_numpy()
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=DATA_SPLIT_SEED)
    folds: list[pandas.DataFrame] = []
    for _, label_indices in skf.split(training_data, labels):
        folds.append(training_data.iloc[label_indices].reset_index(drop=True))
    return folds


def get_device() -> torch.device:
    """
    Prefer CUDA, then Apple MPS, else CPU.
    Emits a warning once per process when falling back to CPU because no accelerator is available.
    """
    global _cpu_fallback_warning_issued
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if not _cpu_fallback_warning_issued:
        warnings.warn(
            "No accelerator available (CUDA and MPS are both unavailable); using CPU.",
            UserWarning,
            stacklevel=2,
        )
        _cpu_fallback_warning_issued = True
    return torch.device("cpu")


def _build_optimizer(model: Model) -> torch.optim.Optimizer:
    assert model.classifier_model is not None
    if model.optimization_function == "adam":
        return torch.optim.Adam(model.classifier_model.parameters(), lr=model.learning_rate)
    if model.optimization_function == "sgd":
        return torch.optim.SGD(model.classifier_model.parameters(), lr=model.learning_rate)
    raise ValueError(f"Unknown optimizer {model.optimization_function}")


def train_model(
    model: Model,
    train_dataframe: pandas.DataFrame,
    validation_dataframe: pandas.DataFrame | None,
    feature_column_names: list[str],
) -> None:
    """
    Fit scaler on train_dataframe only, train BCE-with-logits for num_epochs.
    Appends metric dicts to model.metrics every evaluate_every epochs.
    If validation_dataframe is None, metrics are computed on train_dataframe (for a final full-data fit).
    """
    if model.classifier_model is None:
        raise ValueError("model.classifier_model must be set before train_model")
    compute_device = get_device()
    model.classifier_model = model.classifier_model.to(compute_device)
    model.scaler = StandardScaler()
    training_features, training_labels = extract_feature_matrix_and_labels(
        train_dataframe, feature_column_names
    )
    scaled_training_features = model.scaler.fit_transform(training_features)

    training_features_tensor = torch.tensor(
        scaled_training_features, dtype=torch.float32, device=compute_device
    )
    training_labels_tensor = torch.tensor(
        training_labels, dtype=torch.float32, device=compute_device
    ).unsqueeze(1)

    optimizer = _build_optimizer(model)
    loss_function = nn.BCEWithLogitsLoss()
    model.metrics.clear()
    evaluation_dataframe = (
        validation_dataframe if validation_dataframe is not None else train_dataframe
    )

    for epoch_index in range(1, model.num_epochs + 1):
        model.classifier_model.train()
        optimizer.zero_grad()
        prediction_logits = model.classifier_model(training_features_tensor)
        training_loss = loss_function(prediction_logits, training_labels_tensor)
        training_loss.backward()
        optimizer.step()

        should_record_metrics = (
            epoch_index == 1
            or epoch_index % model.evaluate_every == 0
            or epoch_index == model.num_epochs
        )
        if should_record_metrics:
            model.classifier_model.eval()
            with torch.no_grad():
                epoch_evaluation_metrics = evaluate_model(
                    model,
                    evaluation_dataframe,
                    feature_column_names,
                )
            epoch_evaluation_metrics["epoch"] = float(epoch_index)
            model.metrics.append(epoch_evaluation_metrics)


def evaluate_model(
    model: Model,
    evaluation_dataframe: pandas.DataFrame,
    feature_column_names: list[str],
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 (threshold 0.5), Pearson/Spearman on probabilities vs labels."""
    if model.classifier_model is None or model.scaler is None:
        raise ValueError("model.classifier_model and model.scaler must be set before evaluate_model")
    compute_device = next(model.classifier_model.parameters()).device
    model.classifier_model.eval()
    evaluation_features, true_labels = extract_feature_matrix_and_labels(
        evaluation_dataframe, feature_column_names
    )
    scaled_evaluation_features = model.scaler.transform(evaluation_features)
    evaluation_features_tensor = torch.tensor(
        scaled_evaluation_features, dtype=torch.float32, device=compute_device
    )
    with torch.no_grad():
        prediction_logits = model.classifier_model(evaluation_features_tensor)
        predicted_probabilities = (
            torch.sigmoid(prediction_logits).squeeze(1).cpu().numpy()
        )
    true_labels_integer = true_labels.astype(int)
    predicted_labels_integer = (predicted_probabilities >= 0.5).astype(int)

    accuracy_value = float(
        accuracy_score(true_labels_integer, predicted_labels_integer)
    )
    precision_value = float(
        precision_score(true_labels_integer, predicted_labels_integer, zero_division=0)
    )
    recall_value = float(
        recall_score(true_labels_integer, predicted_labels_integer, zero_division=0)
    )
    f1_value = float(
        f1_score(true_labels_integer, predicted_labels_integer, zero_division=0)
    )

    predicted_probability_series = pandas.Series(predicted_probabilities)
    true_label_float_series = pandas.Series(true_labels_integer.astype(float))
    pearson_correlation = float(
        predicted_probability_series.corr(true_label_float_series, method="pearson")
    )
    if numpy.isnan(pearson_correlation):
        pearson_correlation = 0.0
    spearman_correlation = float(
        predicted_probability_series.corr(true_label_float_series, method="spearman")
    )
    if numpy.isnan(spearman_correlation):
        spearman_correlation = 0.0

    return {
        "accuracy": accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "pearson": pearson_correlation,
        "spearman": spearman_correlation,
    }


def _mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "epoch"]
    return {key: float(numpy.mean([row[key] for row in rows])) for key in keys}


def _run_cv_for_model(
    template: Model,
    train_pool: pandas.DataFrame,
    feature_cols: list[str],
    num_cross_validation_folds: int,
) -> Model:
    folds = cross_validation_split(num_cross_validation_folds, train_pool)
    fold_finals: list[dict[str, float]] = []
    best_f1 = -1.0
    best_state: dict | None = None
    best_scaler: StandardScaler | None = None
    best_metrics_history: list[dict[str, float]] = []

    in_dim = len(feature_cols)
    for fold_index in range(num_cross_validation_folds):
        train_parts = [folds[i] for i in range(num_cross_validation_folds) if i != fold_index]
        train_dataframe = pandas.concat(train_parts, ignore_index=True)
        validation_dataframe = folds[fold_index]
        m = copy.deepcopy(template)
        m.classifier_model = BinaryClassifierMLP(in_dim, m.hidden_sizes, m.dropout)
        m.scaler = StandardScaler()
        train_model(m, train_dataframe, validation_dataframe, feature_cols)
        final = evaluate_model(m, validation_dataframe, feature_cols)
        fold_finals.append(final)
        if final["f1"] > best_f1:
            best_f1 = final["f1"]
            best_state = {kk: vv.cpu().clone() for kk, vv in m.classifier_model.state_dict().items()}
            best_scaler = copy.deepcopy(m.scaler)
            best_metrics_history = copy.deepcopy(m.metrics)

    out = copy.deepcopy(template)
    out.classifier_model = BinaryClassifierMLP(in_dim, out.hidden_sizes, out.dropout)
    if best_state is not None:
        out.classifier_model.load_state_dict(best_state)
    out.scaler = best_scaler
    out.metrics = best_metrics_history
    out.final_cv_fold_scores = fold_finals
    out.mean_cv_scores = _mean_scores(fold_finals)
    return out


def main() -> None:
    feature_csv_file_name = "combined_feature_data.csv"

    parser = argparse.ArgumentParser(description="Train tabular bot classifiers.")
    parser.add_argument(
        "--feature-data-directory",
        type=str,
        default=DEFAULT_FEATURE_DATA_DIRECTORY,
        help="Directory containing feature CSV files",
    )
    parser.add_argument(
        "--feature-csv-file-name",
        type=str,
        default=feature_csv_file_name,
        help="CSV file name under the feature data directory",
    )
    parser.add_argument(
        "--visualizations-root-directory",
        type=str,
        default=DEFAULT_VISUALIZATIONS_ROOT_DIRECTORY,
        help="Root directory for run output (timestamped subfolders are created here)",
    )
    args = parser.parse_args()

    feature_csv_file_path = os.path.join(args.feature_data_directory, args.feature_csv_file_name)
    data = import_data(feature_csv_file_path)
    cols = feature_columns(data)
    train_pool, test_df = test_split(data)

    templates: list[Model] = [
        Model("mlp_small_adam", 80, 5, "adam", 1e-3, (32, 16), 0.1),
        Model("mlp_small_sgd", 80, 5, "sgd", 5e-2, (32, 16), 0.1),
        Model("mlp_wide_adam", 80, 5, "adam", 1e-3, (128, 64), 0.2),
        Model("mlp_deep_adam", 100, 5, "adam", 5e-4, (64, 48, 32), 0.15),
        Model("mlp_tiny_lr", 80, 5, "adam", 1e-4, (48, 24), 0.1),
        Model("mlp_high_dropout", 80, 5, "adam", 1e-3, (64, 32), 0.35),
        Model("mlp_large_batch_style", 60, 4, "adam", 2e-3, (96, 48), 0.1),
        Model("mlp_sgd_deep", 100, 5, "sgd", 1e-1, (72, 36), 0.1),
    ]

    number_of_models = len(templates)
    num_cross_validation_folds = max(2, number_of_models)

    trained: list[Model] = []
    for template in templates:
        trained.append(
            _run_cv_for_model(template, train_pool, cols, num_cross_validation_folds)
        )

    primary = "f1"
    top = max(trained, key=lambda m: m.mean_cv_scores.get(primary, 0.0))
    top_final = copy.deepcopy(top)
    in_dim = len(cols)
    top_final.classifier_model = BinaryClassifierMLP(in_dim, top_final.hidden_sizes, top_final.dropout)
    train_model(top_final, train_pool, None, cols)
    top_final.test_scores = evaluate_model(top_final, test_df, cols)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.visualizations_root_directory, ts)
    os.makedirs(out_dir, exist_ok=True)
    Visualizer(random_state=DATA_SPLIT_SEED).produce_all_visualizations(
        trained, out_dir, top_final, train_pool, cols
    )

    print("Mean CV scores (primary: mean F1 across folds):")
    for m in trained:
        print(f"  {m.model_name}: F1={m.mean_cv_scores.get('f1', 0):.4f}")
    print(f"Top model: {top.model_name}")
    print(f"Holdout test: {top_final.test_scores}")


if __name__ == "__main__":
    main()
