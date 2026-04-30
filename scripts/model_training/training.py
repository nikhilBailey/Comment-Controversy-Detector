from __future__ import annotations
import argparse
import copy
import os
from datetime import datetime
from typing import Any
import joblib
import numpy as numpy
import pandas as pandas
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from Model import (
    CLASSIFIER_STEP,
    LABEL_COL,
    PCA_STEP,
    TEXT_COL,
    ClassifierKind,
    Model,
    build_pipeline,
    extract_feature_matrix_and_labels,
)
from Visualizer import Visualizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FEATURE_DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "feature_data")
DEFAULT_VISUALIZATIONS_ROOT_DIRECTORY = os.path.join(BASE_DIR, "visualizations")

PERCENT_TEST_DATA = 0.1
DATA_SPLIT_SEED = 42
NUM_CV_FOLDS = 5


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
    training_data: pandas.DataFrame,) -> list[pandas.DataFrame]:
    labels = training_data[LABEL_COL].to_numpy()
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=DATA_SPLIT_SEED)
    folds: list[pandas.DataFrame] = []
    for _, label_indices in skf.split(training_data, labels):
        folds.append(training_data.iloc[label_indices].reset_index(drop=True))
    return folds


def _predicted_probabilities_for_positive_class(
    pipeline,
    feature_matrix: numpy.ndarray,) -> numpy.ndarray:
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    probability_matrix = pipeline.predict_proba(feature_matrix)
    classes = list(classifier.classes_)
    if 1 in classes:
        positive_class_column_index = classes.index(1)
    else:
        positive_class_column_index = probability_matrix.shape[1] - 1
    return probability_matrix[:, positive_class_column_index]


def _scores_from_predictions(
    true_labels_integer: numpy.ndarray,
    predicted_probabilities: numpy.ndarray) -> dict[str, float]:
    predicted_labels_integer = (predicted_probabilities >= 0.5).astype(int)

    accuracy_value = float(accuracy_score(true_labels_integer, predicted_labels_integer))
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


def _compute_staged_metrics(
    pipeline,
    evaluation_features_in_pca_space: numpy.ndarray,
    true_labels_integer: numpy.ndarray,) -> list[dict[str, float]]:
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    if not hasattr(classifier, "staged_predict_proba"):
        return []

    metrics_history: list[dict[str, float]] = []
    classes = list(classifier.classes_)
    if 1 in classes:
        positive_class_column_index = classes.index(1)
    else:
        positive_class_column_index = -1

    for stage_index, staged_proba in enumerate(
        classifier.staged_predict_proba(evaluation_features_in_pca_space), start=1):
        positive_probabilities = staged_proba[:, positive_class_column_index]
        stage_scores = _scores_from_predictions(true_labels_integer, positive_probabilities)
        stage_scores["epoch"] = float(stage_index)
        metrics_history.append(stage_scores)
    return metrics_history


def train_model(
    model: Model,
    train_dataframe: pandas.DataFrame,
    validation_dataframe: pandas.DataFrame | None,
    feature_column_names: list[str],) -> None:
    if model.pipeline is None:
        raise ValueError("model.pipeline must be set before train_model")
    training_features, training_labels = extract_feature_matrix_and_labels(
        train_dataframe, feature_column_names
    )
    model.pipeline.fit(training_features, training_labels)
    model.metrics.clear()

    if not model.supports_staged_prediction:
        return

    evaluation_dataframe = (
        validation_dataframe if validation_dataframe is not None else train_dataframe
    )
    evaluation_features, evaluation_labels = extract_feature_matrix_and_labels(
        evaluation_dataframe, feature_column_names
    )
    scaler = model.pipeline.named_steps["scaler"]
    pca = model.pipeline.named_steps[PCA_STEP]
    evaluation_features_in_pca_space = pca.transform(scaler.transform(evaluation_features))

    model.metrics = _compute_staged_metrics(
        model.pipeline, evaluation_features_in_pca_space, evaluation_labels.astype(int)
    )


def evaluate_model(
    model: Model,
    evaluation_dataframe: pandas.DataFrame,
    feature_column_names: list[str],) -> dict[str, float]:
    if model.pipeline is None:
        raise ValueError("model.pipeline must be set before evaluate_model")
    evaluation_features, true_labels = extract_feature_matrix_and_labels(
        evaluation_dataframe, feature_column_names
    )
    predicted_probabilities = _predicted_probabilities_for_positive_class(
        model.pipeline, evaluation_features
    )
    return _scores_from_predictions(true_labels.astype(int), predicted_probabilities)


def _mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "epoch"]
    return {key: float(numpy.mean([row[key] for row in rows])) for key in keys}


def _instantiate_pipeline_for_template(template: Model) -> None:
    template.pipeline = build_pipeline(
        classifier_kind=template.classifier_kind,
        classifier_kwargs=template.classifier_kwargs,
        pca_n_components=template.pca_n_components,
        random_state=DATA_SPLIT_SEED,
    )


def _run_cv_for_model(
    template: Model,
    train_pool: pandas.DataFrame,
    feature_cols: list[str],
    num_cross_validation_folds: int,) -> Model:
    folds = cross_validation_split(num_cross_validation_folds, train_pool)
    fold_finals: list[dict[str, float]] = []
    best_f1 = -1.0
    best_pipeline = None
    best_metrics_history: list[dict[str, float]] = []

    for fold_index in range(num_cross_validation_folds):
        train_parts = [folds[i] for i in range(num_cross_validation_folds) if i != fold_index]
        train_dataframe = pandas.concat(train_parts, ignore_index=True)
        validation_dataframe = folds[fold_index]
        candidate = copy.deepcopy(template)
        _instantiate_pipeline_for_template(candidate)
        train_model(candidate, train_dataframe, validation_dataframe, feature_cols)
        final = evaluate_model(candidate, validation_dataframe, feature_cols)
        fold_finals.append(final)
        if final["f1"] > best_f1:
            best_f1 = final["f1"]
            best_pipeline = copy.deepcopy(candidate.pipeline)
            best_metrics_history = copy.deepcopy(candidate.metrics)

    out = copy.deepcopy(template)
    out.pipeline = best_pipeline
    out.metrics = best_metrics_history
    out.final_cv_fold_scores = fold_finals
    out.mean_cv_scores = _mean_scores(fold_finals)
    return out


def build_templates(num_features: int) -> list[Model]:
    pca_components = num_features
    #Here's the list of models we use
    templates: list[Model] = [
        Model(
            model_name="random_forest_pca",
            classifier_kind="random_forest",
            classifier_kwargs={"n_estimators": 200, "max_depth": None, "n_jobs": -1},
            pca_n_components=pca_components,
        ),
        Model(
            model_name="adaboost_pca",
            classifier_kind="adaboost",
            classifier_kwargs={"n_estimators": 200, "learning_rate": 1.0},
            pca_n_components=pca_components,
        ),
        Model(
            model_name="gradient_boosting_pca",
            classifier_kind="gradient_boosting",
            classifier_kwargs={"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
            pca_n_components=pca_components,
        ),
        Model(
            model_name="svm_rbf_pca",
            classifier_kind="svm",
            classifier_kwargs={"kernel": "rbf", "C": 1.0, "gamma": "scale"},
            pca_n_components=pca_components,
        ),
    ]
    return templates


def _save_models(
    cv_trained_models: list[Model],
    top_final_model: Model,
    output_directory: str,) -> None:

    models_directory = os.path.join(output_directory, "models")
    os.makedirs(models_directory, exist_ok=True)
    for trained_model in cv_trained_models:
        if trained_model.pipeline is None:
            continue
        joblib.dump(
            trained_model.pipeline,
            os.path.join(models_directory, f"{trained_model.model_name}_cv_best.joblib"),
        )
    if top_final_model.pipeline is not None:
        joblib.dump(
            top_final_model.pipeline,
            os.path.join(models_directory, f"{top_final_model.model_name}_refit_full.joblib"),
        )


def main() -> None:
    feature_csv_file_name = "combined_feature_data.csv"

    parser = argparse.ArgumentParser(
        description="Train PCA + ensemble/SVM bot classifiers."
    )
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

    templates = build_templates(num_features=len(cols))

    trained: list[Model] = []
    for template in templates:
        trained.append(_run_cv_for_model(template, train_pool, cols, NUM_CV_FOLDS))

    primary = "f1"
    top = max(trained, key=lambda m: m.mean_cv_scores.get(primary, 0.0))

    #For our top template, we retrain on the full training pool
    top_final = copy.deepcopy(top)
    _instantiate_pipeline_for_template(top_final)
    train_model(top_final, train_pool, None, cols)
    top_final.test_scores = evaluate_model(top_final, test_df, cols)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.visualizations_root_directory, ts)
    os.makedirs(out_dir, exist_ok=True)
    _save_models(trained, top_final, out_dir)
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
