"""
Train tabular MLP classifiers on combined feature CSV (numeric features only).
See project plan: stratified holdout + CV, metrics, visualizations.
"""

from __future__ import annotations

import argparse
import copy
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
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
import torch.nn.functional as torch_functional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FEATURE_DATA_DIRECTORY = os.path.join(BASE_DIR, "data", "feature_data")
DEFAULT_VISUALIZATIONS_ROOT_DIRECTORY = os.path.join(BASE_DIR, "visualizations")

LABEL_COL = "is_bot_annotation"
TEXT_COL = "text"

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


def extract_feature_matrix_and_labels(
    dataframe: pandas.DataFrame,
    feature_columns: list[str],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    feature_matrix = dataframe[feature_columns].to_numpy(dtype=numpy.float64)
    labels = dataframe[LABEL_COL].to_numpy(dtype=numpy.float64)
    return feature_matrix, labels


class BinaryClassifierMLP(nn.Module):
    """MLP on numeric feature vectors; single logit for binary labels. First hidden block is the 'encoder' for viz."""

    def __init__(self, in_dim: int, hidden_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")
        sizes = (in_dim,) + hidden_sizes
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(sizes[-1], 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs
        for linear_layer in self.layers:
            activations = linear_layer(activations)
            activations = torch_functional.relu(activations)
            activations = self.dropout(activations)
        return self.final(activations)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.layers[0](inputs)
        return torch_functional.relu(encoded)


@dataclass
class Model:
    model_name: str
    num_epochs: int
    evaluate_every: int
    optimization_function: Literal["adam", "sgd"]
    learning_rate: float
    hidden_sizes: tuple[int, ...]
    dropout: float
    classifier_model: BinaryClassifierMLP | None = None
    scaler: StandardScaler | None = None
    metrics: list[dict[str, float]] = field(default_factory=list)
    final_cv_fold_scores: list[dict[str, float]] = field(default_factory=list)
    mean_cv_scores: dict[str, float] = field(default_factory=dict)
    test_scores: dict[str, float] = field(default_factory=dict)


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


def filesystem_safe_directory_name(display_name: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in display_name
    )


def produce_model_training_visualizations(model: Model, output_path: str) -> None:
    """
    Write several charts from model.metrics: per-metric panels, normalized overlay,
    precision–recall trajectory across epochs, and metric correlation across checkpoints.
    """
    if not model.metrics:
        return
    os.makedirs(output_path, exist_ok=True)
    safe_name = filesystem_safe_directory_name(model.model_name)
    metric_keys = ["accuracy", "precision", "recall", "f1", "pearson", "spearman"]
    epoch_numbers = numpy.array([float(row["epoch"]) for row in model.metrics])

    figure, axes = pyplot.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for axis, metric_key in zip(axes, metric_keys):
        metric_values = [row[metric_key] for row in model.metrics]
        axis.plot(epoch_numbers, metric_values, marker="o", markersize=2)
        axis.set_title(metric_key)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"Training metrics — {safe_name}")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "training_metrics.png"), dpi=150)
    pyplot.close(figure)

    figure, axis = pyplot.subplots(figsize=(10, 6))
    for metric_key in metric_keys:
        raw_values = numpy.array([row[metric_key] for row in model.metrics], dtype=numpy.float64)
        value_min = raw_values.min()
        value_max = raw_values.max()
        if value_max == value_min:
            normalized_values = numpy.full_like(raw_values, 0.5)
        else:
            normalized_values = (raw_values - value_min) / (value_max - value_min)
        axis.plot(epoch_numbers, normalized_values, marker="o", markersize=3, label=metric_key)
    axis.set_xlabel("epoch")
    axis.set_ylabel("min-max normalized score (per metric)")
    axis.set_title(f"All metrics on comparable scale — {safe_name}")
    axis.legend(loc="best", ncol=2, fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "metrics_normalized_overlay.png"), dpi=150)
    pyplot.close(figure)

    precision_values = numpy.array([row["precision"] for row in model.metrics], dtype=numpy.float64)
    recall_values = numpy.array([row["recall"] for row in model.metrics], dtype=numpy.float64)
    figure, axis = pyplot.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        precision_values,
        recall_values,
        c=epoch_numbers,
        cmap="viridis",
        s=48,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    axis.plot(precision_values, recall_values, "k-", alpha=0.35, linewidth=1, zorder=1)
    axis.set_xlabel("precision")
    axis.set_ylabel("recall")
    axis.set_title(f"Precision vs recall across checkpoints — {safe_name}")
    axis.grid(True, alpha=0.3)
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    figure.colorbar(scatter, ax=axis, label="epoch")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "precision_recall_trajectory.png"), dpi=150)
    pyplot.close(figure)

    metrics_matrix = numpy.array(
        [[row[key] for key in metric_keys] for row in model.metrics],
        dtype=numpy.float64,
    )
    if metrics_matrix.shape[0] >= 2:
        correlation_matrix = numpy.corrcoef(metrics_matrix, rowvar=False)
        masked_correlation = numpy.ma.masked_invalid(correlation_matrix)
        figure, axis = pyplot.subplots(figsize=(7.5, 6.5))
        color_mesh = axis.imshow(masked_correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
        axis.set_xticks(numpy.arange(len(metric_keys)))
        axis.set_yticks(numpy.arange(len(metric_keys)))
        axis.set_xticklabels(metric_keys, rotation=45, ha="right")
        axis.set_yticklabels(metric_keys)
        axis.set_title(f"Metric correlation across logged epochs — {safe_name}")
        for row_index in range(len(metric_keys)):
            for column_index in range(len(metric_keys)):
                cell = correlation_matrix[row_index, column_index]
                if not numpy.isnan(cell):
                    axis.text(
                        column_index,
                        row_index,
                        f"{cell:.2f}",
                        ha="center",
                        va="center",
                        color="black" if abs(cell) < 0.65 else "white",
                        fontsize=8,
                    )
        figure.colorbar(color_mesh, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(os.path.join(output_path, "metric_correlation_heatmap.png"), dpi=150)
        pyplot.close(figure)

    first_row = model.metrics[0]
    last_row = model.metrics[-1]
    figure, axis = pyplot.subplots(figsize=(10, 5))
    bar_x = numpy.arange(len(metric_keys))
    bar_width = 0.35
    first_values = [first_row[key] for key in metric_keys]
    last_values = [last_row[key] for key in metric_keys]
    axis.bar(bar_x - bar_width / 2, first_values, bar_width, label=f"first log (epoch {int(first_row['epoch'])})")
    axis.bar(bar_x + bar_width / 2, last_values, bar_width, label=f"last log (epoch {int(last_row['epoch'])})")
    axis.set_xticks(bar_x)
    axis.set_xticklabels(metric_keys, rotation=20, ha="right")
    axis.set_ylabel("score")
    axis.set_title(f"First vs last logged evaluation — {safe_name}")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "first_vs_last_logged_metrics.png"), dpi=150)
    pyplot.close(figure)


def produce_final_model_evaluation_visualizations(models: list[Model], output_path: str) -> None:
    """
    Several comparison figures from mean CV scores and per-fold CV lists:
    grouped bars, heatmap, fold spread boxplots, precision-recall scatter, F1 ranking.
    """
    if not models:
        return
    os.makedirs(output_path, exist_ok=True)
    names = [filesystem_safe_directory_name(trained_model.model_name) for trained_model in models]
    metric_keys = ["accuracy", "precision", "recall", "f1", "pearson", "spearman"]
    x_positions = numpy.arange(len(names))
    bar_width = 0.12
    figure, axis = pyplot.subplots(figsize=(max(10, len(names) * 1.2), 6))
    for index, metric_key in enumerate(metric_keys):
        values = [trained_model.mean_cv_scores.get(metric_key, 0.0) for trained_model in models]
        axis.bar(x_positions + (index - 2.5) * bar_width, values, bar_width, label=metric_key)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(names, rotation=25, ha="right")
    axis.set_ylabel("score (mean CV)")
    axis.legend(loc="upper right", ncol=2)
    axis.set_title("Model comparison (mean validation across CV folds)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "model_comparison_cv.png"), dpi=150)
    pyplot.close(figure)

    mean_score_matrix = numpy.array(
        [
            [trained_model.mean_cv_scores.get(metric_key, numpy.nan) for metric_key in metric_keys]
            for trained_model in models
        ],
        dtype=numpy.float64,
    )
    color_min = numpy.nanmin(mean_score_matrix)
    color_max = numpy.nanmax(mean_score_matrix)
    if color_max == color_min:
        color_max = color_min + 1e-6
    figure_height = max(4.5, len(models) * 0.45)
    figure, axis = pyplot.subplots(figsize=(9.0, figure_height))
    color_mesh = axis.imshow(
        mean_score_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
        interpolation="nearest",
    )
    axis.set_xticks(numpy.arange(len(metric_keys)))
    axis.set_xticklabels(metric_keys, rotation=35, ha="right")
    axis.set_yticks(numpy.arange(len(models)))
    axis.set_yticklabels(names)
    axis.set_title("Mean CV score heatmap (models × metrics)")
    figure.colorbar(color_mesh, ax=axis, label="score")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "model_comparison_mean_cv_heatmap.png"), dpi=150)
    pyplot.close(figure)

    profile_matrix = mean_score_matrix.copy()
    for row_index in range(profile_matrix.shape[0]):
        row = profile_matrix[row_index, :]
        row_min = numpy.nanmin(row)
        row_max = numpy.nanmax(row)
        if row_max == row_min:
            profile_matrix[row_index, :] = 0.5
        else:
            profile_matrix[row_index, :] = (row - row_min) / (row_max - row_min)
    figure, axis = pyplot.subplots(figsize=(9.0, figure_height))
    color_mesh = axis.imshow(profile_matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_xticks(numpy.arange(len(metric_keys)))
    axis.set_xticklabels(metric_keys, rotation=35, ha="right")
    axis.set_yticks(numpy.arange(len(models)))
    axis.set_yticklabels(names)
    axis.set_title("Per-model metric profile (min-max normalized within each model)")
    figure.colorbar(color_mesh, ax=axis, label="relative strength")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "model_comparison_profile_heatmap.png"), dpi=150)
    pyplot.close(figure)

    figure, axes = pyplot.subplots(2, 3, figsize=(14, 9))
    for subplot_axis, metric_key in zip(axes.flatten(), metric_keys):
        per_model_fold_values: list[list[float]] = []
        for trained_model in models:
            fold_scores = [
                float(fold_row[metric_key])
                for fold_row in trained_model.final_cv_fold_scores
                if metric_key in fold_row
            ]
            if not fold_scores:
                fold_scores = [float(trained_model.mean_cv_scores.get(metric_key, 0.0))]
            per_model_fold_values.append(fold_scores)
        boxplot_result = subplot_axis.boxplot(
            per_model_fold_values,
            showmeans=True,
            meanline=True,
        )
        subplot_axis.set_xticks(numpy.arange(1, len(names) + 1))
        subplot_axis.set_xticklabels(names, rotation=40, ha="right")
        subplot_axis.set_title(f"{metric_key} across CV folds")
        subplot_axis.grid(True, axis="y", alpha=0.3)
        for median_line in boxplot_result.get("medians", []):
            median_line.set_color("darkred")
            median_line.set_linewidth(1.6)
    figure.suptitle("Per-fold spread by model (box = quartiles, red = median)", y=1.02)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "model_comparison_cv_fold_boxplots.png"), dpi=150)
    pyplot.close(figure)

    mean_precision = numpy.array(
        [trained_model.mean_cv_scores.get("precision", 0.0) for trained_model in models],
        dtype=numpy.float64,
    )
    mean_recall = numpy.array(
        [trained_model.mean_cv_scores.get("recall", 0.0) for trained_model in models],
        dtype=numpy.float64,
    )
    mean_f1 = numpy.array(
        [trained_model.mean_cv_scores.get("f1", 0.0) for trained_model in models],
        dtype=numpy.float64,
    )
    figure, axis = pyplot.subplots(figsize=(9, 7))
    scatter = axis.scatter(
        mean_precision,
        mean_recall,
        c=mean_f1,
        cmap="coolwarm",
        s=120,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    for index, label in enumerate(names):
        axis.annotate(
            label,
            (mean_precision[index], mean_recall[index]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            alpha=0.9,
        )
    axis.set_xlabel("mean CV precision")
    axis.set_ylabel("mean CV recall")
    axis.set_title("Models in precision-recall space (color = mean F1)")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    figure.colorbar(scatter, ax=axis, label="mean F1")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "model_comparison_precision_recall_scatter.png"), dpi=150)
    pyplot.close(figure)

    f1_means = [trained_model.mean_cv_scores.get("f1", 0.0) for trained_model in models]
    sorted_indices = numpy.argsort(f1_means)[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_f1 = [f1_means[i] for i in sorted_indices]
    figure, axis = pyplot.subplots(figsize=(max(8, len(models) * 0.55), 5.5))
    y_positions = numpy.arange(len(sorted_names))
    axis.barh(y_positions, sorted_f1, color="steelblue", edgecolor="black", linewidth=0.4)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(sorted_names)
    axis.invert_yaxis()
    axis.set_xlabel("mean CV F1")
    axis.set_title("Models ranked by mean CV F1 (highest at top)")
    axis.set_xlim(0.0, 1.02)
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "model_ranking_mean_f1.png"), dpi=150)
    pyplot.close(figure)


def produce_encoder_visualizations(
    model: Model, output_path: str, sample_df: pandas.DataFrame, feature_cols: list[str]
) -> None:
    """
    Plots for the first linear encoder block: weights, bias, norms, feature attribution,
    activation stats, PCA / optional t-SNE, and an activation heatmap on a sample batch.
    """
    if model.classifier_model is None or model.scaler is None:
        return
    os.makedirs(output_path, exist_ok=True)
    compute_device = next(model.classifier_model.parameters()).device
    first_linear_layer = model.classifier_model.layers[0]
    weight_matrix = first_linear_layer.weight.detach().cpu().numpy()

    figure, axis = pyplot.subplots(figsize=(10, 6))
    color_mesh = axis.imshow(weight_matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axis.set_xlabel("input feature index (scaled column order)")
    axis.set_ylabel("encoder unit index")
    axis.set_title("Encoder weight matrix (first linear layer)")
    figure.colorbar(color_mesh, ax=axis)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "encoder_weights.png"), dpi=150)
    pyplot.close(figure)

    sample_row_count = min(256, len(sample_df))
    sample_dataframe = sample_df.iloc[:sample_row_count]
    sample_features, _ = extract_feature_matrix_and_labels(sample_dataframe, feature_cols)
    scaled_sample_features = model.scaler.transform(sample_features)
    sample_features_tensor = torch.tensor(
        scaled_sample_features, dtype=torch.float32, device=compute_device
    )
    model.classifier_model.eval()
    with torch.no_grad():
        encoder_activations = model.classifier_model.encode(sample_features_tensor).cpu().numpy()

    if first_linear_layer.bias is not None:
        bias_numpy = first_linear_layer.bias.detach().cpu().numpy()
        figure, axis = pyplot.subplots(figsize=(10, 4))
        axis.bar(numpy.arange(len(bias_numpy)), bias_numpy, color="slategray", edgecolor="black", linewidth=0.3)
        axis.set_xlabel("encoder unit index")
        axis.set_ylabel("bias")
        axis.set_title("First-layer encoder biases")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join(output_path, "encoder_first_layer_bias.png"), dpi=150)
        pyplot.close(figure)

    row_weight_l2_norms = numpy.linalg.norm(weight_matrix, axis=1)
    figure, axis = pyplot.subplots(figsize=(10, 4))
    axis.bar(numpy.arange(len(row_weight_l2_norms)), row_weight_l2_norms, color="teal", edgecolor="black", linewidth=0.3)
    axis.set_xlabel("encoder unit index")
    axis.set_ylabel("L2 norm of weight row")
    axis.set_title("Magnitude of incoming weights per encoder unit")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "encoder_unit_weight_l2_norms.png"), dpi=150)
    pyplot.close(figure)

    column_abs_sum_importance = numpy.sum(numpy.abs(weight_matrix), axis=0)
    figure, axis = pyplot.subplots(figsize=(max(8, len(feature_cols) * 0.45), 5))
    axis.bar(
        numpy.arange(len(feature_cols)),
        column_abs_sum_importance,
        color="coral",
        edgecolor="black",
        linewidth=0.3,
    )
    axis.set_xticks(numpy.arange(len(feature_cols)))
    axis.set_xticklabels(feature_cols, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("sum of |weights|")
    axis.set_title("Input feature attribution proxy (first layer, column L1 mass)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "encoder_input_feature_abs_weight_mass.png"), dpi=150)
    pyplot.close(figure)

    figure, axis = pyplot.subplots(figsize=(8, 5))
    axis.hist(encoder_activations.flatten(), bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    axis.set_xlabel("ReLU encoder activation value")
    axis.set_ylabel("count")
    axis.set_title("Distribution of encoder outputs (sample batch)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "encoder_activation_histogram.png"), dpi=150)
    pyplot.close(figure)

    heatmap_row_count = min(80, encoder_activations.shape[0])
    figure, axis = pyplot.subplots(figsize=(max(8, weight_matrix.shape[1] * 0.25), 7))
    activation_slice = encoder_activations[:heatmap_row_count, :]
    color_mesh = axis.imshow(activation_slice, aspect="auto", cmap="magma", interpolation="nearest")
    axis.set_xlabel("encoder unit index")
    axis.set_ylabel("sample row (subset)")
    axis.set_title("Encoder activations (samples × units)")
    figure.colorbar(color_mesh, ax=axis, label="activation")
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "encoder_activation_heatmap.png"), dpi=150)
    pyplot.close(figure)

    mean_activation_per_unit = numpy.mean(encoder_activations, axis=0)
    std_activation_per_unit = numpy.std(encoder_activations, axis=0)
    unit_indices = numpy.arange(len(mean_activation_per_unit))
    figure, axis = pyplot.subplots(figsize=(10, 4))
    axis.bar(unit_indices, mean_activation_per_unit, yerr=std_activation_per_unit, capsize=2, color="mediumpurple", edgecolor="black", linewidth=0.3, alpha=0.85)
    axis.set_xlabel("encoder unit index")
    axis.set_ylabel("mean activation (+/- std)")
    axis.set_title("Per-unit activation mean on sample (error bars = batch std)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "encoder_unit_activation_mean_std.png"), dpi=150)
    pyplot.close(figure)

    activation_sample_count = encoder_activations.shape[0]
    hidden_unit_count = encoder_activations.shape[1]

    if hidden_unit_count >= 2 and activation_sample_count >= 2:
        try:
            from sklearn.decomposition import PCA

            embedding_two_d = PCA(n_components=2).fit_transform(encoder_activations)
            figure, axis = pyplot.subplots(figsize=(7, 6))
            axis.scatter(embedding_two_d[:, 0], embedding_two_d[:, 1], alpha=0.55, s=16, c="darkblue")
            axis.set_title("Encoder activations (PCA 2D)")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(os.path.join(output_path, "encoder_pca.png"), dpi=150)
            pyplot.close(figure)
        except ImportError:
            pass

    if hidden_unit_count >= 2 and activation_sample_count >= 4:
        try:
            from sklearn.manifold import TSNE

            perplexity = float(min(30, max(2, activation_sample_count - 1)))
            embedding_tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                random_state=DATA_SPLIT_SEED,
            ).fit_transform(encoder_activations)
            figure, axis = pyplot.subplots(figsize=(7, 6))
            axis.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], alpha=0.55, s=16, c="darkgreen")
            axis.set_title(f"Encoder activations (t-SNE 2D, perplexity={perplexity:.0f})")
            axis.set_xlabel("t-SNE 1")
            axis.set_ylabel("t-SNE 2")
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(os.path.join(output_path, "encoder_tsne.png"), dpi=150)
            pyplot.close(figure)
        except ImportError:
            pass
        except ValueError:
            pass


def produce_all_visualizations(
    models: list[Model],
    output_dir: str,
    top_model: Model,
    train_pool: pandas.DataFrame,
    feature_cols: list[str],
) -> None:
    per_root = os.path.join(output_dir, "per_model")
    for m in models:
        sub = os.path.join(per_root, filesystem_safe_directory_name(m.model_name))
        produce_model_training_visualizations(m, sub)
    produce_final_model_evaluation_visualizations(models, os.path.join(output_dir, "final_comparison"))
    top_dir = os.path.join(output_dir, "top_model")
    produce_encoder_visualizations(top_model, top_dir, train_pool, feature_cols)


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
    produce_all_visualizations(trained, out_dir, top_final, train_pool, cols)

    print("Mean CV scores (primary: mean F1 across folds):")
    for m in trained:
        print(f"  {m.model_name}: F1={m.mean_cv_scores.get('f1', 0):.4f}")
    print(f"Top model: {top.model_name}")
    print(f"Holdout test: {top_final.test_scores}")


if __name__ == "__main__":
    main()
