
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy as numpy
import pandas as pandas
from sklearn.inspection import permutation_importance

from Model import (
    CLASSIFIER_STEP,
    PCA_STEP,
    SCALER_STEP,
    Model,
    extract_feature_matrix_and_labels,
)


class Visualizer:
    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    @staticmethod
    def filesystem_safe_directory_name(display_name: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in display_name
        )

    def produce_model_training_visualizations(self, model: Model, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        safe_name = self.filesystem_safe_directory_name(model.model_name)
        metric_keys = ["accuracy", "precision", "recall", "f1"]

        if not model.metrics:
            if not model.mean_cv_scores:
                return
            figure, axis = pyplot.subplots(figsize=(8, 5))
            values = [model.mean_cv_scores.get(key, 0.0) for key in metric_keys]
            axis.bar(metric_keys, values, color="steelblue", edgecolor="black", linewidth=0.4)
            axis.set_ylim(0.0, 1.02)
            axis.set_ylabel("score (mean CV)")
            axis.set_title(f"Final mean CV metrics — {safe_name} (no staged trajectory)")
            axis.grid(True, axis="y", alpha=0.3)
            for tick_label in axis.get_xticklabels():
                tick_label.set_rotation(20)
                tick_label.set_horizontalalignment("right")
            figure.tight_layout()
            figure.savefig(os.path.join(output_path, "final_mean_cv_metrics.png"), dpi=150)
            pyplot.close(figure)
            return

        epoch_numbers = numpy.array([float(row["epoch"]) for row in model.metrics])

        figure, axes = pyplot.subplots(2, 2, figsize=(10, 7))
        axes = axes.flatten()
        for axis, metric_key in zip(axes, metric_keys):
            metric_values = [row[metric_key] for row in model.metrics]
            axis.plot(epoch_numbers, metric_values, marker="o", markersize=2)
            axis.set_title(metric_key)
            axis.set_xlabel("n_estimators (staged)")
            axis.grid(True, alpha=0.3)
        figure.suptitle(f"Staged training metrics — {safe_name}")
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
        axis.set_xlabel("n_estimators (staged)")
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
        axis.set_title(f"Precision vs recall across staged checkpoints — {safe_name}")
        axis.grid(True, alpha=0.3)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        figure.colorbar(scatter, ax=axis, label="n_estimators")
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
            axis.set_title(f"Metric correlation across staged checkpoints — {safe_name}")
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
        axis.bar(bar_x - bar_width / 2, first_values, bar_width, label=f"first stage (n_est {int(first_row['epoch'])})")
        axis.bar(bar_x + bar_width / 2, last_values, bar_width, label=f"last stage (n_est {int(last_row['epoch'])})")
        axis.set_xticks(bar_x)
        axis.set_xticklabels(metric_keys, rotation=20, ha="right")
        axis.set_ylabel("score")
        axis.set_title(f"First vs last staged evaluation — {safe_name}")
        axis.legend()
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join(output_path, "first_vs_last_logged_metrics.png"), dpi=150)
        pyplot.close(figure)

    def produce_final_model_evaluation_visualizations(self, models: list[Model], output_path: str) -> None:
        if not models:
            return
        os.makedirs(output_path, exist_ok=True)
        names = [self.filesystem_safe_directory_name(trained_model.model_name) for trained_model in models]
        metric_keys = ["accuracy", "precision", "recall", "f1"]
        x_positions = numpy.arange(len(names))
        bar_width = 0.18
        figure, axis = pyplot.subplots(figsize=(max(10, len(names) * 1.2), 6))
        for index, metric_key in enumerate(metric_keys):
            values = [trained_model.mean_cv_scores.get(metric_key, 0.0) for trained_model in models]
            axis.bar(x_positions + (index - 1.5) * bar_width, values, bar_width, label=metric_key)
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

        figure, axes = pyplot.subplots(2, 2, figsize=(11, 8))
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

    def produce_pca_and_classifier_visualizations(
        self,
        model: Model,
        output_path: str,
        sample_df: pandas.DataFrame,
        feature_cols: list[str],) -> None:
        if model.pipeline is None:
            return
        os.makedirs(output_path, exist_ok=True)

        pca = model.pipeline.named_steps[PCA_STEP]
        scaler = model.pipeline.named_steps[SCALER_STEP]
        classifier = model.pipeline.named_steps[CLASSIFIER_STEP]

        explained_variance_ratio = numpy.asarray(pca.explained_variance_ratio_)
        cumulative_explained_variance = numpy.cumsum(explained_variance_ratio)
        component_indices = numpy.arange(1, len(explained_variance_ratio) + 1)
        figure, axis = pyplot.subplots(figsize=(9, 5))
        axis.bar(
            component_indices,
            explained_variance_ratio,
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
            label="per-component",
        )
        axis_secondary = axis.twinx()
        axis_secondary.plot(
            component_indices,
            cumulative_explained_variance,
            color="darkorange",
            marker="o",
            label="cumulative",
        )
        axis_secondary.set_ylim(0.0, 1.02)
        axis.set_xlabel("PCA component")
        axis.set_ylabel("explained variance ratio")
        axis_secondary.set_ylabel("cumulative explained variance")
        axis.set_title("PCA explained variance (scree)")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join(output_path, "pca_explained_variance.png"), dpi=150)
        pyplot.close(figure)

        loadings = numpy.asarray(pca.components_)
        figure, axis = pyplot.subplots(
            figsize=(max(8, len(feature_cols) * 0.55), max(4.5, loadings.shape[0] * 0.45))
        )
        color_mesh = axis.imshow(loadings, aspect="auto", cmap="coolwarm", interpolation="nearest")
        axis.set_xticks(numpy.arange(len(feature_cols)))
        axis.set_xticklabels(feature_cols, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(numpy.arange(loadings.shape[0]))
        axis.set_yticklabels([f"PC{idx + 1}" for idx in range(loadings.shape[0])])
        axis.set_title("PCA component loadings (rows = components, cols = input features)")
        figure.colorbar(color_mesh, ax=axis, label="loading")
        figure.tight_layout()
        figure.savefig(os.path.join(output_path, "pca_component_loadings.png"), dpi=150)
        pyplot.close(figure)

        sample_row_count = min(512, len(sample_df))
        sample_dataframe = sample_df.iloc[:sample_row_count]
        sample_features, sample_labels = extract_feature_matrix_and_labels(sample_dataframe, feature_cols)
        sample_in_pca_space = pca.transform(scaler.transform(sample_features))
        if sample_in_pca_space.shape[1] >= 2:
            figure, axis = pyplot.subplots(figsize=(7, 6))
            for label_value, color, marker in [(0, "steelblue", "o"), (1, "crimson", "^")]:
                mask = sample_labels.astype(int) == label_value
                axis.scatter(
                    sample_in_pca_space[mask, 0],
                    sample_in_pca_space[mask, 1],
                    s=22,
                    alpha=0.7,
                    c=color,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.4,
                    label=f"label={label_value}",
                )
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.set_title("Training-pool sample projected to PC1 × PC2")
            axis.legend(loc="best")
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(os.path.join(output_path, "pca_pc1_pc2_by_label.png"), dpi=150)
            pyplot.close(figure)

        component_labels = [f"PC{idx + 1}" for idx in range(loadings.shape[0])]
        component_importance: numpy.ndarray | None = None
        importance_kind = "permutation"
        if hasattr(classifier, "feature_importances_"):
            component_importance = numpy.asarray(classifier.feature_importances_)
            importance_kind = "native (impurity-based)"
        else:
            try:
                permutation_result_pca = permutation_importance(
                    classifier,
                    sample_in_pca_space,
                    sample_labels.astype(int),
                    n_repeats=10,
                    random_state=self._random_state,
                    n_jobs=1,
                )
                component_importance = numpy.asarray(permutation_result_pca.importances_mean)
            except Exception:
                component_importance = None

        if component_importance is not None and len(component_importance) == len(component_labels):
            figure, axis = pyplot.subplots(figsize=(max(8, len(component_labels) * 0.45), 5))
            axis.bar(
                numpy.arange(len(component_labels)),
                component_importance,
                color="coral",
                edgecolor="black",
                linewidth=0.3,
            )
            axis.set_xticks(numpy.arange(len(component_labels)))
            axis.set_xticklabels(component_labels, rotation=35, ha="right")
            axis.set_ylabel("importance")
            axis.set_title(f"Classifier importance per PCA component ({importance_kind})")
            axis.grid(True, axis="y", alpha=0.3)
            figure.tight_layout()
            figure.savefig(os.path.join(output_path, "classifier_pca_component_importance.png"), dpi=150)
            pyplot.close(figure)

        if component_importance is not None and len(component_importance) == loadings.shape[0]:
            attribution_per_input_feature = numpy.abs(loadings).T @ numpy.abs(component_importance)
            figure, axis = pyplot.subplots(figsize=(max(8, len(feature_cols) * 0.45), 5))
            axis.bar(
                numpy.arange(len(feature_cols)),
                attribution_per_input_feature,
                color="teal",
                edgecolor="black",
                linewidth=0.3,
            )
            axis.set_xticks(numpy.arange(len(feature_cols)))
            axis.set_xticklabels(feature_cols, rotation=55, ha="right", fontsize=8)
            axis.set_ylabel("|loading| × |component importance|")
            axis.set_title("Input feature attribution (via PCA loadings × component importance)")
            axis.grid(True, axis="y", alpha=0.3)
            figure.tight_layout()
            figure.savefig(os.path.join(output_path, "classifier_input_feature_attribution.png"), dpi=150)
            pyplot.close(figure)

        if sample_in_pca_space.shape[1] >= 2:
            try:
                from sklearn.base import clone

                two_d_features = sample_in_pca_space[:, :2]
                two_d_labels = sample_labels.astype(int)
                two_d_classifier = clone(classifier)
                two_d_classifier.fit(two_d_features, two_d_labels)
                grid_axis_one_min = float(two_d_features[:, 0].min()) - 0.5
                grid_axis_one_max = float(two_d_features[:, 0].max()) + 0.5
                grid_axis_two_min = float(two_d_features[:, 1].min()) - 0.5
                grid_axis_two_max = float(two_d_features[:, 1].max()) + 0.5
                grid_axis_one_values, grid_axis_two_values = numpy.meshgrid(
                    numpy.linspace(grid_axis_one_min, grid_axis_one_max, 200),
                    numpy.linspace(grid_axis_two_min, grid_axis_two_max, 200),
                )
                grid_points = numpy.column_stack(
                    [grid_axis_one_values.ravel(), grid_axis_two_values.ravel()]
                )
                if hasattr(two_d_classifier, "predict_proba"):
                    classes = list(two_d_classifier.classes_)
                    positive_column_index = classes.index(1) if 1 in classes else -1
                    grid_scores = two_d_classifier.predict_proba(grid_points)[:, positive_column_index]
                else:
                    grid_scores = two_d_classifier.predict(grid_points).astype(float)
                grid_scores = grid_scores.reshape(grid_axis_one_values.shape)

                figure, axis = pyplot.subplots(figsize=(7.5, 6.5))
                contour = axis.contourf(
                    grid_axis_one_values,
                    grid_axis_two_values,
                    grid_scores,
                    levels=20,
                    cmap="coolwarm",
                    alpha=0.7,
                )
                figure.colorbar(contour, ax=axis, label="P(label=1) (2D proxy)")
                for label_value, color, marker in [(0, "navy", "o"), (1, "darkred", "^")]:
                    mask = two_d_labels == label_value
                    axis.scatter(
                        two_d_features[mask, 0],
                        two_d_features[mask, 1],
                        s=18,
                        alpha=0.8,
                        c=color,
                        marker=marker,
                        edgecolors="white",
                        linewidths=0.4,
                        label=f"label={label_value}",
                    )
                axis.set_xlabel("PC1")
                axis.set_ylabel("PC2")
                axis.set_title("Decision surface in PC1 × PC2 (2D proxy refit of top classifier)")
                axis.legend(loc="best")
                figure.tight_layout()
                figure.savefig(os.path.join(output_path, "classifier_decision_surface_pc1_pc2.png"), dpi=150)
                pyplot.close(figure)
            except Exception:
                pass

    def produce_all_visualizations(
        self,
        models: list[Model],
        output_dir: str,
        top_model: Model,
        train_pool: pandas.DataFrame,
        feature_cols: list[str],
    ) -> None:
        per_root = os.path.join(output_dir, "per_model")
        for m in models:
            sub = os.path.join(per_root, self.filesystem_safe_directory_name(m.model_name))
            self.produce_model_training_visualizations(m, sub)
        self.produce_final_model_evaluation_visualizations(
            models, os.path.join(output_dir, "final_comparison")
        )
        top_dir = os.path.join(output_dir, "top_model")
        self.produce_pca_and_classifier_visualizations(top_model, top_dir, train_pool, feature_cols)
