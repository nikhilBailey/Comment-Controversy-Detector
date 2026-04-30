from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as numpy
import pandas as pandas
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABEL_COL = "is_bot_annotation"
TEXT_COL = "text"

ClassifierKind = Literal["random_forest", "adaboost", "gradient_boosting", "svm"]

SCALER_STEP = "scaler"
PCA_STEP = "pca"
CLASSIFIER_STEP = "classifier"


def extract_feature_matrix_and_labels(
    dataframe: pandas.DataFrame,
    feature_column_names: list[str],) -> tuple[numpy.ndarray, numpy.ndarray]:
    feature_matrix = dataframe[feature_column_names].to_numpy(dtype=numpy.float64)
    labels = dataframe[LABEL_COL].to_numpy(dtype=numpy.int64)
    return feature_matrix, labels


def build_classifier(
    classifier_kind: ClassifierKind,
    classifier_kwargs: dict[str, Any],
    random_state: int,
) -> Any:
    merged_kwargs = dict(classifier_kwargs)
    if classifier_kind == "random_forest":
        merged_kwargs.setdefault("random_state", random_state)
        return RandomForestClassifier(**merged_kwargs)
    if classifier_kind == "adaboost":
        merged_kwargs.setdefault("random_state", random_state)
        return AdaBoostClassifier(**merged_kwargs)
    if classifier_kind == "gradient_boosting":
        merged_kwargs.setdefault("random_state", random_state)
        return GradientBoostingClassifier(**merged_kwargs)
    if classifier_kind == "svm":
        merged_kwargs.setdefault("probability", True)
        merged_kwargs.setdefault("random_state", random_state)
        return SVC(**merged_kwargs)
    raise ValueError(f"Unknown classifier_kind {classifier_kind!r}")


def build_pipeline(
    classifier_kind: ClassifierKind,
    classifier_kwargs: dict[str, Any],
    pca_n_components: int,
    random_state: int,
) -> Pipeline:
    #We use standard scaler then PCA and then one of the four classifiers we chose
    classifier = build_classifier(classifier_kind, classifier_kwargs, random_state)
    return Pipeline(
        steps=[
            (SCALER_STEP, StandardScaler()),
            (PCA_STEP, PCA(n_components=pca_n_components, random_state=random_state)),
            (CLASSIFIER_STEP, classifier),
        ]
    )


@dataclass
class Model:
    #This hold a single one of our model templates

    model_name: str
    classifier_kind: ClassifierKind
    classifier_kwargs: dict[str, Any]
    pca_n_components: int
    pipeline: Pipeline | None = None
    metrics: list[dict[str, float]] = field(default_factory=list)
    final_cv_fold_scores: list[dict[str, float]] = field(default_factory=list)
    mean_cv_scores: dict[str, float] = field(default_factory=dict)
    test_scores: dict[str, float] = field(default_factory=dict)

    #This is for the ensemble methods
    @property
    def supports_staged_prediction(self) -> bool:
        return self.classifier_kind in ("gradient_boosting", "adaboost")
