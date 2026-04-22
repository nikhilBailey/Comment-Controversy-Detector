"""Tabular classifier network and training-time model container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as numpy
import pandas as pandas
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from sklearn.preprocessing import StandardScaler

LABEL_COL = "is_bot_annotation"
TEXT_COL = "text"


def extract_feature_matrix_and_labels(
    dataframe: pandas.DataFrame,
    feature_column_names: list[str],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    feature_matrix = dataframe[feature_column_names].to_numpy(dtype=numpy.float64)
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
