"""
Evaluation utilities for classifier comparison and confusion-matrix analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for binary classification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return the 2×2 confusion matrix (cats vs dogs)."""
    return confusion_matrix(y_true, y_pred)


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison DataFrame from per-model metric dicts.

    Parameters
    ----------
    results : dict
        ``{"Model A": {"accuracy": …, "precision": …, …}, "Model B": …}``

    Returns
    -------
    pd.DataFrame with models as rows and metrics as columns,
    sorted descending by accuracy.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    return df.sort_values("accuracy", ascending=False)