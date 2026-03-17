"""Metrics utilities for model evaluation."""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Optional


def compute_metrics_from_cm(
    cm: np.ndarray,
    as_percentage: bool = False,
) -> Dict[str, float]:
    """
    Compute classification metrics from confusion matrix.

    Args:
        cm: Confusion matrix (n_classes x n_classes).
        as_percentage: Whether to return values as percentages.

    Returns:
        Dictionary with Accuracy, Precision, Recall, F1-Score.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.nan_to_num(tp / (tp + fp + 1e-8))
    recall = np.nan_to_num(tp / (tp + fn + 1e-8))
    f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall + 1e-8))
    accuracy = tp.sum() / cm.sum()

    scale = 100 if as_percentage else 1
    return {
        "accuracy": accuracy * scale,
        "precision": np.mean(precision) * scale,
        "recall": np.mean(recall) * scale,
        "f1_score": np.mean(f1) * scale,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict:
    """
    Compute classification metrics from predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes (auto-detected if None).

    Returns:
        Dictionary with metrics and confusion matrices.
    """
    if num_classes is None:
        num_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    # Multiclass confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Binary confusion matrix (class 0 vs rest)
    binary_true = (np.array(y_true) != 0).astype(int)
    binary_pred = (np.array(y_pred) != 0).astype(int)
    binary_cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])

    # Metrics
    metrics = compute_metrics_from_cm(cm)
    metrics["multiclass_cm"] = cm.tolist()
    metrics["binary_cm"] = binary_cm.tolist()

    return metrics
