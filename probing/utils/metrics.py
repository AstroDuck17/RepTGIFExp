"""
utils/metrics.py

Evaluation metrics for binary classification probing experiments.
All functions operate on numpy arrays for compatibility with both
PyTorch training loops and sklearn-based analysis.
"""

from __future__ import annotations
import numpy as np
from typing import Dict

# NumPy 2.0 renamed np.trapz → np.trapezoid (np.trapz removed entirely).
# This shim works on both NumPy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
if _trapezoid is None:
    raise ImportError("NumPy has neither np.trapezoid nor np.trapz — upgrade NumPy.")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy = (TPR + TNR) / 2.
    Random baseline = 0.50 for any class distribution.

    Args:
        y_true: Binary ground-truth labels (0 or 1), shape [N].
        y_pred: Binary predictions (0 or 1), shape [N].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    tpr = (y_pred[pos_mask] == 1).mean() if pos_mask.any() else 0.0
    tnr = (y_pred[neg_mask] == 0).mean() if neg_mask.any() else 0.0

    return float((tpr + tnr) / 2.0)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard accuracy = correct / total."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float((y_true == y_pred).mean())


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Area under the ROC curve, computed without sklearn.

    Args:
        y_true: Binary ground-truth labels (0 or 1), shape [N].
        y_prob: Predicted probabilities for class 1, shape [N].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate case

    # Stable sort by descending probability (ties broken by original order)
    sort_idx = np.argsort(-y_prob, kind="stable")
    y_true_sorted = y_true[sort_idx]

    # Build cumulative TPR and FPR curves
    tpr = np.cumsum(y_true_sorted) / n_pos
    fpr = np.cumsum(1 - y_true_sorted) / n_neg

    # Prepend (0, 0) origin for proper AUC integration
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Integrate TPR w.r.t. FPR — compatible with NumPy 1.x and 2.x
    auc = float(_trapezoid(tpr, fpr))

    # Clamp to [0, 1] to guard against floating-point drift
    return float(np.clip(auc, 0.0, 1.0))


def bce_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """
    Binary cross-entropy loss.

    Args:
        y_true: Binary ground-truth labels (0 or 1), shape [N].
        y_prob: Predicted probabilities for class 1, shape [N].
        eps: Small constant for numerical stability.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_all_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics from logits.

    Args:
        y_true: Binary ground-truth labels (0 or 1), shape [N].
        logits: Raw model output (before sigmoid), shape [N].
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dictionary with keys: accuracy, balanced_acc, roc_auc, bce_loss.
    """
    y_prob = sigmoid(logits)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy":     accuracy(y_true, y_pred),
        "balanced_acc": balanced_accuracy(y_true, y_pred),
        "roc_auc":      roc_auc(y_true, y_prob),
        "bce_loss":     bce_loss(y_true, y_prob),
    }
