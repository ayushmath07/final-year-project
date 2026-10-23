"""Metrics and threshold calibration for undirected network reconstruction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _validate_binary_adjacency(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) != 0):
        raise ValueError("adjacency must be symmetric with a zero diagonal")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("adjacency must be binary")
    return matrix.astype(int)


def confusion_counts(true_adjacency: np.ndarray, inferred_adjacency: np.ndarray) -> tuple[int, int, int, int]:
    """Return TP, FP, FN and TN, counting each undirected pair exactly once."""
    true = _validate_binary_adjacency(true_adjacency)
    inferred = _validate_binary_adjacency(inferred_adjacency)
    if true.shape != inferred.shape:
        raise ValueError("matrices must have the same shape")
    y_true, y_pred = _upper_values(true), _upper_values(inferred)
    return (int(np.sum((y_true == 1) & (y_pred == 1))),
            int(np.sum((y_true == 0) & (y_pred == 1))),
            int(np.sum((y_true == 1) & (y_pred == 0))),
            int(np.sum((y_true == 0) & (y_pred == 0))))


def precision_recall_f1(true_adjacency: np.ndarray, inferred_adjacency: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn, tn = confusion_counts(true_adjacency, inferred_adjacency)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    loss = (fp + fn) / total
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "loss": loss}


def ranking_metrics(true_adjacency: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    true = _validate_binary_adjacency(true_adjacency)
    if scores.shape != true.shape or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and match adjacency")
    y_true, y_score = _upper_values(true), _upper_values(scores)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score))}


def calibrate_threshold(score_matrices: list[np.ndarray], true_adjacencies: list[np.ndarray]) -> float:
    """Choose the development-only threshold with best mean F1."""
    if not score_matrices or len(score_matrices) != len(true_adjacencies):
        raise ValueError("provide equal non-empty score and truth lists")
    candidates = np.unique(np.concatenate([_upper_values(np.asarray(s, float)) for s in score_matrices]))
    candidates = np.r_[np.nextafter(candidates.min(), -np.inf), candidates,
                       np.nextafter(candidates.max(), np.inf)]
    best = (-1.0, float(candidates[0]))
    from src.network_builder import threshold_scores
    for threshold in candidates:
        mean_f1 = float(np.mean([
            precision_recall_f1(truth, threshold_scores(scores, float(threshold)))["f1"]
            for scores, truth in zip(score_matrices, true_adjacencies)
        ]))
        if mean_f1 > best[0]:
            best = (mean_f1, float(threshold))
    return best[1]
