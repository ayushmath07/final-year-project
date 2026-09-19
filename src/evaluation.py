"""Evaluation metrics for synthetic network reconstruction."""

from __future__ import annotations

import numpy as np


def confusion_counts(
    true_adjacency: np.ndarray,
    inferred_adjacency: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Compare the true and inferred adjacency matrices.

    Returns
    -------
    tuple
        (TP, FP, FN, TN)
    """

    if true_adjacency.shape != inferred_adjacency.shape:
        raise ValueError("Matrices must have the same shape.")

    # Ignore the diagonal because self-connections are not part
    # of the prototype network.
    mask = ~np.eye(
        true_adjacency.shape[0],
        dtype=bool,
    )

    true = true_adjacency[mask]
    inferred = inferred_adjacency[mask]

    tp = int(np.sum((true == 1) & (inferred == 1)))
    fp = int(np.sum((true == 0) & (inferred == 1)))
    fn = int(np.sum((true == 1) & (inferred == 0)))
    tn = int(np.sum((true == 0) & (inferred == 0)))

    return tp, fp, fn, tn


def precision_recall_f1(
    true_adjacency: np.ndarray,
    inferred_adjacency: np.ndarray,
) -> dict[str, float]:
    """
    Calculate precision, recall, and F1 for edge recovery.
    """

    tp, fp, fn, tn = confusion_counts(
        true_adjacency,
        inferred_adjacency,
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }