"""Convert pair-score matrices into undirected inferred networks."""

import numpy as np


def threshold_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    symmetric = (scores + scores.T) / 2.0
    inferred = (symmetric >= threshold).astype(int)
    np.fill_diagonal(inferred, 0)
    return inferred


def top_k_scores(scores: np.ndarray, edge_count: int) -> np.ndarray:
    """Select exactly edge_count undirected pairs by score."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite square matrix")
    pairs = np.triu_indices(scores.shape[0], k=1)
    if not 0 <= edge_count <= len(pairs[0]):
        raise ValueError("edge_count is outside the available pair range")
    adjacency = np.zeros_like(scores, dtype=int)
    if edge_count:
        order = np.argsort(scores[pairs], kind="stable")[-edge_count:]
        adjacency[pairs[0][order], pairs[1][order]] = 1
        adjacency += adjacency.T
    return adjacency
