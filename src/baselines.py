"""Comparable score matrices for network-reconstruction baselines."""

from __future__ import annotations

import numpy as np


def _check(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 6 or states.shape[1] < 2:
        raise ValueError("states must have at least 6 rows and 2 columns")
    if not np.isfinite(states).all():
        raise ValueError("states must be finite")
    return states


def random_scores(states: np.ndarray, seed: int = 0) -> np.ndarray:
    states = _check(states)
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((states.shape[1], states.shape[1])), 1)
    return upper + upper.T


def pearson_scores(states: np.ndarray) -> np.ndarray:
    states = _check(states)
    scores = np.abs(np.corrcoef(states, rowvar=False))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)
    return scores


def max_lagged_correlation_scores(states: np.ndarray, max_lag: int = 5) -> np.ndarray:
    states = _check(states)
    if not 1 <= max_lag < len(states) - 2:
        raise ValueError("max_lag is outside the usable range")
    n = states.shape[1]
    scores = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            values = []
            for lag in range(1, max_lag + 1):
                for left, right in ((states[:-lag, i], states[lag:, j]),
                                    (states[:-lag, j], states[lag:, i])):
                    if left.std() > 0 and right.std() > 0:
                        values.append(abs(float(np.corrcoef(left, right)[0, 1])))
            scores[i, j] = scores[j, i] = max(values, default=0.0)
    return scores


def _predictive_gain(target: np.ndarray, source: np.ndarray, max_lag: int) -> float:
    y = target[max_lag:]
    own = np.column_stack([target[max_lag - lag:-lag] for lag in range(1, max_lag + 1)])
    cross = np.column_stack([source[max_lag - lag:-lag] for lag in range(1, max_lag + 1)])
    intercept = np.ones((len(y), 1))
    restricted = np.column_stack([intercept, own])
    unrestricted = np.column_stack([intercept, own, cross])
    rss_r = float(np.sum((y - restricted @ np.linalg.lstsq(restricted, y, rcond=None)[0]) ** 2))
    rss_u = float(np.sum((y - unrestricted @ np.linalg.lstsq(unrestricted, y, rcond=None)[0]) ** 2))
    return max(0.0, (rss_r - rss_u) / max(rss_r, np.finfo(float).eps))


def granger_scores(states: np.ndarray, max_lag: int = 2) -> np.ndarray:
    """Pairwise predictive improvement, symmetrized by maximum direction."""
    states = _check(states)
    if not 1 <= max_lag < (len(states) - 2) // 2:
        raise ValueError("max_lag is outside the usable range")
    n = states.shape[1]
    scores = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            score = max(_predictive_gain(states[:, j], states[:, i], max_lag),
                        _predictive_gain(states[:, i], states[:, j], max_lag))
            scores[i, j] = scores[j, i] = score
    return scores


def all_method_scores(states: np.ndarray, events: list[tuple[int, int, float]], seed: int = 0,
                      max_lag: int = 5) -> dict[str, np.ndarray]:
    from src.response_score import build_srcs_matrix
    return {
        # Do not reuse the simulator seed stream: identical first draws would make
        # random scores mechanically dependent on edge generation.
        "Random": random_scores(states, seed + 1_000_003),
        "Pearson": pearson_scores(states),
        "Lagged correlation": max_lagged_correlation_scores(states, max_lag),
        "Granger-style": granger_scores(states, min(2, max_lag)),
        "SRCS": build_srcs_matrix(states, events, max_lag),
    }
