"""Leakage-aware machine-learning inference for hidden undirected edges.

The model never receives the hidden adjacency as an input feature.  Adjacency is
used only to construct labels for synthetic training/calibration/evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.baselines import granger_scores, max_lagged_correlation_scores, pearson_scores
from src.evaluation import calibrate_threshold
from src.response_score import (
    build_srcs_matrix,
    event_response_scores,
    standardize_states,
)
from src.synthetic_generator import SyntheticConfig, SyntheticData, simulate


FEATURE_COLUMNS = (
    "srcs",
    "pearson",
    "max_lagged_correlation",
    "granger_gain",
    "response_mean_min",
    "response_mean_max",
    "response_consistency_min",
    "response_consistency_max",
    "response_asymmetry",
    "event_coverage_min",
    "event_coverage_max",
    "event_count_balance",
    "raw_volatility_min",
    "raw_volatility_max",
    "mean_absolute_difference",
)


@dataclass
class EdgeMLModel:
    """Fitted edge classifier plus its independently calibrated threshold."""

    estimator: HistGradientBoostingClassifier
    threshold: float
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    train_seeds: tuple[int, ...] = ()
    calibration_seeds: tuple[int, ...] = ()


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _safe_consistency(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values >= threshold)) if values.size else 0.0


def build_pair_features(
    states: np.ndarray,
    events: list[tuple[int, int, float]],
    lag_window: int = 5,
    decay: float = 0.5,
    response_threshold: float = 1.0,
) -> pd.DataFrame:
    """Create one ground-truth-free feature row per undirected company pair."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < max(6, lag_window + 3) or states.shape[1] < 2:
        raise ValueError("states are too small for pair-feature extraction")
    if not np.isfinite(states).all():
        raise ValueError("states must be finite")
    if lag_window < 1 or decay < 0 or response_threshold < 0:
        raise ValueError("invalid event-response parameters")

    z_states = standardize_states(states)
    srcs = build_srcs_matrix(states, events, lag_window, decay, response_threshold)
    pearson = pearson_scores(states)
    lagged = max_lagged_correlation_scores(states, lag_window)
    granger = granger_scores(states, min(2, lag_window))
    event_counts = np.bincount(
        [int(company) for time, company, _ in events if 0 <= int(time) < len(states) - lag_window],
        minlength=states.shape[1],
    ).astype(float)
    total_events = max(float(event_counts.sum()), 1.0)
    volatility = states.std(axis=0)

    rows: list[dict[str, float | int]] = []
    for left in range(states.shape[1]):
        for right in range(left + 1, states.shape[1]):
            forward = event_response_scores(z_states, events, left, right, lag_window, decay)
            reverse = event_response_scores(z_states, events, right, left, lag_window, decay)
            mean_forward, mean_reverse = _safe_mean(forward), _safe_mean(reverse)
            consistency_forward = _safe_consistency(forward, response_threshold)
            consistency_reverse = _safe_consistency(reverse, response_threshold)
            coverage_left, coverage_right = event_counts[left] / total_events, event_counts[right] / total_events
            rows.append({
                "company_i": left,
                "company_j": right,
                "srcs": float(srcs[left, right]),
                "pearson": float(pearson[left, right]),
                "max_lagged_correlation": float(lagged[left, right]),
                "granger_gain": float(granger[left, right]),
                "response_mean_min": min(mean_forward, mean_reverse),
                "response_mean_max": max(mean_forward, mean_reverse),
                "response_consistency_min": min(consistency_forward, consistency_reverse),
                "response_consistency_max": max(consistency_forward, consistency_reverse),
                "response_asymmetry": abs(mean_forward - mean_reverse),
                "event_coverage_min": min(coverage_left, coverage_right),
                "event_coverage_max": max(coverage_left, coverage_right),
                "event_count_balance": 1.0 - abs(coverage_left - coverage_right),
                "raw_volatility_min": float(min(volatility[left], volatility[right])),
                "raw_volatility_max": float(max(volatility[left], volatility[right])),
                "mean_absolute_difference": float(np.mean(np.abs(z_states[:, left] - z_states[:, right]))),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.loc[:, FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("feature extraction produced empty or non-finite data")
    return frame


def labelled_pair_dataset(datasets: Iterable[SyntheticData]) -> pd.DataFrame:
    """Combine complete simulations without splitting pairs from the same seed."""
    frames = []
    for data in datasets:
        frame = build_pair_features(data.states, data.events)
        frame["edge"] = [
            int(data.adjacency[int(row.company_i), int(row.company_j)])
            for row in frame.itertuples(index=False)
        ]
        frame["seed"] = data.config.seed
        frames.append(frame)
    if not frames:
        raise ValueError("at least one synthetic dataset is required")
    result = pd.concat(frames, ignore_index=True)
    if result["edge"].nunique() != 2:
        raise ValueError("training data must contain both edges and non-edges")
    return result


def diversify_config(base: SyntheticConfig, seed: int) -> SyntheticConfig:
    """Deterministically sample a broader, still-stable synthetic regime."""
    rng = np.random.default_rng(seed + 7_919)
    persistence = float(rng.uniform(0.08, 0.30))
    propagation = float(rng.uniform(0.25, min(0.72, 0.94 - persistence)))
    max_shocks = len(np.arange(base.event_margin, base.steps - base.event_margin)[::base.min_shock_gap])
    return replace(
        base,
        companies=int(rng.integers(8, 16)),
        edge_probability=float(rng.uniform(0.12, 0.38)),
        persistence=persistence,
        propagation=propagation,
        noise_std=float(rng.uniform(0.04, 0.28)),
        shock_size=float(rng.uniform(1.8, 3.8)),
        shocks=int(rng.integers(max(8, min(15, max_shocks)), max_shocks + 1)),
        seed=int(seed),
    )


def generate_diverse_corpus(
    seeds: Iterable[int], base_config: SyntheticConfig | None = None
) -> list[SyntheticData]:
    """Generate domain-randomized networks for ML training or evaluation."""
    base = base_config or SyntheticConfig()
    return [simulate(diversify_config(base, int(seed))) for seed in seeds]


def fit_edge_classifier(
    training: Iterable[SyntheticData],
    random_state: int = 42,
) -> tuple[HistGradientBoostingClassifier, pd.DataFrame]:
    """Fit a regularized nonlinear classifier on synthetic pair features."""
    frame = labelled_pair_dataset(training)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=random_state,
    )
    estimator.fit(frame.loc[:, FEATURE_COLUMNS], frame["edge"])
    return estimator, frame


def predict_edge_scores(
    estimator: HistGradientBoostingClassifier,
    states: np.ndarray,
    events: list[tuple[int, int, float]],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return a symmetric matrix of learned edge probabilities."""
    frame = build_pair_features(states, events)
    probabilities = estimator.predict_proba(frame.loc[:, FEATURE_COLUMNS])[:, 1]
    matrix = np.zeros((states.shape[1], states.shape[1]), dtype=float)
    for row, probability in zip(frame.itertuples(index=False), probabilities):
        i, j = int(row.company_i), int(row.company_j)
        matrix[i, j] = matrix[j, i] = float(probability)
    frame = frame.copy()
    frame["edge_probability"] = probabilities
    return matrix, frame


def train_ml_network_model(
    training: Iterable[SyntheticData],
    calibration: Iterable[SyntheticData],
    random_state: int = 42,
) -> tuple[EdgeMLModel, pd.DataFrame]:
    """Fit on training networks and calibrate the edge cutoff on disjoint networks."""
    training = list(training)
    calibration = list(calibration)
    train_seeds = {item.config.seed for item in training}
    calibration_seeds = {item.config.seed for item in calibration}
    if not training or not calibration:
        raise ValueError("non-empty training and calibration datasets are required")
    if train_seeds & calibration_seeds:
        raise ValueError("training and calibration seeds must be disjoint")
    estimator, training_frame = fit_edge_classifier(training, random_state)
    calibration_scores = [predict_edge_scores(estimator, item.states, item.events)[0]
                          for item in calibration]
    threshold = calibrate_threshold(calibration_scores, [item.adjacency for item in calibration])
    model = EdgeMLModel(
        estimator=estimator,
        threshold=threshold,
        train_seeds=tuple(sorted(train_seeds)),
        calibration_seeds=tuple(sorted(calibration_seeds)),
    )
    return model, training_frame
