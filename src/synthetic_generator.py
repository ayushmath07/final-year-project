"""Deterministic synthetic company-network and shock-propagation simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    companies: int = 10
    steps: int = 260
    edge_probability: float = 0.22
    persistence: float = 0.20
    propagation: float = 0.55
    noise_std: float = 0.10
    shock_size: float = 2.5
    shocks: int = 30
    seed: int = 42
    min_shock_gap: int = 6
    event_margin: int = 8
    allow_overlapping_shocks: bool = False

    def validate(self) -> None:
        if self.companies < 2:
            raise ValueError("companies must be at least 2")
        if self.steps < 20:
            raise ValueError("steps must be at least 20")
        if not 0 <= self.edge_probability <= 1:
            raise ValueError("edge_probability must be in [0, 1]")
        if not 0 <= self.persistence < 1:
            raise ValueError("persistence must be in [0, 1)")
        if self.propagation < 0 or self.persistence + self.propagation >= 1:
            raise ValueError("require propagation >= 0 and persistence + propagation < 1")
        if self.noise_std < 0 or self.shock_size <= 0:
            raise ValueError("noise_std must be non-negative and shock_size positive")
        if self.shocks < 0 or self.min_shock_gap < 1 or self.event_margin < 1:
            raise ValueError("shocks must be non-negative; gaps and margins must be positive")


@dataclass
class SyntheticData:
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    states: np.ndarray
    shock_matrix: np.ndarray
    events: list[tuple[int, int, float]]
    config: SyntheticConfig

    def metadata(self) -> dict[str, int | float | bool]:
        return asdict(self.config)


def make_undirected_graph(
    companies: int, edge_probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a symmetric binary adjacency matrix with no self-connections."""
    if companies < 2 or not 0 <= edge_probability <= 1:
        raise ValueError("invalid graph parameters")
    upper = np.triu(rng.random((companies, companies)) < edge_probability, k=1)
    adjacency = (upper | upper.T).astype(float)
    if not adjacency.any():
        adjacency[0, 1] = adjacency[1, 0] = 1.0
    return adjacency


def row_normalize(adjacency: np.ndarray) -> np.ndarray:
    """Row-normalize a finite square non-negative adjacency matrix."""
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("adjacency must contain finite non-negative values")
    degrees = adjacency.sum(axis=1, keepdims=True)
    return np.divide(adjacency, degrees, out=np.zeros_like(adjacency), where=degrees > 0)


def _event_times(config: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    available = np.arange(config.event_margin, config.steps - config.event_margin)
    if config.allow_overlapping_shocks:
        if config.shocks > len(available):
            raise ValueError("too many shocks for the available time range")
        return np.sort(rng.choice(available, config.shocks, replace=False))
    candidates = available[:: config.min_shock_gap]
    if config.shocks > len(candidates):
        raise ValueError("too many non-overlapping shocks for steps, margin and min_shock_gap")
    return np.sort(rng.choice(candidates, config.shocks, replace=False))


def add_shocks(
    config: SyntheticConfig, rng: np.random.Generator, eligible: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Create deterministic signed shocks while respecting overlap policy."""
    shock_matrix = np.zeros((config.steps, config.companies), dtype=float)
    if config.shocks == 0:
        return shock_matrix, []
    if len(eligible) == 0:
        raise ValueError("at least one connected company is required for shocks")
    events: list[tuple[int, int, float]] = []
    for time in _event_times(config, rng):
        company = int(rng.choice(eligible))
        signed_size = float(config.shock_size * rng.choice((-1.0, 1.0)))
        shock_matrix[time, company] += signed_size
        events.append((int(time), company, signed_size))
    return shock_matrix, events


def simulate(config: SyntheticConfig = SyntheticConfig()) -> SyntheticData:
    """Generate a graph, event list and stable linear propagation process."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    adjacency = make_undirected_graph(config.companies, config.edge_probability, rng)
    normalized = row_normalize(adjacency)
    eligible = np.flatnonzero(adjacency.sum(axis=1) > 0)
    shock_matrix, events = add_shocks(config, rng, eligible)
    states = np.zeros((config.steps, config.companies), dtype=float)
    for time in range(1, config.steps):
        noise = rng.normal(0.0, config.noise_std, config.companies)
        states[time] = (
            config.persistence * states[time - 1]
            + config.propagation * (normalized @ states[time - 1])
            + shock_matrix[time]
            + noise
        )
    if not np.isfinite(states).all():
        raise FloatingPointError("simulation produced non-finite values")
    return SyntheticData(adjacency, normalized, states, shock_matrix, events, config)
