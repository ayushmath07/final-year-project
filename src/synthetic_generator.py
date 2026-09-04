"""Small, reproducible simulator for Step 1."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    companies: int = 8
    steps: int = 160
    edge_probability: float = 0.28
    persistence: float = 0.25
    propagation: float = 0.45
    noise_std: float = 0.08
    shock_size: float = 1.6
    shocks: int = 12
    seed: int = 42


@dataclass
class SyntheticData:
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    states: np.ndarray
    shock_matrix: np.ndarray
    events: list[tuple[int, int, float]]  # (time, company, signed shock)


def make_undirected_graph(companies: int, edge_probability: float, rng: np.random.Generator) -> np.ndarray:
    """Return a symmetric binary adjacency matrix with no self-connections."""
    upper = rng.random((companies, companies)) < edge_probability
    adjacency = np.triu(upper, k=1).astype(float)
    adjacency += adjacency.T
    return adjacency


def row_normalize(adjacency: np.ndarray) -> np.ndarray:
    """Normalize each non-empty row so high-degree nodes do not dominate."""
    degrees = adjacency.sum(axis=1, keepdims=True)
    return np.divide(adjacency, degrees, out=np.zeros_like(adjacency), where=degrees != 0)


def add_shocks(
    companies: int, steps: int, count: int, size: float, rng: np.random.Generator, eligible: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Create well-separated shocks on connected companies for a visible demo."""
    shock_matrix = np.zeros((steps, companies))
    possible_times = np.arange(8, steps - 8, 10)
    times = rng.choice(possible_times, size=min(count, len(possible_times)), replace=False)
    events = []
    for time in sorted(times):
        company = int(rng.choice(eligible))
        signed_size = size * int(rng.choice([-1, 1]))
        shock_matrix[time, company] = signed_size
        events.append((int(time), company, float(signed_size)))
    return shock_matrix, events


def simulate(config: SyntheticConfig = SyntheticConfig()) -> SyntheticData:
    """Generate graph, shocks, and states using the documented propagation model."""
    if config.companies < 2 or config.steps < 20:
        raise ValueError("Use at least 2 companies and 20 time steps.")

    rng = np.random.default_rng(config.seed)
    adjacency = make_undirected_graph(config.companies, config.edge_probability, rng)
    if not adjacency.any():  # Keeps an unusually sparse random draw demonstrable.
        adjacency[0, 1] = adjacency[1, 0] = 1
    normalized = row_normalize(adjacency)
    connected = np.flatnonzero(adjacency.sum(axis=1))
    shock_matrix, events = add_shocks(
        config.companies, config.steps, config.shocks, config.shock_size, rng, connected
    )

    states = np.zeros((config.steps, config.companies))
    for time in range(1, config.steps):
        noise = rng.normal(0, config.noise_std, config.companies)
        states[time] = (
            config.persistence * states[time - 1]
            + config.propagation * (normalized @ states[time - 1])
            + shock_matrix[time]
            + noise
        )

    return SyntheticData(adjacency, normalized, states, shock_matrix, events)

