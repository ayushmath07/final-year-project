import numpy as np

from src.synthetic_generator import SyntheticConfig, row_normalize, simulate


def test_graph_is_undirected_and_has_no_self_connections():
    data = simulate()
    assert np.array_equal(data.adjacency, data.adjacency.T)
    assert np.all(np.diag(data.adjacency) == 0)


def test_row_normalization_preserves_non_empty_rows():
    normalized = row_normalize(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.allclose(normalized.sum(axis=1), [1.0, 1.0])


def test_same_seed_creates_identical_simulation():
    config = SyntheticConfig(seed=9)
    first, second = simulate(config), simulate(config)
    assert np.array_equal(first.adjacency, second.adjacency)
    assert np.array_equal(first.states, second.states)
    assert first.events == second.events


def test_simulation_has_expected_shape_and_events():
    config = SyntheticConfig(companies=6, steps=80, shocks=4)
    data = simulate(config)
    assert data.states.shape == (80, 6)
    assert len(data.events) == 4
    assert data.shock_matrix.shape == (80, 6)

