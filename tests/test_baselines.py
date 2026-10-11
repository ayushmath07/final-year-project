import numpy as np

from src.baselines import granger_scores, max_lagged_correlation_scores, pearson_scores, random_scores


def test_all_baselines_return_symmetric_finite_scores():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(80, 4))
    for scores in (random_scores(states, 1), pearson_scores(states),
                   max_lagged_correlation_scores(states), granger_scores(states)):
        assert scores.shape == (4, 4)
        assert np.allclose(scores, scores.T)
        assert np.allclose(np.diag(scores), 0)
        assert np.isfinite(scores).all()


def test_random_scores_are_deterministic():
    states = np.ones((20, 3))
    assert np.array_equal(random_scores(states, 4), random_scores(states, 4))
