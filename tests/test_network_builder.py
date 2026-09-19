import numpy as np

from src.network_builder import threshold_scores


def test_threshold_scores_creates_binary_matrix():
    scores = np.array([
        [0.0, 0.5, 1.5],
        [0.5, 0.0, 2.0],
        [1.5, 2.0, 0.0],
    ])

    inferred = threshold_scores(
        scores,
        threshold=1.0,
    )

    expected = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0],
    ])

    assert np.array_equal(inferred, expected)


def test_threshold_scores_has_no_self_connections():
    scores = np.ones((4, 4))

    inferred = threshold_scores(
        scores,
        threshold=0.5,
    )

    assert np.all(np.diag(inferred) == 0)