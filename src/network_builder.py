"""Convert SRCS scores into an inferred network."""

import numpy as np


def threshold_scores(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert a symmetric SRCS score matrix into
    a binary undirected adjacency matrix.
    """

    inferred = (scores >= threshold).astype(int)

    # No self-connections.
    np.fill_diagonal(inferred, 0)

    # Keep the prototype undirected.
    inferred = np.maximum(inferred, inferred.T)

    return inferred