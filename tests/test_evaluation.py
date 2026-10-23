import numpy as np

from src.evaluation import (
    confusion_counts,
    precision_recall_f1,
    probability_log_loss,
)


def test_confusion_counts():
    true = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    inferred = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ])

    tp, fp, fn, tn = confusion_counts(
        true,
        inferred,
    )

    assert tp == 1
    assert fp == 1
    assert fn == 1
    assert tn == 0


def test_perfect_reconstruction_has_perfect_metrics():
    true = np.array([
        [0, 1],
        [1, 0],
    ])

    metrics = precision_recall_f1(
        true,
        true,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_no_predicted_edges_have_zero_recall():
    true = np.array([
        [0, 1],
        [1, 0],
    ])

    inferred = np.zeros((2, 2))

    metrics = precision_recall_f1(
        true,
        inferred,
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 0.0


def test_probability_log_loss_rewards_correct_probabilities():
    true = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    good = np.array([
        [0.0, 0.9, 0.1],
        [0.9, 0.0, 0.8],
        [0.1, 0.8, 0.0],
    ])
    bad = 1.0 - good
    np.fill_diagonal(bad, 0.0)

    assert probability_log_loss(true, good) < probability_log_loss(true, bad)
