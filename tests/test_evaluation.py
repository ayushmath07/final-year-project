import numpy as np

from src.evaluation import (
    confusion_counts,
    precision_recall_f1,
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
    assert metrics["loss"] == 0.0


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
    assert metrics["loss"] == 1.0


def test_accuracy_and_loss_are_complements():
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

    metrics = precision_recall_f1(true, inferred)

    assert metrics["accuracy"] == 1 / 3
    assert metrics["loss"] == 2 / 3
    assert metrics["accuracy"] + metrics["loss"] == 1.0
