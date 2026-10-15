import numpy as np
import pytest

from src.ml_inference import (
    FEATURE_COLUMNS,
    build_pair_features,
    generate_diverse_corpus,
    labelled_pair_dataset,
    predict_edge_scores,
    train_ml_network_model,
)
from src.network_builder import threshold_scores
from src.synthetic_generator import SyntheticConfig, simulate


def test_pair_features_are_complete_finite_and_ground_truth_free():
    data = simulate(SyntheticConfig(companies=6, steps=120, shocks=15, seed=8))
    frame = build_pair_features(data.states, data.events)
    assert len(frame) == 15
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert "edge" not in frame.columns
    assert np.isfinite(frame.loc[:, FEATURE_COLUMNS].to_numpy()).all()


def test_pair_features_are_invariant_to_company_relabelling():
    data = simulate(SyntheticConfig(companies=6, steps=120, shocks=15, seed=18))
    original = build_pair_features(data.states, data.events)
    permutation = np.array([3, 0, 5, 1, 4, 2])
    inverse = np.argsort(permutation)
    relabelled_events = [(time, int(inverse[company]), magnitude)
                         for time, company, magnitude in data.events]
    relabelled = build_pair_features(data.states[:, permutation], relabelled_events)

    old_pair = original[(original.company_i == 1) & (original.company_j == 4)].iloc[0]
    new_i, new_j = sorted((int(inverse[1]), int(inverse[4])))
    new_pair = relabelled[(relabelled.company_i == new_i) &
                          (relabelled.company_j == new_j)].iloc[0]
    assert np.allclose(old_pair.loc[list(FEATURE_COLUMNS)].astype(float),
                       new_pair.loc[list(FEATURE_COLUMNS)].astype(float))


def test_labelled_dataset_counts_each_undirected_pair_once():
    datasets = [simulate(SyntheticConfig(companies=6, steps=120, shocks=15, seed=seed))
                for seed in (10, 11)]
    frame = labelled_pair_dataset(datasets)
    assert len(frame) == 30
    assert set(frame["edge"].unique()) == {0, 1}
    assert frame.groupby("seed").size().to_dict() == {10: 15, 11: 15}


def test_diverse_corpus_is_deterministic_and_changes_regimes():
    first = generate_diverse_corpus([1, 2])
    second = generate_diverse_corpus([1, 2])
    assert np.array_equal(first[0].states, second[0].states)
    assert first[0].config != first[1].config


def test_ml_pipeline_has_disjoint_splits_and_symmetric_probabilities():
    training = generate_diverse_corpus(range(0, 5))
    calibration = generate_diverse_corpus(range(100, 103))
    model, frame = train_ml_network_model(training, calibration)
    test = generate_diverse_corpus([200])[0]
    scores, pair_frame = predict_edge_scores(model.estimator, test.states, test.events)
    inferred = threshold_scores(scores, model.threshold)

    assert len(frame) > 100
    assert scores.shape == test.adjacency.shape
    assert np.allclose(scores, scores.T)
    assert np.allclose(np.diag(scores), 0)
    assert ((pair_frame["edge_probability"] >= 0) &
            (pair_frame["edge_probability"] <= 1)).all()
    assert set(np.unique(inferred)).issubset({0, 1})


def test_ml_pipeline_rejects_seed_leakage():
    shared = generate_diverse_corpus([7])
    with pytest.raises(ValueError, match="disjoint"):
        train_ml_network_model(shared, shared)
