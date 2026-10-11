from src.evaluation import calibrate_threshold, precision_recall_f1
from src.network_builder import threshold_scores
from src.response_score import build_srcs_matrix
from src.synthetic_generator import SyntheticConfig, simulate


def test_independent_calibration_and_inference_pipeline():
    development = [simulate(SyntheticConfig(seed=seed)) for seed in (11, 12)]
    threshold = calibrate_threshold([build_srcs_matrix(d.states, d.events) for d in development],
                                    [d.adjacency for d in development])
    test = simulate(SyntheticConfig(seed=90))
    inferred = threshold_scores(build_srcs_matrix(test.states, test.events), threshold)
    metrics = precision_recall_f1(test.adjacency, inferred)
    assert 0 <= metrics["f1"] <= 1
    assert metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == 45
