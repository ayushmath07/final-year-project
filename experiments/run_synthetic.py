"""Run one complete calibrated synthetic reconstruction and save its evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import calibrate_threshold, precision_recall_f1, ranking_metrics
from src.network_builder import threshold_scores
from src.response_score import build_srcs_matrix
from src.synthetic_generator import SyntheticConfig, simulate
from src.visualization import save_figures


def main() -> None:
    config = SyntheticConfig(seed=42)
    development = [simulate(replace(config, seed=seed)) for seed in range(100, 110)]
    threshold = calibrate_threshold([build_srcs_matrix(d.states, d.events) for d in development],
                                    [d.adjacency for d in development])
    data = simulate(config)
    scores = build_srcs_matrix(data.states, data.events)
    inferred = threshold_scores(scores, threshold)
    metrics = {**precision_recall_f1(data.adjacency, inferred),
               **ranking_metrics(data.adjacency, scores), "threshold": threshold}
    output = ROOT / "reports" / "results" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data.states, columns=[f"C{i}" for i in range(config.companies)]).to_csv(
        output / "observations.csv", index_label="time")
    pd.DataFrame(data.events, columns=["time", "company", "signed_shock"]).to_csv(
        output / "events.csv", index=False)
    pd.DataFrame(scores).to_csv(output / "srcs_scores.csv", index=False)
    pd.DataFrame(inferred).to_csv(output / "inferred_adjacency.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_figures(data, ROOT / "reports" / "figures", inferred, scores)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
