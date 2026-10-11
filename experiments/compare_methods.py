"""Calibrate and compare all network-reconstruction methods on shared datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import all_method_scores
from src.evaluation import calibrate_threshold, precision_recall_f1, ranking_metrics
from src.network_builder import threshold_scores
from src.synthetic_generator import SyntheticConfig, simulate


def run_comparison(output_dir: Path = ROOT / "reports" / "results" / "comparison",
                   dev_seeds: list[int] | None = None,
                   test_seeds: list[int] | None = None,
                   config: SyntheticConfig | None = None) -> pd.DataFrame:
    dev_seeds = dev_seeds or list(range(100, 110))
    test_seeds = test_seeds or list(range(200, 210))
    config = config or SyntheticConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = [simulate(replace(config, seed=seed)) for seed in dev_seeds]
    test = [simulate(replace(config, seed=seed)) for seed in test_seeds]
    dev_scores = [all_method_scores(d.states, d.events, d.config.seed) for d in dev]
    thresholds = {method: calibrate_threshold([s[method] for s in dev_scores],
                                               [d.adjacency for d in dev])
                  for method in dev_scores[0]}
    rows = []
    rankings = []
    for data in test:
        scores_by_method = all_method_scores(data.states, data.events, data.config.seed)
        pairs = [(i, j) for i in range(data.config.companies)
                 for j in range(i + 1, data.config.companies)]
        for method, scores in scores_by_method.items():
            inferred = threshold_scores(scores, thresholds[method])
            row = {"seed": data.config.seed, "method": method,
                   "threshold": thresholds[method]}
            row.update(precision_recall_f1(data.adjacency, inferred))
            row.update(ranking_metrics(data.adjacency, scores))
            rows.append(row)
            for rank, (i, j) in enumerate(sorted(pairs, key=lambda p: scores[p], reverse=True), 1):
                rankings.append({"seed": data.config.seed, "method": method, "rank": rank,
                                 "company_i": i, "company_j": j, "score": scores[i, j],
                                 "true_edge": int(data.adjacency[i, j])})
    raw = pd.DataFrame(rows)
    summary = raw.groupby("method", as_index=False).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"),
    ).sort_values("f1_mean", ascending=False)
    raw.to_csv(output_dir / "method_results_raw.csv", index=False)
    summary.to_csv(output_dir / "method_comparison_summary.csv", index=False)
    pd.DataFrame(rankings).to_csv(output_dir / "score_rankings.csv", index=False)
    (output_dir / "configuration.json").write_text(json.dumps({
        "config": config.metadata() if hasattr(config, "metadata") else config.__dict__,
        "dev_seeds": dev_seeds, "test_seeds": test_seeds, "thresholds": thresholds,
    }, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["method"], summary["f1_mean"], yerr=summary["f1_std"], capsize=4)
    ax.set(ylabel="F1 (mean ± SD)", xlabel="Method", ylim=(0, 1),
           title="Synthetic hidden-edge reconstruction on independent test seeds")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(output_dir / "method_comparison_f1.png", dpi=180); plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use 3 development and 3 test seeds")
    args = parser.parse_args()
    seeds = (list(range(100, 103)), list(range(200, 203))) if args.quick else (None, None)
    print(run_comparison(dev_seeds=seeds[0], test_seeds=seeds[1]).to_string(index=False))


if __name__ == "__main__":
    main()
