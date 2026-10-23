"""Repeated SRCS robustness experiments with development-only calibration."""

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

from src.evaluation import calibrate_threshold, precision_recall_f1, ranking_metrics
from src.network_builder import threshold_scores
from src.response_score import build_srcs_matrix
from src.synthetic_generator import SyntheticConfig, simulate


PRIMARY = {
    "noise_std": [0.05, 0.10, 0.20, 0.35],
    "propagation": [0.20, 0.35, 0.55, 0.70],
    "shocks": [10, 20, 30, 40],
}


def run_robustness(output_dir: Path = ROOT / "reports" / "results" / "robustness",
                   seeds_per_setting: int = 10) -> pd.DataFrame:
    if seeds_per_setting < 1:
        raise ValueError("seeds_per_setting must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = SyntheticConfig()
    rows = []
    for experiment, values in PRIMARY.items():
        for value_index, value in enumerate(values):
            scores, truths, test_items = [], [], []
            for offset in range(seeds_per_setting):
                dev_cfg = replace(base, **{experiment: value}, seed=1000 + value_index * 100 + offset)
                test_cfg = replace(base, **{experiment: value}, seed=5000 + value_index * 100 + offset)
                dev, test = simulate(dev_cfg), simulate(test_cfg)
                scores.append(build_srcs_matrix(dev.states, dev.events))
                truths.append(dev.adjacency)
                test_items.append(test)
            threshold = calibrate_threshold(scores, truths)
            for data in test_items:
                test_scores = build_srcs_matrix(data.states, data.events)
                metrics = precision_recall_f1(data.adjacency, threshold_scores(test_scores, threshold))
                metrics.update(ranking_metrics(data.adjacency, test_scores))
                rows.append({"experiment": experiment, "value": value, "seed": data.config.seed,
                             "threshold": threshold, **metrics})
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["experiment", "value"], as_index=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        accuracy_mean=("accuracy", "mean"), loss_mean=("loss", "mean"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"))
    raw.to_csv(output_dir / "robustness_raw.csv", index=False)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    (output_dir / "configuration.json").write_text(json.dumps({
        "base_config": base.__dict__, "settings": PRIMARY,
        "seeds_per_setting": seeds_per_setting,
        "seed_policy": "development 1000-series; independent test 5000-series",
    }, indent=2) + "\n", encoding="utf-8")
    labels = {"noise_std": "Noise standard deviation",
              "propagation": "Propagation strength", "shocks": "Shock count"}
    for experiment, label in labels.items():
        subset = summary[summary.experiment == experiment]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.errorbar(subset.value, subset.f1_mean, yerr=subset.f1_std, marker="o", capsize=4)
        ax.set(xlabel=label, ylabel="F1 (mean ± SD)", ylim=(0, 1),
               title=f"SRCS robustness to {label.lower()}")
        ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(output_dir / f"f1_vs_{experiment}.png", dpi=180); plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    count = 3 if args.quick else args.seeds
    print(run_robustness(seeds_per_setting=count).to_string(index=False))


if __name__ == "__main__":
    main()
