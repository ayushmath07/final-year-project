"""Run SRCS candidate ranking on cached or uploaded genuine prices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_price_csv
from src.preprocessing import preprocess_prices
from src.response_score import build_srcs_matrix, event_evidence_table
from src.shock_detector import detect_shocks, to_srcs_events


def run_real_data(price_path: Path, market: str = "SPY", threshold: float = 2.5,
                  cooldown: int = 5, event_window: int = 5,
                  output_dir: Path = ROOT / "reports" / "results" / "real_data") -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_price_csv(price_path)
    abnormal, standardized = preprocess_prices(prices, market)
    events = detect_shocks(standardized, threshold, cooldown, event_window)
    if events.empty:
        raise RuntimeError("no shocks detected; lower threshold or provide a longer dataset")
    srcs_events = to_srcs_events(events)
    scores = build_srcs_matrix(standardized.to_numpy(), srcs_events, event_window)
    pairs = []
    columns = list(standardized.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            pairs.append({"company_a": columns[i], "company_b": columns[j], "score": scores[i, j],
                          "events_a": int((events.company_index == i).sum()),
                          "events_b": int((events.company_index == j).sum())})
    ranking = pd.DataFrame(pairs).sort_values("score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    evidence = []
    case_pairs = pd.concat([ranking.head(2), ranking.tail(1)]).copy()
    case_pairs["case_type"] = ["high-ranked", "high-ranked", "ambiguous-low-ranked"]
    for row in case_pairs.itertuples(index=False):
        i, j = columns.index(row.company_a), columns.index(row.company_b)
        for origin, responder in ((i, j), (j, i)):
            for item in event_evidence_table(standardized.to_numpy(), srcs_events, origin, responder,
                                             event_window):
                item.update({"origin_ticker": columns[origin], "responder_ticker": columns[responder],
                             "case_type": row.case_type})
                evidence.append(item)
    events.to_csv(output_dir / "shock_events.csv", index=False)
    ranking.to_csv(output_dir / "ranked_candidate_connections.csv", index=False)
    pd.DataFrame(evidence).to_csv(output_dir / "event_level_evidence.csv", index=False)
    standardized.to_csv(output_dir / "standardized_abnormal_returns.csv", index_label="Date")
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(scores, cmap="viridis")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    ax.set_title("Real-data SRCS candidate-connection scores")
    fig.colorbar(image, ax=ax, label="SRCS")
    fig.tight_layout(); fig.savefig(output_dir / "real_data_srcs_heatmap.png", dpi=180); plt.close(fig)
    for case_index, pair in enumerate(case_pairs.itertuples(index=False), 1):
        fig, ax = plt.subplots(figsize=(9, 4))
        standardized[[pair.company_a, pair.company_b]].plot(ax=ax)
        pair_events = events[events.company.isin([pair.company_a, pair.company_b])]
        for event in pair_events.itertuples(index=False):
            ax.axvline(event.date, color="red", alpha=0.12)
        ax.set(title=f"Case {case_index}: {pair.company_a} and {pair.company_b} ({pair.case_type})",
               ylabel="Abnormal return z-score")
        fig.tight_layout(); fig.savefig(output_dir / f"case_study_{case_index}.png", dpi=180); plt.close(fig)
    provenance_path = price_path.with_suffix(".provenance.json")
    metadata = {"input": str(price_path), "market": market, "threshold": threshold,
                "cooldown": cooldown, "event_window": event_window,
                "observations": len(prices), "companies": columns,
                "event_count": len(events), "verified_real_data": provenance_path.exists(),
                "interpretation": "Ranked statistical evidence only; no causal or contractual claim."}
    if provenance_path.exists():
        metadata["provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"metadata": metadata, "events": events, "ranking": ranking, "evidence": pd.DataFrame(evidence)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "real" / "us_large_caps_2022_2025.csv")
    parser.add_argument("--market", default="SPY")
    parser.add_argument("--threshold", type=float, default=2.5)
    args = parser.parse_args()
    result = run_real_data(args.prices, args.market, args.threshold)
    print(result["ranking"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
