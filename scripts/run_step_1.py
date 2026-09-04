"""Run the complete Step 1 synthetic demonstration."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_generator import SyntheticConfig, simulate
from src.visualization import save_figures


def main() -> None:
    data = simulate(SyntheticConfig())
    data_dir = ROOT / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        data_dir / "step_1_sample.npz",
        adjacency=data.adjacency,
        normalized_adjacency=data.normalized_adjacency,
        states=data.states,
        shocks=data.shock_matrix,
    )
    with (data_dir / "step_1_events.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "company", "signed_shock"])
        writer.writerows(data.events)

    save_figures(data, ROOT / "reports" / "figures")
    edges = int(data.adjacency.sum() / 2)
    print(f"Created {edges} true edges, {len(data.events)} shocks, and Step 1 figures.")


if __name__ == "__main__":
    main()

