import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_generator import SyntheticConfig, simulate
from src.response_score import build_srcs_matrix


def main():
    data = simulate(SyntheticConfig())

    scores = build_srcs_matrix(
        states=data.states,
        events=data.events,
        lag_window=5,
        decay=0.5,
        response_threshold=1.0,
    )

    np.set_printoptions(precision=3, suppress=True)

    print("\nSRCS score matrix:\n")
    print(scores)


if __name__ == "__main__":
    main()