"""Company-specific shock detection with per-company cooldowns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_shocks(z_returns: pd.DataFrame, threshold: float = 2.5, cooldown: int = 5,
                  event_window: int = 5) -> pd.DataFrame:
    if threshold <= 0 or cooldown < 0 or event_window < 1:
        raise ValueError("threshold/window must be positive and cooldown non-negative")
    if not np.isfinite(z_returns.to_numpy(dtype=float)).all():
        raise ValueError("standardized returns must be finite")
    rows: list[dict[str, object]] = []
    for company, series in z_returns.items():
        last = -cooldown - 1
        for position, (date, value) in enumerate(series.items()):
            if position + event_window >= len(series):
                continue
            if abs(float(value)) >= threshold and position - last > cooldown:
                rows.append({"time": position, "date": date, "company": company,
                             "company_index": z_returns.columns.get_loc(company),
                             "z_score": float(value), "signed_shock": float(value)})
                last = position
    return pd.DataFrame(rows, columns=["time", "date", "company", "company_index", "z_score", "signed_shock"])


def to_srcs_events(events: pd.DataFrame) -> list[tuple[int, int, float]]:
    return [(int(row.time), int(row.company_index), float(row.signed_shock))
            for row in events.itertuples(index=False)]
