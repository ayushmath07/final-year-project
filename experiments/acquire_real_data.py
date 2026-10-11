"""Acquire and cache genuine adjusted historical prices."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import download_yahoo_prices

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "WMT", "XOM", "JNJ", "SPY"]


def main() -> None:
    target = ROOT / "data" / "real" / "us_large_caps_2022_2025.csv"
    prices = download_yahoo_prices(TICKERS, "2022-01-01", "2026-01-01", target)
    print(f"Saved {len(prices)} observations for {len(prices.columns)} series to {target}")


if __name__ == "__main__":
    main()
