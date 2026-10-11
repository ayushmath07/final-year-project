"""Historical-price loading with explicit provenance and offline cache support."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_price_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("price CSV must contain dates and at least two series")
    return frame.apply(pd.to_numeric, errors="coerce")


def save_price_cache(prices: pd.DataFrame, csv_path: str | Path, provenance: dict[str, object]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(csv_path, index_label="Date")
    csv_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8"
    )


def download_yahoo_prices(tickers: list[str], start: str, end: str,
                          cache_path: str | Path) -> pd.DataFrame:
    """Download adjusted prices using yfinance; raises rather than fabricating data."""
    if len(tickers) < 2 or len(set(tickers)) != len(tickers):
        raise ValueError("provide at least two unique tickers")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("install yfinance to acquire historical data") from exc
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                      group_by="column", threads=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers).dropna(how="all")
    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no price observations")
    save_price_cache(prices, cache_path, {"source": "Yahoo Finance via yfinance",
        "tickers": tickers, "start": start, "end": end,
        "downloaded_at_utc": pd.Timestamp.utcnow().isoformat()})
    return prices
