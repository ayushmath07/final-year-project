"""Preprocessing for aligned financial prices and abnormal returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_prices(prices: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    frame = prices.copy().sort_index()
    frame = frame.loc[:, frame.notna().mean() >= min_coverage]
    frame = frame.ffill(limit=2).dropna()
    if frame.shape[0] < 3 or frame.shape[1] < 2:
        raise ValueError("insufficient aligned price observations")
    if (frame <= 0).any().any():
        raise ValueError("prices must be positive")
    return frame


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = align_prices(prices)
    return np.log(prices).diff().dropna()


def market_adjust_returns(returns: pd.DataFrame, market_column: str) -> pd.DataFrame:
    if market_column not in returns:
        raise ValueError(f"market column {market_column!r} is missing")
    market = returns[market_column].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(market)), market])
    abnormal: dict[str, np.ndarray] = {}
    for column in returns.columns:
        if column == market_column:
            continue
        values = returns[column].to_numpy(dtype=float)
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        abnormal[column] = values - design @ beta
    if not abnormal:
        raise ValueError("at least one company column is required")
    return pd.DataFrame(abnormal, index=returns.index)


def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    std = returns.std(ddof=0).replace(0, np.nan)
    return ((returns - returns.mean()) / std).fillna(0.0)


def preprocess_prices(prices: pd.DataFrame, market_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = log_returns(prices)
    abnormal = market_adjust_returns(returns, market_column)
    return abnormal, standardize_returns(abnormal)
