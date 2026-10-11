import numpy as np
import pandas as pd

from src.preprocessing import align_prices, log_returns, preprocess_prices


def sample_prices():
    index = pd.date_range("2024-01-01", periods=20)
    market = 100 * np.exp(np.linspace(0, .1, 20))
    return pd.DataFrame({"A": market * np.exp(np.sin(np.arange(20)) * .01),
                         "B": market * np.exp(np.cos(np.arange(20)) * .01),
                         "MKT": market}, index=index)


def test_preprocessing_shapes_and_finite_values():
    prices = sample_prices()
    abnormal, standardized = preprocess_prices(prices, "MKT")
    assert list(abnormal) == ["A", "B"]
    assert abnormal.shape == standardized.shape == (19, 2)
    assert np.isfinite(standardized.to_numpy()).all()


def test_alignment_fills_short_gap():
    prices = sample_prices(); prices.iloc[3, 0] = np.nan
    assert not align_prices(prices).isna().any().any()
    assert len(log_returns(prices)) == 19
