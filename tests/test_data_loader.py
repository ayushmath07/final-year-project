import json

import pandas as pd

from src.data_loader import load_price_csv, save_price_cache


def test_cache_round_trip_with_provenance(tmp_path):
    frame = pd.DataFrame({"A": [1.0, 2.0], "MKT": [3.0, 4.0]},
                         index=pd.date_range("2024-01-01", periods=2))
    path = tmp_path / "prices.csv"
    save_price_cache(frame, path, {"source": "unit test"})
    loaded = load_price_csv(path)
    assert loaded.shape == frame.shape
    assert json.loads(path.with_suffix(".provenance.json").read_text())["source"] == "unit test"
