import pandas as pd

from src.shock_detector import detect_shocks, to_srcs_events


def test_shock_detection_respects_cooldown_and_window():
    frame = pd.DataFrame({"A": [0, 3, 4, 0, 0, -3, 0, 0, 0, 4],
                          "B": [0] * 10})
    events = detect_shocks(frame, threshold=2.5, cooldown=2, event_window=2)
    assert list(events.time) == [1, 5]
    assert to_srcs_events(events) == [(1, 0, 3.0), (5, 0, -3.0)]
