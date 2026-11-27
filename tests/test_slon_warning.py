import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.slon_warning import gauge_features, quarter_start


def test_legacy_boundaries_distinguish_february_and_other_months():
    assert quarter_start(2000, 2) == pd.Timestamp("2000-01-09")
    assert quarter_start(2000, 6) == pd.Timestamp("2000-02-08")
    assert quarter_start(2000, 8) == pd.Timestamp("2000-02-22")


def test_gauge_delay_excludes_latest_completed_period_and_flags():
    dates = pd.date_range("1999-12-01", "2000-02-12")
    series = pd.DataFrame({"value": np.arange(len(dates), dtype=float), "flag": None}, index=dates)
    frame = pd.DataFrame({"Year": [2000] * 5, "QM": [1, 2, 3, 4, 5]})
    timely, delayed = (gauge_features(series, frame, d) for d in (0, 1))
    assert timely.iloc[-1].gauge_last_day == pd.Timestamp("2000-01-31")
    assert delayed.iloc[-1].gauge_last_day == pd.Timestamp("2000-01-23")
    changed = series.copy()
    changed.loc["2000-01-24":, "value"] += 500
    pd.testing.assert_series_equal(gauge_features(changed, frame, 1).iloc[-1], delayed.iloc[-1])
    changed.loc["2000-01-20", "flag"] = "Estimated"
    assert pd.isna(gauge_features(changed, frame, 1).iloc[-1].britannia_mean)
