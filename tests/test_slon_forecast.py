import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.slon_forecast import aligned_pairs, benchmark


def series(years):
    k = np.arange(48 * len(years))
    return pd.DataFrame({
        "Year": np.repeat(years, 48), "QM": k % 48 + 1,
        "stlouisontOut": 1000. + 200. * np.cos(k / 9) + 15. * (k % 11),
    })


def test_availability_and_target_cross_new_year():
    frame = series([2000, 2001])
    issue = 48
    for delay, latest_qm in ((0, 48), (1, 47)):
        pairs = aligned_pairs(frame, lead=4, delay=delay)
        row = pairs.iloc[issue]
        assert (row.issue_year, row.issue_qm) == (2001, 1)
        assert (row.last_available_year, row.last_available_qm) == (2000, latest_qm)
        assert (row.target_year, row.target_qm) == (2001, 4)
        assert row.persistence == frame.stlouisontOut.iloc[issue - 1 - delay]
    changed = frame.copy()
    changed.loc[issue - 1:, "stlouisontOut"] += 1000
    features = ["persistence", "recent_change", "recent_mean"]
    pd.testing.assert_series_equal(
        aligned_pairs(frame, 1, 1).loc[issue, features],
        aligned_pairs(changed, 1, 1).loc[issue, features],
    )
    assert aligned_pairs(frame, 1, 0).loc[issue, "persistence"] != aligned_pairs(changed, 1, 0).loc[issue, "persistence"]


def test_rescoring_future_cannot_change_fit_or_current_forecasts():
    frame = series(np.arange(1900, 1907))
    protocol = {"training_years": [1900, 1902], "development_years": [1903, 1903],
                "reserved_test_years": [1904], "development_events": [1905, 1906]}
    before, _, threshold, fits = benchmark(frame, protocol)
    issue = 48 * 4 + 4
    changed = frame.copy()
    changed.loc[issue:, "stlouisontOut"] += 1000
    after, _, changed_threshold, changed_fits = benchmark(changed, protocol)
    assert threshold == changed_threshold and fits == changed_fits
    fields = ["persistence", "seasonal_mean", "seasonal_history"]
    pd.testing.assert_frame_equal(before.loc[before.issue_index.eq(issue), fields],
                                  after.loc[after.issue_index.eq(issue), fields])
    assert before.loc[before.issue_index.eq(issue), "target_m3_s"].ne(
        after.loc[after.issue_index.eq(issue), "target_m3_s"]).all()
