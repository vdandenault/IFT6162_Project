import numpy as np
import pandas as pd
import pytest

from reservoir_control.systems.ottawa.routing_validation import (
    CALIBRATION, HOLDOUT, STATIONS, aligned_data, compare_routing, compare_tradeoff, years,
)


def test_routing_lag_selection_and_fit_are_independent_of_holdout_levels(tmp_path):
    dates = pd.date_range("1989-10-01", "1991-04-30")
    k = np.arange(len(dates))
    raw = pd.DataFrame({"date": dates})
    values = {
        "carillon": 1900 + 700 * np.sin(k / 8.) + 140 * np.sin(k / 2.),
        "cornwall": 7100 + 300 * np.cos(k / 13.) + 200 * np.cos(k / 3.),
        "pointe_calumet": 22 + .1 * np.sin(k / 7.),
        "pointe_claire": 22 + .1 * np.sin(k / 7.) + .3 * np.cos(k / 8.),
    }
    for name, (value, flag) in STATIONS.items():
        raw[value] = values[name]
        raw[flag] = None
    observed = tmp_path / "historical.csv"
    raw.to_csv(observed, index=False)
    matched = aligned_data(observed)
    base, _, _, _, protocol = compare_routing(matched)
    raw.loc[raw.date.dt.year.eq(HOLDOUT[0]), STATIONS["pointe_claire"][0]] += .8
    raw.to_csv(observed, index=False)
    updated, _, _, _, updated_protocol = compare_routing(aligned_data(observed))
    assert protocol == updated_protocol
    pd.testing.assert_series_equal(base.two_flow_selected_lag, updated.two_flow_selected_lag)
    assert years(base.index, CALIBRATION).any()


def test_tradeoff_rejects_duplicate_period_disguised_as_complete_year(tmp_path):
    trace = pd.DataFrame({
        "delay": 0, "year": 2020,
        "method": np.repeat(["persistence", "seasonal_half_upper"], 48),
        "period": np.tile(range(1, 49), 2),
        "ontario_exposure_m": 0., "downstream_exposure_m": 1.,
    })
    source = tmp_path / "reserved.csv"
    trace.to_csv(source, index=False)
    assert len(compare_tradeoff(source)[0]) == 1
    trace.loc[48, "period"] = 2
    trace.to_csv(source, index=False)
    with pytest.raises(ValueError, match="matched complete years"):
        compare_tradeoff(source)
