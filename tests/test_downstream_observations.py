import pandas as pd
import pytest

from reservoir_control.systems.ottawa.downstream_observations import (
    check_datum,
    read_cornwall,
    summarize,
    validation,
)


def test_cornwall_uses_published_units_and_qualifiers():
    rows = [
        {"value": "1000", "dateTime": "2021-04-01T00:00:00.000", "qualifiers": ["A", "e"]},
        {"value": "-999999", "dateTime": "2021-04-02T00:00:00.000", "qualifiers": ["A"]},
    ]
    payload = {"value": {"timeSeries": [{
        "variable": {"unit": {"unitCode": "ft3/s"}},
        "values": [{"value": rows}],
    }]}}
    discharge = read_cornwall(payload)
    assert discharge.cornwall.iloc[0] == pytest.approx(28.316846592)
    assert discharge.cornwall_flags.iloc[0] == "A|e"
    assert pd.isna(discharge.cornwall.iloc[1])


def test_datum_check_rejects_shift_and_short_overlap():
    dates = pd.date_range("2021-04-01", periods=35)
    frame = pd.DataFrame({"pointe_claire": 22.0, "pointe_claire_flag": None}, index=dates)
    assert check_datum(frame, frame.pointe_claire + .01)["datum_check_passed"]
    assert not check_datum(frame, frame.pointe_claire + .1)["datum_check_passed"]
    assert not check_datum(frame, frame.pointe_claire.iloc[:29])["datum_check_passed"]


def test_holdout_and_flagged_data_are_excluded_from_fit_and_periods():
    dates = pd.date_range("2000-01-01", "2025-12-31")
    frame = pd.DataFrame({
        "lasalle": 10000., "pointe_claire": 22., "cornwall": 8000.,
        "lasalle_flag": None, "pointe_claire_flag": None, "cornwall_flags": "A",
    }, index=dates)
    frame.loc["2022-05-02", "lasalle_flag"] = "Estimated"
    frame.loc["2022-05-10", "cornwall_flags"] = "A|e"
    _, original_fit = validation(frame)
    frame.loc["2022", "pointe_claire"] += .5
    data, new_fit = validation(frame)
    assert new_fit == original_fit
    _, periods = summarize(data)
    may = periods.loc[(periods.year == 2022) & periods.qm.isin([17, 18])]
    assert pd.isna(may.iloc[0].pointe_claire_mean_m)
    assert may.iloc[1].pointe_claire_mean_m == pytest.approx(22.5)
    assert pd.isna(may.iloc[1].cornwall_mean_m3_s)
