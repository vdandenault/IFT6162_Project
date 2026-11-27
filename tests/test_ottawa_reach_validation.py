import numpy as np
import pandas as pd
import pytest

from reservoir_control.systems.ottawa.reach_validation import (
    CAP,
    fit_flow,
    predictors,
    record_metrics,
)


def test_daily_lag_preserves_a_missing_observation():
    dates = pd.date_range("2022-05-01", periods=3)
    flows = pd.DataFrame({"cornwall": [8000., np.nan, 8500.], "britannia": 3000.}, index=dates)
    features = predictors(flows)
    assert features.cornwall_1.iloc[1] == 8000.
    assert pd.isna(features.cornwall_1.iloc[2])
    assert features.cornwall_2.iloc[2] == 8000.


def test_reach_fit_uses_only_development_and_recovers_known_contributions():
    dates = pd.date_range("2016-01-01", periods=730)
    index = np.arange(len(dates))
    data = pd.DataFrame({"cornwall": 7800. + index, "britannia": 2500. + 50 * (index % 3),
                         "prairies": 1200. + 70 * (index % 5)}, index=dates)
    data["lasalle"] = data.cornwall + .8 * data.britannia - .25 * data.prairies + 50.
    frame = predictors(data)
    predicted, fitted = fit_flow(frame, (2016, 2016), 0, 0, True)
    assert fitted["britannia_0"] == pytest.approx(.8)
    assert fitted["prairies"] == pytest.approx(-.25)
    frame.loc["2017", "lasalle"] += 2000.
    repeated, _ = fit_flow(frame, (2016, 2016), 0, 0, True)
    pd.testing.assert_series_equal(predicted, repeated)


def test_room_call_uses_observed_current_stage_as_one_sided_check():
    observed = pd.Series([CAP + .04, CAP, CAP - .10, CAP + .02])
    modeled = pd.Series([CAP - .08, CAP - .08, CAP - .07, CAP + .01])
    flow = pd.Series([10000.] * len(observed))
    scores = record_metrics(observed, flow, modeled, modeled + .06, flow)
    assert scores["observed_exceedances"] == 2
    assert scores["missed_exceedances"] == 1
    assert scores["modeled_plus_200_room_calls"] == 3
    assert scores["modeled_room_with_observed_stage_at_or_above_cap"] == 2
