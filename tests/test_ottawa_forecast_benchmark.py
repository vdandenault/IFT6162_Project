import numpy as np
import pandas as pd

from reservoir_control.systems.ottawa.forecast_benchmark import build_pairs, forecast_pairs


def test_future_records_cannot_change_forecast_features_or_fit(tmp_path):
    days = pd.date_range("1991-11-01", "1992-03-01")
    n = np.arange(len(days))
    raw = pd.DataFrame({
        "date": days, "02LB024_discharge": 1200. + n + np.sin(n / 3) * 80.,
        "02KF005_discharge": 900. + n + np.cos(n / 5) * 50.,
        "02LB024_flag": None, "02KF005_flag": None,
    })
    path = tmp_path / "daily.csv"
    raw.to_csv(path, index=False)
    original = build_pairs(path, 3)
    predictions, fits = forecast_pairs(original)
    issue = pd.Timestamp("1992-01-15")
    raw.loc[raw.date.gt(issue), ["02LB024_discharge", "02KF005_discharge"]] *= 2
    raw.to_csv(path, index=False)
    revised = build_pairs(path, 3)
    changed_predictions, changed_fits = forecast_pairs(revised)
    forecast_columns = ["persistence", "carillon_history", "carillon_and_britannia"]
    assert fits == changed_fits
    pd.testing.assert_series_equal(predictions.loc[issue, forecast_columns],
                                   changed_predictions.loc[issue, forecast_columns])
    assert revised.loc[issue, "target_flow"] == 2 * original.loc[issue, "target_flow"]
