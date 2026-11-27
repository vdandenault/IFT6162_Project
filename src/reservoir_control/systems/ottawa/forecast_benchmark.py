"""Compare causal Carillon forecasts on a fixed chronological HYDAT split."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .reconstruction import digest

LEADS = (1, 3, 7)
FIT_START = "1987-01-01"
FIT_END = "1991-12-31"
TEST_START = "1992-01-01"
TEST_END = "1994-12-31"
MODELS = ("persistence", "carillon_history", "carillon_and_britannia")


def build_pairs(source, lead):
    frame = pd.read_csv(source, parse_dates=["date"]).set_index("date")
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max())):
        raise ValueError("Expected a complete, ordered daily calendar")
    carillon = frame["02LB024_discharge"].where(frame["02LB024_flag"].isna())
    britannia = frame["02KF005_discharge"].where(frame["02KF005_flag"].isna())
    season = 2 * np.pi * (frame.index.day_of_year - 1) / 365.25
    return pd.DataFrame({
        "target_date": frame.index + pd.Timedelta(days=lead),
        "target_flow": carillon.shift(-lead),
        "persistence": carillon,
        "carillon_change_1d": carillon - carillon.shift(1),
        "carillon_change_3d": carillon - carillon.shift(3),
        "britannia_current": britannia,
        "britannia_change_1d": britannia - britannia.shift(1),
        "britannia_change_3d": britannia - britannia.shift(3),
        "sin_season": np.sin(season),
        "cos_season": np.cos(season),
    }).dropna()


def forecast_pairs(pairs):
    train = pairs[(pairs.index >= pd.Timestamp(FIT_START)) & pairs.target_date.le(FIT_END)]
    test = pairs[(pairs.index >= pd.Timestamp(TEST_START)) & pairs.target_date.le(TEST_END)].copy()
    if train.empty or test.empty:
        raise ValueError("The frozen split requires data in 1987–1991 and 1992–1994")
    features = ["persistence", "carillon_change_1d", "carillon_change_3d", "sin_season", "cos_season"]
    brit = ["britannia_current", "britannia_change_1d", "britannia_change_3d"]
    reference = train.target_flow - train.persistence
    coefficients = {}
    for name, columns in zip(MODELS[1:], (features, features + brit), strict=True):
        values = train[columns].to_numpy()
        design = np.column_stack([np.ones(len(values)), values])
        beta, *_ = np.linalg.lstsq(design, reference, rcond=None)
        test[name] = np.maximum(0., test.persistence + beta[0] + test[columns].to_numpy() @ beta[1:])
        coefficients[name] = dict(zip(["intercept", *columns], beta.tolist(), strict=True))
    cutoff = float(train.target_flow.quantile(.9))
    test["high_flow"] = test.target_flow.ge(cutoff)
    return test, {"fit_rows": len(train), "high_flow_threshold_m3_s": cutoff, "coefficients": coefficients}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/ottawa/flow_source_audit_v1/historical_daily.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = digest(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    fits, predictions, scores = {}, [], []
    for lead in LEADS:
        test, fits[lead] = forecast_pairs(build_pairs(args.source, lead))
        predictions.append(test.assign(lead_days=lead).rename_axis("issued_after_date").reset_index())
        groups = {"all": test, "high flow": test[test.high_flow]}
        groups.update({str(y): group for y, group in test.groupby(test.target_date.dt.year)})
        for group, sample in groups.items():
            for model in MODELS:
                errors = sample[model] - sample.target_flow
                scores.append({
                    "lead_days": lead, "sample": group, "forecast": model, "days": len(sample),
                    "mae_m3_s": errors.abs().mean(), "rmse_m3_s": np.sqrt(errors.pow(2).mean()),
                    "bias_m3_s": errors.mean(),
                })
    metrics = pd.DataFrame(scores)
    metrics.to_csv(args.output / "scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    assert digest(args.source) == source_hash
    (args.output / "protocol.json").write_text(json.dumps({
        "source": str(args.source), "source_sha256": source_hash, "fit_start": FIT_START, "fit_end": FIT_END,
        "test_start": TEST_START, "test_end": TEST_END, "lead_days": LEADS, "fits": fits,
    }, indent=2) + "\n")
    lines = [
        "# Carillon daily flow forecast benchmark", "",
        "Each daily forecast is issued after the labeled day ends, assuming that day's mean has arrived. "
        "It uses flow records through the issue day. Fit targets end in 1991. Test issues and targets "
        "fall in 1992–1994. All three forecasts are scored on matched dates after excluding missing "
        "and flagged values needed by any model.", "",
        "Persistence repeats the last Carillon flow. The two regression models forecast the change from "
        "that flow, using recent Carillon changes and seasonal terms. The Britannia model adds the latest "
        "upstream flow and its recent changes. Negative regressions are floored at zero. The high-flow "
        "subset uses the training target's 90th percentile at each lead.", "",
        "| Sample | Lead | Matched days | Persistence MAE | Carillon-history MAE | With Britannia MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "high flow"):
        for lead, group in metrics[metrics["sample"].eq(name)].groupby("lead_days", sort=True):
            values = group.set_index("forecast").loc[list(MODELS), "mae_m3_s"]
            lines.append(f"| {name} | {lead} d | {group.days.iloc[0]} | "
                         + " | ".join(f"{v:.1f}" for v in values) + " |")
    lines += ["", "Errors are in m³/s. See scores.csv for year-specific, RMSE and bias results. "
              "This is a daily historical benchmark with assumed availability of approved daily means. "
              "The comparison does not estimate hydraulic travel time or modern quarter-month forecast skill.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
