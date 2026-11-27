"""Validate offline reach flow and Pointe-Claire headroom with frozen gauges."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .downstream_observations import CAP, FIRST, LAST, SHIFT
from .flow_source_audit import LICENCE, address, check_records, fetch
from .reconstruction import digest

GAUGES = {"britannia": "02KF005", "prairies": "02OA004"}
FIT = (2000, 2009)
SELECT = (2010, 2016)
DEVELOPMENT = (FIT[0], SELECT[1])
HOLDOUT = (2021, 2025)
LAGS = range(8)
NEAR_CAP = .10


def years(index, bounds):
    return (index.year >= bounds[0]) & (index.year <= bounds[1])


def gauge_queries():
    for name, station in GAUGES.items():
        for year in range(FIRST, LAST + 1, 5):
            window = (f"{year}-01-01", f"{min(year + 4, LAST)}-12-31")
            url = address("hydrometric-daily-mean", station, datetime="/".join(window), limit=2200, sortby="DATE")
            yield name, station, window, url


def load_gauges(downstream, raw, queries):
    data = pd.read_csv(downstream / "daily.csv", parse_dates=["date"]).set_index("date")
    if not data.index.equals(pd.date_range(f"{FIRST}-01-01", f"{LAST}-12-31")):
        raise ValueError("Lag calculations require a complete daily calendar")
    for name, station, window, _ in queries:
        year = window[0][:4]
        values = check_records(json.loads((raw / f"{station}_{year}.json").read_bytes()),
                               station, "DISCHARGE", window)
        days = pd.to_datetime(values.index)
        data.loc[days, name] = values.value.to_numpy(dtype=float)
        data.loc[days, f"{name}_flag"] = values.flag.to_numpy()
    for name in GAUGES:
        data[name] = data[name].where(data[f"{name}_flag"].isna())
    data["cornwall"] = data.cornwall.where(data.cornwall_flags.eq("A"))
    data["lasalle"] = data.lasalle.where(data.local_quality)
    data["pointe_claire"] = data.pointe_claire.where(data.local_quality)
    return data


def predictors(data):
    phase = 2 * np.pi * (data.index.day_of_year - 1) / np.where(data.index.is_leap_year, 366, 365)
    features = data.assign(offset=1., sin=np.sin(phase), cos=np.cos(phase))
    for name in ("cornwall", "britannia"):
        for lag in LAGS:
            features[f"{name}_{lag}"] = data[name].shift(lag)
    return features


def fit_flow(frame, bounds, cornwall, ottawa, include_branch):
    cols = ["offset", "sin", "cos"]
    lower, upper = [-np.inf] * 3, [np.inf] * 3
    if ottawa is not None:
        cols.append(f"britannia_{ottawa}")
        lower.append(0.)
        upper.append(np.inf)
    if include_branch:
        cols.append("prairies")
        lower.append(-1.)
        upper.append(0.)
    design = frame[cols]
    reference = frame[f"cornwall_{cornwall}"]
    residual = frame.lasalle - reference
    sample = years(frame.index, bounds) & design.notna().all(axis=1) & residual.notna()
    if sample.sum() < len(cols):
        raise ValueError("Insufficient development overlap")
    fit = lsq_linear(design.loc[sample], residual.loc[sample], bounds=(lower, upper))
    if not fit.success:
        raise ArithmeticError(fit.message)
    return reference + design @ fit.x, dict(zip(cols, fit.x, strict=True))


def select_lags(frame, include_branch):
    lag_cols = [f"{name}_{lag}" for name, lag in product(("cornwall", "britannia"), LAGS)]
    common = frame[["lasalle", *lag_cols, *(["prairies"] if include_branch else [])]].notna().all(axis=1)
    tune = (years(frame.index, SELECT) & frame.index.month.isin((4, 5, 6)) & common)
    if not tune.any():
        raise ValueError("No common spring dates to compare lags")
    rows = []
    for cornwall, ottawa in product(LAGS, repeat=2):
        prediction, _ = fit_flow(frame, FIT, cornwall, ottawa, include_branch)
        error = (prediction - frame.lasalle).loc[tune]
        rows.append({"cornwall_days": cornwall, "britannia_days": ottawa, "selection_days": len(error),
                     "flow_mae_m3_s": error.abs().mean()})
    scores = pd.DataFrame(rows).sort_values(["flow_mae_m3_s", "cornwall_days", "britannia_days"])
    lag = scores.iloc[0]
    return (int(lag.cornwall_days), int(lag.britannia_days)), scores


def stage(flow, factor):
    if flow.dropna().le(0).any():
        raise ValueError("Stage proxy received nonpositive flow")
    return 16.57 + factor * (flow / 604.) ** .58


def record_metrics(observed, flow, modeled, after, actual_flow):
    error = modeled - observed
    near = observed.sub(CAP).abs().le(NEAR_CAP)
    high = observed.gt(CAP)
    room = observed.lt(CAP)
    return {
        "samples": len(observed), "flow_mae_m3_s": (flow - actual_flow).abs().mean(),
        "stage_mae_m": error.abs().mean(), "stage_bias_m": error.mean(),
        "near_cap_samples": int(near.sum()), "near_cap_mae_m": error[near].abs().mean(),
        "p95_absolute_stage_error_m": error.abs().quantile(.95),
        "observed_exceedances": int(high.sum()), "missed_exceedances": int((high & modeled.le(CAP)).sum()),
        "false_exceedances": int((~high & modeled.gt(CAP)).sum()),
        "modeled_plus_200_room_calls": int(after.le(CAP).sum()),
        "modeled_room_with_observed_stage_at_or_above_cap": int((~room & after.le(CAP)).sum()),
    }


def evaluate(data, predictions, quarter_calendar):
    scores, periods = [], []
    cohorts = {
        "core": ["concurrent_lasalle", "cornwall_seasonal"],
        "ottawa": ["concurrent_lasalle", "cornwall_seasonal", "ottawa_same_day", "ottawa_selected_lags"],
        "branch": list(predictions),
    }
    for cohort, names in cohorts.items():
        daily = data[["pointe_claire", "lasalle"]].join(pd.concat([predictions[name] for name in names], axis=1))
        for begin, end in ((2021, 2025), (2017, 2017), (2019, 2019)):
            spring = daily.loc[years(daily.index, (begin, end)) & daily.index.month.isin((4, 5, 6))]
            matched = spring.dropna()
            if matched.empty:
                raise ValueError(f"No matched spring dates for {cohort} in {begin}–{end}")
            dates = quarter_calendar.loc[quarter_calendar.start.dt.year.between(begin, end)]
            for row in dates.itertuples():
                days = pd.date_range(row.start, row.end_exclusive, inclusive="left")
                if days.isin(matched.index).all():
                    periods.append({"cohort": cohort, "start": row.start, "end_exclusive": row.end_exclusive,
                                    **matched.loc[days].mean().to_dict()})
            for name in names:
                scores.append({"cohort": cohort, "split": "holdout" if begin == 2021 else str(begin),
                               "model": name, **record_metrics(
                                   matched.pointe_claire, matched[f"{name}_flow"], matched[f"{name}_stage"],
                                   matched[f"{name}_plus_200"], matched.lasalle,
                               )})
    return pd.DataFrame(scores), pd.DataFrame(periods)


def write_report(output, scores, periods, options, coefficients):
    holdout = scores.loc[scores.split.eq("holdout")]
    lines = [
        "# Downstream routing and headroom check", "",
        "## Observed boundaries", "",
        "Cornwall supplies the St. Lawrence boundary. LaSalle measures discharge downstream of "
        "Lake St. Louis. Britannia is upstream of Carillon. The Prairies gauge samples one of the "
        "routes that bypass Lake St. Louis. Sainte-Anne and Vaudreuil feed Lake St. Louis, and the "
        "Mille Îles is a second bypass. Québec reports that the share through the two Lake St. Louis "
        "channels varied from about 12% in a low-water example to about 48% at higher levels. A fixed "
        "Ottawa fraction has no general physical support. [Québec outlet study, section 2.2.1 and "
        "Figure 7](https://www.cehq.gouv.qc.ca/mille-iles/etude-solutions-soutien-etiages-critiques.pdf) · "
        "[Gauge locations and 2017 survey, annexes D–E](https://www.mrcautray.qc.ca/wp-content/uploads/2024/03/annexes-complet.pdf)",
        "",
        "All records use daily means matched by reported date. A daily lag search cannot resolve "
        "subdaily arrival times. We shift by full calendar days and exclude "
        "source-flagged or missing values. The public HYDAT Carillon discharge series ends in 1994, "
        "and the added upstream gauges returned no discharge for 2025. [Carillon archive](https://wateroffice.ec.gc.ca/report/data_availability_e.html?parameter_type=Flow&station=02LB024&type=historical)", "",
        "## Offline reach comparison", "",
        "Fit seasonal and flow coefficients on 2000–2009. Select lags from 0 to 7 days using "
        "common spring dates in 2010–2016 and LaSalle flow error. Refit on 2000–2016 and score "
        "2021–2025 once. The 2017 and 2019 springs are reported separately in the CSV. Cornwall enters "
        "with unit weight. A seasonal residual accounts for other contributions, and a nonnegative "
        "Britannia coefficient measures its association with downstream flow. The branch model tests "
        "a nonpositive same-day Prairies adjustment. The paired branch reference uses the same "
        "lag and training dates with that adjustment omitted. These coefficients describe an empirical proxy. "
        "Reach storage, local tributaries, backwater and use of an upstream Ottawa proxy all affect "
        "the inferred timing and magnitude.", "",
        "| Model | Cornwall / Britannia lag |", "| --- | ---: |",
    ]
    for name, (cornwall, ottawa, _) in options.items():
        lines.append(f"| {name.replace('_', ' ')} | {cornwall} / {ottawa if ottawa is not None else '—'} days |")
    lines += [
        "", "## Spring headroom on the held-out dates", "",
        "Each group is scored on common observed dates. Concurrent LaSalle flow is an offline stage "
        "reference. Stage uses the same frozen rating from the earlier gauge check. Near-threshold "
        "means within 0.10 m of 22.10 m. A false room call means modeled stage with an extra "
        "200 m³/s of local throughflow stayed within the study threshold while current observed stage "
        "was already at or above it.", "",
        "| Matched group | Model | Dates | Flow MAE (m³/s) | Stage MAE (m) | Near-threshold MAE (m) | False room calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in holdout.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.model.replace('_', ' ')} | {row.samples} | "
            f"{row.flow_mae_m3_s:.0f} | {row.stage_mae_m:.3f} | {row.near_cap_mae_m:.3f} | "
            f"{row.modeled_room_with_observed_stage_at_or_above_cap} |"
        )
    branch_coef = coefficients["prairies_adjusted"]["prairies"]
    lines += ["", f"The fitted Prairies adjustment was {branch_coef:.4f} m³/s of LaSalle "
              "flow per m³/s of Prairies flow. The bound at zero allows the model to omit this signal. "
              "This comparison cannot determine the split among the four outlets.", ""]
    held_periods = periods[periods.start.dt.year.between(*HOLDOUT)]
    lines += ["", "### Quarter-month flood cases", "",
              "Period means require a complete set of eligible days. The table lists core periods above "
              "22.10 m and compares estimated stage in those same periods.", "",
              "| Start date | Observed stage (m) | Concurrent LaSalle (m) | Cornwall + season (m) | Lagged Ottawa proxy (m) | Prairies adjustment (m) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    high = held_periods.loc[held_periods.cohort.eq("core") & held_periods.pointe_claire.gt(CAP)]
    for row in high.itertuples():
        estimates = []
        for cohort, key in (("ottawa", "ottawa_selected_lags_stage"), ("branch", "prairies_adjusted_stage")):
            matches = held_periods.loc[held_periods.start.eq(row.start) & held_periods.cohort.eq(cohort), key]
            estimates.append(f"{matches.iloc[0]:.3f}" if len(matches) else "—")
        lines.append(f"| {row.start.date()} | {row.pointe_claire:.3f} | {row.concurrent_lasalle_stage:.3f} "
                     f"| {row.cornwall_seasonal_stage:.3f} | {' | '.join(estimates)} |")
    lines += ["", "## Decision", "",
              "These checks use realized flows and actual release history. The 200 m³/s sensitivity is "
              "an increment to local throughflow. Its size and timing after an upstream release change "
              "would require independent reach and branch hydraulics. The fixed 22.10 m threshold also "
              "differs from operational application of the changing Plan 2014 tier. Keep the frozen controller "
              "comparison for the project handoff. Reopen this question when matched Carillon and "
              "Sainte-Anne/Vaudreuil flow or a validated hydraulic routing model is available.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", type=Path, required=True, help="Frozen downstream observation output")
    parser.add_argument("--raw", type=Path, help="Use saved HYDAT responses")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    queries = list(gauge_queries())
    sources = {f"{station}_{window[0][:4]}.json": url for _, station, window, url in queries}
    raw = args.raw or args.output / "raw"
    if args.raw is None:
        raw.mkdir()
        with ThreadPoolExecutor(max_workers=3) as executor:
            for name, body in zip(sources, executor.map(fetch, sources.values()), strict=True):
                (raw / name).write_bytes(body)
    data = load_gauges(args.downstream, raw, queries)
    frame = predictors(data)
    options = {"cornwall_seasonal": (0, None, False), "ottawa_same_day": (0, 0, False)}
    for key, branch in (("ottawa_selected_lags", False), ("prairies_adjusted", True)):
        lag, scores = select_lags(frame, branch)
        scores.to_csv(args.output / f"{key}_lag_selection.csv", index=False)
        options[key] = (*lag, branch)
    options["paired_branch_reference"] = (*options["prairies_adjusted"][:2], False)

    factor = json.loads((args.downstream / "rating_fit.json").read_text())["rating_factor"]
    flows = {"concurrent_lasalle": data.lasalle}
    coefficients = {}
    branch_sample = frame.assign(lasalle=frame.lasalle.where(frame.prairies.notna()))
    for key, (cornwall, ottawa, branch) in options.items():
        training_frame = branch_sample if key == "paired_branch_reference" else frame
        flows[key], coefficients[key] = fit_flow(training_frame, DEVELOPMENT, cornwall, ottawa, branch)
    predictions = {
        name: pd.DataFrame({f"{name}_flow": flow, f"{name}_stage": stage(flow, factor),
                            f"{name}_plus_200": stage(flow + SHIFT, factor)})
        for name, flow in flows.items()
    }
    calendar = pd.read_csv(args.downstream / "quarter_months.csv", parse_dates=["start", "end_exclusive"])
    scores, periods = evaluate(data, predictions, calendar)
    data.join(pd.concat(predictions.values(), axis=1)).to_csv(args.output / "daily.csv")
    scores.to_csv(args.output / "scores.csv", index=False)
    periods.to_csv(args.output / "spring_quarter_months.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "fit": FIT, "select": SELECT, "development_refit": DEVELOPMENT, "holdout": HOLDOUT,
        "lags_days": list(LAGS), "options_cornwall_ottawa_prairies": options, "coefficients": coefficients,
        "stage_rating_factor": factor, "fixed_threshold_m": CAP, "throughflow_increment_m3_s": SHIFT,
        "near_cap_m": NEAR_CAP, "licence": LICENCE,
        "downstream": {name: digest(args.downstream / name)
                       for name in ("daily.csv", "quarter_months.csv", "rating_fit.json", "protocol.json")},
        "gauges": {name: {"url": url, "sha256": digest(raw / name)} for name, url in sources.items()},
    }, indent=2) + "\n")
    write_report(args.output, scores, periods, options, coefficients)


if __name__ == "__main__":
    main()
