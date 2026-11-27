"""Test a causal Britannia gauge signal for high-SLON onset."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.ottawa.flow_source_audit import LICENCE, address, check_records, fetch
from .data import load_inputs
from .native import INPUT_FILE, NativeReference, sha256
from .slon_forecast import DELAYS, aligned_pairs, load_protocol

STATION = "02KF005"
LEADS = (2, 3, 4)
FIT = (1961, 1980)
CALIBRATION = (1981, 2000)
FALSE_ALARM_BUDGET = .10
MODELS = ("archived_slon", "slon_only", "slon_with_britannia")
FLOW_COLS = ("britannia_mean", "britannia_change")


def read_gauge(raw, *, download=False):
    values, provenance = [], {}
    for first in range(1960, 2021, 5):
        window = (f"{first}-01-01", f"{min(first + 4, 2020)}-12-31")
        url = address("hydrometric-daily-mean", STATION, datetime="/".join(window), sortby="DATE", limit=2200)
        path = raw / f"{STATION}_{first}.json"
        if download:
            path.write_bytes(fetch(url))
        values.append(check_records(json.loads(path.read_bytes()), STATION, "DISCHARGE", window))
        provenance[path.name] = {"url": url, "sha256": sha256(path)}
    series = pd.concat(values)
    series.index = pd.to_datetime(series.index)
    if series.index.has_duplicates:
        raise ValueError("Duplicate Britannia observations")
    return series.reindex(pd.date_range("1960-01-01", "2020-12-31", name="date")), provenance


def quarter_start(year, qm):
    month, quarter = divmod(qm - 1, 4)
    # Legacy IJC quarter-month day boundaries (Plan 1958-A, section II).
    return pd.Timestamp(year, month + 1, (1, 8, 15, 22)[quarter] if month == 1 else (1, 9, 16, 24)[quarter])


def gauge_features(series, frame, delay):
    discharges = series.value.where(series.flag.isna())
    weekly = discharges.rolling(7).mean()
    history = [quarter_start(int(year), int(qm)) for year, qm in zip(frame.Year, frame.QM, strict=True)]
    border = pd.Series(history, index=frame.index).shift(delay) - pd.Timedelta(days=1)
    cutoff = pd.DatetimeIndex(border)
    return pd.DataFrame({
        "gauge_last_day": cutoff,
        "britannia_mean": weekly.reindex(cutoff).to_numpy(),
        "britannia_change": (weekly - weekly.shift(7)).reindex(cutoff).to_numpy(),
    }, index=frame.index)


def fit_score(train, data, columns, *, label="onset"):
    fit_x = train[columns].to_numpy()
    center, scale = fit_x.mean(axis=0), fit_x.std(axis=0)
    scale[scale == 0.] = 1.
    design = np.column_stack([np.ones(len(train)), (fit_x - center) / scale])
    penalty = np.diag([0., *([1.] * len(columns))])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train[label].to_numpy(float))
    score = coefficients[0] + (data[columns].to_numpy() - center) / scale @ coefficients[1:]
    return score, {"features": columns, "center": center.tolist(), "scale": scale.tolist(),
                   "coefficients": coefficients.tolist()}


def metrics(sample, model, cutoff, *, label="onset"):
    alarm = sample[model].ge(cutoff)
    positive = sample[label]
    tp = int((alarm & positive).sum())
    fp = int((alarm & ~positive).sum())
    return {
        "model": model, "periods": len(sample), "onsets" if label == "onset" else "positives": int(positive.sum()),
        "hits": tp, "misses": int(positive.sum()) - tp, "false_alarms": fp,
        "recall": tp / positive.sum() if positive.any() else np.nan,
        "false_alarm_rate": fp / (~positive).sum() if (~positive).any() else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
    }


def benchmark(frame, series, archived, threshold, protocol):
    upstream = {delay: gauge_features(series, frame, delay) for delay in DELAYS}
    usable = upstream[0].notna().all(axis=1) & upstream[1].notna().all(axis=1)
    cohorts = {
        "calibration": range(CALIBRATION[0], CALIBRATION[1] + 1),
        "reserved": protocol["reserved_test_years"],
        **{str(year): [year] for year in protocol["development_events"]},
    }
    rows, predictions, fits = [], [], {}
    for lead in LEADS:
        before_target = frame.stlouisontOut.shift(2 - lead)
        for delay in DELAYS:
            matched = aligned_pairs(frame, lead, delay).join(upstream[delay]).loc[usable].dropna().copy()
            matched["onset"] = matched.target_m3_s.ge(threshold) & before_target.loc[matched.index].lt(threshold)
            target_angle = 2 * np.pi * (matched.target_qm - 1) / 48
            matched["season_sin"], matched["season_cos"] = np.sin(target_angle), np.cos(target_angle)
            start, end = FIT
            training = matched[matched.issue_year.between(start, end) & matched.target_year.between(start, end)]
            base_features = ["persistence", "recent_change", "recent_mean", "season_sin", "season_cos"]
            primary = archived[archived.lead.eq(lead) & archived.delay.eq(delay)][
                ["issue_index", "seasonal_history"]
            ].rename(columns={"seasonal_history": "archived_slon"})
            evaluated = matched.merge(primary, on="issue_index", how="inner", validate="one_to_one")
            model_fits = {}
            for name, cols in (("slon_only", base_features), ("slon_with_britannia", [*base_features, *FLOW_COLS])):
                evaluated[name], model_fits[name] = fit_score(training, evaluated, cols)
            calibration = evaluated[evaluated.target_year.between(*CALIBRATION) & ~evaluated.onset]
            cutoffs = {name: float(np.nextafter(calibration[name].quantile(1. - FALSE_ALARM_BUDGET,
                                                                          interpolation="higher"), np.inf))
                       for name in MODELS}
            fits[f"delay_{delay}_lead_{lead}"] = {"models": model_fits, "alert_cutoffs": cutoffs,
                                                "training_periods": len(training),
                                                "training_onsets": int(training.onset.sum())}
            for cohort, years in cohorts.items():
                sample = evaluated[evaluated.target_year.isin(years)].copy()
                for name, cutoff in cutoffs.items():
                    sample[f"{name}_alert"] = sample[name].ge(cutoff)
                    rows.append({"cohort": cohort, "lead": lead, "delay": delay,
                                 **metrics(sample, name, cutoff)})
                predictions.append(sample.assign(cohort=cohort, lead=lead, delay=delay))
    output = pd.concat(predictions, ignore_index=True)
    matches = output.groupby(["cohort", "lead", "issue_index"]).size()
    assert matches.eq(2).all()
    return pd.DataFrame(rows), output, fits


def match_witnesses(path, predictions):
    hindsight = pd.read_csv(path)
    witnesses = hindsight[hindsight.found & hindsight.within_four_period_plan.eq(True)].copy()
    witnesses["lead"] = witnesses.later_period - witnesses.earlier_period + 1
    return witnesses.merge(
        predictions, left_on=["year", "delay", "lead", "earlier_period", "later_period"],
        right_on=["issue_year", "delay", "lead", "issue_qm", "target_qm"],
        how="left", validate="one_to_one",
    )


def report(scores, predictions, threshold, series, witnesses):
    lines = [
        "# Earlier warning of high downstream flow", "",
        "The target is the first quarter-month with high SLON following a lower period. "
        f"High SLON is at least {threshold:,.0f} m³/s using the existing training percentile. "
        "SLON represents aggregate non-Ontario flow in the historical archive.", "",
        "We compare the archived seasonal-history score with two otherwise identical linear onset scores "
        "fitted to 1961–1980. The Britannia score adds the last seven fully observed days of flow and the "
        "change from the preceding seven days. Both weeks must be complete and unflagged. "
        "Alert cutoffs use 1981–2000 non-onset periods to target at most 10% false alarms; "
        "the 18 reserved years and the two flood years are held out from both fits and alert calibration.", "",
        "Gauge data end before the issue period, and the delayed case withholds the entire last completed "
        "quarter-month from both gauge and SLON inputs. Date alignment uses the documented legacy IJC "
        "quarter-month boundaries as a proxy for the LOSLR calendar. Actual publication timestamps are unavailable.", "",
        f"HYDAT provided {int(series.value.notna().sum()):,} daily discharge values and "
        f"{int(series.flag.notna().sum()):,} flagged dates. Comparisons use the same eligible dates for all "
        "scores and both delays. These are retrospective flow alerts and do not predict absolute stage.", "",
        "| Delay | Lead | Onsets | Archived hits / false alarms | SLON-only hits / false alarms | With Britannia hits / false alarms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    reserved = scores[scores.cohort.eq("reserved")]
    for (delay, lead), group in reserved.groupby(["delay", "lead"]):
        ordered = group.set_index("model").loc[list(MODELS)]
        counts = (f"{r.hits}/{r.false_alarms}" for r in ordered.itertuples())
        lines.append(f"| {delay} | {lead} | {ordered.onsets.iloc[0]} | " + " | ".join(counts) + " |")
    held = predictions[predictions.cohort.eq("reserved") & predictions.onset]
    lines += ["", "An onset is warned early if it has an alert at any tested lead. "
              "The pooled false-alarm rate is over non-onset issue/lead pairs.", "",
              "| Delay | Onsets with at least one matched lead | Archived warned | SLON-only warned | With Britannia warned | Pooled false-alarm rates in the same order |",
              "|---:|---:|---:|---:|---:|---|"]
    passes = []
    for delay, group in held.groupby("delay"):
        alerts = [f"{name}_alert" for name in MODELS]
        per_onset = group.groupby(["target_year", "target_qm"])[alerts].any()
        counts = per_onset.sum()
        part = reserved[reserved.delay.eq(delay)].groupby("model").agg(
            false_alarms=("false_alarms", "sum"), periods=("periods", "sum"), onsets=("onsets", "sum"))
        rates = part.false_alarms / (part.periods - part.onsets)
        rate_text = " / ".join(f"{rates[name]:.1%}" for name in MODELS)
        lines.append(f"| {delay} | {len(per_onset)} | " + " | ".join(str(n) for n in counts)
                     + f" | {rate_text} |")
        better = counts.iloc[-1] > counts.iloc[:-1].max()
        cost_ok = rates["slon_with_britannia"] <= min(rates["archived_slon"], rates["slon_only"]) + .02
        passes.append(bool(better and cost_ok))
    matched = witnesses[witnesses.onset.notna()]
    warning_counts = [int(matched[f"{name}_alert"].eq(True).sum()) for name in MODELS]
    warning_text = " / ".join(map(str, warning_counts))
    lines += ["", "The predeclared release-search gate requires more onsets warned than either comparator "
              "at both delays, with a pooled held-out false-alarm rate within two percentage points of "
              f"the better comparator at each delay. Gate passed: {len(passes) == 2 and all(passes)}.", "",
              f"The retrospective development check matched {len(matched)} of {len(witnesses)} "
              "hindsight release shifts within the tested leads. Only "
              f"{int(matched.onset.eq(True).sum())} later cuts coincided with a high-SLON onset. "
              f"The three models alerted on {warning_text} of these opportunities, respectively. "
              "A generic SLON onset and a downstream exceedance at a given Ontario release "
              "are different targets.", "",
              "Data source: Environment and Climate Change Canada, Water Survey of Canada, HYDAT station "
              "02KF005. [Station archive](https://wateroffice.ec.gc.ca/report/data_availability_e.html?parameter_type=Flow+and+Level&station=02KF005&type=historical) · "
              f"[ECCC data licence]({LICENCE}) · [IJC quarter-month calendar, section II](https://ijc.org/sites/default/files/L2.pdf)", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Reuse the saved HYDAT responses instead of downloading")
    parser.add_argument("--benchmark", type=Path, default=Path("results/loslr/slon_forecasts_verified_v1"))
    parser.add_argument("--witnesses", type=Path, default=Path("results/loslr/release_timing_final_v1/hindsight.csv"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.raw or args.output / "raw"
    if args.raw is None:
        raw.mkdir()
    series, requests = read_gauge(raw, download=args.raw is None)
    reference = NativeReference()
    frame = load_inputs(reference).iloc[48:].reset_index(drop=True)
    frozen = json.loads((args.benchmark / "protocol.json").read_text())
    assert sha256(reference.root / INPUT_FILE) == frozen["source_sha256"]
    threshold = frozen["high_flow_threshold_m3_s"]
    archived = pd.read_csv(args.benchmark / "predictions.csv")
    scores, predictions, fits = benchmark(frame, series, archived, threshold, load_protocol())
    witnesses = match_witnesses(args.witnesses, predictions)
    scores.to_csv(args.output / "scores.csv", index=False)
    predictions.to_csv(args.output / "alerts.csv", index=False)
    series.to_csv(args.output / "britannia_daily.csv")
    witnesses.to_csv(args.output / "witness_alerts.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "gauge_source": "ECCC / Water Survey of Canada / HYDAT",
        "licence": LICENCE, "archive_source_sha256": frozen["source_sha256"], "gauge_requests": requests,
        "archived_prediction_sha256": sha256(args.benchmark / "predictions.csv"),
        "hindsight_sha256": sha256(args.witnesses),
        "fit": FIT, "alert_calibration": CALIBRATION, "leads": LEADS, "delays": DELAYS,
        "false_alarm_budget": FALSE_ALARM_BUDGET, "high_flow_threshold_m3_s": threshold, "fits": fits,
    }, indent=2) + "\n")
    text = report(scores, predictions, threshold, series, witnesses)
    (args.output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
