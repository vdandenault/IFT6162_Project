"""Screen preregistered one-sided SLON bounds and gate annual confirmation."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_inputs
from .native import INPUT_FILE, NativeReference, sha256
from .planning import MPCSettings
from .slon_forecast import DELAYS, aligned_pairs, fit_forecasts, load_protocol
from .slon_mpc_replay import replay_year, start_sessions

FIT_END = 1960
RESIDUAL_QUANTILE = .90
RISK_QUANTILE = .75
UPSTREAM_ALLOWANCE = .10
VARIANTS = ("seasonal_upper_all", "seasonal_upper_risk", "persistence_upper_risk")
FOLLOW_UP = ("seasonal_half_upper", "seasonal_floor_persistence")
REFERENCES = ("persistence", "seasonal_history")


def calibrate(frame, training):
    start, end = training
    fit = frame[frame.Year.between(start, FIT_END)]
    seasonal = fit.groupby("QM").stlouisontOut.mean()
    risk = float(fit.stlouisontOut.quantile(RISK_QUANTILE))
    results = {}
    for delay in DELAYS:
        predicted, _ = fit_forecasts(aligned_pairs(frame, 1, delay), [start, FIT_END], seasonal)
        sample = predicted[predicted.target_year.between(FIT_END + 1, end)]
        is_risk = sample[["persistence", "seasonal_history"]].max(axis=1).ge(risk)
        results[delay] = {"risk_threshold_m3_s": risk, "calibration_periods": len(sample),
                          "risk_periods": int(is_risk.sum())}
        for variant in VARIANTS:
            method = "persistence" if variant.startswith("persistence") else "seasonal_history"
            subset = sample.loc[is_risk] if variant.endswith("risk") else sample
            residual = subset.target_m3_s - subset[method]
            results[delay][variant] = max(0., float(residual.quantile(RESIDUAL_QUANTILE, interpolation="higher")))
    return results


def make_paths(bank, delay, method, bounds):
    issued = bank[bank.delay.eq(delay)]
    column = "persistence" if method.startswith("persistence") else "seasonal_history"
    wide = issued.pivot(index="issue_index", columns="lead", values=column)
    if method in VARIANTS:
        mask = np.ones(len(wide), dtype=bool)
        if method.endswith("risk"):
            lead1 = issued[issued.lead.eq(1)].set_index("issue_index").loc[wide.index]
            mask = lead1[["persistence", "seasonal_history"]].max(axis=1).ge(bounds["risk_threshold_m3_s"])
        wide.loc[mask, 1] += bounds[method]
    if method == "seasonal_half_upper":
        wide[1] += .5 * bounds["seasonal_upper_all"]
    if method == "seasonal_floor_persistence":
        persisted = issued[issued.lead.eq(1)].set_index("issue_index").persistence
        wide[1] = np.maximum(wide[1], persisted.loc[wide.index])
    assert wide.shape[1] == 4 and wide.notna().all().all()
    return dict(zip(wide.index, wide.to_numpy(), strict=True))


def run_sample(starts, frame, settings, bank, calibration, archived, methods):
    records = []
    for delay in DELAYS:
        for method in methods:
            if method in REFERENCES:
                existing = archived[archived.delay.eq(delay) & archived.method.eq(method) & archived.year.isin(starts)]
                assert len(existing) == 48 * len(starts)
                records.extend(existing.to_dict("records"))
                continue
            paths = make_paths(bank, delay, method, calibration[delay])
            for year, start in starts.items():
                records.extend(replay_year(start, frame, settings, paths, year, method, delay))
            print(f"{method}, delay {delay}: {len(starts)} years complete", flush=True)
    return pd.DataFrame(records)


def aggregate(trace):
    return trace.groupby(["delay", "method"], as_index=False).agg(
        years=("year", "nunique"), periods=("period", "size"),
        upstream=("ontario_exposure_m", "sum"), downstream=("downstream_exposure_m", "sum"),
        minimum_downstream_headroom=("realized_pc_headroom_m", "min"),
        feasible_plans=("plan_feasible", "sum"),
    )


def select(summary, variants):
    metrics = summary.set_index(["method", "delay"])
    baseline = metrics.loc["persistence"]
    passes = []
    for method in variants:
        row = metrics.loc[method]
        if (row.periods.eq(baseline.periods).all() and row.downstream.lt(baseline.downstream).all()
                and row.upstream.le(baseline.upstream + UPSTREAM_ALLOWANCE).all()):
            passes.append(method)
    return min(passes, key=lambda method: metrics.loc[method].downstream.sum()) if passes else None


def table(summary):
    lines = ["| Data delay | Forecast | Completed years | Ontario exposure | Lake St. Louis exposure | Worst downstream headroom (m) |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.delay} | {row.method} | {row.years} | {row.upstream:.3f} | "
                     f"{row.downstream:.3f} | {row.minimum_downstream_headroom:+.2f} |")
    return lines


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("results/loslr/slon_mpc_replay_verified_v1"))
    parser.add_argument("--follow-up", action="store_true", help="Screen smaller, predeclared alternatives")
    args = parser.parse_args()
    files = {name: args.archive / name for name in ("protocol.json", "forecast_bank.csv", "trajectories.csv")}
    hashes = {name: sha256(path) for name, path in files.items()}
    archived_plan = json.loads(files["protocol.json"].read_text())
    protocol = load_protocol()
    reference = NativeReference()
    assert sha256(reference.root / INPUT_FILE) == archived_plan["source_sha256"]
    frame = load_inputs(reference)
    history = frame.iloc[48:][["Year", "QM", "stlouisontOut"]].reset_index(drop=True)
    bounds = calibrate(history, protocol["training_years"])
    args.output.mkdir(parents=True, exist_ok=False)
    variants = FOLLOW_UP if args.follow_up else VARIANTS
    frozen = {"fit_end": FIT_END, "training_years": protocol["training_years"],
              "residual_quantile": RESIDUAL_QUANTILE, "risk_quantile": RISK_QUANTILE,
              "upstream_allowance_m_periods_per_development_cohort": UPSTREAM_ALLOWANCE,
              "gate": "Both delays complete and improve downstream relative to persistence within upstream allowance",
              "variants": variants, "calibration": bounds, "archive_sha256": hashes}
    (args.output / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    archived = pd.read_csv(files["trajectories.csv"])
    bank = pd.read_csv(files["forecast_bank.csv"])
    settings = MPCSettings(**archived_plan["settings"])
    development_years = archived_plan["cohorts"]["development"]
    reserved_years = archived_plan["cohorts"]["reserved"]
    starts = start_sessions(reference, frame, development_years + reserved_years)
    development = run_sample({year: starts[year] for year in development_years}, frame, settings, bank,
                             bounds, archived, (*REFERENCES, *variants))
    summary = aggregate(development)
    winner = select(summary, variants)
    development.to_csv(args.output / "development_trajectories.csv", index=False)
    summary.to_csv(args.output / "development_summary.csv", index=False)
    lines = [
        "# First-period upper SLON forecasts", "",
        "The first forecast period alone can receive a nonnegative margin. Later leads, the controller, "
        "realized flows, limits, costs and warm starts match the archived replay. The risk variants "
        "activate when the larger of persistence and seasonal history exceeds the training 75th percentile. "
        "Margin candidates use the 90th percentile of the applicable late-training residuals, with the "
        "seasonal model fitted only through 1960 for calibration. The replay uses the frozen refit through 1980.", "",
        "The development gate requires lower total downstream exposure at both delays, full completion, "
        "and at most 0.10 metre-periods of added upstream exposure across the 20 years at either delay. "
        "Exposure totals are in metre-periods.", "", *table(summary), "",
    ]
    if args.follow_up:
        lines[4:4] = ["This second development screen tests half the unconditional seasonal margin "
                      "and a floor at persistence, applied to lead one only.", ""]
    confirmed = None
    if winner:
        print(f"{winner} passed the development gate. Running reserved comparison.", flush=True)
        reserved = run_sample({year: starts[year] for year in reserved_years}, frame, settings, bank,
                              bounds, archived, (*REFERENCES, winner))
        comparison = aggregate(reserved)
        comparison.to_csv(args.output / "reserved_summary.csv", index=False)
        reserved.to_csv(args.output / "reserved_trajectories.csv", index=False)
        lines += [f"{winner} passed the development gate and was frozen for the reserved replay.", "",
                  "## Reserved years", "", *table(comparison), ""]
        confirmed = select(comparison, (winner,)) == winner
        lines += [f"Reserved replay met the same exposure and completion gate: {confirmed}.", ""]
    else:
        lines += ["No variant passed the development gate. The reserved comparison did not trigger.", ""]
    assert hashes == {name: sha256(path) for name, path in files.items()}
    (args.output / "decision.json").write_text(json.dumps({
        "selected_variant": winner, "confirmed_on_reserved": confirmed,
    }, indent=2) + "\n")
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
