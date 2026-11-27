"""Compare targeted delay recovery with archived guarded runs on the frozen matrix."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import ONTARIO_COST_COMPONENTS
from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import run_policy
from bilevel_reservoirs.solvers.fixed_schedule_replay import CONFIG, digest, json_value, validate_config
from bilevel_reservoirs.solvers.stress_ottawa_response import HOLDOUT_SEEDS, SCENARIOS

YEARS = (1961, 2017, 2019)
BASELINE = "guard_after_retry"


def verify_prefix(archived, replayed, periods, tolerance):
    """Catch trajectory drift before the optional search changes a decision."""
    for name, atol in {
        "ontario_applied_m3_s": tolerance["flow_m3_s"],
        "ottawa_executed_release_m3_s": tolerance["flow_m3_s"],
        "ottawa_end_storage_m3": tolerance["storage_m3"],
        "ottawa_end_transit_m3": tolerance["storage_m3"],
        "st_louis_flow_m3_s": tolerance["flow_m3_s"],
        "ontario_end_m": 0.500001e-6,
    }.items():
        np.testing.assert_allclose(archived[name].iloc[:periods], replayed[name].iloc[:periods],
                                   rtol=0, atol=atol, err_msg=name)


def run_case(trajectory, config, scenario, bank, history, archived, baseline):
    audit = []
    start = perf_counter()
    trace, plans = run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits",
        align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
        recovery_guard_periods=1, replan_guard="after_retry", delay_headroom_search=True, **history,
    )
    elapsed = perf_counter() - start
    tried = sorted({entry["period"] for entry in audit if entry["recovery_stage"] == "delay_headroom"})
    selected = [entry["period"] for entry in audit
                if entry["recovery_stage"] == "delay_headroom" and entry["selected"]]
    replayed = pd.DataFrame(trace)
    executed = replayed.loc[replayed.status == "executed"]
    archived_executed = archived.loc[archived.status == "executed"]
    if len(archived_executed) != baseline.executed_periods or list(archived.period) != list(range(1, len(archived) + 1)):
        raise ArithmeticError("Archived summary and trace disagree")
    tolerance = config["acceptance"]
    if selected:
        prefix = selected[0] - 1
        if len(archived_executed) < prefix or len(executed) < prefix:
            raise ArithmeticError("The policy diverged before its first targeted selection")
        verify_prefix(archived_executed, executed, prefix, tolerance)
    else:
        if (len(executed) != len(archived_executed)
                or not replayed[["period", "status"]].equals(archived[["period", "status"]])):
            raise ArithmeticError("The original outcome changed without a targeted selection")
        verify_prefix(archived_executed, executed, len(executed), tolerance)
    # Common executed periods are the only periods for which paired costs exist.
    matched = min(len(executed), len(archived_executed))
    costs = {
        f"delta_{key}": float(executed[key].iloc[:matched].sum() - archived_executed[key].iloc[:matched].sum())
        for key in ONTARIO_COST_COMPONENTS
    } if matched else {}
    storage_violation = float(executed["ottawa_storage_violation_m3"].max()) if len(executed) else 0.0
    downstream_violation = max(float((executed.st_louis_flow_m3_s - executed.capacity_m3_s).max()), 0.0) if len(executed) else 0.0
    status = trace[-1]["status"]
    if storage_violation > tolerance["storage_m3"]:
        status = "realized_storage_violation"
    elif downstream_violation > tolerance["flow_m3_s"]:
        status = "realized_downstream_violation"
    elif len(executed) == len(trajectory):
        status = "completed"
    summary = {
        "baseline_status": baseline.status, "targeted_status": status,
        "baseline_executed": int(baseline.executed_periods), "targeted_executed": len(executed),
        "targeted_stop_period": trace[-1]["period"] if status != "completed" else None,
        "search_periods": tried, "selection_periods": selected,
        "baseline_persistence_retries": int(baseline.persistence_retry_selections),
        "search_candidate_count": sum(entry["recovery_stage"] == "delay_headroom" for entry in audit),
        "baseline_candidate_solves": int(baseline.candidate_ottawa_solves),
        "targeted_candidate_solves": sum("ottawa_candidate_status" in entry for entry in audit),
        "wall_seconds": elapsed, "matched_periods": matched,
        "max_storage_violation_m3": storage_violation, "max_downstream_violation_m3_s": downstream_violation,
        "terminal_transit_m3": float(executed.ottawa_end_transit_m3.iloc[-1]) if len(executed) else None,
        **costs,
    }
    return summary, replayed, plans, audit


def write_report(output, table):
    before = table.baseline_status == "completed"
    after = table.targeted_status == "completed"
    tried = table.search_periods.map(bool)
    selected = table.selection_periods.map(bool)
    recovered = ~before & after
    regressed = before & ~after
    lines = [
        "# Targeted Ontario commitment search under delayed Ottawa routing", "",
        f"The guarded baseline completed **{before.sum()}/{len(table)}** noisy cases. "
        f"The same policy with the targeted search completed **{after.sum()}/{len(table)}**. "
        f"It recovered {recovered.sum()} cases and lost {regressed.sum()} previously completed cases.", "",
        "| Ontario window | Archived guard | Guard with targeted search |", "|---|---:|---:|",
    ]
    for year in YEARS:
        in_year = table.year == year
        lines.append(f"| {year} | {(before & in_year).sum()}/{in_year.sum()} | "
                     f"{(after & in_year).sum()}/{in_year.sum()} |")
    lines.extend([
        "", "The additional search is eligible only at the guarded decision immediately after "
        "a successful persistence retry, once the original candidate sets have been exhausted. "
        f"It evaluated candidates in {tried.sum()} cases and selected a schedule in {selected.sum()}.", "",
        f"The maximum realized Ottawa storage violation was {table.max_storage_violation_m3.max():.6g} m³. "
        f"The maximum realized downstream violation was {table.max_downstream_violation_m3_s.max():.6g} m³/s "
        "(compare these with the tolerances in the study configuration).", "",
        f"Full-window Ontario cost comparisons are available for {(before & after).sum()} matched cases. "
        "Other cost comparisons cover only the shared executed prefix. Candidate solve counts count screening "
        "solves. Wall times are recorded for the new replay.", "",
        "All cases use the archived Ontario inputs, inherited Ontario release and ice state, "
        "and the original saved Ottawa forecasts. Ottawa inflows, routing and capacities are synthetic. "
        "The study preserves the four-lead Ontario and five-lead Ottawa horizons, the +700 m³/s check, "
        "and the 12-period endpoint with water in transit. The search is opt-in.", "",
        "## Cases with a search attempt or an incomplete outcome", "",
        "| Window | Scenario | Seed | Archived periods | Targeted periods | Archived retries | Search periods | Selected periods |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ])
    for case in table.loc[tried | ~before | ~after].itertuples():
        attempts = ", ".join(map(str, case.search_periods)) or "None"
        selections = ", ".join(map(str, case.selection_periods)) or "None"
        lines.append(f"| {case.year} | {case.scenario} | {case.seed} | {case.baseline_executed} | "
                     f"{case.targeted_executed} | {case.baseline_persistence_retries} | {attempts} | {selections} |")
    (output / "report.md").write_text("\n".join([*lines, ""]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    config_hash = digest(CONFIG)
    provenance = {str(CONFIG): config_hash}
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for year in YEARS:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        window = args.output / str(year)
        window.mkdir()

        def archived_file(name):
            path = archive / name
            provenance[str(path)] = digest(path)
            return path

        settings = json.loads(archived_file("settings.json").read_text())
        if (config_hash not in settings["input_sha256"].values()
                or settings["ordinary_replan_guards"][BASELINE] != "after_retry"
                or settings["recovery_guard_periods"][BASELINE] != 1
                or settings["recovery_guard_excess_inflow_m3_s"] != config["forecast_noise"]["standard_deviation_m3_s"]
                or (settings["ontario_horizon"], settings["ottawa_horizon"],
                    settings["align_f"], settings["candidate_family"]) != (4, 5, True, "limits")):
            raise ValueError(f"Archived settings do not match the targeted search study for {year}")
        history = settings["ontario_window"]["initial_history"]
        trajectory = pd.read_csv(archived_file("ontario_inputs.csv"))
        if len(trajectory) != 12 or not (trajectory.Year == year).all():
            raise ValueError(f"Invalid Ontario replay window for {year}")
        config["known_ontario_executable_schedule_m3_s"] = (FLOW_SCALE * trajectory.ontFlow).tolist()
        index = pd.read_csv(archived_file("summary.csv"))
        baseline = index.loc[(index.policy == BASELINE) & (index.forecast_mode == "noisy")].set_index(["scenario", "seed"])
        expected = {(name, seed) for name in SCENARIOS for seed in HOLDOUT_SEEDS}
        if set(baseline.index) != expected or len(baseline) != len(expected):
            raise ValueError(f"Archived noisy matrix is incomplete for {year}")
        for scenario in config["scenarios"]:
            name = scenario["id"]
            if name not in SCENARIOS:
                continue
            for seed in HOLDOUT_SEEDS:
                bank = np.load(archived_file(f"{name}_{seed}_forecasts.npy"))
                archived = pd.read_csv(archived_file(f"{name}_{seed}_{BASELINE}_trace.csv"))
                summary, trace, plans, audit = run_case(
                    trajectory, config, scenario, bank, history, archived, baseline.loc[(name, seed)],
                )
                summaries.append({"year": year, "scenario": name, "seed": seed, **summary})
                stem = window / f"{name}_{seed}"
                trace.to_csv(stem.with_name(f"{stem.name}_trace.csv"), index=False)
                pd.DataFrame(audit).to_csv(stem.with_name(f"{stem.name}_candidates.csv"), index=False)
                stem.with_name(f"{stem.name}_plans.json").write_text(
                    json.dumps(plans, default=json_value, indent=2) + "\n"
                )
                print(f"{year} {name} {seed}: {summary['baseline_status']} -> {summary['targeted_status']}", flush=True)
    for path, checksum in provenance.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"An input changed during the study: {path}")
    table = pd.DataFrame(summaries)
    table.to_csv(args.output / "summary.csv", index=False)
    write_report(args.output, table)
    settings = {"archive_input_sha256": provenance, "years": YEARS, "seeds": HOLDOUT_SEEDS,
                "scenarios": SCENARIOS, "baseline": BASELINE, "delay_headroom_search": True}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
