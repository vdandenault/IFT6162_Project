"""Diagnose period-6 overflows and compare causal storage-guard triggers."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import run_policy
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, FollowerTolerances, digest, json_value, validate_config,
)
from bilevel_reservoirs.solvers.stress_ottawa_response import offline_recoverability
from bilevel_reservoirs.solvers.study_delay_headroom import verify_prefix
from reservoir_control.systems.ottawa.delay_diagnostic import explicit_storage_feasibility
from reservoir_control.systems.ottawa.hard_follower import FollowerWeights, HardFollowerProblem

ARMS = ("after_retry", "always", "tight_headroom")


def run(trajectory, config, scenario, bank, history, guard, *, force_period_five=False):
    audit = []
    trace, plans = run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits",
        align_f=True, ottawa_forecast_bank=bank, recovery="persistence", recovery_guard_periods=1,
        replan_guard=guard, guard_at_periods=(5,) if force_period_five else (),
        delay_headroom_search=True, delay_headroom_on_exhaustion=True,
        matched_response_ranking=True, **history,
    )
    executed = [row for row in trace if row["status"] == "executed"]
    max_storage = max((row["ottawa_storage_violation_m3"] for row in executed), default=0.)
    max_downstream = max((max(0., row["st_louis_flow_m3_s"] - row["capacity_m3_s"])
                          for row in executed), default=0.)
    if max_storage > config["acceptance"]["storage_m3"]:
        status = "realized_storage_violation"
    elif max_downstream > config["acceptance"]["flow_m3_s"]:
        status = "realized_downstream_violation"
    elif len(executed) == len(trajectory):
        status = "completed"
    else:
        status = trace[-1]["status"]
    result = {
        "status": status, "executed_periods": len(executed), "last_decision": trace[-1]["period"],
        "max_storage_violation_m3": max_storage, "max_downstream_violation_m3_s": max_downstream,
        "ordinary_guard_periods": sorted({row["period"] for row in audit
            if "recovery_storage_maximum_m3" in row
            and row["recovery_stage"] not in ("persistence_retry", "delay_headroom_retry")}),
        "candidate_solves": sum("ottawa_candidate_status" in row for row in audit),
    }
    return result, trace, plans, audit


def diagnose_overflow(trajectory, config, scenario, baseline_trace, baseline_plans, forced_trace, history):
    failed = baseline_trace[-1]
    k = failed["period"] - 1
    if k != 5 or failed["ottawa_storage_violation_m3"] <= config["acceptance"]["storage_m3"]:
        raise ValueError("Expected the archived period-6 storage overflow")
    plan = next(plan["problem"] for plan in baseline_plans if plan["period"] == k + 1)
    original = HardFollowerProblem(**{**plan, "weights": FollowerWeights(**plan["weights"])})
    tolerance = FollowerTolerances(**config["acceptance"])
    realized = np.array(scenario["realized_net_inflow_m3_s"][k:k + len(original.duration_seconds)])
    # Preserve the selected Ontario schedule and its forecast F caps in the
    # first check. The other leads keep the same issued Ottawa forecasts.
    first_only = np.array(original.net_inflow_m3_s)
    first_only[0] = realized[0]
    fixed_lp = {
        "saved_forecast": explicit_storage_feasibility(original, tolerance),
        "realized_first_lead": explicit_storage_feasibility(replace(original, net_inflow_m3_s=first_only), tolerance),
        "realized_five_leads": explicit_storage_feasibility(replace(original, net_inflow_m3_s=realized), tolerance),
    }
    if fixed_lp["saved_forecast"]["status"] != "feasible" or any(
        item["status"] == "solver_failure" for item in fixed_lp.values()
    ):
        raise ArithmeticError("The fixed-schedule LP did not reproduce the original planning feasibility")

    # The second check allows a different Ontario schedule, carries the exact
    # observed queue, and uses realized inflows through the full endpoint.
    anchor = config["known_ontario_executable_schedule_m3_s"][k:]
    fixed = [*original.ontario_executable_release_m3_s, *anchor[len(original.duration_seconds):]]
    # Ottawa's first action arrives in lead two. Search for a witness by
    # freeing capacity there, with a few paired first-lead edits for native ramps.
    arrival_edits = (
        [schedule[0] - first, schedule[1] - second, *schedule[2:]]
        for schedule in (fixed, anchor) for first in (0, 100, 200)
        for second in range(10, 701, 10)
        if first <= schedule[0] and second <= schedule[1]
    )
    offline = offline_recoverability(trajectory, config, scenario, baseline_trace[:k],
                                     additional_schedules=arrival_edits, **history)
    dt = original.duration_seconds[0]
    room = original.storage_maximum_m3[1] - original.initial_storage_m3
    required = max(original.release_minimum_m3_s[0], realized[0] - room / dt)
    later_room = (original.downstream_capacity_m3_s[1] - original.ontario_executable_release_m3_s[1]
                  - original.other_downstream_m3_s[1] - original.in_transit_contribution_m3_s[1])
    first_release_ceiling = min(original.release_maximum_m3_s[0], later_room)
    forced_p5 = next(row for row in forced_trace if row["period"] == 5)
    forced_p6 = next((row for row in forced_trace if row["period"] == 6), None)
    witness = offline["fixed_continuation_lp"]
    if offline["finding"] == "joint_alternative_schedule_witness":
        witness = offline["alternative_continuation"]["ottawa_lp"]
    if offline["finding"] in ("joint_fixed_schedule_witness", "joint_alternative_schedule_witness"):
        offline["accepted_terminal_transit_m3"] = witness["release_m3_s"][-1] * dt
    return {
        "period_6_storage_room_m3": room, "required_first_release_m3_s": required,
        "planned_first_release_m3_s": failed["ottawa_executed_release_m3_s"],
        "fixed_schedule_first_release_ceiling_m3_s": first_release_ceiling,
        "original_forecast_m3_s": original.net_inflow_m3_s,
        "realized_ottawa_m3_s": realized, "fixed_lp": fixed_lp, "full_remaining_offline": offline,
        "baseline_period_5": baseline_trace[k - 1], "baseline_period_6": failed,
        "forced_guard_period_5": forced_p5, "forced_guard_period_6": forced_p6,
    }


def write_report(output, diagnostics, table):
    lines = [
        "# Ottawa storage guard study", "",
        "The period-6 checks freeze the inherited storage, previous releases and routing queue. "
        "The first LP preserves the selected Ontario commitments and forecast caps. The full-period "
        "offline check can use a different Ontario schedule and uses realized future inflows. "
        "Its witness preserves end-of-window transit.", "",
        "| Ontario window and scenario | Required Ottawa release (m³/s) | Ceiling with selected schedule (m³/s) | Fixed schedule with realized first lead | Full remaining offline finding | Period-5 guard outcome | Change in period-6 storage room (million m³) |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for case in diagnostics:
        forced = case["forced_guard_period_6"]
        change = ((case["baseline_period_6"]["ottawa_start_storage_m3"] - forced["ottawa_start_storage_m3"]) / 1e6
                  if forced is not None and "ottawa_start_storage_m3" in forced else None)
        room = f"{change:.2f}" if change is not None else "—"
        lines.append(f"| {case['year']} {case['scenario']} | {case['required_first_release_m3_s']:.1f} | "
                     f"{case['fixed_schedule_first_release_ceiling_m3_s']:.1f} | "
                     f"{case['fixed_lp']['realized_first_lead']['status']} | {case['full_remaining_offline']['finding']} | "
                     f"{case['forced_outcome']['status']} | {room} |")
    lines.extend([
        "", "The comparison uses the same 36 forecast-noise cases. The tight-headroom trigger retains "
        "the after-retry check and also applies the guard when current storage room, less the positive "
        "forecast net fill at the existing reference release, is at most one stress-period volume. "
        "All arms use the same +700 m³/s stress, physical limits, costs and search rules.", "",
        "| Ordinary guard rule | Completed | Storage overflows | No feasible candidate | Stops after baseline completion | Such stops with a continuation witness |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        rows = table[table.arm == arm]
        lines.append(f"| {arm} | {(rows.status == 'completed').sum()}/{len(rows)} | "
                     f"{(rows.status == 'realized_storage_violation').sum()} | "
                     f"{(rows.status == 'no_admissible_candidate').sum()} | "
                     f"{rows.stops_after_baseline_completion.sum()} | {rows.continuation_witness.sum()} |")
    lines.extend([
        "", "A stop has a continuation witness if the causal baseline completed from the same inherited "
        "state or a separate offline solve found a continuation through period 12. "
        "The forecast seeds are reused, the Ontario forcing is historical, and the Ottawa conditions are synthetic.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    sources = {}

    def source(path):
        sources[str(path)] = digest(path)
        return path

    source(CONFIG)
    prior = args.archive_root / "ottawa_candidate_proposals_fresh_verified"
    prior_settings = json.loads(source(prior / "settings.json").read_text())
    if sources[str(CONFIG)] not in prior_settings["inputs_sha256"].values():
        raise ValueError("Follower configuration differs from the archived comparison")
    index = pd.read_csv(source(prior / "runs.csv"))
    baseline = index[index.arm == "baseline"]
    if len(baseline) != 36 or baseline.duplicated(["year", "scenario", "seed"]).any():
        raise ValueError("Expected 36 distinct archived baseline cases")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics = [], []
    for year, cases in baseline.groupby("year"):
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        settings = json.loads(source(archive / "settings.json").read_text())
        history = settings["ontario_window"]["initial_history"]
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        window = args.output / str(year)
        window.mkdir()
        for case in cases.itertuples():
            scenario = next(row for row in config["scenarios"] if row["id"] == case.scenario)
            stem = f"{case.scenario}_{case.seed}"
            bank = np.load(source(prior / str(year) / f"{stem}_forecasts.npy"))
            saved = pd.read_csv(source(prior / str(year) / f"{stem}_baseline_trace.csv"))
            states = {}
            for arm in ARMS:
                outcome, trace, plans, audit = run(trajectory, local, scenario, bank, history, arm)
                frame = pd.DataFrame(trace)
                if arm == "after_retry":
                    if outcome["status"] != case.status or outcome["executed_periods"] != case.executed_periods:
                        raise ArithmeticError("Archived baseline outcome did not reproduce")
                    verify_prefix(saved, frame, case.executed_periods, config["acceptance"])
                stopped = case.status == "completed" and outcome["status"] == "no_admissible_candidate"
                shared_prefix = False
                if stopped:
                    try:
                        verify_prefix(saved, frame, outcome["executed_periods"], config["acceptance"])
                        shared_prefix = True
                    except AssertionError:
                        pass  # A different inherited state needs its own offline witness.
                offline = (offline_recoverability(trajectory, local, scenario, trace[:-1], **history)
                           if stopped and not shared_prefix else None)
                has_witness = shared_prefix or offline is not None and offline["finding"] in (
                    "joint_fixed_schedule_witness", "joint_alternative_schedule_witness",
                )
                summaries.append({
                    "year": year, "scenario": case.scenario, "seed": case.seed, "arm": arm,
                    "stops_after_baseline_completion": stopped, "continuation_witness": has_witness,
                    "continuation_finding": "completed_causal_baseline_with_same_prefix" if shared_prefix
                    else offline["finding"] if offline else None, **outcome,
                })
                states[arm] = outcome["status"]
                frame.to_csv(window / f"{stem}_{arm}_trace.csv", index=False)
                if stopped:
                    (window / f"{stem}_{arm}_offline.json").write_text(
                        json.dumps({"stop": trace[-1], "offline": offline,
                                    "same_prefix_baseline_trace": str(window / f"{stem}_after_retry_trace.csv")
                                    if shared_prefix else None}, default=json_value, indent=2) + "\n"
                    )
                if arm == "after_retry" and case.status == "realized_storage_violation":
                    forced, forced_trace, forced_plans, _ = run(
                        trajectory, local, scenario, bank, history, arm, force_period_five=True,
                    )
                    verify_prefix(frame, pd.DataFrame(forced_trace), 4, config["acceptance"])
                    diagnostic = diagnose_overflow(trajectory, local, scenario, trace, plans, forced_trace, history)
                    diagnostics.append({"year": year, "scenario": case.scenario, "seed": case.seed,
                                        "forced_outcome": forced, **diagnostic})
                    pd.DataFrame(forced_trace).to_csv(window / f"{stem}_forced_period_5_trace.csv", index=False)
                    (window / f"{stem}_forced_period_5_plans.json").write_text(
                        json.dumps(forced_plans, default=json_value, indent=2) + "\n"
                    )
            print(year, stem, states, flush=True)
    if len(diagnostics) != 3 or {case["seed"] for case in diagnostics} != {115}:
        raise ArithmeticError("Expected the three seed-115 diagnostics")
    for path, checksum in sources.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"An archived input changed: {path}")
    table = pd.DataFrame(summaries)
    table.to_csv(args.output / "runs.csv", index=False)
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, default=json_value, indent=2) + "\n")
    (args.output / "settings.json").write_text(json.dumps({"input_sha256": sources}, indent=2) + "\n")
    write_report(args.output, diagnostics, table)


if __name__ == "__main__":
    main()
