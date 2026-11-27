"""Diagnose the archived 2019 seed-101 guarded replan and replay one recovery."""

from argparse import ArgumentParser
from ast import literal_eval
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from gymnasium.spaces import Box
import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import AdjustFlow
from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import (
    capped_follower, forecast_trajectory, run_policy,
)
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, FollowerTolerances, TransitState, digest, json_value, make_problem,
    solve_hard_follower, validate_config,
)

SCENARIO = "one_period_delay"
SEED = 101
STEM = f"{SCENARIO}_{SEED}_guard_after_retry"


def evaluate_schedules(cases, rows, state, schedules, tolerance):
    """Solve the follower and check Ontario with its planned Ottawa arrivals."""
    mechanism = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0., 1200., (1,)))
    results = []
    for name, problem in cases.items():
        for schedule in schedules:
            fixed = capped_follower(problem, rows, schedule, state[0])
            response = solve_hard_follower(fixed, tolerance)
            status = response.status
            if response.success:
                arrivals = fixed.routing_matrix @ response.release_m3_s + fixed.in_transit_contribution_m3_s
                _, status, _ = forecast_trajectory(
                    rows, schedule, arrivals, fixed.other_downstream_m3_s,
                    fixed.downstream_capacity_m3_s, state, mechanism, tolerance,
                )
            elif status != "infeasible":
                raise RuntimeError(f"Unresolved Ottawa solve for {name}: {response.message}")
            results.append({
                "case": name, "ontario_schedule_m3_s": schedule, "status": status,
                "ottawa_solver_status": response.status,
                "capacity_m3_s": fixed.downstream_capacity_m3_s.tolist(),
                "ottawa_release_m3_s": response.release_m3_s.tolist() if response.success else None,
                "ottawa_storage_m3": response.storage_m3.tolist() if response.success else None,
                "combined_downstream_m3_s": response.downstream_flow_m3_s.tolist() if response.success else None,
                "residuals": response.residuals,
            })
    return results


def audit_schedules(base, rows, state, schedules, true_inflow, stress, tolerance):
    """Hold inherited state fixed for all three inflow comparisons."""
    stressed_upper = np.asarray(base.storage_maximum_m3).copy()
    stressed_upper[1:] -= stress * base.duration_seconds[0]
    cases = {
        "original": base,
        "guarded": replace(base, storage_maximum_m3=stressed_upper),
        "realized_offline": replace(base, net_inflow_m3_s=true_inflow),
    }
    return evaluate_schedules(cases, rows, state, schedules, tolerance)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/loslr/ottawa_replan_guard_2019"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        name: args.archive / file for name, file in {
            "settings": "settings.json", "ontario": "ontario_inputs.csv",
            "trace": f"{STEM}_trace.csv", "candidates": f"{STEM}_candidates.csv",
            "forecast": f"{SCENARIO}_{SEED}_forecasts.npy",
        }.items()
    }
    provenance = {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()}
    trajectory = pd.read_csv(inputs["ontario"])
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    provenance["config"] = {"path": str(CONFIG), "sha256": digest(CONFIG)}
    config["known_ontario_executable_schedule_m3_s"] = (FLOW_SCALE * trajectory.ontFlow).tolist()
    scenario = next(row for row in config["scenarios"] if row["id"] == SCENARIO)
    bank = np.load(inputs["forecast"])
    history = json.loads(inputs["settings"].read_text())["ontario_window"]["initial_history"]
    archived = pd.read_csv(inputs["trace"])
    failed = archived.iloc[-1]
    if failed["status"] != "no_admissible_candidate" or failed.period != 6:
        raise ValueError("Expected the archived guarded stop at period 6")
    k = int(failed.period) - 1
    opts = pd.read_csv(inputs["candidates"])
    period_options = opts[(opts.period == k + 1) & opts.recovery_stage.isin(["primary", "expanded"])]
    schedules = sorted({
        tuple(literal_eval(entry.replace("np.float64(", "(")))
        for entry in period_options.loc[period_options.status == "admissible", "schedule_m3_s"]
    })
    queue = TransitState(tuple(literal_eval(failed.ottawa_start_queue_m3)))
    forecast = bank[k, :5]
    base = make_problem(config, scenario, k, failed.ottawa_start_storage_m3,
                        failed.ottawa_previous_release_m3_s, queue, forecast)
    true = np.array(scenario["realized_net_inflow_m3_s"][k:k + len(forecast)])
    rows = trajectory.iloc[k:k + len(forecast)].to_dict("records")
    state = (failed.ontario_start_m, failed.ontario_previous_release_m3_s / FLOW_SCALE,
             failed.ontario_previous_ice)
    stress = config["forecast_noise"]["standard_deviation_m3_s"]
    tolerance = FollowerTolerances(**config["acceptance"])
    cases = audit_schedules(base, rows, state, schedules, true, stress, tolerance)
    counts = {
        name: dict(Counter(row["status"] for row in cases if row["case"] == name))
        for name in ("original", "guarded", "realized_offline")
    }
    if not counts["original"].get("admissible") or counts["guarded"] != {"infeasible": len(schedules)}:
        raise ArithmeticError("The original candidate-set comparison did not reproduce")

    # With a one-period delay, the second-period capacity bounds the first release.
    dt = np.asarray(base.duration_seconds)
    room = base.storage_maximum_m3[1] - base.initial_storage_m3
    first_release_ceiling = max(
        base.downstream_capacity_m3_s[1] - base.other_downstream_m3_s[1]
        - base.in_transit_contribution_m3_s[1] - schedule[1]
        for schedule in schedules
    )
    maximum_release = np.array(base.release_maximum_m3_s[:3])
    maximum_release[0] = min(maximum_release[0], first_release_ceiling)
    required_volume = float(dt[:3] @ forecast[:3] + dt[0] * stress - room)
    upper_volume = float(dt[:3] @ maximum_release)
    if required_volume <= upper_volume:
        raise ArithmeticError("The expected period-3 storage conflict did not reproduce")

    options = dict(align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
                   recovery_guard_periods=1, replan_guard="after_retry", **history)
    baseline, _ = run_policy(trajectory, config, scenario, "external_screened", [], "limits", **options)
    audit = []
    trace, plans = run_policy(trajectory, config, scenario, "external_screened", audit, "limits",
                              delay_headroom_search=True, **options)
    if baseline[-1]["period"] != k + 1 or baseline[-1]["status"] != failed.status:
        raise ArithmeticError("The archived continuation failure did not reproduce")
    prefix_columns = {
        "ontario_applied_m3_s": tolerance.flow_m3_s,
        "ottawa_executed_release_m3_s": tolerance.flow_m3_s,
        "ottawa_end_storage_m3": tolerance.storage_m3,
        "ottawa_end_transit_m3": tolerance.storage_m3,
    }
    for column, atol in prefix_columns.items():
        for run in (baseline, trace):
            np.testing.assert_allclose([row[column] for row in run[:k]], archived[column].iloc[:k],
                                       rtol=0, atol=atol)
    selected = next(entry for entry in audit if entry["period"] == k + 1 and entry["selected"])
    if selected["recovery_stage"] != "delay_headroom" or trace[k]["status"] != "executed":
        raise ArithmeticError("The targeted search did not execute a recovery at period 6")
    witness = audit_schedules(base, rows, state, [selected["schedule_m3_s"]], true, stress, tolerance)
    guarded_witness = next(row for row in witness if row["case"] == "guarded")
    if guarded_witness["status"] != "admissible":
        raise ArithmeticError("The recovery plan failed the forecast checks with Ottawa's planned arrivals")
    np.testing.assert_allclose(trace[k]["ottawa_start_queue_m3"], queue.volumes_m3,
                               rtol=0, atol=tolerance.storage_m3)
    executed = [row for row in trace if row["status"] == "executed"]
    max_storage = max(row["ottawa_storage_violation_m3"] for row in executed)
    max_downstream = max(max(row["st_louis_flow_m3_s"] - row["capacity_m3_s"], 0) for row in executed)
    summary = {
        "period": k + 1, "seed": SEED, "scenario": SCENARIO, "sources": provenance,
        "inherited_state": baseline[-1], "storage_headroom_m3": room,
        "original_forecast_m3_s": forecast, "offline_realized_inflow_m3_s": true,
        "first_period_stress_m3_s": stress, "reserved_storage_m3": dt[0] * stress,
        "original_schedule_count": len(schedules), "outcomes": counts,
        "first_release_ceiling_old_schedules_m3_s": first_release_ceiling,
        "required_release_volume_first_three_periods_m3": required_volume,
        "maximum_release_volume_first_three_periods_old_schedules_m3": upper_volume,
        "new_selected_candidate": selected, "guarded_witness": guarded_witness,
        "recovery_period": trace[k],
        "executed_periods": len(executed), "last_status": trace[-1]["status"],
        "completed_within_tolerances": len(executed) == len(trajectory)
        and max_storage <= tolerance.storage_m3 and max_downstream <= tolerance.flow_m3_s,
        "max_storage_violation_m3": max_storage, "max_downstream_violation_m3_s": max_downstream,
        "final_transit_m3": executed[-1]["ottawa_end_transit_m3"],
    }
    for name, entry in provenance.items():
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise ArithmeticError(f"Source {name} changed during the diagnostic")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("summary", summary), ("frozen_solve_audit", cases), ("recovery_plans", plans)):
        (args.output / f"{name}.json").write_text(json.dumps(payload, default=json_value, indent=2) + "\n")
    pd.DataFrame(audit).to_csv(args.output / "recovery_candidates.csv", index=False)
    pd.DataFrame(trace).to_csv(args.output / "recovery_trace.csv", index=False)


if __name__ == "__main__":
    main()
