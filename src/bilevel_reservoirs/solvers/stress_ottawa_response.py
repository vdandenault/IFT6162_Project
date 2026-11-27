"""Matched noisy inflow stress test for the coupled Ottawa planning policies."""

from argparse import ArgumentParser
from dataclasses import replace
from itertools import chain
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from bilevel_reservoirs.controllers.operator import AdjustFlow, ONTARIO_COST_COMPONENTS
from bilevel_reservoirs.solvers.compare_ottawa_response import (
    CONFIG, FollowerTolerances, digest, forecast_f_ceiling, json_value, native_step, run_policy,
)
from bilevel_reservoirs.solvers.fixed_schedule_replay import TransitState, make_problem, round_d
from bilevel_reservoirs.simulators.physics import CONVERSION, FLOW_SCALE
from reservoir_control.systems.ottawa.delay_diagnostic import explicit_storage_feasibility
from reservoir_control.systems.ottawa.horizon_study import forecast_bank

SCENARIOS = ("one_period_delay_feasible", "one_period_delay")
POLICIES = ("external_screened", "anticipating")
HOLDOUT_SEEDS = (101, 102, 103, 104, 105, 106)


def spring_window(year, config, scenarios):
    """Warm Ontario with the existing causal rule curve up to the window."""
    from bilevel_reservoirs.simulators.native import native_reference
    from reservoir_control.systems.loslr.causal import CausalOntarioSession
    from reservoir_control.systems.loslr.data import load_inputs
    from reservoir_control.systems.loslr.model import OntarioModel, Plan2014
    from reservoir_control.systems.loslr.native import INPUT_FILE

    reference = native_reference()
    frame = load_inputs(reference, last_year=year)
    session = CausalOntarioSession(OntarioModel(reference), frame)
    rule = Plan2014(reference)
    start = frame.index[(frame.Year == year) & (frame.QM == 13)].item()
    while session.index < start:
        decision, time = session.information()
        session.step(rule.action(session.state, decision.forecast, time))
    state = session.state
    window = frame.iloc[start:start + 12].copy()
    window["ontLevelBOQ"] = state.level_m
    # A common nominal Ontario schedule for both scenarios and all seeds. Its
    # anchor only uses the initial release and the declared synthetic envelope.
    ceiling = min(
        cap - background - config["reference_release_m3_s"]
        for scenario in scenarios.values()
        for cap, background in zip(scenario["downstream_capacity_m3_s"],
                                   config["known_other_downstream_forecast_m3_s"], strict=True)
    )
    window["ontFlow"] = min(state.previous_release_m3_s, ceiling) / FLOW_SCALE
    history = {"previous_flow": state.previous_release_m3_s / FLOW_SCALE, "previous_ice": state.previous_ice}
    return window, history, [reference.root / INPUT_FILE]



def offline_recoverability(trajectory, config, scenario, executed, *, previous_flow=595.0, previous_ice=2.0,
                           additional_schedules=()):
    """Test one fixed continuation and a relaxation of all Ontario decisions."""
    k = len(executed)
    if k == len(trajectory):
        return {"finding": "no_periods_remaining"}
    previous = executed[-1] if executed else None
    level = previous["ontario_end_m"] if previous else trajectory.iloc[0]["ontLevelBOQ"]
    previous_flow = previous["ontario_applied_m3_s"] / FLOW_SCALE if previous else previous_flow
    previous_ice = trajectory.iloc[k - 1]["iceInd"] if previous else previous_ice
    storage = previous["ottawa_end_storage_m3"] if previous else config["initial_storage_m3"]
    release = previous["ottawa_executed_release_m3_s"] if previous else config["previous_executed_release_m3_s"]
    transit = previous["ottawa_end_queue_m3"] if previous else scenario["initial_transit_volumes_m3"]
    queue = TransitState(tuple(transit))
    tolerance = FollowerTolerances(**config["acceptance"])
    if not config["storage_minimum_m3"] - tolerance.storage_m3 <= storage <= config["storage_maximum_m3"] + tolerance.storage_m3:
        return {"finding": "already_outside_ottawa_storage_bounds", "storage_m3": storage}

    window = trajectory.iloc[k:].to_dict("records")
    flows = (trajectory["ontFlow"].to_numpy() * FLOW_SCALE).tolist()
    problem = make_problem(
        {**config, "known_ontario_executable_schedule_m3_s": flows},
        scenario, k, storage, release, queue, scenario["realized_net_inflow_m3_s"][k:],
    )
    relaxed = explicit_storage_feasibility(
        replace(problem, ontario_executable_release_m3_s=np.zeros(len(window))), tolerance,
    )
    limits = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0.0, np.inf, (1,), dtype=np.float64))

    def check_continuation(schedule):
        caps = np.array(problem.downstream_capacity_m3_s, dtype=float)
        begin = level
        for lead, (row, flow) in enumerate(zip(window, schedule, strict=True)):
            caps[lead] = min(caps[lead], forecast_f_ceiling(begin, row["ptclaireR"], flow))
            begin += round_d((row["ontNTS"] / FLOW_SCALE - flow / FLOW_SCALE) / CONVERSION, 6)
        fit = explicit_storage_feasibility(
            replace(problem, ontario_executable_release_m3_s=schedule, downstream_capacity_m3_s=caps),
            tolerance,
        )
        if fit["status"] != "feasible":
            return fit, None
        arrivals = problem.routing_matrix @ fit["release_m3_s"] + problem.in_transit_contribution_m3_s
        current = level, previous_flow, previous_ice
        for lead, row in enumerate(window):
            flow = schedule[lead] / FLOW_SCALE
            non_ontario = problem.other_downstream_m3_s[lead] + arrivals[lead]
            forcing = {**row, "stlouisontOut": non_ontario, "slonFlow_QM1": non_ontario}
            _, allowed, regime = native_step(forcing, row["ontNTS_QM1"], flow, current, limits)
            if abs(allowed - flow) * FLOW_SCALE > tolerance.flow_m3_s:
                return fit, {"period": k + lead + 1, "native_regime": regime,
                             "allowed_m3_s": allowed * FLOW_SCALE}
            actual_end = current[0] + round_d((row["ontNTS"] / FLOW_SCALE - flow) / CONVERSION, 6)
            current = actual_end, flow, row["iceInd"]
        return fit, None

    schedule = flows[k:]
    fixed, witness_conflict = check_continuation(schedule)
    alternative = None
    if fixed["status"] != "feasible" or witness_conflict is not None:
        # Offline witness search only. Reduce the first Ontario release in native
        # increments and retain the rest of the verified schedule.
        first_period_edits = ([schedule[0] - reduction, *schedule[1:]]
                              for reduction in range(10, 701, 10) if reduction <= schedule[0])
        for proposed in chain(additional_schedules, first_period_edits):
            fit, conflict = check_continuation(proposed)
            if fit["status"] == "feasible" and conflict is None:
                alternative = {"schedule_m3_s": proposed, "ottawa_lp": fit}
                break
    if relaxed["status"] == "infeasible":
        finding = "infeasible_even_with_zero_ontario_and_study_caps_only"
    elif fixed["status"] == "feasible" and witness_conflict is None:
        finding = "joint_fixed_schedule_witness"
    elif alternative is not None:
        finding = "joint_alternative_schedule_witness"
    else:
        finding = "undetermined_for_alternative_ontario_schedules"
    return {"finding": finding, "from_period": k + 1, "initial_ottawa_storage_m3": storage,
            "initial_queue_m3": transit, "zero_ontario_relaxation_status": relaxed["status"],
            "fixed_continuation_lp": fixed, "fixed_witness_native_conflict": witness_conflict,
            "alternative_continuation": alternative}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--fixed-schedule", type=Path)
    parser.add_argument("--historical-year", type=int, choices=(2017, 2019),
                        help="Use an April–June Ontario window with synthetic Ottawa forcing")
    parser.add_argument("--holdout", action="store_true",
                        help="Frozen one-period versus unguarded retry with fresh seeds")
    parser.add_argument("--replan-guard", action="store_true",
                        help="With --holdout, also check the next replan and all ordinary replans")
    parser.add_argument("--recovery", action="store_true",
                        help="Compare screened baseline, widened search and persistence recovery")
    parser.add_argument("--storage-guard", action="store_true",
                        help="With --recovery, compare one- and two-period retry storage checks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replan_guard and not args.holdout:
        parser.error("--replan-guard requires --holdout")
    if args.storage_guard and not args.recovery:
        parser.error("--storage-guard requires --recovery")
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    config = json.loads(CONFIG.read_text())
    noise = config["forecast_noise"]
    scenarios = {case["id"]: case for case in config["scenarios"] if case["id"] in SCENARIOS}
    if args.holdout and (args.recovery or args.storage_guard):
        parser.error("--holdout supplies its own fixed comparison arms")
    if args.historical_year:
        if not args.holdout or args.trajectory or args.fixed_schedule:
            parser.error("--historical-year requires --holdout and supplies its own trajectory")
        trajectory, history, input_paths = spring_window(args.historical_year, config, scenarios)
    else:
        if not args.trajectory or not args.fixed_schedule:
            parser.error("Supply --trajectory and --fixed-schedule")
        trajectory = pd.read_csv(args.trajectory)
        fixed = pd.read_csv(args.fixed_schedule)
        if not trajectory[["Year", "QM"]].equals(fixed[["Year", "QM"]]):
            parser.error("Trajectory and fixed schedule dates must match")
        trajectory["ontFlow"] = fixed["ontFlow"]
        history = {"previous_flow": 595.0, "previous_ice": 2.0}
        input_paths = [args.trajectory, args.fixed_schedule]
    args.output.mkdir(parents=True, exist_ok=False)
    if args.holdout:
        trajectory.to_csv(args.output / "ontario_inputs.csv", index=False)
    summary, pairs = [], []
    arms = (
        {"external_screened": None, "expanded": "expand", "persistence_retry": "persistence"}
        if args.recovery else dict.fromkeys(POLICIES)
    )
    guard_periods = {"retry_guard_h1": 1, "retry_guard_h2": 2} if args.storage_guard else {}
    arms.update(dict.fromkeys(guard_periods, "persistence"))
    cases = [("noisy", seed) for seed in noise["seeds"]]
    replan_guards = {}
    if args.holdout:
        if noise["standard_deviation_m3_s"] != 700:
            raise ValueError("The frozen +700 m³/s check requires the original noise configuration")
        arms = {"persistence_retry": "persistence", "retry_guard_h1": "persistence"}
        guard_periods = {"retry_guard_h1": 1}
        cases = [("perfect", 0), *(("noisy", seed) for seed in HOLDOUT_SEEDS)]
        if args.replan_guard:
            replan_guards = {"guard_after_retry": "after_retry", "guard_every_decision": "always"}
            arms.update(dict.fromkeys(replan_guards, "persistence"))
            guard_periods.update(dict.fromkeys(replan_guards, 1))
    for name in SCENARIOS:
        scenario = scenarios[name]
        for mode, seed in cases:
            bank = forecast_bank(
                scenario["realized_net_inflow_m3_s"], mode, seed,
                noise["initial_observed_inflow_estimate_m3_s"], noise["standard_deviation_m3_s"],
            )
            np.save(args.output / f"{name}_{seed}_forecasts.npy", bank)
            executed_by_policy = {}
            active_arms = {"external_screened": None} if mode == "perfect" else arms
            for policy, recovery in active_arms.items():
                audit = []
                trace, plans = run_policy(
                    trajectory, config, scenario, "external_screened" if args.recovery or args.holdout else policy,
                    audit, "limits", align_f=True, ottawa_forecast_bank=bank, recovery=recovery,
                    recovery_guard_periods=guard_periods.get(policy, 0),
                    replan_guard=replan_guards.get(policy), **history,
                )
                for entry in audit:
                    entry["policy"] = policy
                executed = [row for row in trace if row["status"] == "executed"]
                executed_by_policy[policy] = executed
                last = trace[-1]
                violation = max((row["ottawa_storage_violation_m3"] for row in executed), default=0.0)
                if violation > config["acceptance"]["storage_m3"]:
                    status = "realized_storage_violation"
                elif len(executed) == len(trajectory):
                    status = "completed"
                else:
                    status = last["status"]
                stem = f"{name}_{seed}_{policy}"
                diagnosis = offline_recoverability(trajectory, config, scenario, executed, **history) if status != "completed" else None
                for suffix, payload in (("plans", plans), ("offline", diagnosis)):
                    (args.output / f"{stem}_{suffix}.json").write_text(
                        json.dumps(payload, default=json_value, indent=2) + "\n"
                    )
                pd.DataFrame(trace).to_csv(args.output / f"{stem}_trace.csv", index=False)
                pd.DataFrame(audit).to_csv(args.output / f"{stem}_candidates.csv", index=False)
                summary.append({
                    "scenario": name, "seed": seed, "forecast_mode": mode, "policy": policy, "status": status,
                    "executed_periods": len(executed), "stop_period": last["period"] if diagnosis else None,
                    "ottawa_solve_status_at_stop": last.get("ottawa_status") if diagnosis else None,
                    "candidate_ottawa_solves": sum("ottawa_candidate_status" in entry for entry in audit),
                    "max_storage_violation_m3": violation,
                    "max_downstream_violation_m3_s": max(
                        (max(row["st_louis_flow_m3_s"] - row["capacity_m3_s"], 0.0) for row in executed),
                        default=0.0,
                    ),
                    "min_downstream_headroom_m3_s": min(
                        (row["capacity_m3_s"] - row["st_louis_flow_m3_s"] for row in executed), default=None,
                    ),
                    "offline_finding": diagnosis["finding"] if diagnosis else None,
                    "expanded_selections": sum(
                        row["selected"] and row.get("recovery_stage") == "expanded" for row in audit
                    ),
                    "persistence_retry_selections": sum(
                        row["selected"] and row.get("recovery_stage") == "persistence_retry" for row in audit
                    ),
                    "guarded_ordinary_selections": sum(
                        row["selected"] and row.get("recovery_stage") != "persistence_retry"
                        and "recovery_storage_maximum_m3" in row for row in audit
                    ),
                    "infeasible_follower_candidates_at_stop": sum(
                        row["period"] == last["period"] and row.get("ottawa_candidate_status") == "infeasible"
                        for row in audit
                    ) if diagnosis else None,
                })
            matched = min(len(rows) for rows in executed_by_policy.values())
            for policy, rows in executed_by_policy.items():
                pairs.append({"scenario": name, "seed": seed, "forecast_mode": mode, "policy": policy, "matched_periods": matched,
                              **{key: sum(row[key] for row in rows[:matched]) if matched else None
                                 for key in ONTARIO_COST_COMPONENTS}})
    pd.DataFrame(pairs).to_csv(args.output / "matched_costs.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(args.output / "summary.csv", index=False)
    settings = {
        "mode": "synthetic causal persistence plus seeded Gaussian noise",
        "forecast_settings": noise,
        "forecast_cases": cases,
        "ontario_window": {"year": int(trajectory.Year.iloc[0]), "first_qm": int(trajectory.QM.iloc[0]),
                           "last_qm": int(trajectory.QM.iloc[-1]), "initial_history": history},
        "reference_schedule": ("Constant anchor from pre-window causal Plan 2014 release and synthetic cap"
                               if args.historical_year else "Previously verified fixed schedule"),
        "recovery_arms": arms,
        "recovery_guard_periods": guard_periods,
        "ordinary_replan_guards": replan_guards,
        "recovery_guard_excess_inflow_m3_s": noise["standard_deviation_m3_s"] if guard_periods else None,
        "ontario_horizon": 4, "ottawa_horizon": 5, "align_f": True, "candidate_family": "limits",
        "input_sha256": {str(path): digest(path) for path in (*input_paths, CONFIG, Path(__file__))},
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(table[["scenario", "seed", "policy", "status", "executed_periods", "offline_finding"]].to_string(index=False))


if __name__ == "__main__":
    main()
