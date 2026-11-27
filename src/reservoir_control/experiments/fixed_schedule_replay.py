"""Replay an Ontario discharge commitment against the hard Ottawa follower."""

from argparse import ArgumentParser
import csv
from dataclasses import asdict
import json
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
from core.mechanism.base import MDPState
from gymnasium.spaces import Box

from reservoir_control.systems.loslr.operator import AdjustFlow, ONTARIO_COST_COMPONENTS, ontario_stage_costs
from reservoir_control.systems.loslr.simulator import LoslrEnv, round_d
from reservoir_control.systems.loslr.physics import CONVERSION, FLOW_SCALE
from reservoir_control.systems.loslr.solvers.rule_curve import FORCING_COLUMNS
from reservoir_control.systems.ottawa.hard_follower_study import (
    CONFIG,
    load_follower_config,
    FollowerTolerances,
    TransitState,
    digest,
    make_problem,
    solve_hard_follower,
    validate_config,
)

LOOKAHEAD = 5
PERIODS = 12


def replay(trajectory, config, scenario, *, previous_flow, previous_ice, choose_schedule=None,
           ottawa_forecast_bank=None):
    """Return decision records and the Ottawa solves, including any rejection."""
    records, plans = [], []
    decisions = replay_steps(
        trajectory, config, scenario, previous_flow=previous_flow, previous_ice=previous_ice,
        ottawa_forecast_bank=ottawa_forecast_bank, records=records, plans=plans,
        scored=choose_schedule is not None,
    )
    try:
        decision = next(decisions)
        while True:
            selected = choose_schedule(*decision) if choose_schedule is not None else decision[1]
            decision = decisions.send(selected)
    except StopIteration:
        return records, plans


def replay_steps(trajectory, config, scenario, *, previous_flow, previous_ice, ottawa_forecast_bank,
                 records, plans, scored=True, expected_periods=PERIODS):
    """Yield causal decision inputs and execute the supplied follower problem."""
    dates = trajectory["Year"] * 48 + trajectory["QM"]
    if len(trajectory) != expected_periods or not np.all(np.diff(dates) == 1):
        raise ValueError(f"Expected {expected_periods} consecutive Ontario quarter-months")
    schedule = trajectory["ontFlow"].to_numpy(dtype=float)
    if not np.isfinite(schedule).all() or (schedule < 0).any():
        raise ValueError("Ontario commitments must be finite and nonnegative")
    config = {**config, "known_ontario_executable_schedule_m3_s": (schedule * FLOW_SCALE).tolist()}
    validate_config(config)

    env = LoslrEnv(
        world=None, mechanism_id=None, agents_cfg_dict={}, horizon=expected_periods,
        stage_init=float(trajectory["ontLevelBOQ"].iloc[0]), action_mode="flow",
    )
    mdp = env.reset(MDPState())
    mechanism = AdjustFlow(
        aid="ont", id="flow", acts_on=("ont", "flow"),
        action_space=Box(0.0, np.inf, shape=(1,), dtype=np.float64),
    )

    dt = config["clock"]["duration_seconds"]
    storage = config["initial_storage_m3"]
    previous_release = config["previous_executed_release_m3_s"]
    queue = TransitState(tuple(scenario["initial_transit_volumes_m3"]))
    tolerance = FollowerTolerances(**config["acceptance"])
    inflows = np.asarray(scenario["realized_net_inflow_m3_s"], dtype=float)

    for k, forcing in enumerate(trajectory.to_dict("records")):
        flow = float(schedule[k])
        committed = flow * FLOW_SCALE
        level = float(mdp.state["ontLevelBOQ"][k])
        background = config["known_other_downstream_forecast_m3_s"][k]
        capacity = scenario["downstream_capacity_m3_s"][k]
        forecast = inflows[k:k + LOOKAHEAD]
        if ottawa_forecast_bank is not None:
            forecast = np.asarray(ottawa_forecast_bank[k, :len(forecast)], dtype=float)
            if len(forecast) != min(LOOKAHEAD, expected_periods - k) or not np.isfinite(forecast).all():
                raise ValueError("Ottawa forecast bank does not cover the decision window")
        problem = make_problem(config, scenario, k, storage, previous_release, queue, forecast)
        selected = yield k, problem, level, previous_flow, previous_ice, mechanism
        if selected is None:
            records.append({
                "period": k + 1, "status": "no_admissible_candidate",
                "reason": "All candidates rejected under the policy forecast",
                "ottawa_start_storage_m3": storage,
                "ottawa_previous_release_m3_s": previous_release,
                "ottawa_start_queue_m3": list(queue.volumes_m3),
                "ontario_start_m": level,
                "ontario_previous_release_m3_s": previous_flow * FLOW_SCALE,
                "ontario_previous_ice": previous_ice,
            })
            break
        problem = selected
        committed = float(problem.ontario_executable_release_m3_s[0])
        flow = committed / FLOW_SCALE
        result = solve_hard_follower(problem, tolerance)
        plans.append({
            "period": k + 1,
            "status": result.status,
            "message": result.message,
            "problem": asdict(problem),
            "residuals": result.residuals,
            "planned_ottawa_release_m3_s": result.release_m3_s,
        })
        record = {
            "period": k + 1, "year": forcing["Year"], "quarter_month": forcing["QM"],
            "lookahead": len(forecast), "ottawa_status": result.status,
            "ontario_committed_m3_s": committed, "ontario_start_m": level,
            "ontario_previous_release_m3_s": previous_flow * FLOW_SCALE,
            "ontario_previous_ice": previous_ice,
            "ontario_forecast_supply_m3_s": forcing["ontNTS_QM1"],
            "ottawa_start_storage_m3": storage, "ottawa_start_transit_m3": queue.total_m3,
            "ottawa_start_queue_m3": list(queue.volumes_m3),
            "ottawa_net_inflow_m3_s": float(inflows[k]),
            "background_m3_s": background, "capacity_m3_s": capacity,
            "follower_capacity_m3_s": problem.downstream_capacity_m3_s[0],
            "status": "ottawa_solve_failed",
        }
        records.append(record)
        if not result.success:
            record["reason"] = result.message
            break

        release = result.first_release_m3_s
        next_queue, arrival, routing_residual = queue.advance(release, dt)
        non_ontario = background + arrival
        combined = committed + non_ontario
        # The arrival at execution is known. Replace the historical aggregate so
        # the Ottawa arrival is counted once.
        forcing.update(stlouisontOut=non_ontario, slonFlow_QM1=non_ontario)
        forecast_level = level + round_d((forcing["ontNTS_QM1"] / FLOW_SCALE - flow) / CONVERSION, 6)
        allowed, regime = mechanism.constrain(
            forcing, level, forecast_level, flow, "RF",
            previous_flow=previous_flow, previous_ice=previous_ice,
        )
        record.update(
            ontario_allowed_m3_s=allowed * FLOW_SCALE, ontario_limit_regime=regime,
            ottawa_proposed_release_m3_s=release, ottawa_prospective_arrival_m3_s=arrival,
            prospective_combined_m3_s=combined, ontario_forecast_end_m=forecast_level,
        )
        if not isclose(allowed * FLOW_SCALE, committed, rel_tol=0, abs_tol=tolerance.flow_m3_s):
            record.update(status="commitment_rejected", reason="Native Ontario limits change the committed flow")
            break

        next_storage = storage + dt * (inflows[k] - release)
        used_forecast = problem.net_inflow_m3_s[0]
        expected_storage = storage + dt * (used_forecast - release)
        balance_residual = (next_storage + next_queue.total_m3
                            - storage - queue.total_m3 - dt * (inflows[k] - arrival))
        if (abs(balance_residual) > tolerance.storage_m3
                or abs(routing_residual) > tolerance.storage_m3
                or abs(expected_storage - result.storage_m3[1]) > tolerance.storage_m3
                or abs(combined - result.downstream_flow_m3_s[0]) > tolerance.flow_m3_s
                or combined > capacity + tolerance.flow_m3_s):
            raise ArithmeticError(f"Coupled balance or capacity mismatch at period {k + 1}")

        mdp.params.update(forcing)
        mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})
        mdp = env.simulate(mdp)
        applied = float(mdp.state["ontFlow"][k + 1]) * FLOW_SCALE
        routed = float(mdp.state["stlouisFlow"][k + 1]) * FLOW_SCALE
        end_level = float(mdp.state["ontLevelBOQ"][k + 1])
        level_residual = end_level - level - (forcing["ontNTS"] - applied) / (FLOW_SCALE * CONVERSION)
        if (applied != committed or abs(routed - combined) > tolerance.flow_m3_s
                or abs(level_residual) > 0.500001e-6):
            raise ArithmeticError(f"Ontario flow, routing or level mismatch at period {k + 1}")

        record.update(
            status="executed", ontario_applied_m3_s=applied,
            ontario_realized_supply_m3_s=forcing["ontNTS"], ontario_end_m=end_level,
            ontario_level_balance_residual_m=level_residual,
            ottawa_executed_release_m3_s=release, ottawa_arrival_m3_s=arrival,
            ottawa_end_storage_m3=float(next_storage), ottawa_end_transit_m3=next_queue.total_m3,
            ottawa_end_queue_m3=list(next_queue.volumes_m3),
            ottawa_system_balance_residual_m3=float(balance_residual),
            st_louis_flow_m3_s=routed, pointe_claire_m=float(mdp.state["ptclaireLevel"][k + 1]),
        )
        if scored:
            cost_inputs = {
                **forcing,
                **{name: series[k + 1] for name, series in mdp.state.data.items()},
                "ontFlow": flow,
                "ontLevel": (level + end_level) / 2,
            }
            costs = ontario_stage_costs({name: [value] for name, value in cost_inputs.items()})[0]
            if not np.isfinite(costs).all():
                raise ArithmeticError(f"Nonfinite executed Ontario costs at period {k + 1}")
            record.update(zip(ONTARIO_COST_COMPONENTS, costs, strict=True))
        storage, queue, previous_release = next_storage, next_queue, release
        previous_flow, previous_ice = flow, forcing["iceInd"]
        if ottawa_forecast_bank is not None:
            violation = max(
                config["storage_minimum_m3"] - storage, storage - config["storage_maximum_m3"], 0.0,
            )
            record.update(ottawa_forecast_inflow_m3_s=float(used_forecast),
                          ottawa_original_forecast_inflow_m3_s=float(forecast[0]),
                          ottawa_predicted_storage_m3=float(result.storage_m3[1]),
                          ottawa_storage_violation_m3=float(violation))
            if violation > tolerance.storage_m3:
                record["reason"] = "Realized Ottawa storage left its bounds"
                break

def repair_offline(trajectory, config, scenario, *, previous_flow, previous_ice):
    """Try fixed schedules from the same initial state, editing the first rejection."""
    candidate = trajectory.copy().reset_index(drop=True)
    candidate["ontFlow"] = candidate["ontFlow"].astype(float)
    seen = set()
    edits = []
    for _ in range(4 * PERIODS):
        key = tuple(candidate["ontFlow"])
        if key in seen:
            raise RuntimeError("Offline schedule repair revisited an earlier candidate")
        seen.add(key)
        records, _ = replay(
            candidate, config, scenario, previous_flow=previous_flow, previous_ice=previous_ice,
        )
        last = records[-1]
        if last["status"] == "executed":
            columns = FORCING_COLUMNS | {"ontLevelBOQ", "ontFlow", "ontNTS_QM1", "foreInd"}
            candidate = candidate.loc[:, [col for col in candidate if col in columns]].copy()
            candidate["ontLevelBOQ"] = [record["ontario_start_m"] for record in records]
            return candidate, edits
        if last["status"] != "commitment_rejected":
            raise RuntimeError(f"Offline schedule repair stopped at period {last['period']}: {last['reason']}")
        edits.append({key: last[key] for key in (
            "period", "ontario_committed_m3_s", "ontario_allowed_m3_s", "ontario_limit_regime"
        )})
        candidate.at[last["period"] - 1, "ontFlow"] = last["ontario_allowed_m3_s"] / FLOW_SCALE
    raise RuntimeError("Offline schedule repair reached its attempt limit")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", default="one_period_delay_feasible")
    parser.add_argument(
        "--repair-offline", action="store_true",
        help="Find a fixed test schedule using repeated perfect-information replays",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    config = load_follower_config()
    scenarios = {case["id"]: case for case in config["scenarios"]}
    if args.scenario not in scenarios:
        parser.error(f"--scenario must be one of {', '.join(scenarios)}")
    trajectory = pd.read_csv(args.trajectory)
    # Match the initial history constants used by the standalone Ontario MPC.
    history = {"previous_flow": 595.0, "previous_ice": 2.0}
    scenario = scenarios[args.scenario]
    if args.repair_offline:
        trajectory, edits = repair_offline(trajectory, config, scenario, **history)
    records, plans = replay(trajectory, config, scenario, **history)
    executed = [row for row in records if row["status"] == "executed"]
    stopped = records[-1] if len(executed) < PERIODS else None
    summary = {
        "status": stopped["status"] if stopped else "completed",
        "executed_periods": len(executed), "requested_periods": PERIODS,
        "first_conflict": stopped, "ottawa_lookahead": LOOKAHEAD,
        "scenario": args.scenario, "ottawa_information": "perfect forecasts",
        "downstream_background": "synthetic; replaces historical St. Louis non-Ontario input",
        "initial_ontario_previous_release_m3_s": 595.0 * FLOW_SCALE,
        "initial_ontario_previous_ice": 2.0,
        "terminal_ottawa_queue_m3": (
            executed[-1]["ottawa_end_queue_m3"] if executed else records[0]["ottawa_start_queue_m3"]
        ),
        "input_sha256": {str(path): digest(path) for path in (args.trajectory, CONFIG, Path(__file__))},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    if args.repair_offline:
        schedule = args.output / "fixed_schedule.csv"
        trajectory.to_csv(schedule, index=False)
        summary.update(
            schedule_construction="offline first-rejection repair with perfect Ottawa forecasts",
            schedule_sha256=digest(schedule),
            offline_edits=edits,
        )
    columns = list(dict.fromkeys(field for row in records for field in row))
    with (args.output / "joint_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    for filename, value in (("summary.json", summary), ("ottawa_plans.json", plans)):
        (args.output / filename).write_text(json.dumps(value, default=json_value, indent=2) + "\n")
    print(f"{summary['status']}: {len(executed)}/{PERIODS} periods. Results in {args.output}")


if __name__ == "__main__":
    main()
