import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.evaluation.provenance import digest
from reservoir_control.study_config import COUPLED_CONFIG, REPO_ROOT, load_settings
from reservoir_control.systems.ottawa.forecasts import inflow_forecast
from reservoir_control.systems.ottawa.hard_follower import (
    FollowerTolerances,
    FollowerWeights,
    HardFollowerProblem,
    InvalidFollowerInput,
    solve_hard_follower,
)
from reservoir_control.systems.ottawa.routing import TransitState

CONFIG = COUPLED_CONFIG


def load_follower_config(path=CONFIG):
    return load_settings(path, "follower")


def make_problem(c, scenario, k, storage, previous, transit, forecast):
    h = len(forecast)
    dt = np.full(h, c["clock"]["duration_seconds"])
    a, carry = transit.forecast(dt)
    return HardFollowerProblem(
        storage,
        previous,
        forecast,
        c["known_ontario_executable_schedule_m3_s"][k : k + h],
        c["known_other_downstream_forecast_m3_s"][k : k + h],
        a,
        carry,
        scenario["downstream_capacity_m3_s"][k : k + h],
        np.full(h + 1, c["storage_minimum_m3"]),
        np.full(h + 1, c["storage_maximum_m3"]),
        np.full(h, c["release_minimum_m3_s"]),
        np.full(h, c["release_maximum_m3_s"]),
        np.full(h + 1, c["storage_target_m3"]),
        np.full(h, c["release_target_m3_s"]),
        dt,
        c["storage_scale_m3"],
        c["flow_scale_m3_s"],
        FollowerWeights(**c["weights"]),
    )


def rollout(c, scenario, policy, mode, seed, *, solver=solve_hard_follower, forecast_bank=None):
    if policy not in ("mpc", "reference"):
        raise ValueError("Unknown policy")
    realized = np.asarray(scenario["realized_net_inflow_m3_s"], float)
    storage, previous = c["initial_storage_m3"], c["previous_executed_release_m3_s"]
    transit = TransitState(tuple(scenario["initial_transit_volumes_m3"]))
    if len(transit.volumes_m3) != scenario["routing_delay_periods"]:
        raise ValueError("Routing queue/delay mismatch")
    rng = np.random.default_rng(seed)
    traces, plans = [], []
    tolerances = FollowerTolerances(**c["acceptance"])
    stop, stop_message = "completed", ""
    n = len(realized)
    initial_system_storage = storage + transit.total_m3
    input_volume, arrived_volume = 0.0, 0.0
    for k in range(n):
        h = min(c["horizon"], n - k)
        if forecast_bank is None:
            forecast = inflow_forecast(
                mode,
                realized[:k],
                c["forecast_noise"]["initial_observed_inflow_estimate_m3_s"],
                h,
                rng,
                c["forecast_noise"]["standard_deviation_m3_s"],
                perfect_values=realized[k : k + h] if mode == "perfect" else None,
            )
        else:
            # Optional frozen bank: no RNG consumption depends on H or early stopping.
            forecast = np.asarray(forecast_bank[k, :h], float).copy()
            if forecast.shape != (h,) or not np.isfinite(forecast).all():
                raise ValueError("Forecast bank must cover every requested time and lead")
        problem = make_problem(c, scenario, k, storage, previous, transit, forecast)
        status, seconds = "fixed_reference", 0.0
        planned_next, predicted_downstream = np.nan, np.nan
        if policy == "mpc":
            try:
                result = solver(problem, tolerances)
            except InvalidFollowerInput as error:
                stop, stop_message = "invalid_input", str(error)
                plans.append(dict(step=k, status=stop, message=stop_message, action=None))
                break
            status, seconds, residuals = result.status, result.elapsed_seconds, result.residuals
            entry = dict(
                step=k,
                status=status,
                message=result.message,
                solver_seconds=seconds,
                objective=result.objective,
                residuals=residuals,
                supplied_problem={
                    name: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for name, value in asdict(problem).items()
                },
                action=result.first_release_m3_s,
            )
            if not result.success:
                if result.first_release_m3_s is not None:
                    raise RuntimeError("Unsuccessful solve supplied an action")
                plans.append(entry)
                stop, stop_message = status, result.message
                break
            release = result.first_release_m3_s
            if release is None or release != result.release_m3_s[0]:
                raise RuntimeError("Missing/inconsistent first action")
            terminal_transit = transit.after_plan(result.release_m3_s, problem.duration_seconds)
            entry.update(
                planned_release_m3_s=result.release_m3_s.tolist(),
                planned_storage_m3=result.storage_m3.tolist(),
                planned_downstream_m3_s=result.downstream_flow_m3_s.tolist(),
                pending_new_release_volume_m3=result.pending_planned_release_volume_m3,
                planned_terminal_transit_m3=terminal_transit.total_m3,
            )
            plans.append(entry)
            planned_next, predicted_downstream = (
                result.storage_m3[1],
                result.downstream_flow_m3_s[0],
            )
        else:
            release = c["reference_release_m3_s"]
            planned_next = storage + problem.duration_seconds[0] * (forecast[0] - release)
            predicted_downstream = (
                problem.ontario_executable_release_m3_s[0]
                + problem.other_downstream_m3_s[0]
                + (
                    problem.routing_matrix @ np.full(h, release)
                    + problem.in_transit_contribution_m3_s
                )[0]
            )
        dt = float(problem.duration_seconds[0])
        old_transit = transit.total_m3
        # Execute exactly one action using realized forcing, without a storage projection.
        following = storage + dt * (realized[k] - release)
        next_transit, controlled_arrival, route_error = transit.advance(release, dt)
        combined = (
            c["known_ontario_executable_schedule_m3_s"][k]
            + c["known_other_downstream_forecast_m3_s"][k]
            + controlled_arrival
        )
        storage_violation = max(
            0.0, c["storage_minimum_m3"] - following, following - c["storage_maximum_m3"]
        )
        down_violation = max(0.0, combined - scenario["downstream_capacity_m3_s"][k])
        release_violation = max(
            0.0, c["release_minimum_m3_s"] - release, release - c["release_maximum_m3_s"]
        )
        w = c["weights"]
        stage = (
            w["storage"] * ((following - c["storage_target_m3"]) / c["storage_scale_m3"]) ** 2
            + w["release"] * ((release - c["release_target_m3_s"]) / c["flow_scale_m3_s"]) ** 2
            + w["change"] * ((release - previous) / c["flow_scale_m3_s"]) ** 2
        )
        input_volume += realized[k] * dt
        arrived_volume += controlled_arrival * dt
        system_error = (
            following
            + next_transit.total_m3
            - initial_system_storage
            - input_volume
            + arrived_volume
        )
        predicted_storage_violation = max(
            0.0, c["storage_minimum_m3"] - planned_next, planned_next - c["storage_maximum_m3"]
        )
        traces.append(
            dict(
                step=k,
                storage_start_m3=storage,
                storage_end_m3=following,
                storage_target_m3=c["storage_target_m3"],
                storage_minimum_m3=c["storage_minimum_m3"],
                storage_maximum_m3=c["storage_maximum_m3"],
                executed_release_m3_s=release,
                previous_executed_release_m3_s=previous,
                planned_first_release_m3_s=release if policy == "mpc" else np.nan,
                release_minimum_m3_s=c["release_minimum_m3_s"],
                release_maximum_m3_s=c["release_maximum_m3_s"],
                realized_net_inflow_m3_s=float(realized[k]),
                forecast_net_inflow_m3_s=float(forecast[0]),
                forecast_error_m3_s=float(forecast[0] - realized[k]),
                planned_next_storage_m3=planned_next,
                predicted_downstream_flow_m3_s=predicted_downstream,
                predicted_first_step_feasible=predicted_storage_violation <= tolerances.storage_m3
                and predicted_downstream
                <= scenario["downstream_capacity_m3_s"][k] + tolerances.flow_m3_s,
                ontario_executable_m3_s=c["known_ontario_executable_schedule_m3_s"][k],
                other_downstream_m3_s=c["known_other_downstream_forecast_m3_s"][k],
                controlled_arrival_m3_s=controlled_arrival,
                combined_downstream_m3_s=combined,
                downstream_capacity_m3_s=scenario["downstream_capacity_m3_s"][k],
                transit_start_m3=old_transit,
                transit_end_m3=next_transit.total_m3,
                transit_queue_end_m3=json.dumps(next_transit.volumes_m3),
                duration_seconds=dt,
                storage_violation_m3=storage_violation,
                downstream_violation_m3_s=down_violation,
                release_violation_m3_s=release_violation,
                stage_cost=stage,
                solver_status=status,
                solver_seconds=seconds,
                routing_balance_error_m3=route_error,
                reservoir_balance_error_m3=following - storage - dt * (realized[k] - release),
                system_balance_error_m3=system_error,
            )
        )
        storage, previous, transit = following, release, next_transit
        if storage_violation > tolerances.storage_m3:
            stop = "physical_storage_violation"
            stop_message = (
                "Realized storage left declared physical bounds; no clipping/emergency policy"
            )
            break
        if release_violation > tolerances.flow_m3_s:
            stop = "release_bound_violation"
            stop_message = "Reference/policy release violates physical outlet bounds"
            break
    trace = pd.DataFrame(traces)
    complete = len(traces) == n and stop == "completed"
    terminal = (
        c["weights"]["terminal"] * ((storage - c["storage_target_m3"]) / c["storage_scale_m3"]) ** 2
        if complete
        else None
    )
    summary = dict(
        scenario=scenario["id"],
        policy=policy,
        forecast_mode=mode,
        seed=int(seed),
        status=stop,
        reason=stop_message,
        complete=complete,
        executed_steps=len(trace),
        requested_steps=n,
        stage_cost=float(trace.stage_cost.sum()) if len(trace) else 0.0,
        terminal_cost=terminal,
        full_objective=float(trace.stage_cost.sum() + terminal) if complete else None,
        final_storage_m3=storage,
        final_transit_volume_m3=transit.total_m3,
        final_transit_queue_m3=list(transit.volumes_m3),
        initial_transit_volume_m3=initial_system_storage - c["initial_storage_m3"],
        cumulative_system_balance_error_m3=storage
        + transit.total_m3
        - initial_system_storage
        - input_volume
        + arrived_volume,
        max_storage_violation_m3=float(trace.storage_violation_m3.max()) if len(trace) else 0.0,
        max_downstream_violation_m3_s=float(trace.downstream_violation_m3_s.max())
        if len(trace)
        else 0.0,
        max_release_violation_m3_s=float(trace.release_violation_m3_s.max()) if len(trace) else 0.0,
        predicted_feasible_realized_violation_steps=int(
            (
                (trace.storage_violation_m3 > tolerances.storage_m3)
                | (trace.downstream_violation_m3_s > tolerances.flow_m3_s)
            )[trace.predicted_first_step_feasible].sum()
        )
        if len(trace)
        else 0,
        release_standard_deviation_m3_s=float(trace.executed_release_m3_s.std(ddof=0))
        if len(trace)
        else None,
        mean_abs_release_change_m3_s=float(
            (trace.executed_release_m3_s - trace.previous_executed_release_m3_s).abs().mean()
        )
        if len(trace)
        else None,
        inflow_forecast_mae_m3_s=float(trace.forecast_error_m3_s.abs().mean())
        if len(trace)
        else None,
        total_solver_seconds=float(sum(x.get("solver_seconds", 0.0) for x in plans)),
        max_solver_storage_residual_m3=max(
            [x.get("residuals", {}).get("storage_violation_m3", 0.0) for x in plans] or [0.0]
        ),
        solver_failures=int(stop in ("infeasible", "solver_failure", "invalid_input")),
    )
    return trace, plans, summary


def validate_config(c):
    if c["schema_version"] != 1 or c["clock"]["duration_seconds"] != 365 * 86400 / 48:
        raise ValueError("This study explicitly uses 48 equal periods per 365-day year (657000 s)")
    if c["horizon"] < 1 or c["reference_release_m3_s"] < 0:
        raise ValueError("Positive horizon and nonnegative reference required")
    n = len(c["known_ontario_executable_schedule_m3_s"])
    if n == 0 or len(c["known_other_downstream_forecast_m3_s"]) != n:
        raise ValueError("Aligned supplied downstream conditions required")
    for case in c["scenarios"]:
        if len(case["realized_net_inflow_m3_s"]) != n or len(case["downstream_capacity_m3_s"]) != n:
            raise ValueError("Complete scenario arrays required")
        if case["routing_delay_periods"] != len(case["initial_transit_volumes_m3"]):
            raise ValueError("Explicit transit queue required")
    if (
        not np.isfinite(c["forecast_noise"]["standard_deviation_m3_s"])
        or c["forecast_noise"]["standard_deviation_m3_s"] < 0
    ):
        raise ValueError("Nonnegative finite noise amplitude required")
    return c


def run_study(config, output, mode, *, lookahead=None):
    c = load_follower_config(config)
    if lookahead is not None:
        if not 1 <= lookahead <= len(c["known_ontario_executable_schedule_m3_s"]):
            raise ValueError("Look-ahead must be between 1 and the number of study periods")
        c["horizon"] = lookahead
    validate_config(c)
    output.mkdir(parents=True, exist_ok=False)
    (output / "configuration.json").write_text(json.dumps(c, indent=2) + "\n")
    seeds = [0] if mode == "perfect" else c["forecast_noise"]["seeds"]
    summaries, pairs, all_plans = [], [], []
    for case in c["scenarios"]:
        for seed in seeds:
            traces, local = {}, {}
            for policy in ("mpc", "reference"):
                trace, plans, summary = rollout(c, case, policy, mode, seed)
                traces[policy], local[policy] = trace, summary
                stem = f"{case['id']}_{seed}_{policy}"
                trace.to_csv(output / f"{stem}.csv", index=False)
                all_plans.append(dict(scenario=case["id"], seed=seed, policy=policy, plans=plans))
                summaries.append(summary)
            common = min(len(t) for t in traces.values())
            pairs.append(
                dict(
                    scenario=case["id"],
                    seed=seed,
                    common_executed_steps=common,
                    mpc_common_stage_cost=float(traces["mpc"].stage_cost.iloc[:common].sum())
                    if common
                    else None,
                    reference_common_stage_cost=float(
                        traces["reference"].stage_cost.iloc[:common].sum()
                    )
                    if common
                    else None,
                    full_cost_comparable=all(s["complete"] for s in local.values()),
                    mpc_full_objective=local["mpc"]["full_objective"],
                    reference_full_objective=local["reference"]["full_objective"],
                )
            )
    results = pd.DataFrame(summaries)
    results.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_comparison.csv", index=False)
    (output / "plans.json").write_text(json.dumps(all_plans, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    sources = [
        Path(__file__),
        Path(__file__).with_name("hard_follower.py"),
        Path(__file__).with_name("routing.py"),
        Path(__file__).with_name("forecasts.py"),
        REPO_ROOT / "src/reservoir_control/controllers/mpc/convex_response.py",
        config,
        REPO_ROOT / "uv.lock",
    ]
    import importlib.metadata

    provenance = dict(
        forecast_mode=mode,
        lookahead=c["horizon"],
        source_sha256={
            str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p): digest(p)
            for p in sources
        },
        scipy_version=importlib.metadata.version("scipy"),
        numpy_version=np.__version__,
        seeds=seeds,
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results.to_csv(sys.stdout, index=False)


def main():
    parser = ArgumentParser(
        description="Standalone synthetic receding-horizon hard-follower demonstrations, no leader."
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--forecast", choices=("perfect", "noisy"), required=True)
    parser.add_argument(
        "--lookahead", type=int, help="Prediction window in periods (default: configured horizon)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_study(
            args.config.resolve(), args.output.resolve(), args.forecast, lookahead=args.lookahead
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
