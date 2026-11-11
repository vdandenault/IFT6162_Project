"""Offline boundary and optimizer audits of the saved, unchanged MPC prototype."""
import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT


# Fixed diagnostic design, not a controller or hyperparameter selection study.
GRID_POINTS = {1: (101, 401), 2: (21, 41), 3: (11, 21)}
SLICE_POINTS = 401


def level_penalties(levels, model):
    """Positive level costs on identical endpoints; other reward terms omitted."""
    levels = np.asarray(levels)
    c = model.coefficients
    flood = np.maximum(levels - model.flood_level, 0.0)
    low = np.maximum(model.low_level - levels, 0.0)
    binary = c["overflow_coeff"] * (levels > model.flood_level) + c[
        "starving_coeff"
    ] * (levels < model.low_level)
    surrogate = (
        c["overflow_coeff"] * (flood / model.config.level_violation_scale_m) ** 2
        + c["starving_coeff"] * (low / model.config.level_violation_scale_m) ** 2
    )
    return binary, surrogate


def boundary_audit(frame, diagnostics, model):
    keys = ["year", "day"]
    if frame.duplicated(keys).any() or diagnostics.duplicated(keys).any():
        raise ValueError("Duplicate year/day rows")
    columns = keys + [
        "predicted_level_m",
        "forecast_inflow_m3_s",
        "forecast_demand_m3_s",
        "solver_success",
        "used_fallback",
        "planned_cost",
    ]
    audit = frame.merge(
        diagnostics[columns],
        on=keys,
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not audit["_merge"].eq("both").all():
        raise ValueError("Trajectory and diagnostics must cover identical days")
    audit = audit.drop(columns="_merge").sort_values(keys).reset_index(drop=True)
    actual, realized_surrogate = level_penalties(audit.level_m, model)
    forecast, forecast_surrogate = level_penalties(audit.predicted_level_m, model)
    audit["actual_level_cost"] = actual
    audit["realized_surrogate_level_cost"] = realized_surrogate
    audit["forecast_binary_level_cost"] = forecast
    audit["forecast_surrogate_level_cost"] = forecast_surrogate
    audit["level_error_m"] = audit.level_m - audit.predicted_level_m
    audit["inflow_error_m3_s"] = audit.inflow_m3_s - audit.forecast_inflow_m3_s
    audit["actual_flood"] = audit.level_m > model.flood_level
    audit["forecast_flood"] = audit.predicted_level_m > model.flood_level
    audit["actual_low"] = audit.level_m < model.low_level
    audit["forecast_low"] = audit.predicted_level_m < model.low_level
    np.testing.assert_allclose(
        actual, -audit.weighted_overflow_reward - audit.weighted_starving_reward
    )
    rows = []
    for kind in ("flood", "low"):
        outside = audit[f"actual_{kind}"]
        distance = (
            (audit.level_m - model.flood_level)
            if kind == "flood"
            else (model.low_level - audit.level_m)
        )
        safe_margin = (
            (model.flood_level - audit.predicted_level_m)
            if kind == "flood"
            else (audit.predicted_level_m - model.low_level)
        )
        misses = outside & ~audit[f"forecast_{kind}"]
        rows.append(
            dict(
                kind=kind,
                violation_days=int(outside.sum()),
                predicted_violation_days=int(
                    (outside & audit[f"forecast_{kind}"]).sum()
                ),
                forecast_missed_days=int(misses.sum()),
                within_1cm_days=int((outside & (distance <= 0.01)).sum()),
                median_exceedance_m=float(distance[outside].median()),
                median_predicted_safe_margin_m=float(safe_margin[misses].median()),
                fallback_violation_days=int((outside & audit.used_fallback).sum()),
            )
        )
    return audit, pd.DataFrame(rows)


def longest_episode(frame, mask):
    """Longest consecutive day run; ties go to the earliest start."""
    selected = frame.loc[mask].sort_values("day")
    if selected.empty:
        raise ValueError("No violation episode available")
    groups = selected.day.diff().ne(1).cumsum()
    episodes = [group for _, group in selected.groupby(groups)]
    return min(episodes, key=lambda group: (-len(group), int(group.day.min())))


def select_cases(audit):
    """Worst-year longest flood/low episodes, first solver failure, safe control."""
    cases, windows = [], []
    for kind in ("flood", "low"):
        counts = audit.groupby("year")[f"actual_{kind}"].sum()
        worst_year = int(counts.sort_index().idxmax())
        year = audit[audit.year == worst_year]
        episode = longest_episode(year, year[f"actual_{kind}"])
        start, end = int(episode.day.min()), int(episode.day.max())
        windows.append(
            dict(
                kind=kind,
                year=worst_year,
                episode_start=start,
                episode_end=end,
                plot_start=max(1, start - 7),
                plot_end=min(365, end + 7),
            )
        )
        for label, index in [("onset", 0), ("persistent", len(episode) // 2)]:
            row = episode.iloc[index]
            cases.append(
                dict(case=f"{kind}_{label}", year=worst_year, day=int(row.day))
            )
    failed = audit[~audit.solver_success]
    if not failed.empty:
        row = failed.iloc[0]
        cases.append(dict(case="iteration_limit", year=int(row.year), day=int(row.day)))
    safe = audit[
        (audit.level_before_m > -0.35)
        & (audit.level_before_m < 1.14)
        & ~audit.actual_low
        & ~audit.actual_flood
        & ~audit.used_fallback
    ]
    row = safe.iloc[0]
    cases.append(dict(case="interior_control", year=int(row.year), day=int(row.day)))
    return pd.DataFrame(cases), windows


def observation(row):
    phase = 2 * math.pi * (int(row.day) - 1) / 365
    # The deployed MPC ignores demand EMA. Zero is sufficient for exact replay.
    return np.array(
        [row.level_before_m, math.sin(phase), math.cos(phase), 0.0], dtype=np.float32
    )


def replay_snapshots(audit, cases, config):
    """Replay saved states, not environment transitions; verify historical actions."""
    snapshots = {}
    checked = 0
    for year in sorted(cases.year.unique()):
        controller = DeterministicMPC.from_training(config)
        limit = int(cases[cases.year == year].day.max())
        selected_days = set(cases[cases.year == year].day)
        prefix = audit[(audit.year == year) & (audit.day <= limit)]
        for row in prefix.itertuples():
            obs = observation(row)
            action = controller(obs)
            np.testing.assert_allclose(
                action[0],
                row.request_m3_s,
                rtol=0,
                atol=3e-5,
                err_msg=f"Historical MPC action mismatch: {year}/{row.day}",
            )
            np.testing.assert_allclose(
                controller.diagnostics[-1]["planned_cost"],
                row.planned_cost,
                rtol=1e-7,
                atol=1e-8,
            )
            checked += 1
            if row.day in selected_days:
                snapshots[(int(year), int(row.day))] = dict(
                    observation=obs, plan=controller.previous_plan.copy()
                )
    return snapshots, checked


def exhaustive_grid(objective, horizon, points):
    """Best point on a finite Cartesian grid; not a continuous-global certificate."""
    if horizon < 1 or points < 2:
        raise ValueError("Positive horizon and at least two grid points required")
    values = np.linspace(0.0, 1.0, points)
    best, best_cost = None, np.inf
    for coordinates in product(values, repeat=horizon):
        cost = objective(np.array(coordinates))
        if not np.isfinite(cost):
            raise ValueError("Nonfinite grid cost")
        if cost < best_cost:
            best, best_cost = coordinates, cost
    return np.array(best), float(best_cost)


def grid_audit(cases, snapshots, config, reference):
    rows, slices, plans = [], [], []
    model, forecast = reference.model, reference.forecast
    params = None
    # Reuse training parameters, never the selected year's future forcing.
    from reservoir_control.systems.lake_como.environment import settings

    params = settings("train")
    for case in cases.itertuples(index=False):
        snapshot = snapshots[(case.year, case.day)]
        obs, deployed = snapshot["observation"], snapshot["plan"]
        storage = model.lake.level_to_storage(float(obs[0]))
        days = np.arange(case.day, case.day + len(deployed))
        inflows, demands = forecast.inflow[days - 1], forecast.demand[days - 1]
        requests = model.low + deployed * (model.high - model.low)
        levels, releases = model.predict(storage, requests, inflows, days)
        binary, surrogate = level_penalties(levels, model)
        for k, day in enumerate(days):
            plans.append(
                dict(
                    case=case.case,
                    year=case.year,
                    decision_day=case.day,
                    step=k + 1,
                    forecast_day=int(day),
                    request_m3_s=requests[k],
                    predicted_level_m=levels[k],
                    predicted_release_m3_s=releases[k],
                    forecast_binary_level_cost=binary[k],
                    forecast_surrogate_level_cost=surrogate[k],
                )
            )
        # A feasible one-coordinate competitor to the original seven-day warm plan.
        original_cost = model.cost(deployed, storage, inflows, demands, days)
        for x in np.linspace(0, 1, SLICE_POINTS):
            trial = deployed.copy()
            trial[0] = x
            slices.append(
                dict(
                    case=case.case,
                    year=case.year,
                    day=case.day,
                    first_request_m3_s=model.low + x * (model.high - model.low),
                    cost=model.cost(trial, storage, inflows, demands, days),
                    deployed_cost=original_cost,
                    deployed_request_m3_s=requests[0],
                )
            )
        for horizon, refinements in GRID_POINTS.items():
            short = DeterministicMPC(
                params, forecast, replace(config, horizon_days=horizon)
            )
            short(obs)  # Same solver, initial candidates, and fallback; cold start.
            horizon_days = np.arange(case.day, case.day + len(short.previous_plan))
            q, demand = (
                forecast.inflow[horizon_days - 1],
                forecast.demand[horizon_days - 1],
            )
            objective = lambda x: short.model.cost(x, storage, q, demand, horizon_days)
            delivered = objective(short.previous_plan)
            diagnostic = short.diagnostics[-1]
            for points in refinements:
                best, best_cost = exhaustive_grid(objective, len(horizon_days), points)
                tolerance = max(1e-4, 1e-3 * abs(best_cost))
                rows.append(
                    dict(
                        case=case.case,
                        year=case.year,
                        day=case.day,
                        horizon=horizon,
                        grid_points=points,
                        grid_evaluations=points ** len(horizon_days),
                        controller_cost=delivered,
                        grid_cost=best_cost,
                        cost_gap=delivered - best_cost,
                        material_gap=delivered - best_cost > tolerance,
                        tolerance=tolerance,
                        solver_success=diagnostic["solver_success"],
                        used_fallback=diagnostic["used_fallback"],
                        controller_first_request_m3_s=model.low
                        + short.previous_plan[0] * (model.high - model.low),
                        grid_first_request_m3_s=model.low
                        + best[0] * (model.high - model.low),
                    )
                )
        print(
            f"Grid checks complete: {case.case} ({case.year}, day {case.day})",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(slices), pd.DataFrame(plans)


def plots(audit, windows, cases, slices, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for window in windows:
        f = audit[
            (audit.year == window["year"])
            & audit.day.between(window["plot_start"], window["plot_end"])
        ]
        fig, axes = plt.subplots(
            4, 1, figsize=(12, 11), sharex=True, layout="constrained"
        )
        kind = window["kind"]
        threshold = 1.24 if kind == "flood" else -0.45
        axes[0].plot(f.day, f.level_m, label="Realized end-of-day level")
        axes[0].plot(f.day, f.predicted_level_m, label="Forecast end-of-day level")
        axes[0].axhline(
            threshold, color="firebrick", ls="--", label="Penalty threshold"
        )
        axes[0].set(ylabel="Level (m)")
        axes[0].legend(fontsize=8)
        axes[1].plot(f.day, f.actual_level_cost, label="Realized binary level cost")
        axes[1].plot(
            f.day, f.forecast_binary_level_cost, label="Forecast binary level cost"
        )
        axes[1].plot(
            f.day,
            f.forecast_surrogate_level_cost,
            label="Forecast surrogate level cost",
        )
        axes[1].set(ylabel="Daily level cost")
        axes[1].legend(fontsize=8)
        axes[2].plot(f.day, f.request_m3_s, label="Requested release")
        axes[2].plot(f.day, f.release_m3_s, label="Actual release")
        axes[2].set(ylabel="Release (m³/s)")
        axes[2].legend(fontsize=8)
        axes[3].plot(f.day, f.inflow_m3_s, label="Realized inflow")
        axes[3].plot(f.day, f.forecast_inflow_m3_s, label="Training-only forecast")
        axes[3].set(xlabel="Day", ylabel="Inflow (m³/s)")
        axes[3].legend(fontsize=8)
        fig.suptitle(
            f"{kind.title()} episode · {window['year']} · longest run in worst-count year\nForecast binary and surrogate costs use the same predicted endpoint"
        )
        fig.savefig(output / f"{kind}_boundary.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, case in zip(axes.flat, cases.itertuples(index=False)):
        f = slices[slices.case == case.case]
        ax.plot(f.first_request_m3_s, f.cost)
        ax.scatter(
            f.deployed_request_m3_s.iloc[0],
            f.deployed_cost.iloc[0],
            color="firebrick",
            label="Deployed warm plan",
        )
        ax.set(
            title=f"{case.case}: {case.year}/{case.day}",
            xlabel="First request (m³/s)",
            ylabel="Discounted surrogate cost / 1000",
        )
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Original seven-day plan: vary only the first request, hold the tail fixed\nA feasible slice check, not a seven-dimensional global-optimality certificate"
    )
    fig.savefig(output / "seven_day_first_action_slices.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results/comparison/mpc_validation_v1",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/diagnosis_v1"
    )
    args = parser.parse_args()
    metadata = json.loads((args.input / "run.json").read_text())
    if (
        metadata["split"] != "validation"
        or metadata["test_evaluated"]
        or not metadata["complete"]
    ):
        raise ValueError("Diagnostics require the completed validation prototype")
    provenance = next(
        c["provenance"] for c in metadata["controllers"] if c["family"] == "mpc"
    )
    config = MPCConfig(**provenance["config"])
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, expected in metadata["hashes"].items():
        if (
            path == "src/reservoir_control/systems/lake_como/mpc.py"
            or path.startswith("external/gym4real/gym4real/envs/dam/")
            or path.startswith("external/gym4real/gym4real/data/dam/")
        ):
            if digest(PROJECT_ROOT / path) != expected:
                raise ValueError(f"Baseline physics/controller/input changed: {path}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; choose a new --output")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = DeterministicMPC.from_training(config)
    trajectory = pd.read_csv(args.input / "trajectories.csv")
    trajectory = trajectory[trajectory.family == "mpc"].copy()
    diagnostics = pd.read_csv(args.input / "mpc_diagnostics.csv")
    audit, boundary = boundary_audit(trajectory, diagnostics, reference.model)
    cases, windows = select_cases(audit)
    plan = dict(
        baseline_config=asdict(config),
        cases=cases.to_dict("records"),
        windows=windows,
        short_horizon_grid_points=GRID_POINTS,
        slice_points=SLICE_POINTS,
        material_gap="controller_cost - grid_cost > max(1e-4, 1e-3*abs(grid_cost)); costs normalized by 1000",
        short_horizon_initialization="Cold start; same demand/low/high candidates and fallback. Not the deployed 7-day problem.",
        seven_day_check="Deployed warm plan; exhaustive first-coordinate slice with tail fixed",
        selection="Worst violation-count year, longest run: onset and middle; earliest solver failure and safe interior day. Ties earliest.",
        new_closed_loop_rollouts=False,
        oracle_forecasts=False,
        deployed_parameters_changed=False,
        source_sha256={
            str(p): digest(p)
            for p in [Path(__file__), PROJECT_ROOT / "src/reservoir_control/systems/lake_como/mpc.py"]
        },
        input_sha256={
            p.name: digest(p)
            for p in [
                args.input / "run.json",
                args.input / "trajectories.csv",
                args.input / "mpc_diagnostics.csv",
            ]
        },
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    audit.to_csv(args.output / "boundary_daily.csv", index=False)
    boundary.to_csv(args.output / "boundary_summary.csv", index=False)
    cases.to_csv(args.output / "selected_cases.csv", index=False)
    snapshots, checked = replay_snapshots(audit, cases, config)
    print(f"Historical replay verified: {checked} daily decisions", flush=True)
    grid, slices, plans = grid_audit(cases, snapshots, config, reference)
    grid.to_csv(args.output / "short_horizon_grids.csv", index=False)
    slices.to_csv(args.output / "seven_day_slices.csv", index=False)
    plans.to_csv(args.output / "selected_plans.csv", index=False)
    plots(audit, windows, cases, slices, args.output)
    fine = grid.groupby(["case", "horizon"], sort=False).tail(1)
    slice_summary = slices.groupby("case", sort=False).agg(
        deployed_cost=("deployed_cost", "first"), best_slice_cost=("cost", "min")
    )
    slice_summary["gap"] = slice_summary.deployed_cost - slice_summary.best_slice_cost
    slice_summary.to_csv(args.output / "seven_day_slice_summary.csv")
    summary = dict(
        complete=True,
        historical_replay_decisions=checked,
        fine_grid_cases=len(fine),
        fine_grid_material_gaps=int(fine.material_gap.sum()),
        largest_fine_grid_gap=float(fine.cost_gap.max()),
        largest_seven_day_slice_gap=float(slice_summary.gap.max()),
        scope="Selected validation snapshots only; short-horizon grids and 1D slices do not certify global 7-day optimization",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(boundary.round(5).to_string(index=False))
    print(fine.round(5).to_string(index=False))
    print(summary)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
