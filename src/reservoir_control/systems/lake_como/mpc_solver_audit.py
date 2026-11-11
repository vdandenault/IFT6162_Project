"""Offline replay of causal MPC solver failures; no controller changes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como import mpc
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc_causal import CausalMPC, fit_horizon_weights
from reservoir_control.systems.lake_como.mpc_diagnostics import observation
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.splits import split_years

KEYS = ["method", "year", "day"]


def select_slice_cases(diagnostics):
    """All persistence failures plus nearest same-year successful nonfallback controls.

    Controls may serve more than one failure; audit each distinct state once.
    Distances tie toward the earlier day. Selection uses solver flags only.
    """
    frame = diagnostics.query('method == "persistence"').sort_values(["year", "day"])
    failures = frame[~frame.solver_success]
    rows = []
    for row in failures.itertuples():
        safe = frame[
            (frame.year == row.year) & frame.solver_success & ~frame.used_fallback
        ].copy()
        if safe.empty:
            raise ValueError("No matched successful control in year")
        safe["distance"] = (safe.day - row.day).abs()
        control = safe.sort_values(["distance", "day"]).iloc[0]
        rows.append(
            dict(
                year=int(row.year),
                failure_day=int(row.day),
                control_day=int(control.day),
            )
        )
    return pd.DataFrame(rows)


def first_action_slice(model, plan, storage, inflows, demands, days, points=401):
    """Fixed-tail slice, not global seven-day optimization or an annual counterfactual."""
    values = np.unique(np.r_[np.linspace(0.0, 1.0, points), plan[0]])
    costs = []
    for value in values:
        candidate = plan.copy()
        candidate[0] = value
        costs.append(model.cost(candidate, storage, inflows, demands, days))
    winner = int(np.argmin(costs))
    original = model.cost(plan, storage, inflows, demands, days)
    return dict(
        original_cost=original,
        slice_cost=costs[winner],
        cost_gain=original - costs[winner],
        original_request_m3_s=model.low + plan[0] * (model.high - model.low),
        slice_request_m3_s=model.low + values[winner] * (model.high - model.low),
        evaluated_points=len(values),
    )


def inspect_result(result, objective, initial_cost):
    """Inspect an optimizer's discarded iterate without changing fallback behavior."""
    x = np.asarray(result.x)
    finite = bool(np.isfinite(x).all())
    within = bool(finite and (x >= -1e-8).all() and (x <= 1 + 1e-8).all())
    raw_cost = float(objective(x)) if finite else float("nan")
    clipped_cost = float(objective(np.clip(x, 0, 1))) if finite else float("nan")
    gain = initial_cost - clipped_cost if within else float("nan")
    return dict(
        raw_finite=finite,
        raw_within_bounds=within,
        raw_cost=raw_cost,
        clipped_raw_cost=clipped_cost,
        raw_gain_over_initial=gain,
        raw_first_normalized=float(x[0]) if finite else float("nan"),
        gradient_inf_norm=float(np.max(np.abs(result.jac)))
        if hasattr(result, "jac")
        else float("nan"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "results/mpc/causal_seven_day_v2"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/mpc/causal_solver_audit_v1",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    original_plan = json.loads((args.input / "plan.json").read_text())
    run = json.loads((args.input / "run.json").read_text())
    if not run["complete"] or run["test_evaluated"] or run["annual_episodes"] != 30:
        raise ValueError("Complete validation-only seven-day study required")
    for key in ["source_sha256", "baseline_sha256"]:
        for name, expected in original_plan[key].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"Execution input changed: {name}")
    daily = pd.read_csv(args.input / "trajectories.csv", float_precision="round_trip")
    diagnostics = pd.read_csv(
        args.input / "mpc_diagnostics.csv", float_precision="round_trip"
    )
    forecasts = pd.read_csv(
        args.input / "forecast_predictions.csv", float_precision="round_trip"
    )
    if diagnostics.duplicated(KEYS).any() or len(diagnostics) != 10950:
        raise ValueError("Expected unique 30 complete annual diagnostics")
    selection = select_slice_cases(diagnostics)
    roles = {}
    for r in selection.itertuples():
        roles[(r.year, r.failure_day)] = "failure"
        roles.setdefault((r.year, r.control_day), "successful_control")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [
        args.input / name
        for name in [
            "plan.json",
            "run.json",
            "trajectories.csv",
            "mpc_diagnostics.csv",
            "forecast_predictions.csv",
        ]
    ]
    plan = dict(
        scope="Offline replay only; no changed settings, costs, forecasts, training, or annual counterfactuals",
        selection="All persistence failures and nearest same-year successful nonfallback control; ties earliest; unique controls",
        slice="401 equally spaced normalized first actions plus executed first action; remaining plan fixed",
        material_cost_gain=">1e-3 normalized units (=1 unit of discounted planning cost before division by 1000); not annual reward",
        raw_iterate="Failed optimizer iterates inspected but never deployed; numerical feasibility is not certification",
        test_evaluated=False,
        source_sha256={str(Path(__file__)): digest(Path(__file__))},
        input_sha256={str(p): digest(p) for p in inputs},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    selection.to_csv(args.output / "case_selection.csv", index=False)
    config = mpc.MPCConfig(**original_plan["config"])
    assert config == mpc.MPCConfig.load()
    params = settings("train")
    seasonal = mpc.DeterministicMPC.from_training(config).forecast
    weights = fit_horizon_weights(params["inflow"], seasonal.inflow)
    np.testing.assert_array_equal(weights, original_plan["anomaly_weights"])
    scipy_minimize = mpc.minimize
    rows = []
    slices = []
    for method in ["seasonal", "persistence", "seasonal_anomaly"]:
        for year in split_years("validation"):
            policy = CausalMPC(params, seasonal, config, method, weights)
            frame = daily[(daily.method == method) & (daily.year == year)].sort_values(
                "day"
            )
            saved = diagnostics[
                (diagnostics.method == method) & (diagnostics.year == year)
            ].set_index("day")
            expected_forecasts = forecasts[
                (forecasts.method == method) & (forecasts.year == year)
            ].groupby("day", sort=False)
            previous_release = None
            for row in frame.itertuples():
                captured = {}

                def recorder(objective, initial, *positional, **kwargs):
                    result = scipy_minimize(objective, initial, *positional, **kwargs)
                    captured.update(
                        inspect_result(result, objective, objective(initial))
                    )
                    captured["raw_plan"] = np.asarray(result.x).tolist()
                    return result

                info = {"storage": row.storage_before_m3}
                if previous_release is not None:
                    info["release"] = previous_release
                with patch.object(mpc, "minimize", recorder):
                    action = policy(observation(row), info)
                previous_release = row.release_m3_s
                current = policy.diagnostics[-1]
                expected = saved.loc[row.day]
                assert float(action[0]) == row.request_m3_s
                for name in [
                    "solver_success",
                    "solver_status",
                    "iterations",
                    "used_fallback",
                    "planned_cost",
                    "initial_cost",
                ]:
                    if isinstance(current[name], (bool, int)):
                        assert current[name] == expected[name], (
                            method,
                            year,
                            row.day,
                            name,
                        )
                    else:
                        np.testing.assert_allclose(
                            current[name], expected[name], rtol=1e-11, atol=1e-11
                        )
                np.testing.assert_array_equal(
                    policy.forecasts[-1],
                    expected_forecasts.get_group(row.day)
                    .sort_values("lead")
                    .forecast_inflow_m3_s,
                )
                captured.update(current)
                captured.update(
                    method=method,
                    year=year,
                    day=row.day,
                    level_before_m=row.level_before_m,
                    actual_flood=row.flood,
                    actual_low_water=row.low_water,
                    actual_deficit_m3=row.deficit_m3,
                    executed_plan=json.dumps(policy.controller.previous_plan.tolist()),
                )
                captured["raw_plan"] = json.dumps(captured["raw_plan"])
                rows.append(captured)
                if method == "persistence" and (year, row.day) in roles:
                    model = policy.controller.model
                    h = len(policy.controller.previous_plan)
                    days = np.arange(row.day, row.day + h)
                    storage = model.lake.level_to_storage(float(observation(row)[0]))
                    result = first_action_slice(
                        model,
                        policy.controller.previous_plan,
                        storage,
                        policy.forecasts[-1],
                        seasonal.demand[days - 1],
                        days,
                    )
                    result.update(
                        method=method,
                        year=year,
                        day=row.day,
                        role=roles[(year, row.day)],
                    )
                    slices.append(result)
            # Durable replay and slice output for completed years.
            pd.DataFrame(rows).to_csv(args.output / "replay.csv", index=False)
            pd.DataFrame(slices).to_csv(
                args.output / "first_action_slices.csv", index=False
            )
            print(method, year, "exact replay complete", flush=True)
    replay = pd.DataFrame(rows)
    slice_frame = pd.DataFrame(slices)
    merged = replay.merge(
        daily[KEYS + ["level_m", "inflow_m3_s", "level_before_m"]],
        on=KEYS,
        validate="one_to_one",
        suffixes=("", "_saved"),
    )
    failed = merged[~merged.solver_success]
    failure_summary = failed.groupby("method").agg(
        failures=("day", "size"),
        flood_days=("actual_flood", "sum"),
        low_water_days=("actual_low_water", "sum"),
        feasible_raw_iterates=("raw_within_bounds", "sum"),
        raw_material_improvements=(
            "raw_gain_over_initial",
            lambda x: int((x > 1e-3).sum()),
        ),
        mean_raw_cost_gain=("raw_gain_over_initial", "mean"),
        max_raw_cost_gain=("raw_gain_over_initial", "max"),
    )
    status = (
        diagnostics.groupby(["method", "solver_status", "solver_message"])
        .size()
        .rename("days")
        .reset_index()
    )
    by_year = (
        failed.groupby(["method", "year"]).size().rename("failure_days").reset_index()
    )
    failure_flags = diagnostics.pivot(
        index=["year", "day"], columns="method", values="solver_success"
    )
    overlap = pd.DataFrame(
        [
            dict(
                comparison=m,
                both_fail=int((~failure_flags.persistence & ~failure_flags[m]).sum()),
                persistence_only=int(
                    (~failure_flags.persistence & failure_flags[m]).sum()
                ),
                other_only=int((failure_flags.persistence & ~failure_flags[m]).sum()),
            )
            for m in ["seasonal", "seasonal_anomaly"]
        ]
    )
    slice_summary = slice_frame.groupby("role").agg(
        states=("day", "size"),
        material_gains=("cost_gain", lambda x: int((x > 1e-3).sum())),
        median_gain=("cost_gain", "median"),
        max_gain=("cost_gain", "max"),
    )
    for name, table in [
        ("failure_summary", failure_summary.reset_index()),
        ("status_counts", status),
        ("failures_by_year", by_year),
        ("failure_overlap", overlap),
        ("slice_summary", slice_summary.reset_index()),
    ]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    by_year.pivot(index="year", columns="method", values="failure_days").fillna(
        0
    ).plot.bar(ax=axes[0])
    axes[0].set_title("Iteration-limit failures by year")
    for method, group in failed.groupby("method"):
        axes[1].scatter(group.day, group.level_before_m, s=12, alpha=0.5, label=method)
    axes[1].axhline(-0.45, color="gray", ls=":")
    axes[1].axhline(1.24, color="gray", ls=":")
    axes[1].set_title("Failure states: season and starting level")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Level (m)")
    for role, group in slice_frame.groupby("role"):
        axes[2].scatter(group.day, group.cost_gain, s=15, alpha=0.6, label=role)
    axes[2].set_yscale("symlog", linthresh=0.001)
    axes[2].set_title("Fixed-tail first-action slice gain")
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("Predicted cost gain / 1000")
    axes[2].legend(fontsize=7)
    fig.savefig(args.output / "solver_failure_audit.png", dpi=160)
    plt.close(fig)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "input_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    report = (
        "# Causal MPC solver-failure audit\n\n"
        + failure_summary.to_string()
        + "\n\nFixed-tail slices:\n"
        + slice_summary.to_string()
        + "\n\nFailure overlap:\n"
        + overlap.to_string(index=False)
        + "\n\nAll 10,950 decisions replayed exactly. No changed controller or annual counterfactual. Slice and raw-iterate gains are predicted local costs, not realized annual reward improvements.\n"
    )
    (args.output / "report.md").write_text(report)
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                test_evaluated=False,
                replayed_decisions=len(rows),
                slice_states=len(slices),
                source_and_inputs_unchanged=unchanged,
                deployed_controller_changed=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
