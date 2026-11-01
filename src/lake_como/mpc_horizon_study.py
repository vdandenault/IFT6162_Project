"""Matched 1/3/7-day persistence MPC horizons on chronological validation years."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import settings
from lake_como.mpc import DeterministicMPC, MPCConfig
from lake_como.mpc_causal import CausalMPC, METRICS, fit_horizon_weights, merge_severity
from lake_como.mpc_hindsight import digest
from lake_como.mpc_retention import RetainingMPC
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import evaluate
from lake_como.splits import split_years

HORIZONS = (1, 3, 7)


def make_policy(params, seasonal, base_config, horizon):
    if horizon not in HORIZONS:
        raise ValueError("Study horizons are fixed at 1, 3, 7")
    config = replace(base_config, horizon_days=horizon)
    weights = fit_horizon_weights(params["inflow"], seasonal.inflow, horizon)
    policy = CausalMPC(params, seasonal, config, "persistence", weights)
    policy.controller = RetainingMPC(
        params, policy.forecast, config, retain_iteration_limit=True
    )
    return policy


def plot_results(summary, output):
    columns = [
        "full_return",
        "flood_days",
        "low_water_days",
        "deficit_million_m3",
        "flood_excess_m_days",
        "low_excess_m_days",
    ]
    titles = [
        "Annual reward ↑",
        "Flood days ↓",
        "Low-water days ↓",
        "Unmet demand (million m³) ↓",
        "Flood exceedance (m·days) ↓",
        "Low-water exceedance (m·days) ↓",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, column, title in zip(axes.flat, columns, titles):
        ax.bar(
            ["1 day", "3 days", "7 days"],
            summary.loc[list(HORIZONS), column],
            color=["#2980b9", "#269376", "#7d8b95"],
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "MPC horizon ablation · same persistence, surrogate, SLSQP cap and retention",
        fontsize=15,
    )
    fig.savefig(output / "horizon_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "results/mpc/iterate_retention_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/horizon_study_v1"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    baseline_plan = json.loads((args.input / "plan.json").read_text())
    baseline_run = json.loads((args.input / "run.json").read_text())
    if (
        not baseline_run["complete"]
        or baseline_run["test_evaluated"]
        or baseline_run["annual_episodes"] != 20
    ):
        raise ValueError("Completed validation retention study required")
    for key in ["source_sha256", "input_sha256"]:
        for name, expected in baseline_plan[key].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"Source/input changed: {name}")
    config = MPCConfig(**baseline_plan["config"])
    assert config == MPCConfig.load() and config.horizon_days == 7
    params = settings("train")
    seasonal = DeterministicMPC.from_training(config).forecast
    old = pd.read_csv(
        args.input / "trajectories.csv", float_precision="round_trip"
    ).query('method == "retain_iterate"')
    old_diag = pd.read_csv(
        args.input / "mpc_diagnostics.csv", float_precision="round_trip"
    ).query('method == "retain_iterate"')
    sources = [Path(__file__), *[Path(p) for p in baseline_plan["source_sha256"]]]
    inputs = [
        args.input / name
        for name in ["plan.json", "run.json", "trajectories.csv", "mpc_diagnostics.csv"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        base_config=asdict(config),
        horizons=HORIZONS,
        years=split_years("validation"),
        test_evaluated=False,
        contrast="Only horizon length changes. Same SLSQP settings/cap (not equal function evaluations), initialization recipe, surrogate, persistence and retention.",
        warm_starts="Shift previous chosen plan; tail padding/shrinking as in the original implementation, sized to each horizon.",
        terminal="Zero terminal value for every horizon; annual resets; no cross-year storage.",
        information="Past inflow reconstructed from completed-day storage/release; seasonal first-day fallback and demand. Extra history versus PPO.",
        selection="Development ablation: report validation mean full reward and paired physical trade-offs; no automatic default or test evaluation.",
        source_sha256={str(p): digest(p) for p in sources},
        input_sha256={str(p): digest(p) for p in inputs},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frames = []
    annuals = []
    traces = []
    for horizon in HORIZONS:
        for year in split_years("validation"):
            policy = make_policy(params, seasonal, config, horizon)
            frame, seconds = evaluate(
                f"mpc_h{horizon}",
                split="validation",
                year=year,
                protocol="chronological",
                correct_terminal_level=True,
                sequential_years=True,
                measure_decisions=True,
                controller_factory=lambda env: policy,
            )
            trace = pd.DataFrame(policy.diagnostics)
            np.testing.assert_allclose(
                trace.previous_inflow_estimate_m3_s.iloc[1:],
                frame.inflow_m3_s.iloc[:-1],
                atol=1e-8,
                rtol=0,
            )
            np.testing.assert_allclose(
                trace.forecast_inflow_m3_s,
                np.r_[seasonal.inflow[0], frame.inflow_m3_s.iloc[:-1]],
                atol=1e-8,
                rtol=0,
            )
            np.testing.assert_array_equal(trace.forecast_demand_m3_s, seasonal.demand)
            np.testing.assert_array_equal(
                trace.horizon_days, np.minimum(horizon, 366 - frame.day)
            )
            if horizon == 7:
                original = old[old.year == year].sort_values("day")
                original_diag = old_diag[old_diag.year == year].sort_values("day")
                for column in [
                    "request_m3_s",
                    "level_m",
                    "full_reward",
                    "inflow_m3_s",
                    "demand_m3_s",
                ]:
                    np.testing.assert_array_equal(frame[column], original[column])
                for column in [
                    "solver_success",
                    "solver_status",
                    "iterations",
                    "used_fallback",
                    "retained_iteration_limit",
                    "planned_cost",
                    "initial_cost",
                ]:
                    np.testing.assert_array_equal(trace[column], original_diag[column])
            kept = trace.retained_iteration_limit
            assert (~trace.loc[kept, "solver_success"]).all()
            assert trace.loc[kept, "solver_status"].eq(9).all()
            assert (
                trace.loc[kept, "planned_cost"] < trace.loc[kept, "initial_cost"] - 1e-9
            ).all()
            metrics = annual_metrics(frame)
            trace["year"] = year
            metrics["seconds"] = seconds
            for name, table in [
                ("trajectories", frame),
                ("annual", metrics),
                ("mpc_diagnostics", trace),
            ]:
                table["horizon"] = horizon
                table["method"] = f"h{horizon}"
                table["controller"] = f"h{horizon}"
                path = args.output / f"{name}.csv"
                table.to_csv(path, mode="a", header=not path.exists(), index=False)
            frames.append(frame)
            annuals.append(metrics)
            traces.append(trace)
            print(horizon, year, f"complete ({seconds:.2f}s)", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    annual = pd.concat(annuals, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    annual = merge_severity(annual, daily)
    summary = annual.groupby("horizon")[METRICS].mean()
    paired = []
    for horizon in [1, 3]:
        delta = (
            annual[annual.horizon == horizon].set_index("year")[METRICS]
            - annual[annual.horizon == 7].set_index("year")[METRICS]
        )
        delta["horizon"] = horizon
        delta["reference_horizon"] = 7
        paired.append(delta.reset_index())
    paired = pd.concat(paired, ignore_index=True)
    optimization = trace.groupby("horizon").agg(
        decisions=("day", "size"),
        solver_failures=("solver_success", lambda x: int((~x).sum())),
        fallback_days=("used_fallback", "sum"),
        retained_days=("retained_iteration_limit", "sum"),
        mean_iterations=("iterations", "mean"),
        mean_function_evaluations=("function_evaluations", "mean"),
    )
    runtime = annual.groupby("horizon")[
        ["decision_mean_us", "decision_p50_us", "decision_p95_us"]
    ].mean()
    for name, table in [
        ("annual", annual),
        ("summary", summary.reset_index()),
        ("paired_annual", paired),
        ("optimization_summary", optimization.reset_index()),
        ("runtime_summary", runtime.reset_index()),
    ]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    plot_results(summary, args.output)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "input_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    result = dict(
        complete=True,
        annual_episodes=30,
        test_evaluated=False,
        seven_day_exactly_reproduced=True,
        source_and_inputs_unchanged=unchanged,
        default_controller_changed=False,
        best_validation_mean_reward_horizon=int(summary.full_return.idxmax()),
    )
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    report = (
        "# Matched MPC horizon ablation\n\n"
        + summary.round(5).to_string()
        + "\n\nSolver diagnostics:\n"
        + optimization.to_string()
        + "\n\nPaired changes versus 7 days:\n"
        + paired.round(5).to_string(index=False)
        + "\n\nSame settings do not imply equal computational effort or globally solved plans. Validation only; defaults unchanged.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
