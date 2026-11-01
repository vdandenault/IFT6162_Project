"""Paired validation of status-9 retention under the persistence forecast."""
import argparse
from dataclasses import asdict
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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for ax, column, title in zip(axes.flat, columns, titles):
        ax.bar(
            ["Original\nfallback", "Cost-checked\nretention"],
            summary.loc[["original_fallback", "retain_iterate"], column],
            color=["#7d8b95", "#2980b9"],
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Seven-day persistence MPC · same cost, forecast and solver budget", fontsize=15
    )
    fig.savefig(output / "retention_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "results/mpc/causal_seven_day_v2"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/iterate_retention_v1"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    baseline_plan = json.loads((args.input / "plan.json").read_text())
    baseline_run = json.loads((args.input / "run.json").read_text())
    if (
        not baseline_run["complete"]
        or baseline_run["test_evaluated"]
        or baseline_run["annual_episodes"] != 30
    ):
        raise ValueError("Completed validation-only causal study required")
    for key in ["source_sha256", "baseline_sha256"]:
        for name, expected in baseline_plan[key].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"Baseline source/input changed: {name}")
    config = MPCConfig(**baseline_plan["config"])
    assert config == MPCConfig.load()
    params = settings("train")
    seasonal = DeterministicMPC.from_training(config).forecast
    weights = fit_horizon_weights(params["inflow"], seasonal.inflow)
    np.testing.assert_array_equal(weights, baseline_plan["anomaly_weights"])
    old = pd.read_csv(
        args.input / "trajectories.csv", float_precision="round_trip"
    ).query('method == "persistence"')
    old_diag = pd.read_csv(
        args.input / "mpc_diagnostics.csv", float_precision="round_trip"
    ).query('method == "persistence"')
    sources = [Path(__file__), PROJECT_ROOT / "src/lake_como/mpc_retention.py"]
    sources += [Path(p) for p in baseline_plan["source_sha256"]]
    inputs = [
        args.input / name
        for name in ["plan.json", "run.json", "trajectories.csv", "mpc_diagnostics.csv"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        config=asdict(config),
        forecast="unchanged causal persistence; seasonal demand; annual resets",
        contrast="Only final-iterate selection changes: allow finite bounded strictly cheaper SLSQP status-9 candidates",
        improvement_tolerance=1e-9,
        feasibility_tolerance=1e-8,
        flags="solver_success/status unchanged; retained_iteration_limit labels acceptance; used_fallback means initialization selected",
        variants=["original_fallback", "retain_iterate"],
        years=split_years("validation"),
        test_evaluated=False,
        source_sha256={str(p): digest(p) for p in sources},
        input_sha256={str(p): digest(p) for p in inputs},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frames = []
    annuals = []
    traces = []
    for variant in plan["variants"]:
        for year in split_years("validation"):
            policy = CausalMPC(params, seasonal, config, "persistence", weights)
            policy.controller = RetainingMPC(
                params,
                policy.forecast,
                config,
                retain_iteration_limit=variant == "retain_iterate",
            )
            frame, seconds = evaluate(
                variant,
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
            np.testing.assert_array_equal(trace.forecast_demand_m3_s, seasonal.demand)
            expected_inflow = np.r_[seasonal.inflow[0], frame.inflow_m3_s.iloc[:-1]]
            np.testing.assert_allclose(
                trace.forecast_inflow_m3_s, expected_inflow, atol=1e-8, rtol=0
            )
            if variant == "original_fallback":
                original = old[old.year == year].sort_values("day")
                for column in [
                    "request_m3_s",
                    "level_m",
                    "full_reward",
                    "inflow_m3_s",
                    "demand_m3_s",
                ]:
                    np.testing.assert_array_equal(frame[column], original[column])
                original_diag = old_diag[old_diag.year == year].sort_values("day")
                for column in [
                    "solver_success",
                    "solver_status",
                    "used_fallback",
                    "iterations",
                    "planned_cost",
                    "initial_cost",
                ]:
                    np.testing.assert_array_equal(trace[column], original_diag[column])
            kept = trace.retained_iteration_limit
            assert (~trace.loc[kept, "solver_success"]).all()
            assert trace.loc[kept, "solver_status"].eq(9).all()
            assert (~trace.loc[kept, "used_fallback"]).all()
            assert (
                trace.loc[kept, "planned_cost"] < trace.loc[kept, "initial_cost"] - 1e-9
            ).all()
            assert (trace.planned_cost <= trace.initial_cost + 1e-9).all()
            metrics = annual_metrics(frame)
            trace["year"] = year
            metrics["seconds"] = seconds
            for name, table in [
                ("trajectories", frame),
                ("annual", metrics),
                ("mpc_diagnostics", trace),
            ]:
                table["method"] = variant
                table["controller"] = variant
                path = args.output / f"{name}.csv"
                table.to_csv(path, mode="a", header=not path.exists(), index=False)
            frames.append(frame)
            annuals.append(metrics)
            traces.append(trace)
            print(variant, year, f"complete ({seconds:.2f}s)", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    annual = pd.concat(annuals, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    annual = merge_severity(annual, daily)
    summary = annual.groupby("method")[METRICS].mean()
    paired = (
        annual.query('method == "retain_iterate"').set_index("year")[METRICS]
        - annual.query('method == "original_fallback"').set_index("year")[METRICS]
    )
    optimization = trace.groupby("method").agg(
        decisions=("day", "size"),
        solver_failures=("solver_success", lambda x: int((~x).sum())),
        fallback_days=("used_fallback", "sum"),
        retained_iteration_limit_days=("retained_iteration_limit", "sum"),
    )
    runtime = annual.groupby("method")[
        ["decision_mean_us", "decision_p50_us", "decision_p95_us"]
    ].mean()
    for name, table in [
        ("annual", annual),
        ("summary", summary.reset_index()),
        ("paired_annual", paired.reset_index()),
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
        annual_episodes=20,
        test_evaluated=False,
        control_exactly_reproduced=True,
        source_and_inputs_unchanged=unchanged,
        retained_days=int(trace.retained_iteration_limit.sum()),
        years_return_improves=int((paired.full_return > 0).sum()),
        mean_full_return_change=float(paired.full_return.mean()),
        default_controller_changed=False,
    )
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    report = (
        "# Cost-checked iteration-limit retention\n\n"
        + summary.round(5).to_string()
        + "\n\nSolver flags retained:\n"
        + optimization.to_string()
        + "\n\nPaired annual changes (retention minus original):\n"
        + paired.round(5).to_string()
        + "\n\nValidation only. Experimental variant; default controller unchanged. Extra past-inflow history versus PPO.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
