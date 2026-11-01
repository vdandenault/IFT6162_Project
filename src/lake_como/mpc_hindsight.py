"""Validation-only perfect-inflow counterfactual; not a deployable MPC option.

Only the inflow forecast changes. Demand climatology, model, objective, optimizer,
horizon, warm starts, float32 observations, and annual resets remain identical.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import settings
from lake_como.mpc import DeterministicMPC, MPCConfig
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import evaluate
from lake_como.splits import split_years

BASELINE = "seasonal"
HINDSIGHT = "perfect_inflow_hindsight"
METRICS = [
    "full_return",
    "objective_return",
    "flood_days",
    "low_water_days",
    "deficit_million_m3",
    "terminal_storage_m3",
]


@dataclass(frozen=True)
class HindsightForecast:
    inflow: np.ndarray
    demand: np.ndarray
    demand_training_years: tuple
    privileged_information: str = "Actual future validation inflows within the unchanged MPC horizon"


def perfect_inflow_controller(year, params, seasonal, config, validation_inflows):
    """Explicit diagnostic injection; never called by normal MPC factories."""
    if year not in split_years("validation"):
        raise ValueError("Hindsight diagnostic is restricted to validation years")
    raw = np.asarray(validation_inflows[year], dtype=float)
    if (
        len(raw) not in (365, 366)
        or not np.isfinite(raw[:365]).all()
        or (raw[:365] < 0).any()
    ):
        raise ValueError("Expected a finite, nonnegative annual inflow profile")
    inflow = raw[:365].copy()
    inflow.setflags(write=False)
    forecast = HindsightForecast(inflow, seasonal.demand, seasonal.training_years)
    return DeterministicMPC(params, forecast, config)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(root):
    metadata = json.loads((root / "run.json").read_text())
    if (
        not metadata["complete"]
        or metadata["split"] != "validation"
        or metadata["test_evaluated"]
    ):
        raise ValueError("A completed validation-only MPC baseline is required")
    if metadata["years"] != split_years("validation"):
        raise ValueError("Baseline must cover all ten chronological validation years")
    spec = next(c for c in metadata["controllers"] if c["family"] == "mpc")
    config = MPCConfig(**spec["provenance"]["config"])
    for name, expected in metadata["hashes"].items():
        if (
            name
            in [
                "src/lake_como/mpc.py",
                "src/lake_como/environment.py",
                "src/lake_como/reproduce.py",
                "configs/mpc.json",
                "configs/splits.json",
                "uv.lock",
            ]
            or name.startswith("external/gym4real/gym4real/envs/dam/")
            or name.startswith("external/gym4real/gym4real/data/dam/")
        ):
            if digest(PROJECT_ROOT / name) != expected:
                raise ValueError(f"Baseline code/config/input changed: {name}")
    for package in ("numpy", "scipy", "gymnasium"):
        if version(package) != metadata["versions"][package]:
            raise ValueError(f"Baseline library version changed: {package}")
    frame = pd.read_csv(root / "trajectories.csv", float_precision="round_trip")
    frame = frame[frame.family == "mpc"].copy()
    diagnostics = pd.read_csv(
        root / "mpc_diagnostics.csv", float_precision="round_trip"
    )
    if len(frame) != 3650 or frame.duplicated(["year", "day"]).any():
        raise ValueError("Baseline annual episodes are incomplete or duplicated")
    return metadata, config, frame, diagnostics


def collect_diagnostics(controller, frame):
    diagnostics = pd.DataFrame(controller.diagnostics)
    if len(diagnostics) != len(frame):
        raise ValueError("One optimizer record is required for every decision")
    diagnostics["year"] = frame.year.to_numpy()
    diagnostics["actual_level_m"] = frame.level_m.to_numpy()
    diagnostics["level_error_m"] = (
        diagnostics.actual_level_m - diagnostics.predicted_level_m
    )
    diagnostics["inflow_error_m3_s"] = (
        frame.inflow_m3_s.to_numpy() - diagnostics.forecast_inflow_m3_s
    )
    np.testing.assert_array_equal(diagnostics.forecast_inflow_m3_s, frame.inflow_m3_s)
    return diagnostics


def plot_results(annual, trajectories, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = {
        BASELINE: "Seasonal MPC baseline",
        HINDSIGHT: "Perfect inflow · HINDSIGHT ONLY",
    }
    for ax, metric, title in zip(
        axes.flat,
        ["full_return", "flood_days", "low_water_days", "deficit_million_m3"],
        [
            "Full annual return (higher is better)",
            "Flood days",
            "Low-water days",
            "Unmet demand (million m³)",
        ],
    ):
        for variant in (BASELINE, HINDSIGHT):
            f = annual[annual.variant == variant].sort_values("year")
            ax.plot(f.year, f[metric], marker="o", label=labels[variant])
        ax.set(title=title, xlabel="Validation year")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Perfect-inflow diagnostic · same cost, solver, demand forecast, and seven-day horizon"
    )
    fig.savefig(output / "annual_comparison.png", dpi=170)
    plt.close(fig)
    # Windows are defined from the original baseline, before oracle outcomes.
    original = trajectories[trajectories.variant == BASELINE]
    from lake_como.mpc_diagnostics import longest_episode

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    for row, (column, threshold, title) in enumerate(
        [("flood", 1.24, "Flood"), ("low_water", -0.45, "Low water")]
    ):
        year = int(original.groupby("year")[column].sum().sort_index().idxmax())
        f = original[original.year == year]
        episode = longest_episode(f, f[column].astype(bool))
        start = max(1, int(episode.day.min()) - 7)
        end = min(365, int(episode.day.max()) + 7)
        for variant in (BASELINE, HINDSIGHT):
            window = trajectories[
                (trajectories.variant == variant)
                & (trajectories.year == year)
                & trajectories.day.between(start, end)
            ]
            axes[row, 0].plot(window.day, window.level_m, label=labels[variant])
            axes[row, 1].plot(
                window.day,
                -window.weighted_overflow_reward - window.weighted_starving_reward,
                label=labels[variant],
            )
        axes[row, 0].axhline(threshold, color="gray", ls="--")
        axes[row, 0].set(
            title=f"{title}: {year}, original worst-year episode",
            ylabel="Realized level (m)",
            xlabel="Day",
        )
        axes[row, 1].set(
            title="Realized binary level penalties", ylabel="Daily cost", xlabel="Day"
        )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "HINDSIGHT ONLY · different closed-loop trajectories, identical annual reset conditions"
    )
    fig.savefig(output / "baseline_failure_windows.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "results/comparison/mpc_validation_v1",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/perfect_inflow_v1"
    )
    args = parser.parse_args()
    metadata, config, baseline, baseline_diagnostics = load_baseline(args.baseline)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; select a new --output")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = DeterministicMPC.from_training(config)
    params = settings("train")
    validation_inflows = settings("validation")["inflow"]
    years = split_years("validation")
    plan = dict(
        label="PERFECT INFLOW / HINDSIGHT DIAGNOSTIC ONLY",
        years=years,
        config=asdict(config),
        changed="Only forecast inflow: actual future validation values within unchanged horizon",
        unchanged=[
            "training-only demand climatology",
            "hourly planning hydraulics",
            "surrogate and true scoring reward",
            "SLSQP settings and initialization/fallback",
            "warm-start logic",
            "float32 level observations",
            "annual initial/reset conditions",
            "seven-day horizon and zero terminal value",
        ],
        demand_training_years=list(reference.forecast.training_years),
        test_evaluated=False,
        candidate_selection=False,
        deployed_controller_changed=False,
        limitations=[
            "Privileged information: not deployable, not a guaranteed performance bound.",
            "Trajectories and optimizer difficulty may change; this is not a variance decomposition.",
            "Observation/action rounding and hydraulic discontinuities may leave level prediction error.",
        ],
        baseline_file_sha256={
            p.name: digest(p)
            for p in [
                args.baseline / "run.json",
                args.baseline / "trajectories.csv",
                args.baseline / "annual.csv",
                args.baseline / "mpc_diagnostics.csv",
            ]
        },
        source_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__),
                PROJECT_ROOT / "src/lake_como/mpc.py",
                PROJECT_ROOT / "src/lake_como/reproduce.py",
            ]
        },
        versions={p: version(p) for p in ("numpy", "scipy", "gymnasium")},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frames = []
    diagnostics = []
    times = []
    for year in years:
        controller = perfect_inflow_controller(
            year, params, reference.forecast, config, validation_inflows
        )
        frame, seconds = evaluate(
            "mpc_perfect_inflow_hindsight",
            split="validation",
            year=year,
            protocol="chronological",
            correct_terminal_level=True,
            sequential_years=True,
            measure_decisions=True,
            controller_factory=lambda env: controller,
        )
        diagnostic = collect_diagnostics(controller, frame)
        np.testing.assert_array_equal(
            diagnostic.forecast_demand_m3_s,
            reference.forecast.demand[frame.day.to_numpy() - 1],
        )
        frames.append(frame)
        diagnostics.append(diagnostic)
        times.append(dict(year=year, seconds=seconds))
        print(f"HINDSIGHT ONLY: {year} complete ({seconds:.2f}s)", flush=True)
    hindsight = pd.concat(frames, ignore_index=True)
    oracle_diagnostics = pd.concat(diagnostics, ignore_index=True)
    baseline["controller"] = "mpc_seasonal"
    baseline["variant"] = BASELINE
    hindsight["controller"] = "mpc_perfect_inflow_hindsight"
    hindsight["variant"] = HINDSIGHT
    combined = pd.concat([baseline, hindsight], ignore_index=True)
    verify_matched_inputs(combined, years)
    annual_parts = []
    for variant, frame in [(BASELINE, baseline), (HINDSIGHT, hindsight)]:
        a = annual_metrics(frame)
        a["variant"] = variant
        annual_parts.append(a)
    annual = pd.concat(annual_parts, ignore_index=True)
    saved = pd.read_csv(args.baseline / "annual.csv")
    saved = saved[saved.family == "mpc"]
    pd.testing.assert_frame_equal(
        annual_parts[0][["year", *METRICS]].reset_index(drop=True),
        saved[["year", *METRICS]].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-7,
    )
    paired = (
        annual_parts[1].set_index("year")[METRICS]
        - annual_parts[0].set_index("year")[METRICS]
    )
    summary = annual.groupby("variant", sort=False)[METRICS].mean()
    baseline_diagnostics["variant"] = BASELINE
    oracle_diagnostics["variant"] = HINDSIGHT
    all_diagnostics = pd.concat(
        [baseline_diagnostics, oracle_diagnostics], ignore_index=True
    )
    optimization = all_diagnostics.groupby("variant", sort=False).agg(
        decisions=("day", "size"),
        solver_failure_days=("solver_success", lambda s: int((~s).sum())),
        fallback_days=("used_fallback", "sum"),
        inflow_mae_m3_s=("inflow_error_m3_s", lambda s: s.abs().mean()),
        level_mae_m=("level_error_m", lambda s: s.abs().mean()),
        max_level_error_m=("level_error_m", lambda s: s.abs().max()),
    )
    combined.to_csv(args.output / "trajectories.csv", index=False)
    annual.to_csv(args.output / "annual.csv", index=False)
    paired.to_csv(args.output / "paired_annual.csv")
    summary.to_csv(args.output / "summary.csv")
    all_diagnostics.to_csv(args.output / "mpc_diagnostics.csv", index=False)
    optimization.to_csv(args.output / "optimization_summary.csv")
    pd.DataFrame(times).to_csv(args.output / "runtimes.csv", index=False)
    plot_results(annual, combined, args.output)
    report = f"""# Perfect-inflow diagnostic — HINDSIGHT ONLY

Only future inflow values within the existing horizon were replaced. Demand
forecast, surrogate, model, solver, fallback, horizon, and annual resets stayed
fixed. Validation years: 1988–1997. No chronological test evaluation or tuning.
The original baseline was reused after source/config/input/version checks.

## Annual means

```text
{summary.round(3).to_string()}
```

## Solver and forecast diagnostics

```text
{optimization.round(6).to_string()}
```

Differences in paired_annual.csv are hindsight minus seasonal baseline. Higher
returns and lower physical penalties are better. These are different closed-loop
trajectories under common annual forcing and reset conditions. Better forecasts
can change optimizer difficulty too. The results neither provide a statistical
variance decomposition nor establish a deployable performance level or global
upper bound. Observation/action quantization can leave residual model error.

[Annual comparison](annual_comparison.png) · [Original failure windows](baseline_failure_windows.png)
"""
    (args.output / "report.md").write_text(report)
    result = dict(
        complete=True,
        label=plan["label"],
        test_evaluated=False,
        deployed_controller_changed=False,
        mean_full_return_change=float(paired.full_return.mean()),
        years_with_better_full_return=int((paired.full_return > 0).sum()),
        years=len(years),
        perfect_inflow_verified=True,
        demand_forecast_unchanged=True,
        matched_forcing_and_resets=True,
        baseline_file_sha256_after={
            p.name: digest(p)
            for p in [
                args.baseline / "run.json",
                args.baseline / "trajectories.csv",
                args.baseline / "annual.csv",
                args.baseline / "mpc_diagnostics.csv",
            ]
        },
    )
    if result["baseline_file_sha256_after"] != plan["baseline_file_sha256"]:
        raise ValueError("Original baseline files changed during diagnostic")
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.round(2).to_string())
    print(optimization.round(6).to_string())
    print(result)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
