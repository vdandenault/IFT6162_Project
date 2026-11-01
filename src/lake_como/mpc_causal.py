"""Causal seven-day forecasts around the unchanged deterministic MPC solver."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, paired_metrics, verify_matched_inputs
from lake_como.environment import settings
from lake_como.mpc import DeterministicMPC, MPCConfig
from lake_como.mpc_hindsight import digest, load_baseline
from lake_como.mpc_objective_closed_loop import level_severity
from lake_como.paths import DAM_CONFIG, DAM_DATA, PROJECT_ROOT, UPSTREAM_ROOT
from lake_como.reproduce import evaluate
from lake_como.splits import split_years

METHODS = ("seasonal", "persistence", "seasonal_anomaly")
METRICS = [
    "full_return",
    "objective_return",
    "flood_days",
    "low_water_days",
    "deficit_million_m3",
    "flood_excess_m_days",
    "low_excess_m_days",
    "terminal_storage_m3",
    "max_level_m",
    "min_level_m",
]


def fit_horizon_weights(training_inflows, climatology, horizon=7):
    """Direct anomaly regressions at lags 1..H, with no cross-year pairs.

    For a decision on day d, the last known inflow is day d-1. Lead k
    predicts day d+k from that observation, so its training lag is k+1.
    Coefficients are constrained to [0,1], as in the one-day experiment.
    """
    if set(training_inflows) != set(split_years("train")):
        raise ValueError("Exact chronological training years required")
    if not 1 <= horizon < 365:
        raise ValueError("Invalid horizon")
    matrix = np.array(
        [training_inflows[y][:365] for y in split_years("train")], dtype=float
    )
    if matrix.shape != (42, 365) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Finite nonnegative training profiles required")
    anomalies = matrix - climatology
    weights = []
    for lag in range(1, horizon + 1):
        x = anomalies[:, :-lag].ravel()
        target = anomalies[:, lag:].ravel()
        denominator = np.dot(x, x)
        weights.append(
            float(np.clip(np.dot(x, target) / denominator, 0, 1))
            if denominator
            else 0.0
        )
    result = np.array(weights)
    result.setflags(write=False)
    return result


class CausalMPC:
    """Update only the forecast; call the original MPC implementation unchanged.

    Inflow reconstruction assumes disabled evaporation and exact completed-day
    storage/release telemetry. This adds history beyond current PPO inputs.
    """

    def __init__(self, params, seasonal, config, method, weights):
        if method not in METHODS:
            raise ValueError("Unknown forecast method")
        if len(weights) != config.horizon_days or not np.isfinite(weights).all():
            raise ValueError("One finite coefficient per planning lead required")
        self.method = method
        self.weights = np.array(weights, copy=True)
        self.seasonal = seasonal
        self.forecast = SimpleNamespace(
            inflow=seasonal.inflow.copy(), demand=seasonal.demand
        )
        self.controller = DeterministicMPC(params, self.forecast, config)
        self.previous_storage = None
        self.previous_day = None
        self.forecasts = []

    @property
    def diagnostics(self):
        return self.controller.diagnostics

    def __call__(self, observation, previous_info):
        obs = np.asarray(observation)
        if obs.shape != (4,) or not np.isfinite(obs).all():
            raise ValueError("Raw finite four-number observation required")
        day0 = (
            int(np.rint((np.arctan2(obs[1], obs[2]) % (2 * np.pi)) * 365 / (2 * np.pi)))
            % 365
        )
        storage = float(previous_info["storage"])
        if day0 == 0:
            self.previous_storage = None
        elif self.previous_day != day0 - 1:
            raise ValueError("Consecutive daily observations required")
        horizon = min(len(self.weights), 365 - day0)
        predictions = self.seasonal.inflow[day0 : day0 + horizon].copy()
        estimate = None
        if self.previous_storage is not None and self.method != "seasonal":
            estimate = (storage - self.previous_storage) / 86400 + float(
                previous_info["release"]
            )
            if self.method == "persistence":
                predictions[:] = max(0.0, estimate)
            else:
                anomaly = estimate - self.seasonal.inflow[day0 - 1]
                predictions = np.maximum(
                    0.0, predictions + self.weights[:horizon] * anomaly
                )
        self.forecast.inflow[day0 : day0 + horizon] = predictions
        action = self.controller(observation, previous_info)
        self.forecasts.append(predictions.copy())
        self.diagnostics[-1]["previous_inflow_estimate_m3_s"] = estimate
        self.previous_storage, self.previous_day = storage, day0
        return action


def plot_results(summary, output):
    labels = ["Seasonal", "Persistence", "Seasonal\nanomaly"]
    titles = [
        "Annual reward ↑",
        "Flood days ↓",
        "Low-water days ↓",
        "Unmet demand (million m³) ↓",
        "Flood exceedance (m·days) ↓",
        "Low-water exceedance (m·days) ↓",
    ]
    columns = [
        "full_return",
        "flood_days",
        "low_water_days",
        "deficit_million_m3",
        "flood_excess_m_days",
        "low_excess_m_days",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, column, title in zip(axes.flat, columns, titles):
        ax.bar(
            labels,
            summary.loc[list(METHODS), column],
            color=["#7d8b95", "#2980b9", "#269376"],
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Causal seven-day MPC · unchanged surrogate cost and SLSQP · validation means",
        fontsize=15,
    )
    fig.savefig(output / "causal_mpc_comparison.png", dpi=170)
    plt.close(fig)


def merge_severity(annual, daily):
    """Add exceedance integrals without duplicating existing annual extrema."""
    severities = []
    for method, group in daily.groupby("method"):
        group = group.assign(objective="surrogate")
        severity = (
            level_severity(group)
            .reset_index()
            .drop(columns=["objective", "max_level_m", "min_level_m"])
        )
        severity["method"] = method
        severities.append(severity)
    return annual.merge(
        pd.concat(severities), on=["method", "year"], validate="one_to_one"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/causal_seven_day_v2"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "results/comparison/mpc_validation_v1",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    baseline, config, baseline_daily, _ = load_baseline(args.baseline)
    if config != MPCConfig.load() or config.horizon_days != 7:
        raise ValueError("Original seven-day baseline config required")
    params = settings("train")
    seasonal = DeterministicMPC.from_training(config).forecast
    weights = fit_horizon_weights(
        params["inflow"], seasonal.inflow, config.horizon_days
    )
    files = [
        Path(__file__),
        PROJECT_ROOT / "src/lake_como/mpc.py",
        PROJECT_ROOT / "src/lake_como/reproduce.py",
        PROJECT_ROOT / "src/lake_como/environment.py",
        PROJECT_ROOT / "src/lake_como/compare.py",
        PROJECT_ROOT / "src/lake_como/mpc_objective_closed_loop.py",
        PROJECT_ROOT / "configs/mpc.json",
        PROJECT_ROOT / "configs/splits.json",
        PROJECT_ROOT / "uv.lock",
        UPSTREAM_ROOT / "gym4real/envs/dam/env.py",
        UPSTREAM_ROOT / "gym4real/envs/dam/lake.py",
    ]
    files += list(DAM_DATA.rglob("*.csv")) + list(DAM_CONFIG.glob("*.yaml"))
    original_files = [
        args.baseline / "run.json",
        args.baseline / "trajectories.csv",
        args.baseline / "annual.csv",
        args.baseline / "mpc_diagnostics.csv",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        config=asdict(config),
        methods=METHODS,
        training_years=split_years("train"),
        validation_years=split_years("validation"),
        anomaly_weights=weights.tolist(),
        forecast_definition="Persistence: flat last observed daily inflow across horizon. Anomaly: direct lead-specific training regression, not recursively fitted.",
        information="Only completed-day storage/release telemetry; no observed current/future inflows. January 1 seasonal fallback. Annual resets.",
        fixed="Original cost, SLSQP settings, model, demand climatology, warm starts, fallback, observation/action precision; horizon shrinks at year end.",
        comparison="Forecast ablation only. Extra history versus PPO; not a matched PPO comparison.",
        test_evaluated=False,
        source_sha256={str(p): digest(p) for p in files},
        baseline_sha256={str(p): digest(p) for p in original_files},
        versions={p: version(p) for p in ["numpy", "scipy", "gymnasium"]},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    annuals = []
    frames = []
    traces = []
    errors = []
    for method in METHODS:
        for year in split_years("validation"):
            policy = CausalMPC(params, seasonal, config, method, weights)
            frame, seconds = evaluate(
                "mpc_" + method,
                split="validation",
                year=year,
                protocol="chronological",
                correct_terminal_level=True,
                sequential_years=True,
                measure_decisions=True,
                controller_factory=lambda env: policy,
            )
            trace = pd.DataFrame(policy.diagnostics)
            if method != "seasonal":
                np.testing.assert_allclose(
                    trace.previous_inflow_estimate_m3_s.iloc[1:],
                    frame.inflow_m3_s.iloc[:-1],
                    atol=1e-8,
                    rtol=0,
                )
            np.testing.assert_array_equal(trace.forecast_demand_m3_s, seasonal.demand)
            for day0, predictions in enumerate(policy.forecasts):
                for lead, prediction in enumerate(predictions):
                    errors.append(
                        dict(
                            method=method,
                            year=year,
                            day=day0 + 1,
                            lead=lead + 1,
                            forecast_inflow_m3_s=prediction,
                            actual_inflow_m3_s=frame.inflow_m3_s.iloc[day0 + lead],
                            error_m3_s=prediction - frame.inflow_m3_s.iloc[day0 + lead],
                        )
                    )
            metrics = annual_metrics(frame)
            for table in (metrics, frame, trace):
                table["method"] = method
                table["controller"] = method
            trace["year"] = year
            trace["level_error_m"] = frame.level_m.to_numpy() - trace.predicted_level_m
            metrics["seconds"] = seconds
            # Save completed rollouts before final aggregation.
            for name, table in [
                ("trajectories", frame),
                ("annual", metrics),
                ("mpc_diagnostics", trace),
            ]:
                path = args.output / f"{name}.csv"
                table.to_csv(path, mode="a", header=not path.exists(), index=False)
            annuals.append(metrics)
            frames.append(frame)
            traces.append(trace)
            print(method, year, f"complete ({seconds:.2f}s)", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    annual = pd.concat(annuals, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    # Seasonal wrapper must reproduce the preserved original MPC, not just its mean.
    actual = daily.query('method == "seasonal"').sort_values(["year", "day"])
    expected = baseline_daily.sort_values(["year", "day"])
    for column in [
        "request_m3_s",
        "level_m",
        "full_reward",
        "inflow_m3_s",
        "demand_m3_s",
    ]:
        np.testing.assert_array_equal(actual[column], expected[column])
    annual = merge_severity(annual, daily)
    summary = annual.groupby("method")[METRICS].mean()
    optimization = trace.groupby("method").agg(
        decisions=("day", "size"),
        solver_failures=("solver_success", lambda x: (~x).sum()),
        fallback_days=("used_fallback", "sum"),
        level_mae_m=("level_error_m", lambda x: abs(x).mean()),
    )
    errors = pd.DataFrame(errors)
    accuracy = errors.groupby(["method", "lead"]).error_m3_s.agg(
        mae=lambda x: abs(x).mean(), rmse=lambda x: np.sqrt(np.mean(x * x)), bias="mean"
    )
    paired = paired_metrics(annual, METRICS)
    for name, table in [
        ("trajectories", daily),
        ("annual", annual),
        ("mpc_diagnostics", trace),
        ("forecast_predictions", errors),
        ("paired_metrics", paired),
        ("summary", summary.reset_index()),
        ("optimization_summary", optimization.reset_index()),
        ("forecast_accuracy", accuracy.reset_index()),
    ]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    plot_results(summary, args.output)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "baseline_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    result = dict(
        complete=True,
        test_evaluated=False,
        annual_episodes=30,
        source_and_baseline_unchanged=unchanged,
        seasonal_baseline_exactly_reproduced=True,
        deployed_controller_changed=False,
    )
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    report = (
        "# Causal seven-day MPC forecast ablation\n\n"
        + summary.round(5).to_string()
        + "\n\nSolver diagnostics:\n"
        + optimization.to_string()
        + "\n\nOriginal cost/solver preserved. Extra past-inflow history versus PPO; validation only.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
