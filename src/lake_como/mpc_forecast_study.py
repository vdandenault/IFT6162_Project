"""Causal one-day forecast diagnostic; no held-out future controller inputs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import settings
from lake_como.mpc import DeterministicMPC, MPCConfig
from lake_como.mpc_hindsight import digest
from lake_como.mpc_objective_closed_loop import OneDayGridPolicy, level_severity
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import evaluate
from lake_como.splits import split_years


def fit_anomaly_weight(training_inflows, seasonal):
    """Least-squares AR(1) seasonal anomaly coefficient, restricted to [0,1].

    Only within-year day pairs are used; no invented December/January continuity.
    """
    if set(training_inflows) != set(split_years("train")):
        raise ValueError("Exact chronological training years required")
    anomalies = (
        np.array([training_inflows[y][:365] for y in split_years("train")]) - seasonal
    )
    x, target = anomalies[:, :-1].ravel(), anomalies[:, 1:].ravel()
    return float(np.clip(np.dot(x, target) / np.dot(x, x), 0, 1))


class HistoricalInflowPolicy:
    """Recover only the completed day's inflow from observed mass balance.

    Assumes evaporation disabled, exact daily storage/release telemetry. This is
    additional history beyond the default PPO observation, not an operational
    forecast product or a claim about actual Lake Como telemetry availability.
    """

    def __init__(
        self, params, seasonal, config, objective, method, alpha, points=10001
    ):
        if method not in ("seasonal", "persistence", "seasonal_anomaly"):
            raise ValueError("Unknown forecast method")
        if params.get("evaporation", False):
            raise ValueError("Mass-balance inference requires evaporation disabled")
        self.method, self.alpha = method, alpha
        self.seasonal = seasonal
        self.forecast = SimpleNamespace(
            inflow=seasonal.inflow.copy(), demand=seasonal.demand
        )
        self.policy = OneDayGridPolicy(params, self.forecast, config, objective, points)
        self.previous_storage = None
        self.last_day = None

    @property
    def diagnostics(self):
        return self.policy.diagnostics

    def __call__(self, observation, previous_info):
        day = (
            int(
                np.rint(
                    (np.arctan2(observation[1], observation[2]) % (2 * np.pi))
                    * 365
                    / (2 * np.pi)
                )
            )
            % 365
            + 1
        )
        storage = float(previous_info["storage"])
        if day == 1:
            self.previous_storage = None
        elif self.last_day != day - 1:
            raise ValueError("Consecutive daily observations required")
        estimate = None
        prediction = float(self.seasonal.inflow[day - 1])
        if self.previous_storage is not None:
            estimate = (storage - self.previous_storage) / 86400 + float(
                previous_info["release"]
            )
            if self.method == "persistence":
                prediction = max(0.0, estimate)
            elif self.method == "seasonal_anomaly":
                prediction = max(
                    0.0,
                    prediction
                    + self.alpha * (estimate - self.seasonal.inflow[day - 2]),
                )
        self.forecast.inflow[day - 1] = prediction
        action = self.policy(observation, previous_info)
        self.diagnostics[-1]["previous_inflow_estimate_m3_s"] = estimate
        self.previous_storage, self.last_day = storage, day
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/forecast_study_v1"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    config = replace(MPCConfig.load(PROJECT_ROOT / "configs/mpc.json"), horizon_days=1)
    params = settings("train")
    seasonal = DeterministicMPC.from_training(config).forecast
    alpha = fit_anomaly_weight(params["inflow"], seasonal.inflow)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        PROJECT_ROOT / "src/lake_como/mpc_objective_closed_loop.py",
        PROJECT_ROOT / "src/lake_como/mpc.py",
        PROJECT_ROOT / "src/lake_como/reproduce.py",
        PROJECT_ROOT / "configs/mpc.json",
        PROJECT_ROOT / "configs/splits.json",
    ]
    plan = dict(
        training_years=split_years("train"),
        validation_years=split_years("validation"),
        alpha=alpha,
        search="same one-day 10001-action grid for all six arms",
        first_day="seasonal fallback; annual resets",
        information="past-day inflow reconstructed from exact storage/release; evaporation disabled; extra history versus PPO",
        test_evaluated=False,
        source_sha256={str(p): digest(p) for p in sources},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    daily, annual, diagnostics = [], [], []
    for method in ("seasonal", "persistence", "seasonal_anomaly"):
        for objective in ("surrogate", "binary"):
            for year in split_years("validation"):
                policy = HistoricalInflowPolicy(
                    params, seasonal, config, objective, method, alpha
                )
                frame, _ = evaluate(
                    f"{method}_{objective}",
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
                trace["error_m3_s"] = (
                    trace.forecast_inflow_m3_s.to_numpy() - frame.inflow_m3_s.to_numpy()
                )
                metrics = annual_metrics(frame)
                for table in (frame, metrics, trace):
                    table["objective"] = objective
                    table["forecast"] = method
                frame["controller"] = f"{method}_{objective}"
                trace["year"] = year
                daily.append(frame)
                annual.append(metrics)
                diagnostics.append(trace)
                print(method, objective, year, "complete", flush=True)
    daily = pd.concat(daily, ignore_index=True)
    annual = pd.concat(annual, ignore_index=True)
    trace = pd.concat(diagnostics, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    severity = pd.concat(
        [
            level_severity(g).reset_index().assign(forecast=m)
            for m, g in daily.groupby("forecast")
        ]
    )
    annual = annual.merge(
        severity, on=["forecast", "objective", "year"], validate="one_to_one"
    )
    summary = annual.groupby(["forecast", "objective"])[
        [
            "full_return",
            "flood_days",
            "low_water_days",
            "deficit_million_m3",
            "flood_excess_m_days",
            "low_excess_m_days",
        ]
    ].mean()
    accuracy = trace.groupby(["forecast", "objective"]).error_m3_s.agg(
        mae=lambda x: abs(x).mean(), rmse=lambda x: np.sqrt(np.mean(x * x)), bias="mean"
    )
    for name, table in [
        ("trajectories", daily),
        ("annual", annual),
        ("diagnostics", trace),
        ("summary", summary.reset_index()),
        ("forecast_accuracy", accuracy.reset_index()),
    ]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    labels = [f"{m}\n{o}" for m, o in summary.index]
    for ax, column in zip(axes.flat, summary.columns):
        ax.bar(labels, summary[column])
        ax.set_title(column)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(
        "Causal past-inflow forecasts · same one-day grid · validation annual means"
    )
    fig.tight_layout()
    fig.savefig(args.output / "forecast_comparison.png", dpi=160)
    plt.close(fig)
    unchanged = all(digest(Path(p)) == h for p, h in plan["source_sha256"].items())
    assert unchanged
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                annual_episodes=60,
                test_evaluated=False,
                source_unchanged=unchanged,
            ),
            indent=2,
        )
        + "\n"
    )
    report = (
        "# Causal forecast study\n\n"
        + summary.to_string()
        + "\n\nForecast errors (m³/s):\n"
        + accuracy.to_string()
        + "\n\nAdditional past-inflow information; not a matched comparison with current PPO.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
