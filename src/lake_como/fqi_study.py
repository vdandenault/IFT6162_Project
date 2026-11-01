"""Small fixed-data FQI pilot under frozen PPO/MPC information and evaluation tracks.

Run via: uv run --group reproduction --with scikit-learn==1.5.2
    --with numpy==2.2.4 --with scipy==1.15.3 python -m lake_como.fqi_study
The overlay keeps the historical project lock and baseline environments unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import make_env, settings
from lake_como.fqi import FQIController, bellman_targets, greedy_values
from lake_como.inflow_observation import PastInflowObservation
from lake_como.mpc import MPCConfig, PlanningModel, SeasonalForecast
from lake_como.mpc_hindsight import digest
from lake_como.mpc_objective_closed_loop import level_severity
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import evaluate
from lake_como.safety_filter import FloodSafetyFilter, fit_error_limits
from lake_como.splits import split_years


def collect_dataset(config, output, fallback, high):
    """Training years only; all transitions are original unfiltered DamEnv reward."""
    grid = np.linspace(0, high, config["action_grid_points"]).astype(np.float32)
    rng = np.random.default_rng(config["dataset_seed"])
    states = []
    next_states = []
    actions = []
    rewards = []
    done = []
    years = []
    episode_ids = []
    coverage = np.zeros(len(grid), dtype=int)
    for year in split_years("train"):
        for episode in range(config["episodes_per_training_year"]):
            env = PastInflowObservation(make_env("train", year), fallback)
            try:
                obs, info = env.reset()
                for day in range(1, 366):
                    draw = rng.random()
                    if episode == 0 or draw < 0.5:
                        index = int(rng.integers(len(grid)))
                    else:
                        target = float(info["exponential_average_demand"])
                        if draw >= 0.75:
                            target += float(
                                env.unwrapped.lake.min_env_flow[day - 1]
                            ) + 200 * (float(obs[0]) - 0.35)
                        index = int(np.argmin(abs(grid - np.clip(target, 0, high))))
                    action = np.array([grid[index]], dtype=np.float32)
                    following, reward, terminated, truncated, info = env.step(action)
                    states.append(obs.copy())
                    next_states.append(following.copy())
                    actions.append(grid[index] / high)
                    rewards.append(reward / config["reward_scale"])
                    done.append(terminated or truncated)
                    years.append(year)
                    episode_ids.append(f"{year}_{episode}")
                    coverage[index] += 1
                    obs = following
                assert truncated and not terminated
            finally:
                env.close()
    dataset = dict(
        states=np.asarray(states, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float64),
        done=np.asarray(done, dtype=bool),
        years=np.asarray(years),
        episode_ids=np.asarray(episode_ids),
    )
    assert set(dataset["years"]) == set(split_years("train")) and coverage.min() > 0
    assert len(states) == 30660 and sum(done) == 84
    np.savez_compressed(output / "dataset.npz", **dataset)
    pd.DataFrame(dict(request_m3_s=grid, transitions=coverage)).to_csv(
        output / "action_coverage.csv", index=False
    )
    return dataset


def worker(job_path):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    job = json.loads(job_path.read_text())
    config = job["config"]
    output = Path(job["output"])
    root = output.parent
    data = np.load(root / "dataset.npz")
    assert digest(root / "dataset.npz") == job["dataset_sha256"]
    for name, expected in job["source_sha256"].items():
        assert digest(Path(name)) == expected, name
    x = np.column_stack([data["states"], data["actions"]]).astype(np.float32)
    normalized = np.linspace(0, 1, config["action_grid_points"]).astype(np.float32)
    model = None
    records = []
    started = time.perf_counter()
    for iteration in range(1, config["iterations"] + 1):
        values = (
            np.zeros(len(x))
            if model is None
            else greedy_values(model, data["next_states"], normalized)
        )
        targets = bellman_targets(
            data["rewards"], data["done"], values, config["discount_factor"]
        )
        previous = np.zeros(len(x)) if model is None else model.predict(x)
        model = ExtraTreesRegressor(random_state=job["seed"], **config["forest"])
        model.fit(x, targets)
        predictions = model.predict(x)
        records.append(
            dict(
                iteration=iteration,
                mean_absolute_training_prediction_change=float(
                    abs(predictions - previous).mean()
                ),
                fit_rmse=float(np.sqrt(np.mean((predictions - targets) ** 2))),
                target_min=float(targets.min()),
                target_max=float(targets.max()),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        pd.DataFrame(records).to_csv(output / "fitting_trace.csv", index=False)
        if iteration % 10 == 0:
            print(job["seed"], "fit", iteration, flush=True)
    joblib.dump(model, output / "final.joblib")
    (output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                seed=job["seed"],
                iterations=config["iterations"],
                training_seconds=time.perf_counter() - started,
                checkpoint_sha256=digest(output / "final.joblib"),
                dataset_sha256=job["dataset_sha256"],
                test_evaluated=False,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    import joblib
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/fqi/pilot_v1"
    )
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    config = json.loads((PROJECT_ROOT / "configs/fqi.json").read_text())
    freeze_path = PROJECT_ROOT / "configs/experiments/control_protocol_v1.json"
    freeze = json.loads(freeze_path.read_text())
    for name, expected in freeze["source_sha256"].items():
        assert digest(PROJECT_ROOT / name) == expected, name
    assert (
        version("scikit-learn") == "1.5.2"
        and version("numpy") == freeze["versions"]["numpy"]
        and version("scipy") == freeze["versions"]["scipy"]
    )
    params = settings("train")
    seasonal = SeasonalForecast.fit(params["inflow"], params["demand"])
    fallback = float(seasonal.inflow[0])
    model = PlanningModel(params, MPCConfig.load())
    errors = fit_error_limits(params["inflow"], seasonal.inflow)
    args.output.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        PROJECT_ROOT / "src/lake_como/fqi.py",
        PROJECT_ROOT / "configs/fqi.json",
        freeze_path,
    ]
    files += [PROJECT_ROOT / p for p in freeze["source_sha256"]]
    plan = dict(
        config=config,
        train_years=split_years("train"),
        validation_years=split_years("validation"),
        test_evaluated=False,
        scope="First fixed-data FQI pilot. Three regression seeds on shared data; not three independent datasets, budget-matched PPO, convergence proof, or reservoir-paper reproduction.",
        equation="Q0=0; y=r/1000 + gamma*(not terminated/truncated)*max_a Qprev(s_next,a); refit ExtraTrees on [raw five features, normalized action].",
        limits="60 Bellman fits, 21 discrete actions, 30660 transitions. Depth/coverage and residual approximation need investigation before mature benchmark claims.",
        sources=[
            "https://jmlr.org/papers/volume6/ernst05a/ernst05a.pdf",
            "https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
        ],
        source_sha256={str(p): digest(p) for p in files},
        versions={
            p: version(p) for p in ["scikit-learn", "numpy", "scipy", "gymnasium"]
        },
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    collect_dataset(config, args.output, fallback, model.high)
    jobs = []
    for seed in config["regressor_seeds"]:
        output = args.output / f"seed_{seed}"
        output.mkdir()
        job = dict(
            seed=seed,
            config=config,
            output=str(output.resolve()),
            dataset_sha256=digest(args.output / "dataset.npz"),
            source_sha256=plan["source_sha256"],
        )
        path = output / "job.json"
        path.write_text(json.dumps(job, indent=2) + "\n")
        jobs.append(path)

    def launch(path):
        with (path.parent / "console.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lake_como.fqi_study",
                    "--worker-config",
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(launch, jobs))
    frames = []
    annuals = []
    diagnostics = []
    for seed in config["regressor_seeds"]:
        regressor = joblib.load(args.output / f"seed_{seed}/final.joblib")
        for variant in ["plain", "shared_filter"]:
            filt = (
                FloodSafetyFilter(model, fallback, errors)
                if variant == "shared_filter"
                else None
            )
            controller = FQIController(
                regressor,
                np.linspace(0, 1, config["action_grid_points"]),
                model.high,
                fallback,
                filt,
            )
            frame, _ = evaluate(
                "fqi",
                split="validation",
                protocol="chronological",
                correct_terminal_level=True,
                sequential_years=True,
                measure_decisions=True,
                controller_factory=lambda env: controller,
            )
            frame["controller"] = f"fqi_{seed}_{variant}"
            frame["regressor_seed"] = seed
            frame["variant"] = variant
            annual = annual_metrics(frame)
            severity = (
                level_severity(frame.assign(objective=variant))
                .reset_index()
                .drop(columns=["objective", "max_level_m", "min_level_m"])
            )
            annual = annual.merge(severity, on="year", validate="one_to_one")
            annual["regressor_seed"] = seed
            annual["variant"] = variant
            frames.append(frame)
            annuals.append(annual)
            if filt is not None:
                diag = pd.DataFrame(filt.diagnostics)
                diag["regressor_seed"] = seed
                diagnostics.append(diag)
            print("Evaluated FQI", seed, variant, flush=True)
    daily = pd.concat(frames, ignore_index=True)
    annual = pd.concat(annuals, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    columns = [
        "full_return",
        "flood_days",
        "low_water_days",
        "deficit_million_m3",
        "flood_excess_m_days",
        "low_excess_m_days",
    ]
    means = annual.groupby(["variant", "regressor_seed"])[columns].mean()
    summary = means.groupby("variant").mean()
    variability = means.groupby("variant").full_return.agg(
        regressor_seed_sd="std", weakest_seed="min"
    )
    for name, table in [
        ("trajectories", daily),
        ("annual", annual),
        ("seed_means", means.reset_index()),
        ("summary", summary.reset_index()),
        ("variability", variability.reset_index()),
        ("filter_diagnostics", pd.concat(diagnostics, ignore_index=True)),
    ]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, column in zip(axes, ["full_return", "flood_days", "deficit_million_m3"]):
        ax.bar(
            ["Plain", "Shared filter"], summary.loc[["plain", "shared_filter"], column]
        )
        ax.set_title(column)
    fig.suptitle(
        "First FQI pilot · 3 forest seeds, one shared training dataset · validation only"
    )
    fig.savefig(args.output / "fqi_pilot.png", dpi=170)
    plt.close(fig)
    unchanged = all(digest(Path(p)) == h for p, h in plan["source_sha256"].items())
    assert unchanged
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                test_evaluated=False,
                training_interactions=30660,
                regressor_seeds=config["regressor_seeds"],
                annual_episodes=60,
                source_unchanged=True,
                defaults_changed=False,
                pilot=True,
            ),
            indent=2,
        )
        + "\n"
    )
    report = (
        "# First tree-based FQI pilot\n\n"
        + summary.to_string()
        + "\n\n"
        + variability.to_string()
        + "\n\nThree regressor seeds share one 30660-transition training dataset. Final 60th fit, 21 actions; not convergence or sample-efficiency evidence.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
