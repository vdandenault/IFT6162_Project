"""Five-seed, fixed-budget PPO observation ablation; validation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import make_env, settings
from lake_como.experiment_plans import record_plan
from lake_como.inflow_observation import PastInflowObservation
from lake_como.mpc import SeasonalForecast
from lake_como.mpc_hindsight import digest
from lake_como.mpc_objective_closed_loop import level_severity
from lake_como.paths import PROJECT_ROOT, UPSTREAM_ROOT, DAM_DATA, DAM_CONFIG
from lake_como.reproduce import CheckpointPolicy
from lake_como.splits import split_years
from lake_como.stabilize import COMMON, agent_config

STEPS = 1000000
SEEDS = [123, 124, 125, 126, 127]
METRICS = [
    "full_return",
    "objective_return",
    "flood_days",
    "low_water_days",
    "deficit_million_m3",
    "flood_excess_m_days",
    "low_excess_m_days",
]


def first_day_fallback():
    params = settings("train")
    return float(SeasonalForecast.fit(params["inflow"], params["demand"]).inflow[0])


def training_env(fallback):
    from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction

    env = PastInflowObservation(
        make_env("train", protocol="chronological", correct_terminal_level=True),
        fallback,
    )
    return RescaleAction(RecordEpisodeStatistics(env), -1.0, 1.0)


def evaluate_checkpoint(checkpoint, with_inflow, fallback):
    """Same deterministic FP32 policy/scaler inference as the shared runner."""
    rows = []
    for year in split_years("validation"):
        base = make_env(
            "validation", year, protocol="chronological", correct_terminal_level=True
        )
        sim = base.unwrapped
        env = PastInflowObservation(base, fallback) if with_inflow else base
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
        try:
            policy = CheckpointPolicy(
                env.observation_space, env.action_space, checkpoint
            )
            obs, info = env.reset()
            policy(obs)
            for day in range(1, 366):
                storage_before = info["storage"]
                level_before = info["level"]
                start = time.perf_counter_ns()
                action = policy(obs)
                decision_ns = time.perf_counter_ns() - start
                obs, reward, terminated, truncated, info = env.step(action)
                if with_inflow:
                    np.testing.assert_allclose(
                        obs[-1], sim.inflow[day - 1], rtol=1e-6, atol=1e-5
                    )
                weighted = info["weighted_reward"]
                np.testing.assert_allclose(
                    sum(weighted.values()), reward, rtol=1e-10, atol=1e-7
                )
                deficit = max(
                    info["demand"]
                    - max(info["release"] - sim.lake.min_env_flow[day - 1], 0),
                    0,
                )
                rows.append(
                    dict(
                        policy="ppo",
                        seed=123,
                        year=year,
                        day=day,
                        full_reward=reward,
                        objective_reward=sum(
                            weighted[k]
                            for k in [
                                "overflow_reward",
                                "starving_reward",
                                "daily_deficit_reward",
                            ]
                        ),
                        level_m=info["level"],
                        release_m3_s=info["release"],
                        request_m3_s=float(np.asarray(info["action"]).item()),
                        inflow_m3_s=sim.inflow[day - 1],
                        demand_m3_s=info["demand"],
                        deficit_m3=deficit * 86400,
                        flood=int(info["level"] > sim.flood_level),
                        low_water=int(info["level"] < sim.starving_level),
                        storage_m3=info["storage"],
                        storage_before_m3=storage_before,
                        level_before_m=level_before,
                        balance_error_m3=info["storage"]
                        - storage_before
                        - 86400 * (sim.inflow[day - 1] - info["release"]),
                        decision_ns=decision_ns,
                        **{f"weighted_{k}": v for k, v in weighted.items()},
                    )
                )
                assert not terminated and truncated == (day == 365)
        finally:
            env.close()
    return pd.DataFrame(rows)


def verify_control(seed):
    root = PROJECT_ROOT / f"results/stabilization/duration_1m_v1/seed_{seed}"
    metadata = json.loads((root / "run.json").read_text())
    if (
        not metadata["complete"]
        or metadata["test_evaluated"]
        or metadata["job"]["common"] != COMMON
    ):
        raise ValueError("Completed matched control required")
    assert metadata["training_years"] == split_years("train")
    assert metadata["validation_years"] == split_years("validation")
    assert max(metadata["job"]["budgets"]) == STEPS
    checkpoint = root / f"step_{STEPS}.pt"
    assert digest(checkpoint) == metadata["checkpoint_sha256"][str(STEPS)]
    assert metadata["versions"] == {
        p: version(p) for p in ["torch", "skrl", "gymnasium", "numpy"]
    }
    return checkpoint


def worker(job_path):
    import torch
    from gym4real.algorithms.dam.ppo_skrl import Policy, Value
    from skrl.agents.torch.ppo import PPO
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    job = json.loads(job_path.read_text())
    output = Path(job["output"])
    checkpoint = output / "final.pt"
    assert job["steps"] == STEPS and job["seed"] in SEEDS and job["common"] == COMMON
    control = verify_control(job["seed"])
    for name, expected in job["source_sha256"].items():
        assert digest(Path(name)) == expected, name
    torch.set_num_threads(1)
    set_seed(job["seed"])
    env = wrap_env(training_env(job["fallback"]))
    models = dict(
        policy=Policy(
            env.observation_space,
            env.action_space,
            device="cpu",
            net_arch=[16, 16],
            initial_log_std=-0.5,
        ),
        value=Value(
            env.observation_space, env.action_space, device="cpu", net_arch=[16, 16]
        ),
    )
    cfg = agent_config("capped_scheduler", output)
    cfg["state_preprocessor_kwargs"]["size"] = 5

    class CapturePPO(PPO):
        def post_interaction(self, timestep, timesteps):
            super().post_interaction(timestep, timesteps)
            step = timestep + 1
            if step % 10000 == 0:
                (output / "progress.json").write_text(
                    json.dumps(
                        dict(
                            steps=step,
                            total=STEPS,
                            elapsed_seconds=time.perf_counter() - started,
                        )
                    )
                    + "\n"
                )
            if step == STEPS:
                self.save(str(checkpoint))

    agent = CapturePPO(
        models=models,
        memory=RandomMemory(memory_size=2048, num_envs=env.num_envs, device="cpu"),
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu",
    )
    metadata = dict(
        job=job,
        complete=False,
        test_evaluated=False,
        training_years=split_years("train"),
        validation_years=split_years("validation"),
        control_checkpoint=str(control),
        control_sha256=digest(control),
        versions={p: version(p) for p in ["torch", "skrl", "gymnasium", "numpy"]},
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.perf_counter()
    SequentialTrainer(
        cfg={"timesteps": STEPS, "headless": True, "disable_progressbar": True},
        env=env,
        agents=agent,
    ).train()
    metadata["training_seconds"] = time.perf_counter() - started
    metadata["checkpoint_sha256"] = digest(checkpoint)
    # Evaluate after training, so inference RNG cannot influence learning.
    frame = evaluate_checkpoint(checkpoint, True, job["fallback"])
    frame.to_csv(output / "validation_trajectories.csv", index=False)
    annual_metrics(frame).to_csv(output / "validation_annual.csv", index=False)
    metadata["complete"] = True
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Complete seed", job["seed"], flush=True)


def summarize_study(output, plan):
    import torch

    torch.set_num_threads(1)
    all_frames = []
    all_annual = []
    saved = pd.read_csv(
        PROJECT_ROOT / "results/comparison/validation_v1/trajectories.csv",
        float_precision="round_trip",
    ).query('family == "ppo"')
    for seed in SEEDS:
        control = verify_control(seed)
        original = evaluate_checkpoint(control, False, plan["fallback"])
        reference = saved[saved.training_seed == seed].sort_values(["year", "day"])
        for column in [
            "request_m3_s",
            "level_m",
            "full_reward",
            "inflow_m3_s",
            "demand_m3_s",
        ]:
            np.testing.assert_array_equal(original[column], reference[column])
        directory = output / f"seed_{seed}"
        metadata = json.loads((directory / "run.json").read_text())
        assert (
            metadata["complete"]
            and digest(directory / "final.pt") == metadata["checkpoint_sha256"]
        )
        candidate = pd.read_csv(
            directory / "validation_trajectories.csv", float_precision="round_trip"
        )
        for variant, frame in [("four_features", original), ("past_inflow", candidate)]:
            frame = frame.copy()
            frame["variant"] = variant
            frame["training_seed"] = seed
            frame["controller"] = f"{variant}_{seed}"
            annual = annual_metrics(frame)
            severity = (
                level_severity(frame.assign(objective=variant))
                .reset_index()
                .drop(columns=["objective", "max_level_m", "min_level_m"])
            )
            annual = annual.merge(severity, on="year", validate="one_to_one")
            annual["variant"] = variant
            annual["training_seed"] = seed
            all_frames.append(frame)
            all_annual.append(annual)
    daily = pd.concat(all_frames, ignore_index=True)
    annual = pd.concat(all_annual, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    means = annual.groupby(["variant", "training_seed"])[METRICS].mean()
    summary = means.groupby("variant")[METRICS].mean()
    variability = means.groupby("variant").full_return.agg(
        seed_sd="std", weakest_seed="min", strongest_seed="max"
    )
    paired = (
        annual.query('variant == "past_inflow"').set_index(["training_seed", "year"])[
            METRICS
        ]
        - annual.query('variant == "four_features"').set_index(
            ["training_seed", "year"]
        )[METRICS]
    )
    for name, table in [
        ("trajectories", daily),
        ("annual", annual),
        ("seed_means", means.reset_index()),
        ("summary", summary.reset_index()),
        ("variability", variability.reset_index()),
        ("paired_annual", paired.reset_index()),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for variant, g in means.reset_index().groupby("variant"):
        axes[0].plot(g.training_seed, g.full_return, "o-", label=variant)
    axes[0].set(
        title="Annual return by training seed", xlabel="Seed", ylabel="Full reward"
    )
    axes[0].legend()
    for ax, metric, title in zip(
        axes[1:],
        ["flood_days", "low_water_days"],
        ["Flood days/year", "Low-water days/year"],
    ):
        ax.bar(
            ["Four features", "Past inflow"],
            summary.loc[["four_features", "past_inflow"], metric],
        )
        ax.set_title(title)
    fig.savefig(output / "ppo_inflow_comparison.png", dpi=170)
    plt.close(fig)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "control_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    (output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                test_evaluated=False,
                control_exactly_reproduced=True,
                source_and_controls_unchanged=True,
                seeds=SEEDS,
                annual_episodes=100,
            ),
            indent=2,
        )
        + "\n"
    )
    report = (
        "# PPO past-inflow observation ablation\n\n"
        + summary.to_string()
        + "\n\nSeed variability:\n"
        + variability.to_string()
        + "\n\nExisting controls reused after checkpoint verification and exact daily evaluation replay. Five seed means, not 50 independent observations. No test evaluation.\n"
    )
    (output / "report.md").write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/ppo/past_inflow_v1"
    )
    parser.add_argument("--workers", type=int, default=5, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config)
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    fallback = first_day_fallback()
    controls = [verify_control(seed) for seed in SEEDS]
    sources = [
        Path(__file__),
        PROJECT_ROOT / "src/lake_como/inflow_observation.py",
        PROJECT_ROOT / "src/lake_como/environment.py",
        PROJECT_ROOT / "src/lake_como/stabilize.py",
        PROJECT_ROOT / "src/lake_como/reproduce.py",
        PROJECT_ROOT / "configs/ppo.json",
        PROJECT_ROOT / "configs/splits.json",
        PROJECT_ROOT / "uv.lock",
        UPSTREAM_ROOT / "gym4real/algorithms/dam/ppo_skrl.py",
    ]
    sources += list(DAM_DATA.rglob("*.csv")) + list(DAM_CONFIG.glob("*.yaml"))
    plan = dict(
        steps=STEPS,
        seeds=SEEDS,
        common=COMMON,
        scheduler="capped_scheduler",
        training_years=split_years("train"),
        validation_years=split_years("validation"),
        fallback=fallback,
        contrast="Append only completed-day inflow reconstructed from exact storage/release. Same hidden layers; extra input changes first-layer parameter count and RNG draws.",
        checkpoint_selection="Final 1M step only; no intermediate or best-seed selection",
        test_evaluated=False,
        controls="Reuse verified 1M controls, reproduce saved daily validation outputs; not fresh control retraining",
        observation_timing="After step t, append reconstructed inflow t for next decision; reset uses training climatology, never previous-year leakage.",
        source_sha256={str(p): digest(p) for p in sources},
        control_sha256={str(p): digest(p) for p in controls},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    record_plan("past_inflow_observations_v1", plan)
    jobs = []
    for seed in SEEDS:
        directory = args.output / f"seed_{seed}"
        directory.mkdir()
        job = dict(
            seed=seed,
            steps=STEPS,
            common=COMMON,
            fallback=fallback,
            output=str(directory.resolve()),
            source_sha256=plan["source_sha256"],
        )
        path = directory / "job.json"
        path.write_text(json.dumps(job, indent=2) + "\n")
        jobs.append(path)

    def launch(path):
        with (path.parent / "console.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lake_como.ppo_inflow_study",
                    "--worker-config",
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(launch, jobs))
    summarize_study(args.output, plan)


if __name__ == "__main__":
    main()
