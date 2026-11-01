"""Bounded common-filter experiment on frozen past-inflow PPO and persistence MPC."""
import argparse
import json
from pathlib import Path
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import make_env, settings
from lake_como.flood_feasibility import grid_rollout
from lake_como.inflow_observation import PastInflowObservation
from lake_como.mpc import DeterministicMPC, MPCConfig, PlanningModel
from lake_como.mpc_causal import CausalMPC, fit_horizon_weights
from lake_como.mpc_hindsight import digest
from lake_como.mpc_objective_closed_loop import level_severity
from lake_como.mpc_retention import RetainingMPC
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import CheckpointPolicy
from lake_como.safety_filter import FloodSafetyFilter, fit_error_limits
from lake_como.splits import split_years

METRICS = [
    "full_return",
    "flood_days",
    "low_water_days",
    "deficit_million_m3",
    "flood_excess_m_days",
    "low_excess_m_days",
    "max_level_m",
]


def main():
    import torch

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/safety/common_filter_v1"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    ppo_root = PROJECT_ROOT / "results/ppo/past_inflow_v1"
    mpc_root = PROJECT_ROOT / "results/mpc/iterate_retention_v1"
    roots = [ppo_root, mpc_root]
    for root in roots:
        run = json.loads((root / "run.json").read_text())
        assert run["complete"] and not run["test_evaluated"]
        source = json.loads((root / "plan.json").read_text())
        for key in ["source_sha256", "control_sha256", "input_sha256"]:
            for name, expected in source.get(key, {}).items():
                assert digest(Path(name)) == expected, name
    params = settings("train")
    config = MPCConfig.load()
    seasonal = DeterministicMPC.from_training(config).forecast
    weights = fit_horizon_weights(params["inflow"], seasonal.inflow)
    errors = fit_error_limits(params["inflow"], seasonal.inflow)
    model = PlanningModel(params, config)
    old_ppo = pd.read_csv(
        ppo_root / "trajectories.csv", float_precision="round_trip"
    ).query('variant == "past_inflow"')
    old_mpc = pd.read_csv(
        mpc_root / "trajectories.csv", float_precision="round_trip"
    ).query('method == "retain_iterate"')
    checkpoints = [
        ppo_root / f"seed_{seed}/final.pt" for seed in [123, 124, 125, 126, 127]
    ]
    inputs = [
        root / name
        for root in roots
        for name in ["plan.json", "run.json", "trajectories.csv"]
    ] + checkpoints
    sources = [
        Path(__file__),
        PROJECT_ROOT / "src/lake_como/safety_filter.py",
        PROJECT_ROOT / "src/lake_como/flood_feasibility.py",
    ]
    for root in roots:
        sources += [
            Path(p)
            for p in json.loads((root / "plan.json").read_text())["source_sha256"]
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        scenario_error_quantiles=[0.01, 0.99],
        training_error_limits_m3_s=errors,
        action_grid_points=401,
        criterion="All hourly model endpoints including initial level <=1.24m under five interval grid scenarios plus persistence; numeric tolerance 1e-7m.",
        fallback="Minimize worst scenario peak, then worst endpoint, then change from clipped proposal; log no sampled safe action.",
        priority="Flood criterion before closeness to original irrigation/reward proposal. No low-water constraint.",
        scope="Finite-scenario conditional filter, not robust guarantee over inflow interval or unknown real inflows. No parameter sweep, retraining, or test evaluation.",
        controls="Rerun all 5 PPO seeds and MPC; exact saved daily reproduction required.",
        test_evaluated=False,
        source_sha256={str(p): digest(p) for p in sources},
        input_sha256={str(p): digest(p) for p in inputs},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frames = []
    annuals = []
    diagnostics = []
    for family, seeds in [("ppo", [123, 124, 125, 126, 127]), ("mpc", [0])]:
        for seed in seeds:
            for variant in ["unfiltered", "filtered"]:
                for year in split_years("validation"):
                    env = make_env("validation", year, correct_terminal_level=True)
                    sim = env.unwrapped
                    filter_policy = FloodSafetyFilter(model, seasonal.inflow[0], errors)
                    if family == "ppo":
                        env = PastInflowObservation(env, seasonal.inflow[0])
                        rescaler = gym.wrappers.RescaleAction(env, -1.0, 1.0)
                        policy = CheckpointPolicy(
                            rescaler.observation_space,
                            rescaler.action_space,
                            ppo_root / f"seed_{seed}/final.pt",
                        )
                    else:
                        policy = CausalMPC(
                            params, seasonal, config, "persistence", weights
                        )
                        policy.controller = RetainingMPC(
                            params, policy.forecast, config, retain_iteration_limit=True
                        )
                    rows = []
                    try:
                        obs, info = env.reset()
                        if family == "ppo":
                            policy(obs)
                        for day in range(1, 366):
                            storage = info["storage"]
                            level = info["level"]
                            start = time.perf_counter_ns()
                            proposed = (
                                rescaler.action(policy(obs))
                                if family == "ppo"
                                else policy(obs, info)
                            )
                            action = (
                                filter_policy(proposed, obs, info)
                                if variant == "filtered"
                                else proposed
                            )
                            elapsed = time.perf_counter_ns() - start
                            obs, reward, terminated, truncated, info = env.step(action)
                            weighted = info["weighted_reward"]
                            np.testing.assert_allclose(
                                sum(weighted.values()), reward, atol=1e-7, rtol=1e-10
                            )
                            q = sim.inflow[day - 1]
                            endpoint, peak = grid_rollout(
                                model.lake,
                                np.array([storage]),
                                np.array([q]),
                                np.array([day]),
                                np.array([sim.actions[-1]]),
                            )
                            np.testing.assert_allclose(
                                endpoint[0, 0], info["level"], atol=1e-10, rtol=0
                            )
                            deficit = max(
                                info["demand"]
                                - max(
                                    info["release"] - sim.lake.min_env_flow[day - 1], 0
                                ),
                                0,
                            )
                            rows.append(
                                dict(
                                    policy=family,
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
                                    request_m3_s=float(
                                        np.asarray(info["action"]).item()
                                    ),
                                    inflow_m3_s=q,
                                    demand_m3_s=info["demand"],
                                    deficit_m3=deficit * 86400,
                                    flood=int(info["level"] > model.flood_level),
                                    low_water=int(info["level"] < model.low_level),
                                    storage_m3=info["storage"],
                                    storage_before_m3=storage,
                                    level_before_m=level,
                                    decision_ns=elapsed,
                                    balance_error_m3=info["storage"]
                                    - storage
                                    - 86400 * (q - info["release"]),
                                    actual_hourly_peak_m=peak[0, 0],
                                    hourly_violation=int(
                                        peak[0, 0] > model.flood_level
                                    ),
                                    **{f"weighted_{k}": v for k, v in weighted.items()},
                                )
                            )
                            assert not terminated and truncated == (day == 365)
                    finally:
                        env.close()
                    frame = pd.DataFrame(rows)
                    if variant == "unfiltered":
                        old = (
                            old_ppo[
                                (old_ppo.training_seed == seed) & (old_ppo.year == year)
                            ]
                            if family == "ppo"
                            else old_mpc[old_mpc.year == year]
                        ).sort_values("day")
                        for column in [
                            "request_m3_s",
                            "level_m",
                            "full_reward",
                            "inflow_m3_s",
                            "demand_m3_s",
                        ]:
                            np.testing.assert_array_equal(frame[column], old[column])
                    annual = annual_metrics(frame)
                    for table in [frame, annual]:
                        table["family"] = family
                        table["variant"] = variant
                        table["training_seed"] = seed
                        table["controller"] = f"{family}_{seed}_{variant}"
                    hourly = frame.hourly_violation.sum()
                    annual["hourly_violation_days"] = hourly
                    if variant == "filtered":
                        diag = pd.DataFrame(filter_policy.diagnostics)
                        diag["family"] = family
                        diag["training_seed"] = seed
                        diag["year"] = year
                        diag["actual_flood"] = frame.flood
                        diag["actual_hourly_violation"] = frame.hourly_violation
                        diag["actual_inflow_m3_s"] = frame.inflow_m3_s
                        diag["actual_inflow_outside_scenario_range"] = (
                            frame.inflow_m3_s < diag.scenario_lower_m3_s
                        ) | (frame.inflow_m3_s > diag.scenario_upper_m3_s)
                        diagnostics.append(diag)
                        path = args.output / "filter_diagnostics.csv"
                        diag.to_csv(
                            path, mode="a", header=not path.exists(), index=False
                        )
                    for name, table in [("trajectories", frame), ("annual", annual)]:
                        path = args.output / f"{name}.csv"
                        table.to_csv(
                            path, mode="a", header=not path.exists(), index=False
                        )
                    frames.append(frame)
                    annuals.append(annual)
                print(
                    family, seed, variant, "all validation years complete", flush=True
                )
    daily = pd.concat(frames, ignore_index=True)
    annual = pd.concat(annuals, ignore_index=True)
    diag = pd.concat(diagnostics, ignore_index=True)
    verify_matched_inputs(daily, split_years("validation"))
    severities = []
    for controller, g in daily.groupby("controller"):
        sev = (
            level_severity(g.assign(objective=controller))
            .reset_index()
            .drop(columns=["objective", "max_level_m", "min_level_m"])
        )
        sev["controller"] = controller
        severities.append(sev)
    annual = annual.merge(
        pd.concat(severities), on=["controller", "year"], validate="one_to_one"
    )
    summary = annual.groupby(["family", "variant"])[
        METRICS + ["hourly_violation_days"]
    ].mean()
    paired = (
        annual.query('variant == "filtered"').set_index(
            ["family", "training_seed", "year"]
        )[METRICS + ["hourly_violation_days"]]
        - annual.query('variant == "unfiltered"').set_index(
            ["family", "training_seed", "year"]
        )[METRICS + ["hourly_violation_days"]]
    )
    interventions = diag.groupby("family").agg(
        decisions=("day", "size"),
        interventions=("intervened", "sum"),
        no_sampled_safe_action=("sampled_safe_action_found", lambda x: int((~x).sum())),
        starts_flooded=("starts_above_threshold", "sum"),
        outside_scenario_range=("actual_inflow_outside_scenario_range", "sum"),
        actual_flood_days=("actual_flood", "sum"),
        actual_hourly_violation_days=("actual_hourly_violation", "sum"),
    )
    for name, table in [
        ("annual", annual),
        ("summary", summary.reset_index()),
        ("paired_annual", paired.reset_index()),
        ("intervention_summary", interventions.reset_index()),
    ]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, family in enumerate(["ppo", "mpc"]):
        for ax, metric in zip(
            axes[row], ["flood_days", "low_water_days", "deficit_million_m3"]
        ):
            ax.bar(
                ["Unfiltered", "Filtered"],
                summary.loc[family].loc[["unfiltered", "filtered"], metric],
                color=["#267ead", "#e78b27"],
            )
            ax.set_title(f"{family.upper()}: {metric}")
    fig.suptitle(
        "Common finite-scenario flood filter · validation annual means · no robust safety guarantee"
    )
    fig.savefig(args.output / "common_filter_comparison.png", dpi=170)
    plt.close(fig)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "input_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                annual_episodes=120,
                test_evaluated=False,
                controls_exactly_reproduced=True,
                source_and_inputs_unchanged=True,
                defaults_changed=False,
            ),
            indent=2,
        )
        + "\n"
    )
    report = (
        "# Common finite-scenario safety filter\n\n"
        + summary.to_string()
        + "\n\nInterventions (pooled counts, 5 PPO policies versus 1 MPC):\n"
        + interventions.to_string()
        + "\n\nOriginal reward and all physical metrics retained. No retraining, guarantee, or default promotion.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
