import logging
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from reservoir_control.controllers.ppo.agent import PPOAgent, PPOConfig
from reservoir_control.experiments.final_coupled_env import OBSERVATION_DIM, HardCoupledEnv

LOG = logging.getLogger(__name__)

ARMS = ("rule_curve", "external_mpc", "stackelberg_mpc", "ppo")
PPO = PPOConfig(
    observation_dim=OBSERVATION_DIM,
    hidden=32,
    rollout=256,
    epochs=4,
    batch_size=64,
    learning_rate=1e-4,
    reward_scale=1.0,
)


def train(env, seed, budget, out, *, checkpoint_steps=()):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    agent = PPOAgent(PPO, action_low=-2000.0, action_high=2000.0, seed=seed)
    obs, _ = env.reset(seed=seed)
    log = []
    while agent.steps < budget:
        rollout = {
            key: []
            for key in (
                "observation",
                "latent",
                "log_probability",
                "value",
                "reward",
                "next_value",
                "terminated",
                "truncated",
            )
        }
        counts = {"episodes": 0, "incomplete": 0, "projection": 0}
        reward_sum = 0.0
        for _ in range(min(PPO.rollout, budget - agent.steps)):
            action, latent, probability, value = agent.act(obs)
            following, reward, terminated, truncated, row = env.step(action)
            for key, value in zip(
                rollout,
                (
                    obs,
                    latent,
                    probability,
                    value,
                    reward,
                    0.0 if terminated else agent.value(following),
                    terminated,
                    truncated,
                ),
                strict=True,
            ):
                rollout[key].append(value)
            reward_sum += reward
            counts["episodes"] += int(terminated)
            counts["incomplete"] += int(
                terminated and (row["status"] != "executed" or row["period"] != 48)
            )
            counts["projection"] += int((row.get("intervention_m3_s") or 0.0) > 1e-5)
            obs = env.reset()[0] if terminated or truncated else following
            agent.steps += 1
        entry = {
            "steps": agent.steps,
            "rollout_reward": reward_sum,
            **counts,
            **agent.update(rollout),
        }
        if not np.isfinite(list(entry.values())).all():
            raise ArithmeticError("Nonfinite PPO training diagnostic")
        log.append(entry)
        pd.DataFrame(log).to_csv(out / "training.csv", index=False)
        if agent.steps in checkpoint_steps:
            agent.save(out / f"checkpoint_{agent.steps}.pt")
        if agent.steps % 2048 == 0 or agent.steps == budget:
            LOG.info("PPO seed %d: %d/%d steps", seed, agent.steps, budget)
    agent.save(out / "final.pt")


def evaluate(
    model, cases, costs, agents, destination, *, recovery_mode="original", controllers=ARMS
):
    env = HardCoupledEnv(model, cases, costs, native_costs=True, recovery_mode=recovery_mode)
    policies = [(arm, None) for arm in controllers if arm != "ppo"]
    if "ppo" in controllers:
        policies.extend(("ppo", agent) for agent in agents)
    trajectories, results = [], []
    for index, case in enumerate(cases):
        for arm, agent in policies:
            started = perf_counter()
            obs, _ = env.reset(options={"case": index})
            done = False
            while not done:
                if agent is not None:
                    action, *_ = agent.act(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(action)
                else:
                    obs, _, done, _, _ = env.apply(arm)
            executed = [row for row in env.records if "ontario_applied_m3_s" in row]
            n = len(case["rows"])
            seed = agent.seed if agent else -1
            status = env.records[-1]["status"]
            complete = status == "executed" and len(executed) == n
            meta = {
                "case": case["name"],
                "year": case["year"],
                "design": case["design"],
                "arm": arm,
                "seed": seed,
            }
            for row in env.records:
                trajectories.append({**meta, **row})
            results.append(
                {
                    **meta,
                    "status": "completed" if complete else status,
                    "executed_periods": len(executed),
                    "planned_periods": n,
                    "full_cost": sum(row["cost"] for row in executed) if complete else None,
                    "upstream_exposure_m_periods": sum(
                        max(row["ontario_end_m"] - costs["ontario_flood_cap_m"], 0.0)
                        for row in executed
                    ),
                    "downstream_exposure_m_periods": sum(
                        max(row["pointe_claire_m"] - row["downstream_target_m"], 0.0)
                        for row in executed
                    ),
                    "storage_violation_m3": max(
                        (row["ottawa_storage_violation_m3"] for row in executed), default=0.0
                    ),
                    "downstream_violation_m3_s": max(
                        (
                            max(0.0, row["st_louis_flow_m3_s"] - row["capacity_m3_s"])
                            for row in executed
                        ),
                        default=0.0,
                    ),
                    "ottawa_end_storage_m3": executed[-1]["ottawa_end_storage_m3"]
                    if executed
                    else None,
                    "terminal_transit_m3": executed[-1]["ottawa_end_transit_m3"]
                    if complete
                    else None,
                    "ottawa_service_shortfall_million_m3": sum(
                        max(0.0, 900.0 - row["ottawa_executed_release_m3_s"]) * 657000.0 / 1e6
                        for row in executed
                    ),
                    "energy_value": -sum(row["negative_energy_value"] for row in executed),
                    "request_interventions": sum(
                        (row["intervention_m3_s"] or 0.0) > 1e-5 for row in executed
                    ),
                    "runtime_seconds": perf_counter() - started,
                }
            )
        statuses = ", ".join(row["status"] for row in results[-len(policies) :])
        LOG.info("%s: %s", case["name"], statuses)
    pd.DataFrame(trajectories).to_csv(destination / "trajectories.csv", index=False)
    pd.DataFrame(results).to_csv(destination / "runs.csv", index=False)
