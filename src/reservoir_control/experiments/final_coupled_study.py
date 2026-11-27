import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import reservoir_control.systems.ottawa.forecasts as forecasts
from reservoir_control.controllers.ppo.agent import PPOAgent, PPOConfig
from reservoir_control.experiments.final_coupled_env import OBSERVATION_DIM, OFFSETS, HardCoupledEnv
from reservoir_control.experiments.fixed_schedule_replay import (
    CONFIG,
    digest,
    json_value,
    load_follower_config,
)
from reservoir_control.study_config import load_settings
from reservoir_control.systems.loslr.coupled import load_protocol, start_states
from reservoir_control.systems.loslr.data import load_inputs
from reservoir_control.systems.loslr.model import OntarioModel
from reservoir_control.systems.loslr.native import NativeReference
from reservoir_control.systems.loslr.physics import FLOW_SCALE

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


def prepared_case(
    name, year, trajectory, start, base, inflows, capacities, design_label, *, bank=None
):
    trajectory = trajectory.copy()
    p = len(trajectory)
    background = 2000.0
    reference = base["reference_release_m3_s"]
    trajectory["ontLevelBOQ"] = start.level_m
    # The fifth Ontario commitment is anchored by the known initial release
    # and the declared synthetic capacity. No later observed release enters.
    trajectory["ontFlow"] = (
        np.minimum(start.previous_release_m3_s, np.asarray(capacities) - background - reference)
        / FLOW_SCALE
    )
    scenario = {
        "id": name,
        "realized_net_inflow_m3_s": list(inflows),
        "downstream_capacity_m3_s": list(capacities),
        "routing_delay_periods": 1,
        "initial_transit_volumes_m3": [reference * base["clock"]["duration_seconds"]],
    }
    config = {
        **base,
        "known_other_downstream_forecast_m3_s": [background] * p,
        "scenarios": [scenario],
    }
    if bank is None:
        noise = config["forecast_noise"]
        bank = forecasts.forecast_bank(
            inflows,
            "noisy",
            year,
            noise["initial_observed_inflow_estimate_m3_s"],
            noise["standard_deviation_m3_s"],
        )
    return {
        "name": name,
        "year": year,
        "design": design_label,
        "trajectory": trajectory,
        "rows": trajectory.to_dict("records"),
        "start": start,
        "config": config,
        "scenario": scenario,
        "bank": bank,
    }


def annual_cases(frame, states, years, base, design):
    packed = []
    for year in years:
        state, index = states[year]
        window = frame.iloc[index : index + 48].copy()
        if len(window) != 48 or not window.Year.eq(year).all():
            raise ValueError(f"Incomplete annual forcing for {year}")
        for j in range(48):
            previous = frame.iloc[index + j - 1]
            i = window.index[j]
            for lead in range(1, 5):
                window.loc[i, f"ontNTS_QM{lead}"] = previous.ontNTS
            window.loc[i, "forNTS"] = frame.ontNTS.iloc[index + j - 48 : index + j].mean()
            for key in ("iceInd", "ptclaireR", "longsaultR", "saundershwR", "saunderstwR"):
                window.loc[i, key] = previous[key]
            window.loc[i, "foreInd"] = 1
        shapes = list(design["annual_design"])
        shape = shapes[year % len(shapes)]
        inflows = np.interp(np.arange(48), np.arange(12) * 4, design["annual_design"][shape])
        packed.append(
            prepared_case(
                f"annual_{year}",
                year,
                window,
                state,
                base,
                inflows,
                [design["annual_downstream_capacity_m3_s"]] * 48,
                shape,
            )
        )
    return packed


def stress_cases(root, frame, states, base, design_path=None):
    directory = root / "ottawa_two_path_wider_fresh_v1"
    frozen = (
        load_settings(design_path, "confirmation_stress")
        if design_path
        else json.loads((directory / "settings.json").read_text())["design"]
    )
    caps = next(
        case["downstream_capacity_m3_s"]
        for case in base["scenarios"]
        if case["id"] == "one_period_delay"
    )
    cases = []
    for year in (1961, 2017, 2019):
        source = root / f"ottawa_replan_guard_{year}"
        rows = pd.read_csv(source / "ontario_inputs.csv")
        history = json.loads((source / "settings.json").read_text())["ontario_window"][
            "initial_history"
        ]
        state, index = states[year]
        offset = int(rows.QM.iloc[0]) - 1
        # Retain the archived initial commitments and warm the rule-curve
        # history with completed historical mean levels up to this window.
        prior = frame.ontLevelMOQ.iloc[index + offset - 48 : index + offset].to_numpy(dtype=float)
        if np.isfinite(prior).all():
            state = replace(state, mean_level_history_m=tuple(prior))
        state = replace(
            state,
            level_m=rows.ontLevelBOQ.iloc[0],
            previous_release_m3_s=history["previous_flow"] * FLOW_SCALE,
            previous_ice=history["previous_ice"],
        )
        for j in range(len(rows)):
            i = index + offset + j
            rows.loc[j, "forNTS"] = frame.ontNTS.iloc[i - 48 : i].mean()
        for pair in frozen["pairs"]:
            for shape in ("pulse", "plateau"):
                for seed in frozen["seeds"]:
                    stem = f"{pair['name']}_{shape}_{seed}"
                    inflows = pair[shape]
                    if design_path:
                        noise = base["forecast_noise"]
                        raw = forecasts.forecast_bank(
                            inflows,
                            "noisy",
                            seed,
                            noise["initial_observed_inflow_estimate_m3_s"],
                            noise["standard_deviation_m3_s"],
                        )
                        bank = forecasts.recession_bank(inflows, raw, jump_minimum_m3_s=700.0)
                    else:
                        with np.load(directory / str(year) / f"{stem}_forecasts.npz") as archive:
                            bank = archive["receding"]
                    cases.append(
                        prepared_case(
                            f"stress_{year}_{stem}",
                            year,
                            rows,
                            state,
                            base,
                            pair[shape],
                            caps,
                            shape,
                            bank=bank,
                        )
                    )
    return cases


def train(env, seed, budget, out, *, checkpoint_steps=()):
    import torch

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


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Train and evaluate four Ontario controllers under one hard Ottawa execution contract."
    )
    parser.add_argument(
        "--mode", choices=("smoke", "train", "development", "reserved", "stress"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument(
        "--controller", choices=ARMS, help="Evaluate one controller (default: all four)"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--recovery",
        choices=(
            "original",
            "audit",
            "reserve",
            "profiles",
            "proactive_profiles",
            "peak_protected",
        ),
        default="original",
    )
    parser.add_argument(
        "--event-only",
        action="store_true",
        help="Evaluate only the 2017 and 2019 development years",
    )
    parser.add_argument(
        "--stress-design", type=Path, help="Use the declared alternative synthetic stress inputs"
    )
    args = parser.parse_args()
    protocol, design = load_protocol(), load_settings(CONFIG, "annual")
    costs = {**protocol, "incomplete_penalty": design["incomplete_penalty"]}
    base = load_follower_config()
    training = args.mode in ("smoke", "train")
    controllers = (args.controller,) if args.controller else ARMS
    if training and args.controller:
        parser.error("--controller applies only to evaluation")
    if training and args.recovery not in ("original", "profiles"):
        parser.error("PPO training supports the original or profiles execution contract")
    if args.event_only and args.mode != "development":
        parser.error("--event-only applies to development")
    if args.stress_design and args.mode != "stress":
        parser.error("--stress-design applies to stress")
    if training and args.mode == "train" and args.seed not in protocol["seeds"]:
        parser.error("Select one declared PPO seed with --seed")
    if not training and "ppo" in controllers and args.training_root is None:
        parser.error("PPO evaluation requires --training-root")
    years = (
        list(range(protocol["training_years"][0], protocol["training_years"][1] + 1))
        if training
        else sorted(
            set(range(protocol["development_years"][0], protocol["development_years"][1] + 1))
            | set(protocol["development_events"])
        )
        if args.mode == "development"
        else protocol["reserved_test_years"]
        if args.mode == "reserved"
        else [1961, 2017, 2019]
    )
    if args.event_only:
        years = protocol["development_events"]
    reference = NativeReference()
    frame = load_inputs(reference)
    model = OntarioModel(reference)
    states = start_states(model, frame, years)
    cases = (
        stress_cases(Path("results/loslr"), frame, states, base, args.stress_design)
        if args.mode == "stress"
        else annual_cases(frame, states, years, base, design)
    )
    budget = 256 if args.mode == "smoke" else design["training_decisions_per_seed"]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        CONFIG,
        Path(__file__),
        Path(__file__).with_name("final_coupled_env.py"),
        Path(__file__).with_name("fixed_schedule_replay.py"),
        Path(forecasts.__file__),
    ]
    if args.stress_design and args.stress_design.resolve() != CONFIG:
        sources.append(args.stress_design)
    plan = {
        "mode": args.mode,
        "recovery": args.recovery,
        "years": years,
        "case_names": [case["name"] for case in cases],
        "controllers": ("ppo",) if training else controllers,
        "design": design,
        "training_budget": budget,
        "ppo": asdict(PPO),
        "seeds": protocol["seeds"],
        "candidate_offsets_m3_s": OFFSETS,
        "costs": costs,
        "upstream": reference.provenance(),
        "sources_sha256": {str(path): digest(path) for path in sources},
    }
    (args.output / "plan.json").write_text(json.dumps(plan, default=json_value, indent=2) + "\n")
    if training:
        train(
            HardCoupledEnv(model, cases, costs, recovery_mode=args.recovery),
            args.seed or protocol["seeds"][0],
            budget,
            args.output / "agent",
        )
    else:
        agents = []
        if "ppo" in controllers:
            import torch

            torch.set_num_threads(1)
            agents = [
                PPOAgent.load(args.training_root / f"seed_{seed}/agent/final.pt")
                for seed in protocol["seeds"]
            ]
            if any(agent.steps != budget for agent in agents):
                raise ValueError("The saved PPO endpoint does not match the frozen budget")
        evaluate(
            model,
            cases,
            costs,
            agents,
            args.output,
            recovery_mode=args.recovery,
            controllers=controllers,
        )


if __name__ == "__main__":
    main()
