import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import torch

import reservoir_control.systems.ottawa.forecasts as forecasts
from reservoir_control.controllers.ppo.agent import PPOAgent
from reservoir_control.evaluation.provenance import digest, json_value
from reservoir_control.experiments.coupled_cases import annual_cases, stress_cases
from reservoir_control.experiments.coupled_planning import OFFSETS
from reservoir_control.experiments.coupled_runs import ARMS, PPO, evaluate, train
from reservoir_control.experiments.final_coupled_env import HardCoupledEnv
from reservoir_control.study_config import COUPLED_CONFIG as CONFIG
from reservoir_control.study_config import load_follower_config, load_settings
from reservoir_control.systems.loslr.coupled import load_protocol, start_states
from reservoir_control.systems.loslr.data import load_inputs
from reservoir_control.systems.loslr.model import OntarioModel
from reservoir_control.systems.loslr.native import NativeReference

LOG = logging.getLogger(__name__)


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
    if training:
        first, last = protocol["training_years"]
        years = list(range(first, last + 1))
    elif args.event_only:
        years = protocol["development_events"]
    elif args.mode == "development":
        first, last = protocol["development_years"]
        years = sorted(set(range(first, last + 1)).union(protocol["development_events"]))
    elif args.mode == "reserved":
        years = protocol["reserved_test_years"]
    else:
        years = [1961, 2017, 2019]
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
        Path(__file__).with_name("coupled_cases.py"),
        Path(__file__).with_name("coupled_forecast.py"),
        Path(__file__).with_name("coupled_planning.py"),
        Path(__file__).with_name("coupled_runs.py"),
        Path(__file__).with_name("coupled_transition.py"),
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
