"""Train matched PPO endpoints under peak protection and gate held-out evaluation."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pandas as pd
import torch

from bilevel_reservoirs.solvers.final_coupled_env import HardCoupledEnv
from bilevel_reservoirs.solvers.final_coupled_study import (
    CHRONOLOGY, CONFIG, DESIGN, NativeReference, OntarioModel, annual_cases, evaluate,
    load_inputs, load_protocol, start_states, train,
)
from bilevel_reservoirs.solvers.fixed_schedule_replay import digest
from reservoir_control.controllers.ppo.agent import PPOAgent

METRICS = ("full_cost", "upstream_exposure_m_periods", "downstream_exposure_m_periods")


def prepare(years):
    protocol = load_protocol()
    config, design = (json.loads(path.read_text()) for path in (CONFIG, DESIGN))
    costs = {**protocol, "incomplete_penalty": design["incomplete_penalty"]}
    reference = NativeReference()
    model = OntarioModel(reference)
    frame = load_inputs(reference)
    return model, annual_cases(frame, start_states(model, frame, years), years, config, design), costs


def train_seed(job):
    seed, first, second, years, output = job
    model, cases, costs = prepare(years)
    env = HardCoupledEnv(model, cases, costs, recovery_mode="peak_protected")
    train(env, seed, second, output / f"seed_{seed}", checkpoint_steps=(first,))
    return seed


def compare(output, years, seeds, first, second, checkpoints):
    model, cases, costs = prepare(years)
    samples = []
    for steps, filename in ((first, f"checkpoint_{first}.pt"), (second, "final.pt")):
        agents = [PPOAgent.load(checkpoints / f"seed_{seed}" / filename) for seed in seeds]
        if any(agent.seed != seed or agent.steps != steps for agent, seed in zip(agents, seeds, strict=True)):
            raise ValueError("Unexpected checkpoint")
        path = output / f"steps_{steps}"
        path.mkdir(parents=True, exist_ok=False)
        evaluate(model, cases, costs, agents, path, recovery_mode="peak_protected")
        samples.append(pd.read_csv(path / "runs.csv").assign(steps=steps))
    runs = pd.concat(samples, ignore_index=True)
    ppo = runs[runs.arm.eq("ppo")].copy()
    ppo["study_arm"] = ppo.steps.map({first: "earlier", second: "later"})
    reference = runs[(runs.steps.eq(first)) & runs.arm.eq("external_mpc")].assign(study_arm="external_mpc")
    table = pd.concat([ppo, reference], ignore_index=True)
    complete = table[table.status.eq("completed") & table.executed_periods.eq(table.planned_periods)]
    count = complete.groupby("year").size()
    complete_years = count[count.eq(2 * len(seeds) + 1)].index
    means = complete[complete.year.isin(complete_years)].groupby(["year", "study_arm"])[list(METRICS)].mean().groupby(
        "study_arm",
    ).mean().reindex(["earlier", "later", "external_mpc"])
    old, new, mpc = (means.loc[arm] for arm in ("earlier", "later", "external_mpc"))
    later = table[table.study_arm.eq("later")]
    no_violations = not ((later.storage_violation_m3 > .01) | (later.downstream_violation_m3_s > 1e-5)).any()
    passes = bool(len(complete_years) == len(years) and no_violations and new.full_cost < old.full_cost
                  and new.full_cost < mpc.full_cost and (new.iloc[1:] <= old.iloc[1:] + 1e-9).all())
    table.to_csv(output / "runs.csv", index=False)
    lines = [f"Matched complete years: {len(complete_years)}/{len(years)}.", "",
             "| Controller | Completed | Mean annual cost | Ontario exposure | Lake St. Louis exposure |",
             "|---|---:|---:|---:|---:|"]
    for name, scores in means.iterrows():
        subset = table[table.study_arm.eq(name)]
        lines.append(f"| {name} | {subset.status.eq('completed').sum()}/{len(subset)} | "
                     + " | ".join(f"{value:.5f}" for value in scores) + " |")
    return passes, lines


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4, 5), default=3)
    args = parser.parse_args()
    protocol = load_protocol()
    first = json.loads(DESIGN.read_text())["training_decisions_per_seed"]
    second = 2 * first
    training = list(range(protocol["training_years"][0], protocol["training_years"][1] + 1))
    development = list(range(protocol["development_years"][0], protocol["development_years"][1] + 1))
    reserved = protocol["reserved_test_years"]
    seeds = protocol["seeds"]
    checkpoints = args.output / "training"
    checkpoints.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "training": training, "development": development, "reserved": reserved, "recovery": "peak_protected",
        "seeds": seeds, "endpoints": [first, second], "configuration_sha256": {
            str(path): digest(path) for path in (CHRONOLOGY, CONFIG, DESIGN)},
        "gate": "All years complete, no later violations, later cost below earlier and external MPC, "
                "both later flood exposures at or below earlier PPO",
    }, indent=2) + "\n")
    jobs = [(seed, first, second, training, checkpoints) for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for seed in workers.map(train_seed, jobs):
            print(f"Both endpoints trained for seed {seed}", flush=True)
    torch.set_num_threads(1)
    passed, development_lines = compare(args.output / "development", development, seeds, first, second, checkpoints)
    lines = [
        "# PPO duration under the peak-protected filter", "",
        f"Each of five seeds trained to {second:,} decisions with the same policy, observations, "
        f"objectives and execution contract. The earlier checkpoint at {first:,} decisions is the exact "
        "prefix of that seed's longer run. These are fresh paired runs under the peak-protected contract. "
        "Frozen original-contract policies remain archived.", "",
        "## Development", "", *development_lines, "",
    ]
    confirmed = None
    if passed:
        confirmed, reserved_lines = compare(args.output / "reserved", reserved, seeds, first, second, checkpoints)
        lines += ["## Reserved replay", "", *reserved_lines, "", f"Reserved gate passed: {confirmed}.", ""]
    else:
        lines += ["The duration experiment did not pass the development gate. The reserved replay did not trigger.", ""]
    (args.output / "decision.json").write_text(json.dumps({
        "development_passed": passed, "reserved_passed": confirmed,
    }, indent=2) + "\n")
    (args.output / "report.md").write_text("\n".join(lines))
    from bilevel_reservoirs.solvers.report_final_coupled import curve_figure

    logs = {seed: pd.read_csv(checkpoints / f"seed_{seed}" / "training.csv") for seed in seeds}
    curve_figure(logs, args.output)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
