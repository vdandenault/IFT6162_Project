"""Compare deterministic and modal executed-release choices for frozen PPO."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bilevel_reservoirs.solvers.diagnose_ppo_menu import enumerate_menu, release_probabilities
from bilevel_reservoirs.solvers.final_coupled_env import HardCoupledEnv
from bilevel_reservoirs.solvers.final_coupled_study import (
    CHRONOLOGY, CONFIG, DESIGN, NativeReference, OntarioModel, annual_cases, load_inputs,
    load_protocol, start_states,
)
from bilevel_reservoirs.solvers.fixed_schedule_replay import digest
from reservoir_control.controllers.ppo.agent import PPOAgent

ARMS = ("external_mpc", "ppo_deterministic", "ppo_modal")
METRICS = ("cost", "upstream_exposure", "downstream_exposure")


def modal_action(env, agent, observation, deterministic):
    neutral = env.neutral_request()
    request = max(0., neutral + float(deterministic.item()))
    releases, _ = enumerate_menu(env, request, neutral)
    if not releases:
        return deterministic, None
    with torch.no_grad():
        distribution = agent.model.distribution(torch.as_tensor(observation).reshape(1, -1))
    low, high = env.action_space.low.item(), env.action_space.high.item()
    probabilities = release_probabilities(np.array(releases), neutral, distribution.mean.item(),
                                          distribution.stddev.item(), low, high)
    planned = releases[int(probabilities.argmax())]
    action = np.clip(planned - neutral, low, high)
    return np.array([action]), planned


def evaluate(model, cases, costs, agents):
    env = HardCoupledEnv(model, cases, costs, recovery_mode="peak_protected")
    runs, trace = [], []
    for index, case in enumerate(cases):
        for arm in ARMS:
            members = [None] if arm == "external_mpc" else agents
            for agent in members:
                obs, _ = env.reset(options={"case": index})
                done = False
                while not done:
                    if agent is None:
                        obs, _, done, _, _ = env.apply(arm)
                        continue
                    action, *_ = agent.act(obs, deterministic=True)
                    modal = None
                    if arm == "ppo_modal":
                        action, modal = modal_action(env, agent, obs, action)
                    obs, _, done, _, row = env.step(action)
                    if modal is not None and "ontario_applied_m3_s" in row:
                        np.testing.assert_allclose(row["ontario_applied_m3_s"], modal, rtol=0, atol=1e-5)
                executed = [row for row in env.records if "ontario_applied_m3_s" in row]
                completed = len(executed) == len(case["rows"]) and env.records[-1]["status"] == "executed"
                key = {"year": case["year"], "arm": arm, "seed": agent.seed if agent else -1}
                trace.extend({**key, **row} for row in env.records)
                runs.append({**key, "completed": completed, "executed_periods": len(executed),
                             "end_status": env.records[-1]["status"],
                             "cost": sum(row["cost"] for row in executed) if completed else None,
                             "upstream_exposure": sum(max(0., row["ontario_end_m"] - costs["ontario_flood_cap_m"])
                                                      for row in executed) if completed else None,
                             "downstream_exposure": sum(max(0., row["pointe_claire_m"] - row["downstream_target_m"])
                                                        for row in executed) if completed else None,
                             "max_storage_violation_m3": max((row["ottawa_storage_violation_m3"] for row in executed),
                                                             default=0.),
                             "max_downstream_violation_m3_s": max(
                                 (max(0., row["st_louis_flow_m3_s"] - row["capacity_m3_s"]) for row in executed),
                                 default=0.),
                             })
        print(f"PPO execution comparison complete for {case['year']}", flush=True)
    return pd.DataFrame(runs), pd.DataFrame(trace)


def summarize(runs):
    complete = runs[runs.completed]
    joint = complete.groupby("year").size()
    expected = runs.groupby("year").size()
    years = expected.index[joint.reindex(expected.index, fill_value=0).eq(expected)]
    metrics = complete[complete.year.isin(years)].groupby(["year", "arm"])[list(METRICS)].mean()
    means = metrics.groupby("arm").mean().reindex(ARMS)
    modal, base = means.loc["ppo_modal"], means.loc["ppo_deterministic"]
    violations = runs.max_storage_violation_m3.gt(.01) | runs.max_downstream_violation_m3_s.gt(1e-5)
    passed = bool(len(years) == len(expected) and not violations[runs.arm.eq("ppo_modal")].any()
                  and modal.cost < base.cost and modal.upstream_exposure <= base.upstream_exposure + 1e-9
                  and modal.downstream_exposure <= base.downstream_exposure + 1e-9)
    return means, len(years), passed


def table(runs, means):
    lines = ["| Controller | Completed | Mean annual cost | Ontario exposure | Lake St. Louis exposure |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows = runs[runs.arm.eq(arm)]
        values = means.loc[arm, list(METRICS)]
        lines.append(f"| {arm} | {rows.completed.sum()}/{len(rows)} | "
                     + " | ".join(f"{value:.5f}" for value in values) + " |")
    return lines


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    config, design = (json.loads(path.read_text()) for path in (CONFIG, DESIGN))
    costs = {**protocol, "incomplete_penalty": design["incomplete_penalty"]}
    first, last = protocol["development_years"]
    development = list(range(first, last + 1))
    reserved = protocol["reserved_test_years"]
    root = Path("results/loslr/final_coupled_training_v1")
    checkpoints = [root / f"seed_{seed}/agent/final.pt" for seed in protocol["seeds"]]
    hashes = {str(path): digest(path) for path in (CHRONOLOGY, CONFIG, DESIGN, *checkpoints)}
    torch.set_num_threads(1)
    agents = [PPOAgent.load(path) for path in checkpoints]
    assert all(agent.steps == design["training_decisions_per_seed"] for agent in agents)
    reference = NativeReference()
    frame = load_inputs(reference)
    model = OntarioModel(reference)
    states = start_states(model, frame, sorted(set(development + reserved)))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "development": development, "reserved": reserved, "source_sha256": hashes,
        "seeds": protocol["seeds"], "arms": ARMS, "recovery_mode": "peak_protected",
        "gate": "lower cost than deterministic PPO, no added exposure or violations, complete matched years",
    }, indent=2) + "\n")
    runs, trace = evaluate(model, annual_cases(frame, states, development, config, design), costs, agents)
    means, common, passed = summarize(runs)
    runs.to_csv(args.output / "development_runs.csv", index=False)
    trace.to_csv(args.output / "development_trajectories.csv", index=False)
    lines = [
        "# Frozen PPO executed-release mode", "",
        "The deterministic policy transforms the Gaussian mean to a release request. The modal policy "
        "instead picks the highest-probability executed release after integrating that same Gaussian "
        "over the feasibility filter's nearest-release intervals. Checkpoints, information, physical "
        "limits, cost and peak-protected recovery are matched. No policies are retrained.", "",
        f"Development scores use {common} common complete years. PPO seeds are averaged within each "
        "year. Exposure is in metre-periods.", "", *table(runs, means), "",
    ]
    confirmed = None
    if passed:
        runs, trace = evaluate(model, annual_cases(frame, states, reserved, config, design), costs, agents)
        means, common, confirmed = summarize(runs)
        runs.to_csv(args.output / "reserved_runs.csv", index=False)
        trace.to_csv(args.output / "reserved_trajectories.csv", index=False)
        lines += [f"The modal rule passed development. Reserved scores use {common} common complete years.", "",
                  *table(runs, means), "", f"Reserved replay passed the same gate: {confirmed}.", ""]
    else:
        lines += ["The modal rule did not pass the development gate. The reserved comparison did not trigger.", ""]
    assert hashes == {name: digest(Path(name)) for name in hashes}
    (args.output / "decision.json").write_text(json.dumps({
        "development_passed": passed, "reserved_passed": confirmed,
    }, indent=2) + "\n")
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
