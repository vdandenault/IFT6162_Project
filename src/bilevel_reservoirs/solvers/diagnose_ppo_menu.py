"""Enumerate the feasible release menu on frozen-policy development trajectories."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

from bilevel_reservoirs.solvers.diagnose_ppo_filter import PROBE_PERIODS, RESIDUALS
from bilevel_reservoirs.solvers.final_coupled_env import HardCoupledEnv
from bilevel_reservoirs.solvers.final_coupled_study import (
    CHRONOLOGY, CONFIG, DESIGN, NativeReference, OntarioModel, annual_cases,
    load_inputs, load_protocol, start_states,
)
from bilevel_reservoirs.solvers.fixed_schedule_replay import digest
from reservoir_control.controllers.ppo.agent import PPOAgent


def first_release_indices(releases, requests):
    """The filter chooses the nearest first release, breaking ties downwards."""
    midpoints = (releases[:-1] + releases[1:]) / 2
    return np.searchsorted(midpoints, requests, side="left")


def release_probabilities(releases, neutral, mean, sigma, action_low, action_high):
    """Integrate the squashed Gaussian over the filter's nearest-release bins."""
    midpoints = (releases[:-1] + releases[1:]) / 2. - neutral
    scaled = 2. * (midpoints - action_low) / (action_high - action_low) - 1.
    with np.errstate(divide="ignore"):
        latent = np.arctanh(np.clip(scaled, -1., 1.))
    return np.diff(np.r_[0., ndtr((latent - mean) / sigma), 1.])


def enumerate_menu(env, request, neutral):
    candidates = []
    choice, _, stage, _ = env.plan("ppo", request, neutral, candidate_audit=candidates)
    schedules = [tuple(row["schedule_m3_s"]) for row in candidates]
    if not schedules:
        assert choice is None
        return (), stage
    expected = min(schedules, key=lambda schedule: (abs(schedule[0] - request), schedule))
    np.testing.assert_array_equal(choice.ontario_executable_release_m3_s, expected)
    return tuple(sorted(set(schedule[0] for schedule in schedules))), stage


def development_rollouts(model, cases, costs, agents):
    env = HardCoupledEnv(model, cases, costs, recovery_mode="peak_protected")
    states, runs = [], []
    for index, case in enumerate(cases):
        for agent in agents:
            obs, _ = env.reset(options={"case": index})
            done = False
            while not done:
                action, *_ = agent.act(obs, deterministic=True)
                residual = float(np.asarray(action).item())
                period = env.context[0] + 1
                if period in PROBE_PERIODS:
                    neutral = env.neutral_request()
                    request = max(0., neutral + residual)
                    releases, stage = enumerate_menu(env, request, neutral)
                    with torch.no_grad():
                        distribution = agent.model.distribution(torch.as_tensor(obs).reshape(1, -1))
                obs, _, done, _, row = env.step(action)
                if period not in PROBE_PERIODS:
                    continue
                entry = {"year": case["year"], "seed": agent.seed, "period": period,
                         "status": row["status"], "recovery_stage": stage, "neutral_m3_s": neutral,
                         "ppo_requested_m3_s": request, "executed_m3_s": row.get("ontario_applied_m3_s"),
                         "eligible_first_releases": len(releases), "reachable_first_releases": 0,
                         "probe_first_releases": 0, "release_menu_m3_s": json.dumps(releases)}
                if releases:
                    np.testing.assert_allclose(row["requested_release_m3_s"], request, rtol=0., atol=1e-6)
                    predicted = min(releases, key=lambda release: (abs(release - request), release))
                    if "ontario_applied_m3_s" in row:
                        np.testing.assert_allclose(row["ontario_applied_m3_s"], predicted,
                                                   rtol=0., atol=env.tolerance.flow_m3_s)
                        assert row["recovery_stage"] == stage
                    offsets = max(0., neutral - 2000.), max(0., neutral + 2000.)
                    lower, upper = first_release_indices(np.array(releases), offsets)
                    grid = np.maximum(0., neutral + np.array(RESIDUALS))
                    indices = first_release_indices(np.array(releases), grid)
                    probabilities = release_probabilities(
                        np.array(releases), neutral, distribution.mean.item(), distribution.stddev.item(),
                        env.action_space.low.item(), env.action_space.high.item(),
                    )
                    np.testing.assert_allclose(probabilities.sum(), 1., rtol=0, atol=1e-12)
                    modal = int(probabilities.argmax())
                    chosen = releases.index(predicted)
                    entry.update(reachable_first_releases=int(upper - lower + 1),
                                 probe_first_releases=len(np.unique(indices)),
                                 release_probabilities=json.dumps(probabilities.tolist()),
                                 modal_release_m3_s=releases[modal], modal_probability=probabilities[modal],
                                 deterministic_release_probability=probabilities[chosen],
                                 effective_releases_5pct=int((probabilities >= .05).sum()))
                states.append(entry)
            executed = sum("ontario_applied_m3_s" in row for row in env.records)
            runs.append({"year": case["year"], "seed": agent.seed, "executed": executed,
                         "completed": executed == len(case["rows"]) and env.records[-1]["status"] == "executed",
                         "end_status": env.records[-1]["status"]})
        print(f"Audit complete for {case['year']}", flush=True)
    return pd.DataFrame(states), pd.DataFrame(runs)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    first, last = protocol["development_years"]
    years = list(range(first, last + 1))
    config, design = (json.loads(path.read_text()) for path in (CONFIG, DESIGN))
    costs = {**protocol, "incomplete_penalty": design["incomplete_penalty"]}
    root = Path("results/loslr/final_coupled_training_v1")
    checkpoints = [root / f"seed_{seed}/agent/final.pt" for seed in protocol["seeds"]]
    hashes = {str(path): digest(path) for path in [CHRONOLOGY, CONFIG, DESIGN, *checkpoints]}
    torch.set_num_threads(1)
    agents = [PPOAgent.load(path) for path in checkpoints]
    assert all(agent.steps == design["training_decisions_per_seed"] for agent in agents)
    reference = NativeReference()
    model = OntarioModel(reference)
    frame = load_inputs(reference)
    cases = annual_cases(frame, start_states(model, frame, years), years, config, design)
    args.output.mkdir(parents=True, exist_ok=False)
    states, runs = development_rollouts(model, cases, costs, agents)
    assert hashes == {path: digest(Path(path)) for path in hashes}
    states.to_csv(args.output / "states.csv", index=False)
    runs.to_csv(args.output / "runs.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "years": years, "seeds": protocol["seeds"], "recovery": "peak_protected",
        "sampled_periods": PROBE_PERIODS, "residual_probe_m3_s": RESIDUALS, "source_sha256": hashes,
    }, indent=2) + "\n")
    viable = states[states.eligible_first_releases.gt(0)]
    lines = [
        "# PPO feasible release menu on development trajectories", "",
        f"We replayed {len(runs)} frozen-policy episodes from {first}–{last} with peak-protected recovery. "
        f"{runs.completed.sum()} completed. We enumerated all eligible schedules in the first successful "
        f"recovery stage at {len(states)} prespecified year/seed/period states reached by PPO. "
        f"{len(states) - len(viable)} sampled states had an empty menu. Later stages were not probed "
        "when the deployed filter would already have accepted a schedule.", "",
        "| Measure among nonempty states | One first release | Median distinct first releases |",
        "|---|---:|---:|",
    ]
    for name, column in (
        ("Full eligible menu", "eligible_first_releases"),
        ("Reachable anywhere in PPO request range", "reachable_first_releases"),
        ("Selected by nine evenly spaced requests", "probe_first_releases"),
    ):
        count = viable[column].eq(1).sum()
        lines.append(f"| {name} | {count}/{len(viable)} ({count / len(viable):.1%}) | {viable[column].median():g} |")
    missed = viable.eligible_first_releases.gt(viable.reachable_first_releases).sum()
    multi = viable[viable.reachable_first_releases.gt(1)]
    modal_diff = (multi.modal_release_m3_s - multi.executed_m3_s).abs().gt(1e-5).sum()
    lines += [
        "", f"Selecting directly from the same menu would add at least one first-release choice "
        f"at {missed}/{len(viable)} sampled nonempty states. A direct selector would still face "
        "menus with only one eligible release under the current search. This is an action-interface diagnostic with frozen "
        "policies and synthetic Ottawa conditions. Controller performance was not retested by training.", "",
        f"Among {len(multi)} states with multiple reachable releases, the median number receiving at least "
        f"5% of policy mass was {multi.effective_releases_5pct.median():g}. The median most-probable "
        f"release received {multi.modal_probability.median():.1%} of the mass. The highest-probability "
        f"release differed from the deterministic action at {modal_diff} of these states. "
        "Probabilities integrate the saved squashed-Gaussian actor over the filter's release intervals.", "",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
