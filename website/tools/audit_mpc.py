"""Compare both ways of scoring each eligible schedule in the two website replays."""

import json
from argparse import ArgumentParser
from pathlib import Path

from reservoir_control.experiments.coupled_cases import annual_cases
from reservoir_control.experiments.final_coupled_env import HardCoupledEnv
from reservoir_control.study_config import COUPLED_CONFIG, load_follower_config, load_settings
from reservoir_control.systems.loslr.coupled import load_protocol, start_states
from reservoir_control.systems.loslr.data import load_inputs
from reservoir_control.systems.loslr.model import OntarioModel
from reservoir_control.systems.loslr.native import NativeReference


def audit_year(env, index, policies):
    """Compare candidate scores and match the executed first moves to both saved traces."""
    env.reset(options={"case": index})
    counts = dict(dates=0, schedules=0, changed_stage=0, positive_penalty=0, changed_score=0)
    for step, (external, stackelberg) in enumerate(
        zip(
            policies["ordinary"]["lake_ontario_release_m3_s"],
            policies["anticipating"]["lake_ontario_release_m3_s"],
            strict=True,
        )
    ):
        candidates = []
        reference = env.neutral_request()
        plan, *_ = env.plan("stackelberg_mpc", reference, reference, candidate_audit=candidates)
        if not candidates or plan is None:
            raise ValueError(f"Missing eligible schedules at step {step + 1}")
        for row in candidates:
            counts["schedules"] += 1
            counts["changed_stage"] += (
                row["nominal_downstream_stage_m"] != row["downstream_stage_m"]
            )
            counts["positive_penalty"] += (
                max(*row["nominal_downstream_excess_m"], *row["downstream_excess_m"]) > 0
            )
            counts["changed_score"] += row["external_score"] != row["response_score"]
        nominal_best = min(
            candidates, key=lambda row: (row["external_score"], row["schedule_m3_s"])
        )
        _, _, done, _, executed = env.apply("stackelberg_mpc")
        if (
            nominal_best["schedule_m3_s"][0] != external
            or plan.ontario_executable_release_m3_s[0] != stackelberg
            or executed["ontario_applied_m3_s"] != stackelberg
            or executed["status"] != "executed"
            or done != (step == 47)
        ):
            raise ValueError(f"Decision {step + 1} differs from the website replay")
        counts["dates"] += 1
    return counts


def main():
    """Load the same event cases, initial states and recovery mode as the site export."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the aggregate counts for Hugo")
    args = parser.parse_args()
    replay = json.loads((Path(__file__).parents[1] / "static/replay.json").read_text("utf-8"))
    years = [int(year) for year in replay["years"]]
    reference = NativeReference()
    model = OntarioModel(reference)
    frame = load_inputs(reference)
    states = start_states(model, frame, years)
    design = load_settings(COUPLED_CONFIG, "annual")
    cases = annual_cases(frame, states, years, load_follower_config(), design)
    costs = {**load_protocol(), "incomplete_penalty": design["incomplete_penalty"]}
    env = HardCoupledEnv(model, cases, costs, recovery_mode=replay["recovery"])
    result = {"revision": replay["revision"], "trace_sha256": replay["trace_sha256"], "years": {}}
    for index, year in enumerate(years):
        result["years"][str(year)] = audit_year(env, index, replay["years"][str(year)]["policies"])
    summary = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
