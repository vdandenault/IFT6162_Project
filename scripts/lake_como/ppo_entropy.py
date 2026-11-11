"""PPO entropy.

Run: uv run --group reproduction python scripts/lake_como/ppo_entropy.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""
import json
import pandas as pd
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT


def main():
    # PPO stabilization: exploration through an entropy bonus
    #
    # Compare entropy coefficient 0.01 with the completed capped mixed-precision
    # controls (coefficient 0), using the same five learner seeds and 200,000-step
    # budget. The coefficient changes an auxiliary PPO loss, not reservoir reward. Both
    # arms are evaluated using deterministic mean actions, so any difference comes from
    # what training learned.
    #
    # Training: 1946–1987. Validation: 1988–1997. No chronological test evaluation or
    # best-seed selection. Scheduler, precision, architecture, reward weights,
    # environment year sampling, and annual-horizon semantics remain fixed.

    output = PROJECT_ROOT / "results/stabilization/entropy_v1"
    assert (
        output / "selection.json"
    ).exists(), "Run: uv run --group reproduction lake-como-entropy-study --workers 3"
    plan = json.loads((output / "plan.json").read_text())
    assert not plan["test_evaluated"]
    print(plan["entropy"])
    print("Scheduler:", plan["scheduler"])
    print("Learner seeds:", plan["seeds"])

    # Does the bonus preserve exploration?
    #
    # The policy has a learned Gaussian standard deviation in normalized action units. A
    # larger value means a wider training action distribution before action
    # rescaling/hydraulic clipping; it is not measured variability in actual delivered
    # water. The final standard deviation is not used when evaluating mean actions.
    #
    # We need both a successful exploration mechanism and useful validation outcomes.
    # Merely making the distribution wider is not sufficient.

    print("Plot:", str(output / "exploration_traces.png"))
    diagnostics = pd.read_csv(output / "policy_diagnostics.csv")
    print(
        diagnostics.pivot(
            index="training_seed",
            columns="exploration",
            values="final_action_std_normalized",
        )
        .round(4)
        .to_string()
    )

    # Paired validation outcomes
    #
    # Each seed is averaged over the same ten years. We compare mean, sample SD, and
    # weakest seed across the five policies, rather than pooling all seed/year
    # combinations as independent trials. Full return is the primary selection metric;
    # the three-term paper objective and physical outcomes remain visible.

    seeds = pd.read_csv(output / "seed_means.csv")
    annual = pd.read_csv(output / "validation_annual.csv")
    assert len(seeds) == 10
    assert annual.groupby(["exploration", "training_seed"]).size().eq(10).all()
    assert set(annual.year) == set(plan["validation_years"])
    print(
        seeds.pivot(index="training_seed", columns="exploration", values="full_return")
        .round(2)
        .to_string()
    )
    print(
        seeds.groupby("exploration")[["full_return", "objective_return"]]
        .agg(["mean", "std", "min", "max"])
        .round(2)
        .to_string()
    )
    print("Plot:", str(output / "entropy_comparison.png"))

    # Decision fixed before training
    #
    # The bonus qualifies only with non-worse mean and weakest-seed full return and
    # non-increased seed SD. If it fails, retain zero entropy. This tests one
    # coefficient, not all possible exploration strategies, and does not establish
    # statistical significance or generalization to later test years.

    decision = json.loads((output / "selection.json").read_text())
    print("Selected:", decision["selected"])
    print("Bonus qualifies:", decision["entropy_bonus_qualifies"])
    print(
        seeds.groupby("exploration")[
            ["flood_days", "low_water_days", "deficit_million_m3"]
        ]
        .mean()
        .round(2)
        .to_string()
    )
    print(pd.read_csv(output / "paired_changes.csv").round(2).to_string())

    # The original reproduction and previous ablations remain preserved. No precision,
    # duration, reward, or architecture change is combined with the entropy coefficient.
    # Longer training, if warranted by the curves, should be a separate declared
    # comparison.

    # Recorded outcome
    #
    # The bonus increased mean final Gaussian spread from 0.354 to 0.446, but full
    # validation return worsened from −23,515 ± 5,075 to −24,153 ± 4,638 (sample SD
    # across seed means). The worst seed improved modestly, but four of five matched
    # seed means worsened. The predeclared rule retains entropy coefficient 0.
    #
    # This is a useful negative result: more exploration was achieved, but did not
    # improve the learned policy under this coefficient and 200,000-step budget. It does
    # not establish that all entropy regularization is ineffective. The remaining next
    # item is inspecting learning curves before deciding whether longer training is
    # justified; no longer run or test evaluation was performed here.


if __name__ == "__main__":
    main()
