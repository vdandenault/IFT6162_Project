"""PPO stabilization.

Run: uv run --group reproduction python scripts/analysis/ppo_stabilization.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""

import json
import pandas as pd
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT


def main():
    # PPO stabilization: learning-rate ablation
    #
    # We compare three scheduler settings across five learner seeds, changing only the
    # learning-rate schedule. These are new chronological experiments: 1946–1987
    # training, 1988–1997 validation. We do not evaluate the 1998–2010 chronological
    # test years or reuse the reproduction test scores for tuning.
    #
    # The original reproduction snapshot remains unchanged. All new arms use the same
    # corrected final observation, fixed annual reset, and source time-limit
    # bootstrapping setting. These results therefore compare against the new
    # chronological control, not directly against the old mixed-year benchmark.

    output = PROJECT_ROOT / "results/stabilization/lr_v1"
    assert (
        output / "selection.json"
    ).exists(), "Run: uv run --group reproduction lake-como-stabilize --workers 3"
    plan = json.loads((output / "plan.json").read_text())
    assert not plan["test_evaluated"]
    print("Train:", min(plan["train_years"]), "–", max(plan["train_years"]))
    print(
        "Validation:", min(plan["validation_years"]), "–", max(plan["validation_years"])
    )
    print("Learner seeds:", plan["seeds"])
    print("Initial rate:", plan["common"]["learning_rate"])
    print("Variants:", plan["variants"])

    # What changes—and what stays fixed
    #
    # Original: KL-adaptive scheduler, minimum 1e-6, maximum 1e-2. Capped: same
    # scheduler, maximum 1e-4. Fixed: no scheduler, rate 8e-6 throughout.
    #
    # All use 200,000 steps, the same [16,16] actor/critic networks, 2,048-step
    # rollouts, 10 epochs, 32 mini-batches, mixed precision, and zero entropy bonus.
    # Each learner seed is paired across arms, and the environment's year-sampling seed
    # stays at 123. Final-step checkpoints are evaluated with frozen normalization and
    # deterministic mean actions in FP32.

    seeds = pd.read_csv(output / "seed_means.csv")
    annual = pd.read_csv(output / "validation_annual.csv")
    assert len(seeds) == 15
    assert annual.groupby(["variant", "training_seed"]).size().eq(10).all()
    assert set(annual.year) == set(plan["validation_years"])
    print(
        seeds.pivot(index="training_seed", columns="variant", values="full_return")
        .round(2)
        .to_string()
    )
    print(
        seeds.groupby("variant")[["full_return", "objective_return"]]
        .agg(["mean", "std", "min", "max"])
        .round(2)
        .to_string()
    )

    print("Plot:", str(output / "stabilization.png"))

    print("Plot:", str(output / "learning_rate_traces.png"))

    # Selection was specified before training
    #
    # Use full five-term validation return, averaged across ten years per seed. A
    # candidate must have non-worse mean, non-larger sample SD, and non-worse weakest
    # seed than the control. Among qualifying candidates, prefer the higher mean. If
    # none qualifies, retain the control and report no stabilization demonstrated.
    #
    # This is a practical validation rule, not a significance test. The sample SD is
    # across five seed means, not across 50 supposedly independent seed/year rows.

    decision = json.loads((output / "selection.json").read_text())
    print("Qualifying candidates:", decision["qualifying_candidates"])
    print("Validation selection:", decision["selected"])
    print(decision["claim"])
    print(
        seeds.groupby("variant")[["flood_days", "low_water_days", "deficit_million_m3"]]
        .mean()
        .round(2)
        .to_string()
    )
    rules = pd.read_csv(output / "validation_rules.csv")
    print(
        rules.groupby("policy")[
            [
                "full_return",
                "objective_return",
                "flood_days",
                "low_water_days",
                "deficit_million_m3",
            ]
        ]
        .mean()
        .round(2)
        .to_string()
    )

    # What remains untested
    #
    # We have not changed precision, entropy bonus, architecture, training duration,
    # reward weights, or annual-reset semantics in this ablation. Results do not
    # establish that a fixed learning rate in general is good or bad: only the specific
    # 8e-6 rate is tested. Similarly, the 1e-4 scheduler cap is one candidate, not a
    # proven optimum.
    #
    # Any selected setting is provisional until an appropriately frozen final
    # comparison. Keep chronological test years out of further tuning. Do not substitute
    # the strongest seed for the distribution of outcomes.

    # Recorded outcome
    #
    # The capped scheduler is the provisional validation choice. Full return improves
    # from −32,357 ± 6,465 to −23,515 ± 5,075 (sample SD across five seed means), and
    # the worst seed improves. Fixed 8e-6 has smaller variation but poorer average/worst
    # performance and is rejected. The cap improves mean full validation return in all
    # five matched seeds and improves the physical metrics on average.
    #
    # This reduces sensitivity without eliminating it. Freeze the result for subsequent
    # comparisons; do not treat validation selection as proof of performance on later
    # unseen test years. The original reproduction and its snapshot remain unchanged.
    #
    # The capped rate reaches 1e-4 early and stays there. A fixed 1e-4 rate was not
    # tested, so this does not establish that continued adaptation is necessary.


if __name__ == "__main__":
    main()
