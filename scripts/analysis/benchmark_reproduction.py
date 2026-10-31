"""Benchmark reproduction.

Run: uv run --group reproduction python scripts/analysis/benchmark_reproduction.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""

import json
import pandas as pd
from lake_como.paths import PROJECT_ROOT


def main():
    # Gym4ReaL DamEnv: benchmark reproduction
    #
    # We evaluate the four simple rules and supplied PPO checkpoint on the authors'
    # original 13 test years. This is separate from our chronological MPC-versus-PPO
    # experiment. We also trained one PPO from scratch with the source's fixed
    # 200,000-step configuration.
    #
    # Generate results with uv run --group reproduction lake-como-reproduce.
    # Fresh-training results additionally require lake-como-train-reference and
    # evaluating its final checkpoint with a separate output directory. No training or
    # evaluation is run automatically by this script.

    output = PROJECT_ROOT / "results/reproduction"
    assert (
        output / "aggregate.csv"
    ).exists(), "Run: uv run --group reproduction lake-como-reproduce"
    metadata = json.loads((output / "run.json").read_text())
    print("Original test years:", metadata["years"])
    print("Upstream commit:", metadata["upstream_commit"])
    print("Checkpoint hash:", metadata["checkpoint_sha256"])
    summary = pd.read_csv(output / "aggregate.csv")
    print(summary.round(2).to_string())

    # Compare with Figure 1
    #
    # The plotted objective includes only weighted deficit, flood, and low-water
    # penalties. Full reward includes two additional penalties. The curve shows averages
    # across years; the band follows the upstream notebook's pointwise formula and does
    # not represent uncertainty across training seeds.
    #
    # The ordering and scale are visually consistent with Figure 1, but the paper's
    # figure does not supply exact numerical reference values for a tolerance-based
    # reproduction check.

    print("Plot:", str(output / "figure1_reproduction.png"))

    # What the physical outcomes tell us
    #
    # The strongest objective score does not mean the best value of every physical
    # metric. Max avoids floods while allowing low water. EAD leaves less demand unmet
    # than the supplied PPO. Examine those trade-offs alongside the aggregate reward.

    annual = pd.read_csv(output / "annual.csv")
    print(
        annual.pivot(index="year", columns="policy", values="objective_return")
        .round(1)
        .to_string()
    )
    assert len(annual) == 65
    assert annual.groupby("policy").size().eq(13).all()
    assert annual.max_balance_error_m3.max() < 0.0001

    # One fresh training run
    #
    # This uses the authors' original function and fixed configuration, seed 123, CPU,
    # and one thread. Its final 200,000-step checkpoint is evaluated without choosing a
    # checkpoint by test score. It is a training reproducibility check, not a new
    # algorithm or a multiple-seed result.

    fresh = output / "retrained/aggregate.csv"
    if fresh.exists():
        comparison = pd.concat(
            [
                summary.assign(run="Supplied checkpoint"),
                pd.read_csv(fresh).assign(run="Fresh CPU run"),
            ]
        )
        print(comparison[comparison.policy == "ppo"].round(2).to_string())
        training = json.loads((output / "training/training_run.json").read_text())
        print("Training elapsed seconds:", round(training["elapsed_seconds"], 2))
        print("Plot:", str(output / "retrained/figure1_reproduction.png"))
    else:
        print("Fresh training has not been run in this checkout.")

    # Limits of this reproduction
    #
    # Inference uses saved frozen normalization and deterministic mean actions on CPU in
    # FP32. The source agent's mixed-precision context and GPU arithmetic may differ.
    # The random baseline has an explicit seed while the upstream notebook does not
    # explicitly seed its action space. Independent tests check FP32 inference
    # equivalence and reward accounting.
    #
    # Both PPO models used original training years that overlap our chronological test
    # years. Retrain on the chronological training split before the main
    # engineering-versus-learning comparison. The corrected terminal observation used
    # elsewhere in our project is deliberately disabled for these source-reproduction
    # runs.

    # Five training seeds
    #
    # We retain the original seed 123 and add learner seeds 124–127 with the same fixed
    # configuration and final 200,000-step checkpoint rule. Hydrological training-year
    # sampling remains fixed at environment seed 123, so this isolates learner RNG
    # variation; it does not test every source of randomness.
    #
    # Each seed is evaluated on the same 13 original test years. Compute uncertainty
    # across the five seed means, not by treating the 65 seed/year combinations as
    # independent experiments. The supplied checkpoint is a separate reference, not a
    # sixth newly trained seed.

    study = output / "seeds"
    if (study / "aggregate_across_seeds.csv").exists():
        seed_means = pd.read_csv(study / "seed_means.csv")
        seed_variation = pd.read_csv(study / "aggregate_across_seeds.csv")
        paired = pd.read_csv(study / "paired_comparisons.csv")
        print(seed_means.round(2).to_string())
        print(seed_variation.round(2).to_string())
        print(paired.round(2).to_string())
        print("Plot:", str(study / "seed_study.png"))
    else:
        print("Run: uv run --group reproduction lake-como-seed-study --workers 2")

    # Interpretation of the seed check
    #
    # The five policy means average −18,654 ± 3,633 (sample SD). Seed 123 was favorable,
    # rather than representative of every run. Compared with Max, three seeds have a
    # higher mean objective, one is effectively tied, and one has a lower mean
    # objective. Five seeds and these fixed test years do not establish a universal
    # advantage. Keep the physical metrics and both sources of variation—training seed
    # and hydrological year—visible.


if __name__ == "__main__":
    main()
