"""PPO precision.

Run: uv run --group reproduction python scripts/lake_como/ppo_precision.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""

import json
import pandas as pd
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT


def main():
    # PPO stabilization: training precision
    #
    # Five new FP32-training seeds are compared against the completed five capped
    # mixed-precision controls. We reuse the controls without retraining or modifying
    # them. The only training change is disabling CPU autocast; both arms use FP32
    # inference.
    #
    # Training: 1946–1987. Validation: 1988–1997. The chronological test years are not
    # evaluated. The scheduler remains capped at 1e-4, with the same 200,000-step
    # budget, zero entropy bonus, architecture, environment seed, and annual-horizon
    # settings.

    output = PROJECT_ROOT / "results/stabilization/precision_v1"
    assert (
        output / "selection.json"
    ).exists(), "Run: uv run --group reproduction lake-como-precision-study --workers 3"
    plan = json.loads((output / "plan.json").read_text())
    assert not plan["test_evaluated"]
    print("Control:", plan["control_precision"])
    print("Candidate:", plan["candidate_precision"])
    print("Both evaluated with:", plan["evaluation"])
    print("Seeds:", plan["seeds"])

    # Paired learner seeds
    #
    # Each row below averages the same ten validation years. Pair seed 123 with seed
    # 123, and so on. Do not pool all 50 seed/year rows as independent experiments.
    # Lower variance alone is insufficient if mean or worst-case seed performance
    # deteriorates.

    seeds = pd.read_csv(output / "seed_means.csv")
    annual = pd.read_csv(output / "validation_annual.csv")
    assert len(seeds) == 10
    assert annual.groupby(["precision", "training_seed"]).size().eq(10).all()
    assert set(annual.year) == set(plan["validation_years"])
    print(
        seeds.pivot(index="training_seed", columns="precision", values="full_return")
        .round(2)
        .to_string()
    )
    print(
        seeds.groupby("precision")[["full_return", "objective_return"]]
        .agg(["mean", "std", "min", "max"])
        .round(2)
        .to_string()
    )
    print("Plot:", str(output / "precision_comparison.png"))

    # Decision and physical outcomes
    #
    # The selection rule was recorded before training: FP32 must preserve/improve the
    # mean and weakest seed while preserving/reducing sample SD of full validation
    # return. Otherwise retain mixed precision. These five-seed validation results do
    # not establish test-set generalization or statistical significance.

    decision = json.loads((output / "selection.json").read_text())
    print("Selected:", decision["selected"])
    print("FP32 meets all criteria:", decision["fp32_qualifies"])
    print(
        seeds.groupby("precision")[
            ["flood_days", "low_water_days", "deficit_million_m3"]
        ]
        .mean()
        .round(2)
        .to_string()
    )
    print(pd.read_csv(output / "paired_changes.csv").round(2).to_string())

    # Runtime
    #
    # Timing is descriptive: the mixed-precision controls were recorded in an earlier
    # run, so concurrent workload and other system conditions are not perfectly matched.
    # Timing does not drive the performance/stability selection rule.

    runtime = pd.read_csv(output / "runtimes.csv")
    print(
        runtime.groupby("precision")
        .training_seconds.agg(["mean", "median", "min", "max"])
        .round(2)
        .to_string()
    )

    # The reproduction and learning-rate baseline artifacts are preserved. Training
    # precision is the only ablated variable; entropy and training duration have not
    # changed. Keep the chronological test set out of subsequent tuning.

    # Recorded conclusion
    #
    # FP32 training did not demonstrate additional stabilization: full validation return
    # −23,610 ± 5,054 versus −23,515 ± 5,075 for capped mixed precision (sample SD
    # across five seed means). Mean and weakest-seed return were slightly worse. The
    # predeclared rule retains mixed precision.
    #
    # Recorded FP32 training was faster locally, but runtime was not the selection
    # objective and the controls were reused from a different session. The result
    # concerns this CPU configuration, not every device or precision implementation. A
    # small entropy bonus remains a separate next experiment, not a change already
    # tested here.


if __name__ == "__main__":
    main()
