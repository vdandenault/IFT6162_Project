"""PPO duration 1m.

Run: uv run --group reproduction python scripts/analysis/ppo_duration_1m.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""
import json
import pandas as pd
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT


def main():
    # PPO: 800k and 1M versus 600k
    #
    # Five continuous runs with the same capped scheduler, mixed precision, zero
    # entropy, and learner seeds 123–127. All earlier checkpoint states through 600k
    # must match exactly, including optimizer and normalizers. Earlier validation
    # metrics must also match. Evaluate after training ends; no evaluation interrupts
    # training.
    #
    # Training: 1946–1987. Validation: 1988–1997. No chronological test evaluation. Both
    # fixed endpoints qualify independently against 600k only if mean full return, seed
    # SD, and weakest seed are non-worse. Choose the highest mean among qualifying
    # endpoints (ties favor fewer steps), otherwise retain 600k. Stop at 1M. Repeated
    # investigations on the same validation years are exploratory selection, not
    # independent confirmation of generalization.

    root = PROJECT_ROOT / "results/stabilization/duration_1m_v1"
    means = pd.read_csv(root / "seed_means.csv")
    annual = pd.read_csv(root / "validation_annual.csv")
    decision = json.loads((root / "selection.json").read_text())
    checks = [
        json.loads((root / f"seed_{seed}/run.json").read_text())
        for seed in range(123, 128)
    ]
    assert all(
        (
            r["complete"]
            and (not r["test_evaluated"])
            and all(
                (
                    r[k]["exact"]
                    for k in ["prefix_check", "prefix_400k_check", "prefix_600k_check"]
                )
            )
            and all(
                (
                    r[k]
                    for k in [
                        "validation_200k_matches_control",
                        "validation_400k_matches_control",
                        "validation_600k_matches_control",
                    ]
                )
            )
            for r in checks
        )
    )
    print(
        "All five runs exactly match prior checkpoint states through 600k and their validation metrics."
    )
    print(decision["rule"])

    print(
        means.groupby("budget")
        .full_return.agg(["mean", "std", "min", "max"])
        .rename(
            columns={"std": "seed SD", "min": "weakest seed", "max": "strongest seed"}
        )
        .round(2)
        .to_string()
    )

    print("Plot:", str(root / "duration_comparison.png"))

    # Paired changes against 600k
    #
    # Higher reward is better. Lower flood/low-water days and deficit volume are better.
    # The full reward has five terms. The three-term objective excludes waste and
    # clipping penalties and may move differently.

    print("800k minus 600k")
    print(pd.read_csv(root / "paired_changes_800000.csv").round(2).to_string())
    print("1M minus 600k")
    print(pd.read_csv(root / "paired_changes_1000000.csv").round(2).to_string())

    print("Marginal gain: 1M minus 800k")
    indexed = means.set_index(["budget", "training_seed"])
    metrics = [
        "full_return",
        "objective_return",
        "flood_days",
        "low_water_days",
        "deficit_million_m3",
    ]
    print(
        (indexed.loc[1000000, metrics] - indexed.loc[800000, metrics])
        .round(2)
        .to_string()
    )

    print(
        means.groupby("budget")[
            ["objective_return", "flood_days", "low_water_days", "deficit_million_m3"]
        ]
        .mean()
        .round(2)
        .to_string()
    )

    print("Selected endpoint:", decision["selected_steps"])
    for (endpoint, comparison) in decision["comparisons"].items():
        print(endpoint, "qualifies:", comparison["longer_qualifies"])
    print(pd.read_csv(root / "runtimes.csv").round(2).to_string())

    # 800k and 1M use 33% and 67% more simulator interactions than 600k. Wall-clock
    # measurements depend on process concurrency and machine load. All endpoints and
    # seeds are preserved. This study establishes neither an optimal budget nor unseen
    # test performance. See the study notes (../../docs/PPO.md#training-duration).

    # Interpretation
    #
    # The extra gains diminish: mean full return improves by 544 from 600k to 800k, then
    # only 107 from 800k to 1M. Seed 127 accounts for about 85% of the total 600k-to-1M
    # gain and about 92% of the final 800k-to-1M gain. The other four show small, mixed
    # changes; seed 125 is worse than at 600k. This is evidence of the weaker run
    # catching up, rather than uniform improvement across controllers.
    #
    # At 1M versus 600k, average low-water days fall from 8.16 to 5.76/year and average
    # flood days stay at 4.40/year. Unmet demand rises from 512.89 to 514.91 million
    # m³/year; seed 127 alone increases its unmet demand by 19.90 million m³/year while
    # reducing low-water days by 10.4/year. Better weighted reward involves a trade-off.
    #
    # Both endpoints qualify against 600k, and 1M has the higher mean, so the recorded
    # rule selects 1M. The final 200k steps require 25% more interactions than 800k for
    # a small additional reward gain. Stop this duration investigation here and use the
    # selected setting consistently for the MPC/PPO comparison. This does not establish
    # convergence or performance on unseen test years.


if __name__ == "__main__":
    main()
