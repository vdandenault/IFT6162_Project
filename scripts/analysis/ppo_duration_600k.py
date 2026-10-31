"""PPO duration 600k.

Run: uv run --group reproduction python scripts/analysis/ppo_duration_600k.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""

import json
import pandas as pd
from lake_como.paths import PROJECT_ROOT


def main():
    # PPO: does 600k improve on 400k?
    #
    # Five continuous 600k runs preserve the same capped scheduler, mixed precision,
    # zero entropy, and seeds. Both the 200k and 400k saved states must match previous
    # controls exactly. Evaluate only after training finishes. Training years:
    # 1946–1987; validation: 1988–1997. No chronological test evaluation.
    #
    # This follow-up uses the same validation set as earlier investigations. Compare
    # fixed 400k and 600k checkpoints; 200k appears for context. The rule requires
    # non-worse mean full return, seed SD, and weakest seed versus 400k. This does not
    # retroactively revise the earlier comparison's rule.

    root = PROJECT_ROOT / "results/stabilization/duration_600k_v1"
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
            and r["prefix_check"]["exact"]
            and r["prefix_400k_check"]["exact"]
            and r["validation_200k_matches_control"]
            and r["validation_400k_matches_control"]
            and (not r["test_evaluated"])
            for r in checks
        )
    )
    print(
        "All five 200k and 400k checkpoint states and validation results match existing controls."
    )
    print(decision["rule"])

    summary = means.groupby("budget").full_return.agg(["mean", "std", "min", "max"])
    print(
        summary.rename(
            columns={"std": "seed SD", "min": "weakest seed", "max": "strongest seed"}
        )
        .round(2)
        .to_string()
    )

    print("Plot:", str(root / "duration_comparison.png"))

    print(
        "Changes: 600k minus 400k. Higher reward is better; lower physical penalties are better."
    )
    print(pd.read_csv(root / "paired_changes.csv").round(2).to_string())

    # What changed physically?
    #
    # Flood and low-water days, and unmet demand volume, help explain the weighted
    # score. The three-term objective omits waste and clipping penalties included in
    # full return; it is a separate diagnostic. These are annual validation averages.

    print(
        means.groupby("budget")[
            ["objective_return", "flood_days", "low_water_days", "deficit_million_m3"]
        ]
        .mean()
        .round(2)
        .to_string()
    )

    print(
        "Preferred endpoint for this 400k-versus-600k comparison:",
        decision["selected_steps"],
    )
    print(pd.read_csv(root / "runtimes.csv").round(2).to_string())

    # The outcome applies to this fixed pair of budgets and these five seeds. The
    # original 200k-versus-400k outcome is preserved. A better weighted return can
    # coexist with worse demand deficits or flood exposure; inspect all metrics.

    # 600k uses 50% more training interactions than 400k. Wall-clock timing depends on
    # concurrency and machine load. This comparison cannot establish an optimal budget
    # or performance on unseen test years. See the study
    # (../../docs/PPO.md#training-duration).

    # Interpretation
    #
    # The gain is concentrated in seed 127: it improves by 5,680, accounting for about
    # 80% of the summed paired reward improvement. The other four improve by 36–829.
    # This is consistent with slower progress for the weak seed and diminishing gains
    # for the others, rather than evidence that all five have converged.
    #
    # Average low-water days fall from 11.96 to 8.16/year, flood days from 4.44 to 4.40,
    # and unmet demand from 520.46 to 512.89 million m³/year. Individual trade-offs
    # remain: seed 126 has more low-water days and seed 125 has more unmet demand. The
    # three-term objective worsens slightly for seeds 125 and 126 even though full
    # return improves. 600k meets the pairwise mean/SD/weakest-seed rule; this does not
    # justify indefinite extension or establish generalization on the held-out test
    # years.


if __name__ == "__main__":
    main()
