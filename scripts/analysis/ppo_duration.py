"""PPO duration.

Run: uv run --group reproduction python scripts/analysis/ppo_duration.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""

import json
import pandas as pd
from lake_como.paths import PROJECT_ROOT


def main():
    # Does more PPO training help?
    #
    # We compare fixed 200k and 400k checkpoints from five continuous runs. The 200k
    # prefix must match the existing capped controls exactly, including optimizer and
    # normalizers. Only duration changes: mixed precision, zero entropy, and the capped
    # scheduler stay fixed. Training: 1946–1987. Validation: 1988–1997. No test tuning.
    #
    # Select 400k only if mean full validation return, seed SD, and weakest seed are all
    # non-worse. Higher return is better. SD is across five seed means, each evaluated
    # on the same ten years—not 50 independent samples.

    root = PROJECT_ROOT / "results/stabilization/duration_v1"
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
            and r["validation_200k_matches_control"]
            and (not r["test_evaluated"])
            for r in checks
        )
    )
    print(
        "All five 200k checkpoint states and validation results match existing controls."
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

    print("Selected training budget:", decision["selected_steps"])
    print(pd.read_csv(root / "runtimes.csv").round(2).to_string())

    # The 400k endpoint requires twice as many environment interactions. Wall-clock
    # times are descriptive because process concurrency and machine load differ. This
    # bounded comparison does not establish an optimal training duration or performance
    # on unseen test years. All seeds and both endpoints are retained. See the duration
    # study (../../docs/PPO.md#training-duration) for the recorded decision.

    # Interpretation
    #
    # Every seed improved its full validation return. Mean return improved by 4,214, and
    # the weakest seed improved by 2,997. However, seed SD rose from 5,075 to 5,289
    # (about 4.2%); four seeds clustered near -17k while seed 127 remained near -29k.
    # Longer training improved performance here but did not resolve seed sensitivity.
    # The conservative stability rule therefore retains 200k; this does not imply that
    # its average decisions were better.
    #
    # Average low-water days fell from 26.96 to 11.96 per year. Flood days rose slightly
    # from 4.32 to 4.44, and unmet demand rose from 510.03 to 520.46 million m³/year.
    # The reward improved through a trade-off, rather than improving every physical
    # metric. Keep both endpoints available for review.


if __name__ == "__main__":
    main()
