"""Controller comparison.

Run: uv run --group reproduction python scripts/lake_como/controller_comparison.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""
import json

import pandas as pd
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.splits import split_years


def main():
    # Chronological validation: rules versus selected PPO
    #
    # Random, Mean, Max, EAD, and all five 1M PPO seeds run on the same ten years,
    # 1988–1997. All use identical initial conditions and forcing. No training or test
    # split evaluation occurs here. Generate results with uv run --group reproduction
    # lake-como-compare; this script reads saved results.
    #
    # The PPO setting was selected using these validation years, so this is a
    # development comparison, not an unbiased estimate of test performance. The rules
    # are sanity baselines; MPC remains the required engineering comparison.

    root = PROJECT_ROOT / "results/comparison/validation_v1"
    metadata = json.loads((root / "run.json").read_text())
    assert (
        metadata["complete"]
        and metadata["split"] == "validation"
        and (not metadata["test_evaluated"])
    )
    assert metadata["years"] == split_years("validation")
    annual = pd.read_csv(root / "annual.csv")
    summary = pd.read_csv(root / "summary.csv")
    family_years = pd.read_csv(root / "family_years.csv")
    paired = pd.read_csv(root / "paired_family_annual.csv")
    assert (
        len(annual) == 90 and annual.groupby("controller").year.nunique().eq(10).all()
    )
    cols = [
        "year",
        "full_return",
        "objective_return",
        "flood_days",
        "low_water_days",
        "deficit_million_m3",
        "terminal_storage_m3",
    ]
    for seed in range(123, 128):
        saved = pd.read_csv(
            PROJECT_ROOT
            / f"results/stabilization/duration_1m_v1/seed_{seed}/validation_annual_1000000.csv"
        )
        current = annual[annual.controller == f"ppo_{seed}"]
        pd.testing.assert_frame_equal(
            saved[cols].reset_index(drop=True),
            current[cols].reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
            atol=1e-07,
        )
    print("90 annual episodes; all five PPO scores match the selected 1M study.")

    # Average annual performance
    #
    # Higher return is better. Lower flood/low-water days and unmet demand are better.
    # PPO is averaged equally over five learner seeds and ten years. Each rule has ten
    # years from one controller instance. The full return and three-term objective must
    # be kept separate. Terminal storage below is in million m³ relative to the model's
    # minimum-level datum.

    table = summary[
        [
            "family",
            "mean_full_return",
            "mean_objective_return",
            "mean_flood_days",
            "mean_low_water_days",
            "mean_deficit_million_m3",
            "mean_terminal_storage_m3",
        ]
    ].copy()
    table["mean_terminal_storage_m3"] /= 1000000.0
    table.columns = [
        "Controller",
        "Full return",
        "Three-term objective",
        "Flood days",
        "Low-water days",
        "Unmet demand (million m³)",
        "Terminal storage (million m³)",
    ]
    print(table.round(2).to_string())

    print("Plot:", str(root / "annual_comparison.png"))

    # Paired annual comparisons
    #
    # Subtract each rule's result from PPO on the same year. The line is the mean across
    # five PPO seeds and shading is the observed seed range, not a confidence interval.
    # Positive reward differences favor PPO; negative deficit differences favor PPO. The
    # same years are reused across seeds.

    print("Plot:", str(root / "paired_ppo_vs_rules.png"))

    p = paired[paired.candidate == "ppo"]
    print(
        p.groupby("reference")
        .delta_full_return.agg(
            mean_difference="mean",
            worst_year_difference="min",
            best_year_difference="max",
            years_positive=lambda s: int((s > 0).sum()),
        )
        .round(2)
        .to_string()
    )

    # Seed variation, reward components, and runtime
    #
    # Across-seed SD measures variation among the five annual-mean PPO scores. Year SD
    # measures variation across ten annual seed averages. Neither supplies 50
    # independent observations. Reward components are annual weighted penalties;
    # decision timing excludes simulation, model loading, and per-year PPO warmup.

    print(
        summary[
            [
                "family",
                "seed_sd_full_return",
                "year_sd_full_return",
                "weakest_seed_full_return",
            ]
        ]
        .round(2)
        .to_string()
    )
    print(pd.read_csv(root / "controller_means.csv").round(2).to_string())

    components = [c for c in annual if c.startswith("weighted_")]
    print(annual.groupby("family", sort=False)[components].mean().round(2).to_string())
    print(pd.read_csv(root / "timings.csv").round(3).to_string())

    # The RNG for Random is seeded once across the split. Additional random-rule seeds
    # would be a separate robustness study. Leap years retain the 365-step
    # approximation; years reset independently, so there is no cross-year storage
    # continuity.
    #
    # Next: specify and implement the literature-based MPC controller, then compare
    # under matched decision-time information. Keep the chronological test years
    # untouched while selecting that design. See the runner documentation
    # (../../docs/EVALUATION.md).

    # Interpretation and current findings: ../../docs/EVALUATION.md


if __name__ == "__main__":
    main()
