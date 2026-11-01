"""Explain the frozen PPO observation ablation in wet/dry validation years.

Run with uv run --group reproduction python scripts/analysis/analyze_ppo_inflow.py
Year selection uses only total external inflow; no reward/violation selection.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lake_como.mpc_hindsight import digest
from lake_como.paths import PROJECT_ROOT
from lake_como.splits import split_years


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "results/ppo/past_inflow_v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/ppo/past_inflow_regimes_v1",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    run = json.loads((args.input / "run.json").read_text())
    if not run["complete"] or run["test_evaluated"]:
        raise ValueError("Complete validation-only study required")
    sources = [
        args.input / name
        for name in [
            "plan.json",
            "run.json",
            "trajectories.csv",
            "annual.csv",
            "seed_means.csv",
        ]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        selection="Lowest/highest sum of daily inflow on validation years; ties earliest year. No outcome-dependent case selection.",
        scope="Post-hoc explanation of frozen runs, not new training, causal identification, or test evaluation.",
        test_evaluated=False,
        source_sha256={str(Path(__file__)): digest(Path(__file__))},
        input_sha256={str(p): digest(p) for p in sources},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    f = pd.read_csv(args.input / "trajectories.csv", float_precision="round_trip")
    annual = pd.read_csv(args.input / "annual.csv", float_precision="round_trip")
    forcing = f[["year", "day", "inflow_m3_s"]].drop_duplicates()
    assert not forcing.duplicated(["year", "day"]).any()
    assert forcing.groupby("year").size().eq(365).all()
    assert sorted(forcing.year.unique()) == split_years("validation")
    ranking = (
        forcing.groupby("year")
        .inflow_m3_s.sum()
        .mul(86400 / 1e6)
        .rename("inflow_million_m3")
        .reset_index()
        .sort_values(["inflow_million_m3", "year"])
    )
    dry = int(ranking.iloc[0].year)
    wet = int(
        ranking.sort_values(["inflow_million_m3", "year"], ascending=[False, True])
        .iloc[0]
        .year
    )
    ranking.to_csv(args.output / "inflow_ranking.csv", index=False)
    selection = pd.DataFrame(
        [dict(regime="dry", year=dry), dict(regime="wet", year=wet)]
    )
    selection.to_csv(args.output / "selected_years.csv", index=False)
    metrics = [
        "full_return",
        "flood_days",
        "low_water_days",
        "deficit_million_m3",
        "flood_excess_m_days",
        "low_excess_m_days",
        "terminal_storage_m3",
    ]
    yearly = annual.groupby(["variant", "year"])[metrics].mean()
    delta = yearly.loc["past_inflow"] - yearly.loc["four_features"]
    changes = (
        f.groupby(["variant", "year", "day"])[
            ["request_m3_s", "release_m3_s", "level_m", "deficit_m3"]
        ]
        .mean()
        .unstack("variant")
    )
    action_delta = (
        changes["request_m3_s"]["past_inflow"]
        - changes["request_m3_s"]["four_features"]
    )
    decisions = action_delta.groupby("year").agg(
        mean_request_change_m3_s="mean",
        mean_absolute_request_change_m3_s=lambda x: abs(x).mean(),
        days_request_change_over_10_m3_s=lambda x: int((abs(x) > 10).sum()),
    )
    delta = delta.join(decisions)
    delta.to_csv(args.output / "paired_yearly.csv")
    selected = yearly.reset_index().merge(selection, on="year", validate="many_to_one")
    selected.to_csv(args.output / "selected_metrics.csv", index=False)
    components = [c for c in f if c.startswith("weighted_")]
    contributions = (
        f.groupby(["variant", "training_seed", "year"])[components]
        .sum()
        .groupby(["variant", "year"])
        .mean()
    )
    (contributions.loc["past_inflow"] - contributions.loc["four_features"]).to_csv(
        args.output / "reward_component_changes.csv"
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for row, (regime, year) in enumerate([("dry", dry), ("wet", wet)]):
        for variant, label, color in [
            ("four_features", "Four features", "#267ead"),
            ("past_inflow", "Past inflow", "#e78b27"),
        ]:
            g = f[(f.variant == variant) & (f.year == year)].copy()
            g["cumulative_deficit_million_m3"] = (
                g.groupby("training_seed").deficit_m3.cumsum() / 1e6
            )
            for ax, column, title in zip(
                axes[row],
                ["level_m", "release_m3_s", "cumulative_deficit_million_m3"],
                [
                    "Lake level (m)",
                    "Actual release (m³/s)",
                    "Cumulative unmet demand (million m³)",
                ],
            ):
                stats = g.groupby("day")[column].agg(["mean", "min", "max"])
                ax.plot(stats.index, stats["mean"], label=label, color=color)
                ax.fill_between(
                    stats.index, stats["min"], stats["max"], color=color, alpha=0.12
                )
                ax.set(
                    title=f"{regime.title()} year {year}: {title}",
                    xlabel="Simulation day",
                )
                ax.grid(alpha=0.15)
        axes[row, 0].axhline(1.24, color="#ba554c", ls="--", lw=1)
        axes[row, 0].axhline(-0.45, color="#b78219", ls="--", lw=1)
    axes[0, 0].legend()
    fig.suptitle(
        "Frozen PPO observation experiment · five-seed means and ranges · years selected by inflow only",
        fontsize=14,
    )
    fig.savefig(args.output / "wet_dry_decisions.png", dpi=170)
    plt.close(fig)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "input_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    result = dict(
        complete=True,
        test_evaluated=False,
        source_and_inputs_unchanged=True,
        dry_year=dry,
        wet_year=wet,
        scope="Descriptive post-hoc analysis; seed ranges are not confidence intervals",
    )
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    report = (
        "# PPO past-inflow: wet/dry decision analysis\n\n"
        + selected.round(5).to_string(index=False)
        + "\n\nPaired yearly changes (past inflow minus original):\n"
        + delta.round(5).to_string()
        + "\n\nYears selected only by external inflow total; descriptive, not mechanistic proof or independent replicates.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
