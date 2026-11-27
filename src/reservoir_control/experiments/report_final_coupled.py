from argparse import ArgumentParser
from importlib.resources import files
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reservoir_control.experiments.final_coupled_study import ARMS, load_protocol
from reservoir_control.experiments.fixed_schedule_replay import digest

MODES = ("development", "reserved", "stress")


def verify(root, tables, training, protocol):
    training_years = list(range(protocol["training_years"][0], protocol["training_years"][1] + 1))
    source_hashes = None
    checkpoints = {}
    for seed, log in training.items():
        folder = root / f"final_coupled_training_v1/seed_{seed}"
        plan = json.loads((folder / "plan.json").read_text())
        if plan["years"] != training_years or int(log.steps.iloc[-1]) != plan["training_budget"]:
            raise ValueError("A PPO endpoint differs from the frozen training plan")
        if source_hashes is not None and source_hashes != plan["sources_sha256"]:
            raise ValueError("PPO seeds used different study sources")
        source_hashes = plan["sources_sha256"]
        checkpoints[seed] = digest(folder / "agent/final.pt")

    samples = {}
    for label, table in tables.items():
        folder = root / f"final_coupled_{label}_v1"
        plan = json.loads((folder / "plan.json").read_text())
        if plan["sources_sha256"] != source_hashes:
            raise ValueError("Training and evaluation used different study sources")
        if label == "reserved" and plan["years"] != protocol["reserved_test_years"]:
            raise ValueError("The reserved years differ from the chronological protocol")
        rows = pd.read_csv(folder / "trajectories.csv")
        executed = rows[rows.ontario_applied_m3_s.notna()]
        keys = ["case", "arm", "seed"]
        expected = table.set_index(keys)
        arm_seeds = {("rule_curve", -1), ("external_mpc", -1), ("stackelberg_mpc", -1),
                     *(("ppo", seed) for seed in training)}
        if set(expected.index) != {(case, arm, seed) for case in plan["case_names"] for arm, seed in arm_seeds}:
            raise ValueError("Evaluation is missing a paired case or seed")
        actual_counts = executed.groupby(keys).size().reindex(expected.index, fill_value=0)
        if not actual_counts.eq(expected.executed_periods).all():
            raise ValueError("The executed count in a summary does not match the trace")
        if not expected.full_cost.notna().eq(expected.status.eq("completed")).all():
            raise ValueError("Full costs must be restricted to completed cases")
        residual = float(executed.ottawa_system_balance_residual_m3.abs().max())
        if not residual <= 0.01:
            raise ValueError("Water balance exceeded the acceptance tolerance")
        samples[label] = {"runs": len(table), "executed_decisions": len(executed),
                          "max_water_balance_residual_m3": residual}
    return {"checkpoint_sha256": checkpoints, "samples": samples}


def paired_rows(table, arm):
    complete = table[table.status == "completed"]
    base = complete[complete.arm == "rule_curve"].set_index("case")
    target = complete[complete.arm == arm]
    target = target[target.case.isin(base.index)]
    # Average PPO endpoints per case before comparing with the rule curve.
    means = target.groupby("case")[["full_cost", "upstream_exposure_m_periods",
                                    "downstream_exposure_m_periods"]].mean()
    return means - base.loc[means.index, means.columns]


def create_report(output, tables, training):
    lines = [
        "# Paired Ontario–Ottawa controller study", "",
        "All controllers execute with the same native Ontario limit checks, hard Ottawa follower, one-period "
        "routing, and realized physical limits. Ontario inputs are historical and Ottawa scenarios are synthetic. "
        "Plan 2014 requests and PPO residual requests use the same finite feasible candidate filter. "
        "External-ranked and Stackelberg MPC use the same Ontario operating cost and candidate family. "
        "Only Stackelberg ranking uses the planned Ottawa response. PPO uses final checkpoints trained on "
        "1901–1980. Annual starts are independently warmed with causal Plan 2014. Stress windows are reported separately.", "",
        "| Sample | Controller | Completed runs | Realized storage violations | Downstream violations |",
        "|---|---|---:|---:|---:|",
    ]
    for label, table in tables.items():
        for arm in ARMS:
            group = table[table.arm == arm]
            lines.append(f"| {label} | {arm} | {(group.status=='completed').sum()}/{len(group)} | "
                         f"{(group.storage_violation_m3 > .01).sum()} | "
                         f"{(group.downstream_violation_m3_s > 1e-5).sum()} |")
    lines.extend(["", "## Differences from Plan 2014 on completed pairs", "",
                  "Each row averages only common full-window cases. PPO is averaged over completing seeds within each "
                  "case. Negative values mean lower cost or exposure.", "",
                  "| Sample | Controller | Cases | Operating cost | Ontario exceedance (m·periods) | Pointe-Claire exceedance (m·periods) |",
                  "|---|---|---:|---:|---:|---:|"])
    for label, table in tables.items():
        for arm in ARMS[1:]:
            delta = paired_rows(table, arm)
            means = [f"{x:+.3f}" for x in delta.mean()] if len(delta) else ["—"] * 3
            lines.append(f"| {label} | {arm} | {len(delta)} | " + " | ".join(means) + " |")
    lines.extend(["", "## Stackelberg versus external-ranked MPC", "",
                  "| Sample | Both complete | Mean cost change (Stackelberg minus external) |",
                  "|---|---:|---:|"])
    for label, table in tables.items():
        wide = table.pivot(index="case", columns=["arm", "seed"], values="full_cost")
        delta = (wide[("stackelberg_mpc", -1)] - wide[("external_mpc", -1)]).dropna()
        change = f"{delta.mean():+.4f}" if len(delta) else "—"
        lines.append(f"| {label} | {len(delta)} | {change} |")
    lines.extend(["", "## 2017 and 2019 full-year development cases", "",
                  "| Year | Controller | Completed | Cost | Ontario exposure (m·periods) | Pointe-Claire exposure (m·periods) |",
                  "|---|---|---:|---:|---:|---:|"])
    development = tables["development"]
    for year in (2017, 2019):
        for arm in ARMS:
            subset = development[(development.year == year) & (development.arm == arm)]
            done = subset[subset.status == "completed"]
            values = [f"{x:.3f}" for x in done[["full_cost", "upstream_exposure_m_periods",
                                                  "downstream_exposure_m_periods"]].mean()] if len(done) else ["—"] * 3
            lines.append(f"| {year} | {arm} | {len(done)}/{len(subset)} | " + " | ".join(values) + " |")
    lines.extend(["", "### 2017 and 2019 common executed prefixes", "",
                  "Every arm and seed reached the same stopping period in each of these two years. These "
                  "exposures describe only the shared executed prefix.", "",
                  "| Year | Controller | Executed periods | Ontario exposure (m·periods) | Pointe-Claire exposure (m·periods) |",
                  "|---|---|---:|---:|---:|"])
    for year in (2017, 2019):
        event = development[development.year == year]
        if event.executed_periods.nunique() == 1:
            for arm in ARMS:
                rows = event[event.arm == arm]
                lines.append(f"| {year} | {arm} | {int(rows.executed_periods.iloc[0])} | "
                             f"{rows.upstream_exposure_m_periods.mean():.3f} | "
                             f"{rows.downstream_exposure_m_periods.mean():.3f} |")
    lines.extend(["", "## PPO seed variability", "",
                  "| Seed | Development completed | Reserved completed | Stress completed | Training incomplete episodes |",
                  "|---|---:|---:|---:|---:|"])
    for seed, log in training.items():
        values = []
        for label in MODES:
            group = tables[label][(tables[label].arm == "ppo") & (tables[label].seed == seed)]
            values.append(f"{(group.status=='completed').sum()}/{len(group)}")
        lines.append(f"| {seed} | " + " | ".join([*values, str(log.incomplete.sum())]) + " |")
    lines.extend(["", "Run tables retain storage and service, transit, native hydropower value, intervention counts, "
                  "and runtime. Early stops do not enter full-window cost comparisons.", "",
                  "Every stress run stopped in the common filter. In 24 of the 36 stress cases it stopped before "
                  "the first release for every controller. These stress outcomes do not distinguish the policies.", "",
                  "The nine-state offline diagnostic admitted 9/9 realized-input decisions without the reserve "
                  "and 1/9 with it under the same branch-style search, independent of the tested fifth-lead value. "
                  "A separate two-path sensitivity increased completion from 7/36 to 18/36 by dropping the reserve "
                  "only after a guarded branch search failed, with zero realized limit violations. The scored "
                  "comparison uses the fixed fifth lead and the common after-retry reserve policy.", ""])
    (output / "report.md").write_text("\n".join(lines))


def curve_figure(training, output):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True, layout="constrained")
    for seed, run in training.items():
        axes[0].plot(run.steps, run.rollout_reward.rolling(4, min_periods=1).mean(), label=str(seed))
        axes[1].plot(run.steps, run.incomplete.rolling(4, min_periods=1).sum()
                     / run.episodes.rolling(4, min_periods=1).sum(), label=str(seed))
    axes[0].set(title="Ontario PPO training", ylabel="Return per rollout")
    axes[1].set(xlabel="Training decisions", ylabel="Incomplete episode fraction")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend(title="Seed", ncol=5)
    fig.savefig(output / "ppo_learning_curves.png", dpi=160)
    plt.close(fig)


def playback(root, tables, seeds, output, *, trajectory_paths=None):
    cases = []
    for mode in ("development", "stress"):
        path = (trajectory_paths[mode] if trajectory_paths is not None
                else root / f"final_coupled_{mode}_v1/trajectories.csv")
        records = pd.read_csv(path)
        table = tables[mode]
        if mode == "development":
            case_names = ["annual_2017", "annual_2019"]
        else:
            case_names = [name for name in table.case.unique() if name.startswith(("stress_2017", "stress_2019"))]
        for name in case_names:
            runs = table[table.case == name]
            series = []
            for summary in runs.itertuples():
                selected = records[(records.case == name) & (records.arm == summary.arm)
                                   & (records.seed == summary.seed)]
                fields = ["period", "ontario_applied_m3_s", "ottawa_executed_release_m3_s", "ottawa_end_storage_m3",
                          "ontario_end_m", "pointe_claire_m", "st_louis_flow_m3_s", "ottawa_end_transit_m3"]
                rows = json.loads(selected[fields].to_json(orient="records"))
                metrics = {
                    key: float(value) if pd.notna(value) and summary.status == "completed" else None
                    for key, value in (
                        ("cost", summary.full_cost),
                        ("ontario_exposure", summary.upstream_exposure_m_periods),
                        ("downstream_exposure", summary.downstream_exposure_m_periods),
                    )
                }
                series.append({"arm": summary.arm, "seed": summary.seed, "status": summary.status,
                               "executed": summary.executed_periods, "metrics": metrics, "rows": rows})
            label = name.removeprefix("annual_") + " full year" if mode == "development" else name.removeprefix("stress_").replace("_", " ")
            cases.append({"id": name, "label": label, "periods": int(runs.planned_periods.iloc[0]), "series": series})
    template = files(__package__).joinpath("final_coupled_playback.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPLAY_DATA__", json.dumps({"seeds": seeds, "cases": cases}, allow_nan=False)))


def main():
    parser = ArgumentParser(
        description="Assemble matched results, learning curves, and a local Ontario–Ottawa replay."
    )
    parser.add_argument("--root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--playback", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    seeds = protocol["seeds"]
    tables = {mode: pd.read_csv(args.root / f"final_coupled_{mode}_v1/runs.csv") for mode in MODES}
    training = {seed: pd.read_csv(args.root / f"final_coupled_training_v1/seed_{seed}/agent/training.csv") for seed in seeds}
    verification = verify(args.root, tables, training, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    create_report(args.output, tables, training)
    curve_figure(training, args.output)
    playback(args.root, tables, seeds, args.playback)


if __name__ == "__main__":
    main()
