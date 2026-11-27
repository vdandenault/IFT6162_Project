"""Report matched recovery comparisons and the PPO execution-contract ablation."""
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path

import pandas as pd

from bilevel_reservoirs.solvers.final_coupled_study import ARMS

CONFIGURATIONS = {
    "Flood years": ("original", "audit", "reserve", "profiles", "proactive_profiles", "peak_protected", "retrained"),
    "Archived stress": ("audit", "reserve", "profiles"),
    "Reserved annual replays": ("original", "profiles", "peak_protected", "retrained"),
    "First stress holdout": ("original", "profiles", "proactive_profiles", "peak_protected", "retrained"),
    "Confirmation stress": ("original", "profiles", "peak_protected"),
}
SAMPLES = {"Flood years": "events", "Archived stress": "stress", "Reserved annual replays": "reserved",
           "First stress holdout": "holdout", "Confirmation stress": "confirmation"}
METRICS = ["full_cost", "upstream_exposure_m_periods", "downstream_exposure_m_periods"]
ORDER = ["case", "arm", "seed"]


def run_table(root, mode, sample):
    if mode == "original" and sample in ("events", "reserved"):
        source = "development" if sample == "events" else sample
        frame = pd.read_csv(root / f"final_coupled_{source}_v1/runs.csv")
        return frame[frame.year.isin((2017, 2019))] if sample == "events" else frame
    return pd.read_csv(root / f"final_recovery_{mode}_{sample}_v1/runs.csv")


def paired_delta(table, arm, comparator):
    complete = table[table.status == "completed"]
    base = complete[complete.arm == comparator].set_index("case")[METRICS]
    challenger = complete[complete.arm == arm].groupby("case")[METRICS].mean()
    index = base.index.intersection(challenger.index)
    return challenger.loc[index] - base.loc[index]


def completion(cases):
    return f"{cases.status.eq('completed').sum()}/{len(cases)}"


def numeric(values):
    return " | ".join(f"{v:+.4f}" for v in values)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = {(label, mode): run_table(args.root, mode, SAMPLES[label])
            for label, modes in CONFIGURATIONS.items() for mode in modes}
    for label, group in CONFIGURATIONS.items():
        expected = set(data[label, group[0]].set_index(ORDER).index)
        for mode in group:
            frame = data[label, mode]
            if len(frame) != len(expected) or set(frame.set_index(ORDER).index) != expected:
                raise ValueError(f"{label} {mode} does not cover the same cases")
            if not frame.full_cost.notna().eq(frame.status.eq("completed")).all():
                raise ValueError("Only completed episodes may have a full-window cost")
    for sample, mode in (("development", "events"), ("stress", "stress")):
        original = pd.read_csv(args.root / f"final_coupled_{sample}_v1/runs.csv")
        if mode == "events":
            original = original[original.year.isin((2017, 2019))]
        audit = pd.read_csv(args.root / f"final_recovery_audit_{mode}_v1/runs.csv")
        columns = ["status", "executed_periods", *METRICS]
        pd.testing.assert_frame_equal(original.set_index(ORDER)[columns].sort_index(),
                                      audit.set_index(ORDER)[columns].sort_index())
    lines = [
        "# Shared hard-follower recovery and controller comparison", "",
        "All variants retain the native Ontario checks, hard Ottawa QP bounds, and realized limits. "
        "Reserve recovery tries unguarded persistence only after the original search fails. "
        "Profile recovery also tries gradual Ontario release schedules in that situation. "
        "Proactive profiles offer the wider schedules at every decision. Retrained PPO uses the "
        "profile recovery contract in training, with the original five seeds, 1901–1980 split, and "
        "16,384 decisions per seed. The peak-protected variant adds a first-action storage check "
        "against the last observed inflow when it rose by at least 700 m³/s above the recent observations. "
        "That check is applied after learning about the first holdout overflow. The confirmation sample "
        "changes the synthetic inputs and noise seeds and reuses the historical Ontario windows.", "",
        "| Frozen stop sample | Reopened by reserve only | Reopened by wider ordinary search | Reopened by wider guarded persistence | Probed controller runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in ("events", "stress"):
        frame = pd.read_csv(args.root / f"final_recovery_audit_{sample}_v1/trajectories.csv")
        probes = pd.DataFrame([literal_eval(x) for x in frame.loc[frame.recovery_stage == "stopped", "recovery_probes"]])
        lines.append(f"| {sample} | {probes.reserve_released.sum()} | {probes.wider_ordinary.sum()} | "
                     f"{probes.wider_guarded_persistence.sum()} | {len(probes)} |")
    lines.extend(["", "| Sample | Shared recovery | Filtered Plan 2014 | External MPC | Stackelberg MPC | PPO | Runs with physical violations |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for (label, mode), frame in data.items():
        counts = " | ".join(completion(frame[frame.arm == arm]) for arm in ARMS)
        violations = (frame.storage_violation_m3.gt(.01) | frame.downstream_violation_m3_s.gt(1e-5)).sum()
        lines.append(f"| {label} | {mode} | {counts} | {violations} |")
    lines.extend([
        "", "## Costs and flood exposure on completed pairs", "",
        "Each PPO case averages its completing seeds before comparison. Lower changes favor the challenger. "
        "Annual replays use already observed evaluation years. Fresh stress varies synthetic Ottawa flows "
        "and noise while retaining the same historical Ontario windows.", "",
        "| Sample | Recovery | Challenger vs reference | Cases | Cost change | Ontario exposure change (m·periods) | Pointe-Claire exposure change (m·periods) | Cost wins / losses |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for (label, mode), frame in data.items():
        if mode not in ("profiles", "proactive_profiles", "peak_protected", "retrained"):
            continue
        comparisons = (("ppo", "rule_curve"), ("ppo", "external_mpc")) if mode == "retrained" else (
            ("external_mpc", "rule_curve"), ("stackelberg_mpc", "external_mpc"),
            ("ppo", "rule_curve"), ("ppo", "external_mpc"),
        )
        for arm, base in comparisons:
            paired = paired_delta(frame, arm, base)
            if paired.empty:
                continue
            wins = paired.full_cost.lt(-1e-8).sum()
            losses = paired.full_cost.gt(1e-8).sum()
            lines.append(f"| {label} | {mode} | {arm} vs {base} | {len(paired)} | "
                         f"{numeric(paired.mean())} | {wins} / {losses} |")
    lines.extend([
        "", "## Flood-year costs by controller", "",
        "| Year | Recovery | Plan 2014 | External MPC | Stackelberg MPC | PPO mean |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for mode in ("profiles", "proactive_profiles", "peak_protected", "retrained"):
        subset = data["Flood years", mode]
        for year, frame in subset[subset.status == "completed"].groupby("year"):
            values = frame.groupby("arm").full_cost.mean()
            lines.append(f"| {year} | {mode} | " + " | ".join(f"{values.get(arm, float('nan')):.4f}" for arm in ARMS) + " |")
    (args.output / "report.md").write_text("\n".join([*lines, ""]))
    pd.concat([frame.assign(sample=label, recovery=mode) for (label, mode), frame in data.items()]).to_csv(
        args.output / "runs.csv", index=False,
    )


if __name__ == "__main__":
    main()
