"""Compare generic and Ottawa-guided candidate proposals at equal budgets."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import ONTARIO_COST_COMPONENTS
from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import run_policy
from bilevel_reservoirs.solvers.fixed_schedule_replay import CONFIG, digest, json_value, validate_config
from bilevel_reservoirs.solvers.stress_ottawa_response import SCENARIOS
from bilevel_reservoirs.solvers.study_delay_headroom import YEARS, verify_prefix
from bilevel_reservoirs.solvers.study_response_ranking import PAIRED_METRICS, summarize
from reservoir_control.systems.loslr.scoring import DEFAULT_CONFIG, ScoreSettings
from reservoir_control.systems.ottawa.horizon_study import forecast_bank

ARMS = {"baseline": None, "generic": "generic", "guided": "guided"}
BUDGET = 12
SAVED = ((2017, 101), (2017, 112), (2019, 101), (2019, 103), (2019, 111), (2019, 112))
FRESH_SEEDS = (113, 114, 115, 116, 117, 118)


def run_case(trajectory, config, scenario, bank, history, score, directory, stem):
    summaries, frames, counts = {}, {}, {}
    for label, search in ARMS.items():
        audit = []
        start = perf_counter()
        trace, plans = run_policy(
            trajectory, config, scenario, "external_screened", audit, "limits",
            align_f=True, ottawa_forecast_bank=bank, recovery="persistence", recovery_guard_periods=1,
            replan_guard="after_retry", delay_headroom_search=True, delay_headroom_on_exhaustion=True,
            matched_response_ranking=True, proposal_search=search, proposal_budget=BUDGET, **history,
        )
        elapsed = perf_counter() - start
        frame = pd.DataFrame(trace)
        frames[label] = frame
        proposals = [row for row in audit if "extra_proposal" in row]
        counts[label] = np.bincount([row["period"] for row in proposals], minlength=len(trajectory) + 1)[1:]
        if np.any(counts[label] > BUDGET):
            raise ArithmeticError("Extra candidate budget exceeded")
        result = summarize(frame, audit, config, score, len(trajectory))
        result.update(
            extra_candidates=len(proposals),
            selected_extras=sum(row["selected"] for row in proposals),
            periods_with_extra_candidates=int(np.count_nonzero(counts[label])),
            wall_seconds=elapsed,
        )
        summaries[label] = result
        frame.to_csv(directory / f"{stem}_{label}_trace.csv", index=False)
        (directory / f"{stem}_{label}_audit.json").write_text(
            json.dumps({"candidate_audit": audit, "executed_plans": plans}, default=json_value) + "\n"
        )
    pairs = []
    for reference, comparison in (("baseline", "generic"), ("baseline", "guided"), ("generic", "guided")):
        a, b = summaries[reference], summaries[comparison]
        if a["status"] == b["status"] == "completed":
            pairs.append({
                "base": reference, "comparison": comparison,
                **{metric: b[metric] - a[metric] for metric in PAIRED_METRICS},
                "periods_with_different_ontario_release": int(np.count_nonzero(
                    np.abs(frames[reference].ontario_applied_m3_s - frames[comparison].ontario_applied_m3_s)
                    > config["acceptance"]["flow_m3_s"]
                )),
            })
    budget_match = int(np.count_nonzero(counts["generic"] != counts["guided"]))
    return summaries, pairs, budget_match, frames["baseline"]


def write_report(output, runs, pairs, windows, sample):
    lines = [
        "# Ontario candidate proposal comparison", "",
        "Both expansions draw from the same set of nearby changes to the first four Ontario commitments. "
        "The generic search tries small moves in balanced lead and sign order. The guided search gives "
        "priority to moves that free downstream capacity for Ottawa when its forecast storage is above "
        "target, accounting for the release limit and routing delay. Both use the same external ranking, "
        "native limit checks with the routed Ottawa best response, and recovery stages.", "",
        f"Each expansion may try {BUDGET} new schedules at the stage that first finds a feasible incumbent. "
        "This is a local refinement, so a decision with no feasible incumbent receives no extra proposals. "
        "The budget counts newly proposed schedules before screening. The actual Ottawa solve counts are reported separately.", "",
        f"This report covers the **{sample}** cases. Forecasts are shared within each case. Historical Ontario "
        "forcing is paired with synthetic Ottawa conditions. The fifth Ontario commitment stays fixed.", "",
        "| Arm | Completed | Extra proposals | Extras selected | Ottawa candidate solves | Max storage violation (million m³) | Replay seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        data = runs[runs.arm == arm]
        lines.append(f"| {arm} | {(data.status == 'completed').sum()}/{len(data)} | "
                     f"{data.extra_candidates.sum()} | {data.selected_extras.sum()} | "
                     f"{data.candidate_solves.sum()} | {data.max_storage_violation_m3.max() / 1e6:.2f} | "
                     f"{data.wall_seconds.sum():.1f} |")
    lines.extend([
        "", f"Generic and guided used different proposal counts at {windows.budget_mismatch_periods.sum()} decision times.", "",
        "## Ontario objective in complete pairs", "",
        "These changes use the implemented scalar objective, the sum of the four stage indicators "
        "in their native units. Lower is better on that objective.", "",
        "| Comparison | Complete pairs | Lower | Same | Higher | Mean change | Median change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for (base, comparison), data in pairs.groupby(["base", "comparison"], sort=False):
        change = data[list(ONTARIO_COST_COMPONENTS)].sum(axis=1)
        lower, higher = (change < -1e-8).sum(), (change > 1e-8).sum()
        lines.append(f"| {comparison} versus {base} | {len(change)} | {lower} | "
                     f"{len(change) - lower - higher} | {higher} | {change.mean():.6g} | {change.median():.6g} |")
    lines.extend([
        "",
        "## Guided against generic in complete pairs", "",
    ])
    guided = pairs[(pairs.base == "generic") & (pairs.comparison == "guided")]
    lines.extend([
        f"Paired sample size is {len(guided)}. Positive changes mean a higher value under guided search.", "",
        "| Metric | Mean change | Pairs with lower value | Pairs with higher value |", "|---|---:|---:|---:|",
    ])
    for metric in PAIRED_METRICS:
        values = guided[metric]
        lines.append(f"| {metric.replace('_', ' ')} | {values.mean():.6g} | "
                     f"{(values < -1e-8).sum()} | {(values > 1e-8).sum()} |")
    (output / "report.md").write_text("\n".join([*lines, ""]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--sample", choices=("saved", "fresh"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    score_raw = json.loads(DEFAULT_CONFIG.read_text())
    score = ScoreSettings(**{key: score_raw[key] for key in ScoreSettings.__dataclass_fields__})
    noise = config["forecast_noise"]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def source(path):
        sources[str(path)] = digest(path)
        return path

    source(CONFIG)
    source(DEFAULT_CONFIG)
    runs, deltas, windows = [], [], []
    previous_study = args.archive_root / "ottawa_broader_headroom_validation_v1"
    archived_seeds = set(json.loads(source(previous_study / "settings.json").read_text())["fresh_seeds"])
    years = YEARS if args.sample == "fresh" else tuple(sorted({year for year, _ in SAVED}))
    for year in years:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        settings = json.loads(source(archive / "settings.json").read_text())
        if sources[str(CONFIG)] not in settings["input_sha256"].values():
            raise ValueError(f"Follower configuration changed since the archived experiment for {year}")
        if set(FRESH_SEEDS) & (archived_seeds | {seed for _, seed in settings["forecast_cases"]}):
            raise ValueError("Fresh seeds overlap the reference experiments")
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        if len(trajectory) != 12 or not (trajectory.Year == year).all():
            raise ValueError(f"Unexpected Ontario window for {year}")
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        history = settings["ontario_window"]["initial_history"]
        window = args.output / str(year)
        window.mkdir()
        for scenario in local["scenarios"]:
            name = scenario["id"]
            if name not in SCENARIOS or (args.sample == "saved" and name != "one_period_delay"):
                continue
            seeds = FRESH_SEEDS if args.sample == "fresh" else [seed for case_year, seed in SAVED if case_year == year]
            for seed in seeds:
                stem = f"{name}_{seed}"
                if args.sample == "fresh":
                    bank = forecast_bank(
                        scenario["realized_net_inflow_m3_s"], "noisy", seed,
                        noise["initial_observed_inflow_estimate_m3_s"], noise["standard_deviation_m3_s"],
                    )
                else:
                    bank_dir = (archive if seed in (101, 103) else previous_study / str(year))
                    bank = np.load(source(bank_dir / f"{stem}_forecasts.npy"))
                np.save(window / f"{stem}_forecasts.npy", bank)
                summaries, pairs, mismatches, baseline = run_case(
                    trajectory, local, scenario, bank, history, score, window, stem,
                )
                if args.sample == "saved" and seed in archived_seeds:
                    old = args.archive_root / "ottawa_response_ranking_v1" / str(year) / f"{stem}_external_screened_trace.csv"
                    verify_prefix(pd.read_csv(source(old)), baseline, len(trajectory), config["acceptance"])
                case = {"year": year, "scenario": name, "seed": seed}
                runs.extend({**case, "arm": arm, **result} for arm, result in summaries.items())
                deltas.extend({**case, **pair} for pair in pairs)
                windows.append({**case, "budget_mismatch_periods": mismatches})
                print(year, name, seed, *(result["status"] for result in summaries.values()), flush=True)

    for path, checksum in sources.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"A study input changed: {path}")
    run_table, window_table = pd.DataFrame(runs), pd.DataFrame(windows)
    pair_table = pd.DataFrame(deltas, columns=[
        "year", "scenario", "seed", "base", "comparison", *PAIRED_METRICS, "periods_with_different_ontario_release",
    ])
    run_table.to_csv(args.output / "runs.csv", index=False)
    pair_table.to_csv(args.output / "paired_deltas.csv", index=False)
    window_table.to_csv(args.output / "proposal_budgets.csv", index=False)
    write_report(args.output, run_table, pair_table, window_table, args.sample)
    settings = {"sample": args.sample, "budget": BUDGET, "inputs_sha256": sources,
                "fresh_seeds": FRESH_SEEDS, "saved_cases": SAVED, "score": score_raw}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
