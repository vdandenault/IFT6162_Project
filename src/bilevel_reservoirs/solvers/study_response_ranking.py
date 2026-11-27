"""Compare Ontario's external and best-response rankings with shared safeguards."""

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
from bilevel_reservoirs.solvers.study_delay_headroom import YEARS, verify_prefix
from reservoir_control.systems.loslr.scoring import DEFAULT_CONFIG, ScoreSettings

POLICIES = ("external_screened", "anticipating")
STATE_METRICS = (
    "ontario_peak_m", "pointe_claire_peak_m", "ontario_excess_m_periods",
    "pointe_claire_excess_m_periods", "ontario_periods_above", "pointe_claire_periods_above",
    "ottawa_terminal_storage_m3", "ottawa_storage_tracking_rmse_m3",
    "ottawa_release_tracking_rmse_m3_s",
)
PAIRED_METRICS = (*STATE_METRICS, *ONTARIO_COST_COMPONENTS)


def summarize(trace, audit, config, score, expected_periods):
    executed = trace.loc[trace.status == "executed"]
    result = {
        "executed_periods": len(executed),
        "candidate_solves": sum("ottawa_candidate_status" in row for row in audit),
        "recovery_decisions": sum(row["selected"] and row["recovery_stage"] != "primary" for row in audit),
        "rank_changed_schedule": 0,
        "rank_changed_first_release": 0,
        "max_storage_violation_m3": 0.0,
        "max_downstream_violation_m3_s": 0.0,
        "status": trace.status.iloc[-1],
    }
    for selected in (entry for entry in audit if entry["selected"] and "best_response_score" in entry):
        eligible = (entry for entry in audit if entry["period"] == selected["period"]
                    and entry["recovery_stage"] == selected["recovery_stage"] and "best_response_score" in entry)
        external = min(eligible, key=lambda entry: entry["ottawa_screen_rank"])
        result["rank_changed_schedule"] += selected["schedule_m3_s"] != external["schedule_m3_s"]
        result["rank_changed_first_release"] += abs(
            selected["schedule_m3_s"][0] - external["schedule_m3_s"][0]
        ) > config["acceptance"]["flow_m3_s"]
    if executed.empty:
        return result
    ontario = (executed.ontario_end_m - score.ontario_cap_m).clip(lower=0)
    pointe_claire = (executed.pointe_claire_m - score.pointe_claire_cap_m).clip(lower=0)
    result.update(
        max_storage_violation_m3=float(executed.ottawa_storage_violation_m3.max()),
        max_downstream_violation_m3_s=max(float((executed.st_louis_flow_m3_s - executed.capacity_m3_s).max()), 0.0),
        ontario_peak_m=float(executed.ontario_end_m.max()),
        pointe_claire_peak_m=float(executed.pointe_claire_m.max()),
        ontario_excess_m_periods=float(ontario.sum()),
        pointe_claire_excess_m_periods=float(pointe_claire.sum()),
        ontario_periods_above=int((ontario > 1e-9).sum()),
        pointe_claire_periods_above=int((pointe_claire > 1e-9).sum()),
        ottawa_terminal_storage_m3=float(executed.ottawa_end_storage_m3.iloc[-1]),
        ottawa_terminal_transit_m3=float(executed.ottawa_end_transit_m3.iloc[-1]),
        ottawa_storage_tracking_rmse_m3=float(np.sqrt(np.mean(
            (executed.ottawa_end_storage_m3 - config["storage_target_m3"]) ** 2
        ))),
        ottawa_release_tracking_rmse_m3_s=float(np.sqrt(np.mean(
            (executed.ottawa_executed_release_m3_s - config["release_target_m3_s"]) ** 2
        ))),
        **{key: float(executed[key].sum()) for key in ONTARIO_COST_COMPONENTS},
    )
    if result["max_storage_violation_m3"] > config["acceptance"]["storage_m3"]:
        result["status"] = "realized_storage_violation"
    elif result["max_downstream_violation_m3_s"] > config["acceptance"]["flow_m3_s"]:
        result["status"] = "realized_downstream_violation"
    elif len(executed) == expected_periods:
        result["status"] = "completed"
    return result


def write_report(output, runs, pairs, score):
    lines = [
        "# Matched Ottawa response-ranking comparison", "",
        "Both arms screen the same candidate family against Ontario's external forecast and then require "
        "Ottawa QP feasibility and native Ontario admissibility with Ottawa's routed best response. "
        "The external arm ranks by the external forecast. The anticipating arm ranks the eligible "
        "schedules by the same objective evaluated with Ottawa's best response.", "",
        "| Controller | Completed | Archived external trajectories reproduced | Ottawa candidate solves | Total replay seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        subset = runs[runs.policy == policy]
        reproduced = str(int(subset.matches_archived_baseline.sum())) if policy == POLICIES[0] else "—"
        lines.append(f"| {policy} | {(subset.status == 'completed').sum()}/{len(subset)} | {reproduced} | "
                     f"{subset.candidate_solves.sum()} | {subset.wall_seconds.sum():.1f} |")
    nested = runs[runs.policy == "anticipating"]
    lines.extend([
        "", f"At its own decision states, anticipation changed the selected schedule {nested.rank_changed_schedule.sum()} times "
        f"and the immediate Ontario release {nested.rank_changed_first_release.sum()} times, relative to "
        "externally ranking the same eligible schedules.", "",
        f"The mean changes below use {len(pairs)} pairs completing the full window in both arms. "
        "Changes are anticipating minus external. Diagnostic level thresholds are "
        f"{score.ontario_cap_m:g} m for Ontario and {score.pointe_claire_cap_m:g} m for Pointe-Claire.", "",
        "| Outcome | Mean paired change |", "|---|---:|",
    ])
    for metric in PAIRED_METRICS:
        mean = f"{pairs[metric].mean():.6g}" if not pairs.empty else "—"
        lines.append(f"| {metric.replace('_', ' ')} | {mean} |")
    lines.extend([
        "", "The leader searches a finite set over four Ontario forecast leads, with the fifth "
        "Ontario commitment fixed for the five-period Ottawa solve. Only the first actions execute. "
        "The guards, candidate expansion and persistence retry rules are shared. The Ontario forcing "
        "is historical and the Ottawa aggregate, routing and capacities are synthetic. The matrix reuses "
        "previously exercised noise seeds and Ontario windows.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--baseline", type=Path, default=Path("results/loslr/ottawa_broader_headroom_validation_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    score_raw = json.loads(DEFAULT_CONFIG.read_text())
    score = ScoreSettings(**{key: score_raw[key] for key in ScoreSettings.__dataclass_fields__})
    provenance = {}

    def source(path):
        provenance[str(path)] = digest(path)
        return path

    source(CONFIG)
    source(DEFAULT_CONFIG)
    settings = json.loads(source(args.baseline / "settings.json").read_text())
    baseline_index = pd.read_csv(source(args.baseline / "summary.csv"))
    cases = baseline_index[baseline_index["sample"] == "fresh"]
    if set(cases.year) != set(YEARS) or len(cases) != 36 or cases.duplicated(["year", "scenario", "seed"]).any():
        raise ValueError("Expected the frozen 36-case baseline study")
    if (provenance[str(CONFIG)] not in settings["input_sha256"].values()
            or set(cases.seed) != set(settings["fresh_seeds"])):
        raise ValueError("Forecast or controller configuration differs from the baseline")

    args.output.mkdir(parents=True, exist_ok=False)
    rows, pairs = [], []
    for year in YEARS:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        archive_settings = json.loads(source(archive / "settings.json").read_text())
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        if len(trajectory) != 12 or not (trajectory.Year == year).all():
            raise ValueError(f"Unexpected Ontario window for {year}")
        config["known_ontario_executable_schedule_m3_s"] = (FLOW_SCALE * trajectory.ontFlow).tolist()
        history = archive_settings["ontario_window"]["initial_history"]
        window = args.output / str(year)
        window.mkdir()
        for case in cases.loc[cases.year == year].itertuples():
            scenario = next(item for item in config["scenarios"] if item["id"] == case.scenario)
            key = f"{case.scenario}_{case.seed}"
            bank = np.load(source(args.baseline / str(year) / f"{key}_forecasts.npy"))
            original = pd.read_csv(source(args.baseline / str(year) / f"{key}_broader_trace.csv"))
            matched = {}
            for policy in POLICIES:
                audit = []
                start = perf_counter()
                trace, plans = run_policy(
                    trajectory, config, scenario, policy, audit, "limits",
                    align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
                    recovery_guard_periods=1, replan_guard="after_retry", delay_headroom_search=True,
                    delay_headroom_on_exhaustion=True, matched_response_ranking=True, **history,
                )
                elapsed = perf_counter() - start
                frame = pd.DataFrame(trace)
                result = summarize(frame, audit, config, score, len(trajectory))
                if policy == POLICIES[0]:
                    same = result["status"] == case.broader_status
                    try:
                        verify_prefix(original.loc[original.status == "executed"],
                                      frame.loc[frame.status == "executed"], int(case.broader_executed_periods),
                                      config["acceptance"])
                    except AssertionError:
                        same = False
                    result["matches_archived_baseline"] = int(same)
                matched[policy] = result
                rows.append({"year": year, "scenario": case.scenario, "seed": case.seed, "policy": policy,
                             "wall_seconds": elapsed, **result})
                stem = window / f"{key}_{policy}"
                frame.to_csv(stem.with_name(f"{stem.name}_trace.csv"), index=False)
                for name, value in (("plans", plans), ("audit", audit)):
                    stem.with_name(f"{stem.name}_{name}.json").write_text(
                        json.dumps(value, default=json_value) + "\n"
                    )
            ext, anticipating = (matched[policy] for policy in POLICIES)
            if ext["status"] == anticipating["status"] == "completed":
                pairs.append({"year": year, "scenario": case.scenario, "seed": case.seed,
                              **{metric: anticipating[metric] - ext[metric] for metric in PAIRED_METRICS}})
            print(year, case.scenario, case.seed, ext["status"], anticipating["status"], flush=True)

    for path, checksum in provenance.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"A comparison input changed: {path}")
    runs = pd.DataFrame(rows)
    deltas = pd.DataFrame(pairs, columns=["year", "scenario", "seed", *PAIRED_METRICS])
    runs.to_csv(args.output / "runs.csv", index=False)
    deltas.to_csv(args.output / "paired_deltas.csv", index=False)
    write_report(args.output, runs, deltas, score)
    (args.output / "settings.json").write_text(json.dumps({"input_sha256": provenance}, indent=2) + "\n")


if __name__ == "__main__":
    main()
