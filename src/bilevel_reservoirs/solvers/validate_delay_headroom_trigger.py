"""Compare the original and broader opt-in delay recovery triggers on fresh noise."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import run_policy
from bilevel_reservoirs.solvers.diagnose_remaining_stops import CASES, SCENARIO
from bilevel_reservoirs.solvers.fixed_schedule_replay import CONFIG, digest, json_value, validate_config
from bilevel_reservoirs.solvers.stress_ottawa_response import SCENARIOS
from bilevel_reservoirs.solvers.study_delay_headroom import YEARS, verify_prefix
from reservoir_control.systems.ottawa.horizon_study import forecast_bank

FRESH_SEEDS = (107, 108, 109, 110, 111, 112)
STAGES = ("delay_headroom", "delay_headroom_retry")


def test_case(window, trajectory, config, scenario, bank, history, seed):
    results, traces = {}, {}
    for name in ("after_retry", "broader"):
        audit = []
        trace, plans = run_policy(
            trajectory, config, scenario, "external_screened", audit, "limits",
            align_f=True, ottawa_forecast_bank=bank, recovery="persistence", recovery_guard_periods=1,
            replan_guard="after_retry", delay_headroom_search=True,
            delay_headroom_on_exhaustion=name == "broader", **history,
        )
        frame = pd.DataFrame(trace)
        executed = frame.loc[frame.status == "executed"]
        traces[name] = executed
        storage = float(executed["ottawa_storage_violation_m3"].max()) if len(executed) else 0.0
        downstream = max(float((executed.st_louis_flow_m3_s - executed.capacity_m3_s).max()), 0.0) if len(executed) else 0.0
        tolerance = config["acceptance"]
        status = trace[-1]["status"]
        if storage > tolerance["storage_m3"]:
            status = "realized_storage_violation"
        elif downstream > tolerance["flow_m3_s"]:
            status = "realized_downstream_violation"
        elif len(executed) == len(trajectory):
            status = "completed"
        selected = {entry["period"]: entry for entry in audit if entry["selected"]}
        results[name] = {
            "status": status, "executed_periods": len(executed), "last_decision": trace[-1]["period"],
            "max_storage_violation_m3": storage, "max_downstream_violation_m3_s": downstream,
            "headroom_selected": {p: row["recovery_stage"] for p, row in selected.items()
                                  if row["recovery_stage"] in STAGES},
            "candidate_solves": sum("ottawa_candidate_status" in row for row in audit),
        }
        stem = window / f"{scenario['id']}_{seed}_{name}"
        frame.to_csv(stem.with_name(f"{stem.name}_trace.csv"), index=False)
        stem.with_name(f"{stem.name}_plans.json").write_text(json.dumps(plans, default=json_value, indent=2) + "\n")
    first_difference = trace_length = min(map(len, traces.values()))
    for index in range(trace_length):
        a, b = (frame.iloc[index] for frame in traces.values())
        if (abs(a.ontario_applied_m3_s - b.ontario_applied_m3_s) > config["acceptance"]["flow_m3_s"]
                or abs(a.ottawa_executed_release_m3_s - b.ottawa_executed_release_m3_s) > config["acceptance"]["flow_m3_s"]):
            first_difference = index
            break
    verify_prefix(traces["after_retry"], traces["broader"], first_difference, config["acceptance"])
    if (first_difference < trace_length or results["after_retry"]["status"] != results["broader"]["status"]):
        if not any(period <= first_difference + 1 for period in results["broader"]["headroom_selected"]):
            raise ArithmeticError("The paired rollouts diverged before a broader headroom selection")
    return {f"{arm}_{key}": value for arm, outcome in results.items() for key, value in outcome.items()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    noise = config["forecast_noise"]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(CONFIG): digest(CONFIG)}
    summaries = []
    for year in YEARS:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        settings_path = archive / "settings.json"
        trajectory_path = archive / "ontario_inputs.csv"
        sources.update({str(path): digest(path) for path in (settings_path, trajectory_path)})
        settings = json.loads(settings_path.read_text())
        if sources[str(CONFIG)] not in settings["input_sha256"].values():
            raise ValueError(f"Follower configuration differs from the archived study for {year}")
        if set(FRESH_SEEDS) & {seed for _, seed in settings["forecast_cases"]}:
            raise ValueError("Fresh seeds overlap archived forecasts")
        history = settings["ontario_window"]["initial_history"]
        trajectory = pd.read_csv(trajectory_path)
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        window = args.output / str(year)
        window.mkdir()
        for scenario in config["scenarios"]:
            name = scenario["id"]
            if name not in SCENARIOS:
                continue
            known_seeds = [seed for case_year, seed in CASES if case_year == year and name == SCENARIO]
            for seed in (*known_seeds, *FRESH_SEEDS):
                if seed in known_seeds:
                    path = archive / f"{name}_{seed}_forecasts.npy"
                    sources[str(path)] = digest(path)
                    bank = np.load(path)
                else:
                    bank = forecast_bank(
                        scenario["realized_net_inflow_m3_s"], "noisy", seed,
                        noise["initial_observed_inflow_estimate_m3_s"], noise["standard_deviation_m3_s"],
                    )
                np.save(window / f"{name}_{seed}_forecasts.npy", bank)
                row = test_case(window, trajectory, local, scenario, bank, history, seed)
                summaries.append({"year": year, "scenario": name, "seed": seed,
                                  "sample": "known" if seed in known_seeds else "fresh", **row})
                print(year, name, seed, row["after_retry_status"], row["broader_status"], flush=True)
    for path, checksum in sources.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"An archived input changed during validation: {path}")
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    (args.output / "settings.json").write_text(
        json.dumps({"input_sha256": sources, "fresh_seeds": FRESH_SEEDS, "forecasts": noise}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
