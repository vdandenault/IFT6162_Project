"""Separate forecast persistence from storage reserve at three archived sharp-peak stops."""

from argparse import ArgumentParser
from ast import literal_eval
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.diagnose_replan_stop import evaluate_schedules
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, LOOKAHEAD, FollowerTolerances, TransitState, digest, json_value, make_problem, validate_config,
)
from bilevel_reservoirs.solvers.study_delay_headroom import verify_prefix
from bilevel_reservoirs.solvers.study_storage_guard import run

HYDROLOGY = "early_sharp_peak"
CASES = ((2017, 119), (2017, 121), (2019, 122))


def diagnose(config, scenario, trajectory, history, bank, archived, archived_stop):
    outcome, trace, _, audit = run(trajectory, config, scenario, bank, history, "always")
    stop = trace[-1]
    k = stop["period"] - 1
    if (outcome["status"] != "no_admissible_candidate" or outcome["last_decision"] != 4
            or not pd.DataFrame(trace)[["period", "status"]].equals(archived[["period", "status"]])):
        raise ArithmeticError("The archived sharp-peak stop did not reproduce")
    verify_prefix(archived, pd.DataFrame(trace), k, config["acceptance"])
    tolerance = FollowerTolerances(**config["acceptance"])
    recorded = archived_stop["stop"]
    for key, atol in {
        "ottawa_start_storage_m3": tolerance.storage_m3,
        "ottawa_start_queue_m3": tolerance.storage_m3,
        "ottawa_previous_release_m3_s": tolerance.flow_m3_s,
        "ontario_start_m": 0.500001e-6,
        "ontario_previous_release_m3_s": tolerance.flow_m3_s,
        "ontario_previous_ice": 0.0,
    }.items():
        np.testing.assert_allclose(stop[key], recorded[key], atol=atol, rtol=0)
    np.testing.assert_allclose(stop["ottawa_start_queue_m3"],
                               literal_eval(archived.iloc[-1].ottawa_start_queue_m3),
                               atol=tolerance.storage_m3, rtol=0)

    attempted = [row for row in audit if row["period"] == k + 1]
    schedules = sorted({row["schedule_m3_s"] for row in attempted if row["status"] == "admissible"})
    if not schedules or any(row["selected"] for row in attempted):
        raise ArithmeticError("The archived candidate rejection did not reproduce")
    horizon = min(LOOKAHEAD, len(trajectory) - k)
    forecast = bank[k, :horizon]
    queue = TransitState(tuple(stop["ottawa_start_queue_m3"]))
    problem = make_problem(config, scenario, k, stop["ottawa_start_storage_m3"],
                           stop["ottawa_previous_release_m3_s"], queue, forecast)
    state = (stop["ontario_start_m"], stop["ontario_previous_release_m3_s"] / FLOW_SCALE,
             stop["ontario_previous_ice"])
    rows = trajectory.iloc[k:k + horizon].to_dict("records")
    realized = np.asarray(scenario["realized_net_inflow_m3_s"][k:k + horizon])
    prior = scenario["realized_net_inflow_m3_s"][k - 1]
    inputs = {
        "saved": forecast,
        "persistence": np.full(horizon, prior),
        "realized_first_only": np.r_[realized[0], forecast[1:]],
        "realized_tail_only": np.r_[forecast[0], realized[1:]],
        "realized_all": realized,
    }
    reserve = config["forecast_noise"]["standard_deviation_m3_s"] * problem.duration_seconds[0]
    guarded_upper = np.asarray(problem.storage_maximum_m3).copy()
    guarded_upper[1:] -= reserve
    for entry in attempted:
        np.testing.assert_allclose(entry["recovery_storage_maximum_m3"], guarded_upper,
                                   rtol=0, atol=tolerance.storage_m3)
    problems = {
        f"{name}_{guard}": replace(problem, net_inflow_m3_s=values, storage_maximum_m3=upper)
        for name, values in inputs.items()
        for guard, upper in (("plain", problem.storage_maximum_m3), ("guarded", guarded_upper))
    }
    evaluated = evaluate_schedules(problems, rows, state, schedules, tolerance)
    outcomes = {name: Counter() for name in problems}
    witnesses = {}
    for entry in evaluated:
        cell = entry["case"]
        outcomes[cell][entry["status"]] += 1
        if entry["status"] == "admissible" and cell not in witnesses:
            witnesses[cell] = entry
    if any(outcomes[f"{name}_guarded"]["admissible"] for name in ("saved", "persistence")):
        raise ArithmeticError("A supposedly rejected guarded forecast admitted a schedule")
    if not outcomes["realized_all_plain"]["admissible"]:
        raise ArithmeticError("The offline inflow swap did not recover any tested schedule")
    return {
        "inherited_state": stop, "schedule_count": len(schedules), "inflow_m3_s": inputs,
        "storage_reserve_m3": reserve,
        "tail_excess_inflow_volume_m3": float(problem.duration_seconds[1:] @ (forecast[1:] - realized[1:])),
        "outcomes": outcomes, "example_witnesses": witnesses,
        "archived_full_remaining_finding": archived_stop["finding"],
    }, evaluated


def write_report(output, results):
    lines = [
        "# Sharp-peak stop diagnosis", "",
        "The three archived always-on stops reproduced at period 4. At each stop we held the "
        "inherited Ontario and Ottawa state, routing queue, native flow limits, and candidate set fixed. "
        "Each cell counts candidates accepted by both Ottawa's follower and the native Ontario checks.", "",
        "| Case | Schedules | Saved without / with reserve | Persistence without / with reserve | Realized first lead only without / with reserve | Realized tail only without / with reserve | All realized without / with reserve | Saved tail excess (billion m³) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in results:
        outcomes = case["outcomes"]
        counts = [f"{outcomes[f'{name}_plain'].get('admissible', 0)} / "
                  f"{outcomes[f'{name}_guarded'].get('admissible', 0)}" for name in case["inflow_m3_s"]]
        excess = f"{case['tail_excess_inflow_volume_m3'] / 1e9:.2f}"
        lines.append(f"| {case['year']} seed {case['seed']} | {case['schedule_count']} | "
                     + " | ".join([*counts, excess]) + " |")
    lines.extend([
        "", "Realized-input substitutions are offline checks on the five-lead Ottawa planning window. "
        "The first-lead swap keeps leads 2–5 from the saved forecast. The tail swap keeps the saved "
        "first lead. Ontario forecasts and the Ontario candidate commitments stay fixed in every cell.", "",
        "The +700 m³/s first-lead stress reserves 459.9 million m³ at subsequent storage boundaries. "
        "That reserve is separate from the physical limits, which all cells enforce. "
        "A positive cell supplies a witness within the archived candidate family. "
        "A zero cell establishes rejection within that family and forecast, rather than joint physical "
        "infeasibility. Archived offline witnesses reach the end of period 12.", "",
        "No controller settings were changed. Ottawa inflow, routing, and flow bounds come from the synthetic study.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def source(path):
        sources[str(path)] = digest(path)
        return path

    config = json.loads(source(CONFIG).read_text())
    validate_config(config)
    prior = args.archive_root / "ottawa_storage_guard_hydrology_v1"
    settings = json.loads(source(prior / "settings.json").read_text())
    design = settings["design"]
    if sources[str(CONFIG)] not in settings["input_sha256"].values():
        raise ValueError("Follower configuration differs from the archived sharp-peak study")
    base = next(item for item in config["scenarios"] if item["id"] == design["base_scenario"])
    scenario = {**base, "realized_net_inflow_m3_s": design["inflows_m3_s"][HYDROLOGY]}
    archive_table = pd.read_csv(source(prior / "runs.csv"))
    pairs = archive_table[archive_table.hydrology == HYDROLOGY].pivot(
        index=["year", "seed"], columns="arm", values="status",
    )
    selected = pairs[(pairs.after_retry == "completed") & (pairs.always == "no_admissible_candidate")]
    if tuple(selected.index) != CASES:
        raise ValueError("The archive does not contain the expected three always-on-only stops")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for year, seed in CASES:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        window = prior / str(year)
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        history = json.loads(source(archive / "settings.json").read_text())["ontario_window"]["initial_history"]
        stem = f"{HYDROLOGY}_{seed}"
        bank = np.load(source(window / f"{stem}_forecasts.npy"))
        trace = pd.read_csv(source(window / f"{stem}_always_trace.csv"))
        archived_stop = json.loads(source(window / f"{stem}_always_stop.json").read_text())
        if not archived_stop["witness"]:
            raise ValueError("The archived stop lacks a full continuation witness")
        result, evaluated = diagnose(local, scenario, trajectory, history, bank, trace, archived_stop)
        results.append({"year": year, "seed": seed, **result})
        (args.output / f"{year}_{seed}_candidate_audit.json").write_text(
            json.dumps(evaluated, default=json_value) + "\n"
        )
        print(year, seed, {key: value["admissible"] for key, value in result["outcomes"].items()}, flush=True)
    for path, sha256 in sources.items():
        if digest(Path(path)) != sha256:
            raise ArithmeticError(f"A diagnostic input changed: {path}")
    (args.output / "summary.json").write_text(
        json.dumps({"cases": results, "input_sha256": sources}, default=json_value, indent=2) + "\n"
    )
    write_report(args.output, results)


if __name__ == "__main__":
    main()
