"""Validate the frozen Ottawa storage guard on new flow shapes and noise seeds."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.fixed_schedule_replay import CONFIG, digest, json_value, validate_config
from bilevel_reservoirs.solvers.stress_ottawa_response import offline_recoverability
from bilevel_reservoirs.solvers.study_delay_headroom import verify_prefix
from bilevel_reservoirs.solvers.study_storage_guard import run
from reservoir_control.systems.ottawa.horizon_study import forecast_bank

DESIGN = CONFIG.with_name("storage_guard_hydrology_v1.json")
ARMS = ("after_retry", "always")


def stop_evidence(trajectory, config, scenario, trace, peer_trace, peer_outcome, history):
    """Find one feasible continuation from the actual stopped state, if possible."""
    k = len(trace) - 1
    alternatives = []
    if peer_outcome["status"] == "completed":
        try:
            if k:
                verify_prefix(pd.DataFrame(peer_trace), pd.DataFrame(trace), k, config["acceptance"])
        except AssertionError:
            alternatives.append([row["ontario_applied_m3_s"] for row in peer_trace[k:]])
        else:
            return {
                "finding": "completed_causal_peer_from_same_state",
                "witness": True,
                "continuation": peer_trace[k:],
            }

    anchor = config["known_ontario_executable_schedule_m3_s"][k:]
    if len(anchor) > 1:
        alternatives.extend(
            [anchor[0] - first, anchor[1] - second, *anchor[2:]]
            for first in (0, 100, 200) for second in (100, 200, 300, 400, 500, 600, 700)
            if first <= anchor[0] and second <= anchor[1]
        )
    proof = offline_recoverability(trajectory, config, scenario, trace[:-1],
                                  additional_schedules=alternatives, **history)
    proof["witness"] = proof["finding"] in ("joint_fixed_schedule_witness", "joint_alternative_schedule_witness")
    if proof["witness"]:
        path = (proof["alternative_continuation"]["ottawa_lp"]
                if proof["finding"] == "joint_alternative_schedule_witness" else proof["fixed_continuation_lp"])
        proof["terminal_transit_m3"] = path["release_m3_s"][-1] * config["clock"]["duration_seconds"]
    return proof


def assess(outcome, trace, config):
    storage = [row["ottawa_end_storage_m3"] for row in trace if row["status"] == "executed"]
    lower, upper = config["storage_minimum_m3"], config["storage_maximum_m3"]
    return {
        **outcome,
        "lowest_storage_m3": min(storage, default=None),
        "highest_storage_m3": max(storage, default=None),
        "below_storage_bound_m3": max(0., lower - min(storage, default=lower)),
        "above_storage_bound_m3": max(0., max(storage, default=upper) - upper),
        "terminal_transit_m3": trace[-1].get("ottawa_end_transit_m3") if outcome["status"] == "completed" else None,
    }


def write_report(output, table, design):
    lines = [
        "# Storage guard validation on new Ottawa inflow patterns", "",
        "Each pair uses the same historical Ontario window, synthetic Ottawa inflow, routing, "
        "downstream capacities and causal forecast bank. Only the trigger for the existing "
        f"+{design['guard_stress_m3_s']} m³/s guard changes. The flow shapes and fresh seeds "
        "were recorded before running either arm.", "",
        "| Inflow pattern | Controller | Completed | Upper storage violations | Lower storage violations | Downstream violations | No candidate | Stops with continuation witnesses |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (*design["inflows_m3_s"], "all"):
        group = table if label == "all" else table[table.hydrology == label]
        for arm in ARMS:
            rows = group[group.arm == arm]
            lines.append(f"| {label} | {arm} | {(rows.status == 'completed').sum()}/{len(rows)} | "
                         f"{rows.upper_violation.sum()} | {rows.lower_violation.sum()} | "
                         f"{rows.downstream_violation.sum()} | "
                         f"{(rows.status == 'no_admissible_candidate').sum()} | {rows.stop_witness.sum()} |")
    paired = table.pivot(index=["year", "hydrology", "seed"], columns="arm", values="status")
    wins = ((paired["always"] == "completed") & (paired["after_retry"] != "completed")).sum()
    losses = ((paired["after_retry"] == "completed") & (paired["always"] != "completed")).sum()
    dry = table[(table.hydrology == "dry_spring") & (table.status == "completed")]
    lines.extend([
        "", f"Always-on guarding completed {wins} cases the after-retry arm did not complete. "
        f"The reverse occurred in {losses} cases.", "",
        "| Controller | Completed dry cases | Lowest storage in those cases (million m³) |",
        "|---|---:|---:|",
    ])
    for arm in ARMS:
        rows = dry[dry.arm == arm]
        minimum = f"{rows.lowest_storage_m3.min() / 1e6:.2f}" if len(rows) else "—"
        lines.append(f"| {arm} | {len(rows)} | {minimum} |")
    lines.extend([
        "", "Storage violations describe realized endpoints outside the declared physical storage "
        "bounds. A continuation witness uses the completed causal peer from the same state or a "
        "separate offline LP and native-limit check with realized inflows. Failed searches for an "
        "offline witness leave recoverability unresolved unless the relaxation is itself infeasible.", "",
        "Only synthetic Ottawa flow shapes and forecast noise vary. The cases reuse three Ontario "
        "windows and one outlet and routing configuration. The study does not establish operational "
        "performance.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    design = json.loads(DESIGN.read_text())
    noise = config["forecast_noise"]
    if noise["standard_deviation_m3_s"] != design["guard_stress_m3_s"]:
        raise ValueError("The frozen guard stress must match the follower configuration")
    years, seeds = design["ontario_years"], design["forecast_seeds"]
    base = next(case for case in config["scenarios"] if case["id"] == design["base_scenario"])
    sources = {str(path): digest(path) for path in (CONFIG, DESIGN)}
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for year in years:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        for path in (archive / "settings.json", archive / "ontario_inputs.csv"):
            sources[str(path)] = digest(path)
        archived = json.loads((archive / "settings.json").read_text())
        if sources[str(CONFIG)] not in archived["input_sha256"].values():
            raise ValueError(f"Follower configuration differs from the archived {year} window")
        trajectory = pd.read_csv(archive / "ontario_inputs.csv")
        if not (trajectory.Year == year).all():
            raise ValueError(f"Unexpected Ontario years in the {year} window")
        history = archived["ontario_window"]["initial_history"]
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        window = args.output / str(year)
        window.mkdir()
        for hydrology, inflows in design["inflows_m3_s"].items():
            if len(inflows) != len(trajectory) or not np.isfinite(inflows).all() or min(inflows) < 0:
                raise ValueError(f"Invalid synthetic inflow hydrograph for {hydrology}")
            scenario = {**base, "realized_net_inflow_m3_s": inflows}
            for seed in seeds:
                stem = window / f"{hydrology}_{seed}"
                bank = forecast_bank(inflows, "noisy", seed, noise["initial_observed_inflow_estimate_m3_s"],
                                     noise["standard_deviation_m3_s"])
                np.save(stem.with_name(f"{stem.name}_forecasts.npy"), bank)
                results = {arm: run(trajectory, local, scenario, bank, history, arm) for arm in ARMS}
                for arm, peer in (ARMS, ARMS[::-1]):
                    outcome, trace, plans, _ = results[arm]
                    result = assess(outcome, trace, config)
                    name = f"{stem.name}_{arm}"
                    pd.DataFrame(trace).to_csv(window / f"{name}_trace.csv", index=False)
                    (window / f"{name}_plans.json").write_text(json.dumps(plans, default=json_value) + "\n")
                    proof = None
                    if outcome["status"] == "no_admissible_candidate":
                        peer_outcome, peer_trace, *_ = results[peer]
                        proof = stop_evidence(trajectory, local, scenario, trace, peer_trace, peer_outcome, history)
                        (window / f"{name}_stop.json").write_text(
                            json.dumps({"stop": trace[-1], **proof}, default=json_value, indent=2) + "\n"
                        )
                    rows.append({
                        "year": year, "hydrology": hydrology, "seed": seed, "arm": arm,
                        "upper_violation": result["above_storage_bound_m3"] > config["acceptance"]["storage_m3"],
                        "lower_violation": result["below_storage_bound_m3"] > config["acceptance"]["storage_m3"],
                        "downstream_violation": result["max_downstream_violation_m3_s"] > config["acceptance"]["flow_m3_s"],
                        "stop_witness": bool(proof and proof["witness"]),
                        "stop_finding": proof["finding"] if proof else None,
                        **result,
                    })
                print(year, hydrology, seed, *(results[arm][0]["status"] for arm in ARMS), flush=True)
    for path, checksum in sources.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"An experiment input changed: {path}")
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "runs.csv", index=False)
    write_report(args.output, table, design)
    (args.output / "settings.json").write_text(
        json.dumps({"design": design, "input_sha256": sources}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
