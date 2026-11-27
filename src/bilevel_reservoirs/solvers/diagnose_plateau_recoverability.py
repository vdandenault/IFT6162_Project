"""Trace full-window physical feasibility through archived sustained-rise runs."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, LOOKAHEAD, FollowerTolerances, TransitState, digest, json_value, make_problem, validate_config,
)
from bilevel_reservoirs.solvers.study_scenario_recourse import WIDER_ARM, common_offline, online_case
from reservoir_control.systems.ottawa.delay_diagnostic import explicit_storage_feasibility


def inherited_state(trajectory, config, scenario, history, executed):
    if executed.empty:
        return (float(trajectory.ontLevelBOQ.iloc[0]), history["previous_flow"] * FLOW_SCALE,
                history["previous_ice"], config["initial_storage_m3"], config["previous_executed_release_m3_s"],
                TransitState(tuple(scenario["initial_transit_volumes_m3"])))
    row = executed.iloc[-1]
    return (row.ontario_end_m, row.ontario_applied_m3_s, trajectory.iceInd.iloc[len(executed) - 1],
            row.ottawa_end_storage_m3, row.ottawa_executed_release_m3_s,
            TransitState(tuple(json.loads(row.ottawa_end_queue_m3))))


def check_prefix(trajectory, config, scenario, history, executed, tolerance):
    k = len(executed)
    state = inherited_state(trajectory, config, scenario, history, executed)
    level, flow, ice, storage, release, queue = state
    record = {
        "executed_periods": k, "ontario_level_m": level, "previous_ontario_release_m3_s": flow,
        "previous_ice": ice, "ottawa_storage_m3": storage, "previous_ottawa_release_m3_s": release,
        "transit_queue_m3": queue.volumes_m3,
    }
    if k == len(trajectory):
        return {**record, "classification": "completed"}
    realized = scenario["realized_net_inflow_m3_s"]
    problem = make_problem(config, scenario, k, storage, release, queue, realized[k:])
    # A zero Ontario release removes a nonnegative downstream contribution
    # and drops all native Ontario limits. Failure is a physical certificate.
    optimistic = replace(problem, ontario_executable_release_m3_s=np.zeros(len(realized) - k))
    relaxed = explicit_storage_feasibility(optimistic, tolerance)
    if relaxed["status"] == "solver_failure":
        raise ArithmeticError(relaxed["message"])
    record["relaxation_status"] = relaxed["status"]
    if relaxed["status"] == "infeasible":
        dt = np.asarray(problem.duration_seconds)
        # Optimistic release ceiling, including the one-period arrival.
        ceiling = np.array(problem.release_maximum_m3_s)
        ceiling[:-1] = np.minimum(ceiling[:-1], (
            np.asarray(problem.downstream_capacity_m3_s) - problem.other_downstream_m3_s
            - problem.in_transit_contribution_m3_s
        )[1:])
        overflow = storage + np.cumsum(dt * (np.asarray(realized[k:]) - ceiling)) - config["storage_maximum_m3"]
        record.update(classification="certified_infeasible", relaxation_reason=relaxed["message"],
                      optimistic_max_overflow_m3=float(max(0., np.max(overflow))))
        return record
    witness = common_offline(config, scenario, trajectory.to_dict("records"), k, state, [realized], tolerance,
                             project_first=True)
    record["classification"] = "witnessed_feasible" if witness["status"] == "common_witness" else "unresolved"
    if record["classification"] == "witnessed_feasible":
        record["witness"] = witness
    return record


def extend_prefix_witnesses(prefixes):
    """Prepend valid executed actions to a later continuation witness."""
    later = None
    for index in range(len(prefixes) - 1, -1, -1):
        prefix = prefixes[index]
        if prefix["classification"] in ("witnessed_feasible", "completed"):
            later = index
        elif later is not None:
            if prefix["classification"] == "certified_infeasible":
                raise ArithmeticError("A later feasible continuation contradicts the infeasibility certificate")
            prefixes[index] = {**prefix, "classification": "witnessed_feasible",
                               "witness_via_executed_prefix_to": later}


def first_difference_probe(trajectory, config, scenario, history, bank, saved, stop):
    k = len(saved) - 1
    low, high = (bank[name] for name in ("receding", "sustained"))
    original, original_trace, _, original_audit = online_case(trajectory, config, scenario, history, low, high, WIDER_ARM)
    pd.testing.assert_frame_equal(pd.DataFrame(original_trace)[["period", "status"]], saved[["period", "status"]])
    columns = ["ontario_start_m", "ottawa_start_storage_m3", "ottawa_previous_release_m3_s"]
    np.testing.assert_allclose([original_trace[k][key] for key in columns], [stop[key] for key in columns],
                               atol=1e-5, rtol=0)
    actual = np.asarray(scenario["realized_net_inflow_m3_s"][k:k + LOOKAHEAD])
    oracle_low, oracle_high = low.copy(), high.copy()
    oracle_low[k, :len(actual)] = actual
    oracle_high[k, :len(actual)] = actual
    if (k + 1) in original_audit:
        # The original decision used two-path candidates. Keep that mode for
        # the oracle probe even though both inputs are now the same plateau.
        _, trace, _, audit = online_case(trajectory, config, scenario, history, oracle_low, oracle_high,
                                         WIDER_ARM, forced_branch_periods=(k,))
    else:
        _, trace, _, audit = online_case(trajectory, config, scenario, history, oracle_low, oracle_high, WIDER_ARM)
    for i in range(k):
        for key in ("ontario_applied_m3_s", "ottawa_executed_release_m3_s", "ottawa_end_storage_m3"):
            np.testing.assert_allclose(trace[i][key], original_trace[i][key], atol=1e-5, rtol=0)
    accepted = trace[k]["status"] == "executed"
    return {"issued_status": original["status"], "offline_replan_status": trace[k]["status"],
            "accepted_with_realized_plateau": accepted,
            "oracle_probe_used_branch_search": k + 1 in audit}


def write_report(output, cases):
    counts = Counter(case["finding"] for case in cases)
    probes = [case["forecast_probe"] for case in cases if "forecast_probe" in case]
    lines = [
        "# Fresh sustained-rise recoverability", "",
        "The 18 cases reuse the wider two-path traces. Each prefix starts with the actual storage, "
        "prior releases, lake level, and water already in transit. Full-window offline feasibility "
        "uses realized plateau inflows and physical limits. The forecast reserve remains fixed in "
        "the issued-forecast and realized-inflow controller probes.", "",
        "| Finding | Cases |", "|---|---:|",
    ]
    lines.extend(f"| {label.replace('_', ' ')} | {number} |" for label, number in counts.items())
    lines.extend([
        "", f"Realized-inflow substitution accepted the stopped decision in "
        f"{sum(p['accepted_with_realized_plateau'] for p in probes)}/{len(probes)} probed cases.", "",
        "", "| Window | Hydrograph | Seed | Initial state | Final prefix | Stop decision | Realized-forecast probe |",
        "|---|---|---:|---|---|---:|---|",
    ])
    for case in cases:
        prefixes = case["prefixes"]
        probe = case.get("forecast_probe")
        lines.append(f"| {case['year']} | {case['pair']} | {case['seed']} | {prefixes[0]['classification']} | "
                     f"{prefixes[-1]['classification']} | {case['last_decision'] if case['finding'] != 'completed' else '—'} | "
                     f"{probe['offline_replan_status'] if probe else '—'} |")
    lines.extend([
        "", "A continuation witness satisfies native Ontario limits and Ottawa storage and routing "
        "through period 12 and retains terminal transit. A search without a witness remains unresolved "
        "unless the zero-Ontario physical relaxation is itself infeasible. A realized-forecast probe "
        "makes the plateau inputs available at the stopped decision offline and tests the same search "
        "with the reserve intact. It does not establish causal forecast performance.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = args.archive_root / "ottawa_two_path_wider_fresh_v1"
    inputs = {}

    def source(path):
        inputs[str(path)] = digest(path)
        return path

    config = json.loads(source(CONFIG).read_text())
    validate_config(config)
    settings = json.loads(source(prior / "settings.json").read_text())
    if inputs[str(CONFIG)] not in settings["inputs_sha256"].values():
        raise ValueError("Config differs from the archived run")
    table = pd.read_csv(source(prior / "runs.csv"))
    cases = table[(table.arm == WIDER_ARM) & (table.shape_name == "plateau")]
    if len(cases) != 18 or cases.duplicated(["year", "pair", "seed"]).any():
        raise ValueError("Expected the 18 distinct sustained-rise cases")
    tolerance = FollowerTolerances(**config["acceptance"])
    base = next(row for row in config["scenarios"] if row["id"] == "one_period_delay")
    records = []
    initial = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for year, window in cases.groupby("year"):
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        history = json.loads(source(archive / "settings.json").read_text())["ontario_window"]["initial_history"]
        local = {**config, "known_ontario_executable_schedule_m3_s": (trajectory.ontFlow * FLOW_SCALE).tolist()}
        for case in window.itertuples():
            hydrograph = next(pair for pair in settings["design"]["pairs"] if pair["name"] == case.pair)
            scenario = {**base, "realized_net_inflow_m3_s": hydrograph["plateau"]}
            stem = prior / str(year) / f"{case.pair}_plateau_{case.seed}"
            saved = pd.read_csv(source(stem.with_name(f"{stem.name}_{WIDER_ARM}_trace.csv")))
            executed = saved[saved.status == "executed"]
            if len(executed) != case.executed_periods:
                raise ValueError("Trace and summary disagree on the number of executed decisions")
            key = year, case.pair
            if key not in initial:
                initial[key] = check_prefix(trajectory, local, scenario, history, executed.iloc[:0], tolerance)
            prefixes = [initial[key]]
            for k in range(1, len(executed) + 1):
                prefixes.append(check_prefix(trajectory, local, scenario, history, executed.iloc[:k], tolerance))
            extend_prefix_witnesses(prefixes)
            if case.status == "completed":
                # The executed continuation is its own witness on every prefix.
                for prefix in prefixes:
                    if prefix["classification"] == "certified_infeasible":
                        raise ArithmeticError("An impossible prefix occurs on a completed trajectory")
                finding = "completed"
            elif prefixes[0]["classification"] == "certified_infeasible":
                finding = "initial_physical_infeasibility"
            elif any(p["classification"] == "certified_infeasible" for p in prefixes):
                finding = "certified_loss_after_execution" if prefixes[0]["classification"] == "witnessed_feasible" else "certified_infeasibility_at_prefix"
            elif prefixes[-1]["classification"] == "witnessed_feasible":
                finding = "feasible_when_controller_stopped"
            else:
                finding = "unresolved_when_controller_stopped"
            record = {"year": int(year), "pair": case.pair, "seed": case.seed,
                      "last_decision": case.last_decision, "finding": finding,
                      "archived_trace": str(stem.with_name(f"{stem.name}_{WIDER_ARM}_trace.csv")),
                      "prefixes": prefixes}
            if finding == "feasible_when_controller_stopped":
                with np.load(source(stem.with_name(f"{stem.name}_forecasts.npz"))) as bank:
                    record["forecast_probe"] = first_difference_probe(
                        trajectory, local, scenario, history, bank, saved, saved.iloc[-1],
                    )
            records.append(record)
            print(year, case.pair, case.seed, finding, [p["classification"] for p in prefixes], flush=True)
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ArithmeticError(f"An input changed: {path}")
    (args.output / "prefixes.json").write_text(json.dumps(
        {"cases": records, "inputs_sha256": inputs}, default=json_value, indent=2,
    ) + "\n")
    write_report(args.output, records)


if __name__ == "__main__":
    main()
