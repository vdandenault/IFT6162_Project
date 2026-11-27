"""Offline common-action diagnostic and causal two-path Ottawa replay."""

from argparse import ArgumentParser
import json
from dataclasses import replace
from itertools import chain, product
from pathlib import Path

from gymnasium.spaces import Box
import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import AdjustFlow, ONTARIO_COST_COMPONENTS
from bilevel_reservoirs.simulators.physics import CONVERSION, FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import forecast_f_ceiling, native_step, run_policy
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, FollowerTolerances, TransitState, digest, json_value, make_problem, round_d, validate_config,
)
from bilevel_reservoirs.solvers.scenario_ottawa import (
    choose_two_paths, first_release_interval, fix_first_release, gradual_schedules,
)
from bilevel_reservoirs.solvers.study_ottawa_recession import ISOLATED_DESIGN, recession_bank
from bilevel_reservoirs.solvers.validate_storage_guard import stop_evidence
from reservoir_control.systems.ottawa.delay_diagnostic import explicit_storage_feasibility
from reservoir_control.systems.ottawa.horizon_study import forecast_bank

DESIGN = CONFIG.with_name("scenario_recourse_v1.json")
FRESH_DESIGN = CONFIG.with_name("scenario_recourse_fresh_v1.json")
ARMS = ("isolated_jump", "two_path")
WIDER_ARM = "two_path_wider"
RESERVE_FALLBACK = "two_path_reserve_fallback"
SHAPES = ("pulse", "plateau")


def high_path_bank(persistence, low, extra_periods):
    """Keep persistence longer, then delay the same causal recession adjustment."""
    high = np.array(persistence, copy=True)
    high[:, extra_periods:] -= persistence[:, :-extra_periods] - low[:, :-extra_periods]
    high.flags.writeable = False
    return high


def common_offline(config, scenario, rows, k, inherited, inflows, tolerance, *, project_first=False):
    """Certify a common action using delayed-routing intervals and independent LPs."""
    window = rows[k:]
    level, previous_flow, ice, storage, previous_release, queue = inherited
    state = level, previous_flow / FLOW_SCALE, ice
    mechanism = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0.0, np.inf, (1,), dtype=np.float64))
    branches = [
        make_problem(config, scenario, k, storage, previous_release, queue, values[k:]) for values in inflows
    ]
    anchor = branches[0].ontario_executable_release_m3_s
    start_flows = {max(0.0, previous_flow - drop) for drop in (*range(0, 1401, 50), 700, 710, 1420)}
    if project_first:
        non_ontario = branches[0].other_downstream_m3_s[0] + branches[0].in_transit_contribution_m3_s[0]
        forcing = {**window[0], "stlouisontOut": non_ontario, "slonFlow_QM1": non_ontario}
        for flow in tuple(start_flows):
            _, allowed, _ = native_step(forcing, forcing["ontNTS_QM1"], flow / FLOW_SCALE, state, mechanism)
            if allowed >= 0:
                start_flows.add(allowed * FLOW_SCALE)
    profiles = ([max(target, first - ramp * j) for j in range(len(window))]
                for first, ramp, target in product(sorted(start_flows), (350, 500, 600, 700), (4800, 4950, 5100, 5300, 5500, 5800, 6100)))
    for schedule in chain((anchor,), profiles):
        caps = np.array(branches[0].downstream_capacity_m3_s)
        states, begin, past_flow, past_ice = [], *state
        for j, (row, flow) in enumerate(zip(window, schedule, strict=True)):
            states.append((begin, past_flow, past_ice))
            caps[j] = min(caps[j], forecast_f_ceiling(begin, row["ptclaireR"], flow))
            begin += round_d((row["ontNTS"] - flow) / (FLOW_SCALE * CONVERSION), 6)
            past_flow, past_ice = flow / FLOW_SCALE, row["iceInd"]
        fixed = [replace(branch, ontario_executable_release_m3_s=schedule, downstream_capacity_m3_s=caps)
                 for branch in branches]
        intervals = [first_release_interval(branch, tolerance) for branch in fixed]
        if any(interval is None for interval in intervals):
            continue
        lo, hi = max(value[0] for value in intervals), min(value[1] for value in intervals)
        if hi < lo:
            continue
        shared = (lo + hi) / 2
        proofs = [explicit_storage_feasibility(fix_first_release(branch, shared), tolerance) for branch in fixed]
        if any(proof["status"] != "feasible" for proof in proofs):
            raise ArithmeticError("The independent LP did not confirm the common interval witness")
        for branch, proof in zip(fixed, proofs, strict=True):
            releases = np.asarray(proof["release_m3_s"])
            arrivals = branch.routing_matrix @ releases + branch.in_transit_contribution_m3_s
            for j, row in enumerate(window):
                flow = schedule[j] / FLOW_SCALE
                non_ontario = branch.other_downstream_m3_s[j] + arrivals[j]
                forcing = {**row, "stlouisontOut": non_ontario, "slonFlow_QM1": non_ontario}
                _, allowed, _ = native_step(forcing, row["ontNTS_QM1"], flow, states[j], mechanism)
                if abs(allowed - flow) * FLOW_SCALE > tolerance.flow_m3_s:
                    break
            else:
                proof["terminal_transit_m3"] = float(branch.duration_seconds[-1] * releases[-1])
        if all("terminal_transit_m3" in proof for proof in proofs):
            return {"status": "common_witness", "period": k + 1, "ontario_schedule_m3_s": schedule,
                    "physical_interval_before_native_check_m3_s": [lo, hi],
                    "shared_ottawa_release_m3_s": shared, "paths": proofs}
    return {"status": "unresolved_in_enumerated_schedules", "period": k + 1}


def archived_checks(root, output, config, scenario, source):
    prior = root / "ottawa_isolated_jump_v1"
    old_design = json.loads(source(prior / "settings.json").read_text())
    original = old_design["design"]
    inflows = [original["inflows_m3_s"][name] for name in ("early_sharp_peak", "sharp_rise_stays_high")]
    if inflows[0][:3] != inflows[1][:3]:
        raise ValueError("The archived hydrographs must share the first three observations")
    cases = []
    tolerance = FollowerTolerances(**config["acceptance"])
    for year in original["ontario_years"]:
        archive = root / f"ottawa_replan_guard_{year}"
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        rows = trajectory.to_dict("records")
        for seed in old_design["seeds"]:
            traces = [pd.read_csv(source(prior / str(year) / f"{shape}_{seed}_isolated_jump_trace.csv"))
                      for shape in ("early_sharp_peak", "sharp_rise_stays_high")]
            cols = ["period", "ontario_end_m", "ontario_applied_m3_s", "ottawa_end_storage_m3",
                    "ottawa_executed_release_m3_s", "ottawa_end_queue_m3"]
            pd.testing.assert_frame_equal(traces[0].iloc[:3][cols], traces[1].iloc[:3][cols])
            last = traces[0].iloc[2]
            if last.status != "executed":
                raise ValueError("Expected an executed common history through period three")
            inherited = (last.ontario_end_m, last.ontario_applied_m3_s, rows[2]["iceInd"],
                         last.ottawa_end_storage_m3, last.ottawa_executed_release_m3_s,
                         TransitState(tuple(json.loads(last.ottawa_end_queue_m3))))
            proof = common_offline(local, scenario, rows, 3, inherited, inflows, tolerance)
            cases.append({"year": year, "seed": seed, **proof})
            print("offline", year, seed, proof["status"], flush=True)
    (output / "offline_common_actions.json").write_text(json.dumps(cases, default=json_value, indent=2) + "\n")
    return cases


def online_case(trajectory, config, scenario, history, low, high, arm, *, forced_branch_periods=()):
    branch_audit = {}
    rows = trajectory.to_dict("records")
    tolerance = FollowerTolerances(**config["acceptance"])

    def choose(k, problem, level, previous_flow, previous_ice, mechanism):
        audit = branch_audit.setdefault(k + 1, [])
        window = rows[k:k + len(problem.duration_seconds)]
        state = level, previous_flow, previous_ice
        extra = (gradual_schedules(problem, previous_flow * FLOW_SCALE)
                 if arm in (WIDER_ARM, RESERVE_FALLBACK) else ())
        chosen = choose_two_paths(problem, high[k], window, state, mechanism,
                                 config["reference_release_m3_s"], tolerance, audit, extra_schedules=extra)
        if chosen is None and arm == RESERVE_FALLBACK:
            physical = replace(problem, storage_maximum_m3=np.full(len(problem.storage_maximum_m3),
                                                                    config["storage_maximum_m3"]))
            audit.append({"reserve_fallback": True})
            return choose_two_paths(physical, high[k], window, state, mechanism,
                                    config["reference_release_m3_s"], tolerance, audit, extra_schedules=extra)
        return chosen

    active = np.flatnonzero(np.any(np.nan_to_num(high - low) > 1e-9, axis=1))
    decisions = {int(k): choose for k in (*active, *forced_branch_periods)} if arm in ("two_path", WIDER_ARM, RESERVE_FALLBACK) else None
    audit = []
    trace, plans = run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits", align_f=True,
        ottawa_forecast_bank=low, recovery="persistence", recovery_guard_periods=1,
        replan_guard="always", delay_headroom_search=True, delay_headroom_on_exhaustion=True,
        matched_response_ranking=True, scenario_decisions=decisions, **history,
    )
    executed = [row for row in trace if row["status"] == "executed"]
    upper_violations = sum(row["ottawa_end_storage_m3"] > config["storage_maximum_m3"] + tolerance.storage_m3
                           for row in executed)
    lower_violations = sum(row["ottawa_end_storage_m3"] < config["storage_minimum_m3"] - tolerance.storage_m3
                           for row in executed)
    downstream_violations = sum(row["st_louis_flow_m3_s"] > row["capacity_m3_s"] + tolerance.flow_m3_s
                                for row in executed)
    status = trace[-1]["status"]
    if upper_violations or lower_violations or downstream_violations:
        status = "realized_violation"
    elif len(executed) == len(trajectory):
        status = "completed"
    for period, records in branch_audit.items():
        selected = next((entry for entry in records if entry.get("selected")), None)
        if selected is not None:
            row = trace[period - 1]
            if not np.isclose(row["ottawa_proposed_release_m3_s"], selected["ottawa_first_release_m3_s"],
                              rtol=0, atol=tolerance.flow_m3_s):
                raise ArithmeticError("Replay failed to reproduce the shared Ottawa first action")
    summary = {"status": status, "last_decision": trace[-1]["period"], "executed_periods": len(executed),
               "upper_violations": upper_violations, "lower_violations": lower_violations,
               "downstream_violations": downstream_violations, "branch_decisions": sorted(branch_audit),
               "gradual_selections": [period for period, records in branch_audit.items()
                                      if any(x.get("selected") and x["stage"] == "gradual_reductions" for x in records)],
               **{key: sum(row[key] for row in executed) for key in ONTARIO_COST_COMPONENTS}}
    return summary, trace, plans, branch_audit


def heldout_common_checks(root, output, config, scenario, design, table, source):
    """Look for a physical common witness when the causal branch search stopped."""
    cases = []
    tolerance = FollowerTolerances(**config["acceptance"])
    failed = table[(table.arm == "two_path") & (table.shape_name == "pulse")
                   & (table.status == "no_admissible_candidate") & (table.last_decision == 4)]
    for year, group in failed.groupby("year"):
        archive = root / f"ottawa_replan_guard_{year}"
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        local = {**config, "known_ontario_executable_schedule_m3_s": (trajectory.ontFlow * FLOW_SCALE).tolist()}
        rows = trajectory.to_dict("records")
        for case in group.itertuples():
            pair = next(pair for pair in design["pairs"] if pair["name"] == case.pair)
            trace = pd.read_csv(output / str(year) / f"{case.pair}_pulse_{case.seed}_two_path_trace.csv")
            stop = trace.iloc[-1]
            inherited = (
                stop.ontario_start_m, stop.ontario_previous_release_m3_s, stop.ontario_previous_ice,
                stop.ottawa_start_storage_m3, stop.ottawa_previous_release_m3_s,
                TransitState(tuple(json.loads(stop.ottawa_start_queue_m3))),
            )
            proof = common_offline(local, scenario, rows, 3, inherited, [pair[shape] for shape in SHAPES], tolerance)
            cases.append({"year": int(year), "pair": case.pair, "seed": case.seed, **proof})
    (output / "heldout_stops_common_actions.json").write_text(json.dumps(cases, default=json_value, indent=2) + "\n")
    return cases


def write_report(output, offline, table, heldout=None):
    lines = [
        "# Two-path Ottawa MPC", "",
        f"Found a common first action in {sum(case['status'] == 'common_witness' for case in offline)}/{len(offline)} "
        "archived period-4 states. The offline checks use realized inputs and validate full-window "
        "release plans against the native Ontario limits. They search a common Ontario schedule "
        "with different Ottawa continuations. A failed search leaves existence unresolved.", "",
        "The causal arm uses an isolated-jump recession and a high path that delays recession by "
        "two more periods. Both retain the same first forecast lead and noise draws. At an active "
        "decision each path may select a different Ontario suffix, but the first Ontario and "
        "Ottawa releases must agree. The search ranks schedule pairs by mean external Ontario score. "
        "The shared Ottawa release projects the mean of the unconstrained first best responses "
        "onto the intersection of their feasible intervals. Each branch then solves the existing "
        "follower objective under that shared release.", "",
        "Both controllers retain the +700 m³/s reserve and existing hard limits. At a branch decision "
        "the two-path arm stops if no common admissible candidate is found. In other decisions it uses "
        "the same isolated-jump controller and fallback. The fifth Ontario commitment remains fixed.", "",
        "| Realized inflow | Controller | Completed | Upper / lower / downstream violations | No candidate | Stops with continuation witnesses |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for shape in (*SHAPES, "all"):
        group = table if shape == "all" else table[table.shape_name == shape]
        for arm in ARMS:
            a = group[group.arm == arm]
            violations = " / ".join(str(a[key].sum()) for key in ("upper_violations", "lower_violations", "downstream_violations"))
            lines.append(f"| {shape} | {arm} | {(a.status == 'completed').sum()}/{len(a)} | {violations} | "
                         f"{(a.status == 'no_admissible_candidate').sum()} | {a.stop_witness.sum()} |")
    if heldout:
        count = sum(case["status"] == "common_witness" for case in heldout)
        lines.extend([
            "", f"At the {len(heldout)} held-out period-4 branch stops, the separate realized-inflow "
            f"search found a physical common action in {count} cases. These checks use the actual "
            "stopped state and physical bounds. The forecast reserve applies to the causal planner.", "",
        ])
    pairs = table[table.status == "completed"].pivot(index=["year", "pair", "shape_name", "seed"],
                                                     columns="arm", values=list(ONTARIO_COST_COMPONENTS))
    lines.extend(["", "## Ontario outcomes on completed pairs", "",
                  f"Only pairs completing in both arms enter this comparison. Sample size {(pairs.notna().all(axis=1)).sum()}.", "",
                  "| Outcome | Mean change, two paths minus isolation |", "|---|---:|"])
    for key in ONTARIO_COST_COMPONENTS:
        value = pairs.get((key, "two_path"), pd.Series(dtype=float)) - pairs.get((key, "isolated_jump"), pd.Series(dtype=float))
        lines.append(f"| {key} | {value.mean():.6g} |")
    lines.extend(["", "The forecasts use completed Ottawa observations only. Realized inputs enter the "
                  "simulator and separate offline checks. The holdout varies synthetic heights, durations "
                  "and forecast noise on the same historical Ontario windows.", ""])
    (output / "report.md").write_text("\n".join(lines))


def write_wider_report(output, table, sample, recovered):
    lines = [
        "# Closed-loop test of wider two-path search", "",
        f"Sample: {sample}. The wider arm adds gradual Ontario release candidates after the original "
        "search stages fail. The issued forecasts, +700 m³/s reserve, physical constraints, and "
        "objective weights are matched. The Ottawa scenarios are synthetic.", "",
        "| Inflow | Controller | Completed | Upper / lower / downstream violations | No candidate | Witnessed feasible stops |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for shape in (*SHAPES, "all"):
        subset = table if shape == "all" else table[table.shape_name == shape]
        for arm, group in subset.groupby("arm", sort=False):
            violations = " / ".join(str(group[key].sum()) for key in (
                "upper_violations", "lower_violations", "downstream_violations",
            ))
            lines.append(f"| {shape} | {arm} | {(group.status == 'completed').sum()}/{len(group)} | "
                         f"{violations} | {(group.status == 'no_admissible_candidate').sum()} | {group.stop_witness.sum()} |")
    index = ["year", "pair", "shape_name", "seed"]
    outcomes = table.pivot(index=index, columns="arm", values="status")
    for arm in ARMS:
        wins = ((outcomes[WIDER_ARM] == "completed") & (outcomes[arm] != "completed")).sum()
        losses = ((outcomes[arm] == "completed") & (outcomes[WIDER_ARM] != "completed")).sum()
        lines.extend(["", f"Compared with {arm}, wider search gained {wins} completions and lost {losses}."])
    if recovered:
        completed = sum(case["status"] == "completed" for case in recovered)
        lines.extend(["", f"All {len(recovered)} frozen recoveries selected a gradual release schedule "
                      f"at period 4 in closed loop. Of these, {completed} completed the window."])
    lines.extend(["", "## Ontario costs on completed pairs", "",
                  "Mean changes are wider search minus the reference arm in the original indicator units.", "",
                  "| Reference | Paired windows | Upstream coastal | Downstream coastal | Negative energy value | Recreational boating |",
                  "|---|---:|---:|---:|---:|---:|"])
    complete = table[table.status == "completed"]
    wider = complete[complete.arm == WIDER_ARM].set_index(index)[list(ONTARIO_COST_COMPONENTS)]
    for arm in ARMS:
        other = complete[complete.arm == arm].set_index(index)[list(ONTARIO_COST_COMPONENTS)]
        common = wider.index.intersection(other.index)
        if common.empty:
            values = ["—"] * len(ONTARIO_COST_COMPONENTS)
        else:
            delta = (wider.loc[common] - other.loc[common]).mean()
            values = [f"{delta[key]:.6g}" for key in ONTARIO_COST_COMPONENTS]
        lines.append(f"| {arm} | {len(common)} | " + " | ".join(values) + " |")
    lines.extend(["", "Fresh cases vary Ottawa peak height, high-flow duration, and forecast noise "
                  "while reusing the same historical Ontario windows. Continuation witnesses may "
                  "use hindsight. Controller defaults were preserved.", ""])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--sample", choices=("original", "wider-archived", "wider-fresh"), default="original")
    parser.add_argument("--reserve-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def source(path):
        sources[str(path)] = digest(path)
        return path

    config = json.loads(source(CONFIG).read_text())
    validate_config(config)
    design = json.loads(source(FRESH_DESIGN if args.sample == "wider-fresh" else DESIGN).read_text())
    wider = args.sample != "original"
    arms = (*ARMS, WIDER_ARM) if wider else ARMS
    if args.reserve_fallback:
        if not wider:
            parser.error("--reserve-fallback requires a wider sample")
        arms += (RESERVE_FALLBACK,)
    previous = args.archive_root / "ottawa_two_path_recourse_v1"
    recovered = []
    if args.sample == "wider-archived":
        factorial = args.archive_root / "ottawa_two_path_factorial_v1" / "summary.json"
        recovered_keys = {(case["year"], case["pair"], case["seed"])
                          for case in json.loads(source(factorial).read_text())["cases"]
                          if case["cells"]["wider_guarded"]["accepted"]}
        original = pd.read_csv(source(previous / "runs.csv")).set_index(["year", "pair", "shape_name", "seed", "arm"])
    jump = json.loads(source(ISOLATED_DESIGN).read_text())["jump_minimum_m3_s"]
    scenario = next(row for row in config["scenarios"] if row["id"] == "one_period_delay")
    noise = config["forecast_noise"]
    if noise["standard_deviation_m3_s"] != 700 or jump != 700:
        raise ValueError("Expected the frozen guard and trigger")
    args.output.mkdir(parents=True, exist_ok=False)
    offline = archived_checks(args.archive_root, args.output, config, scenario, source) if not wider else None
    summaries = []
    for year in (1961, 2017, 2019):
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        trajectory = pd.read_csv(source(archive / "ontario_inputs.csv"))
        history = json.loads(source(archive / "settings.json").read_text())["ontario_window"]["initial_history"]
        directory = args.output / str(year)
        directory.mkdir()
        local = {**config, "known_ontario_executable_schedule_m3_s": (trajectory.ontFlow * FLOW_SCALE).tolist()}
        for pair in design["pairs"]:
            if pair["pulse"][:3] != pair["plateau"][:3]:
                raise ValueError("The two realized paths must share their opening")
            for seed in design["seeds"]:
                issued = {}
                for shape in SHAPES:
                    values = pair[shape]
                    origin = forecast_bank(values, "noisy", seed, noise["initial_observed_inflow_estimate_m3_s"],
                                           noise["standard_deviation_m3_s"])
                    low = recession_bank(values, origin, jump_minimum_m3_s=jump)
                    high = high_path_bank(origin, low, design["high_path_extra_periods"])
                    if args.sample == "wider-archived":
                        stem = f"{pair['name']}_{shape}_{seed}"
                        with np.load(source(previous / str(year) / f"{stem}_forecasts.npz")) as saved:
                            np.testing.assert_array_equal(low, saved["receding"])
                            np.testing.assert_array_equal(high, saved["sustained"])
                    issued[shape] = low, high
                for a, b in zip(issued["pulse"], issued["plateau"], strict=True):
                    np.testing.assert_array_equal(a[:4], b[:4])
                pair_traces = {}
                for shape, (low, high) in issued.items():
                    case = {**scenario, "realized_net_inflow_m3_s": pair[shape]}
                    stem = f"{pair['name']}_{shape}_{seed}"
                    np.savez(directory / f"{stem}_forecasts.npz", receding=low, sustained=high)
                    results = {arm: online_case(trajectory, local, case, history, low, high, arm) for arm in arms}
                    pair_traces[shape] = {}
                    for arm in arms:
                        summary, trace, plans, audit = results[arm]
                        frame = pd.DataFrame(trace)
                        pair_traces[shape][arm] = frame
                        trace_path = directory / f"{stem}_{arm}_trace.csv"
                        frame.to_csv(trace_path, index=False)
                        if args.sample == "wider-archived":
                            if arm in ARMS:
                                archived = original.loc[(year, pair["name"], shape, seed, arm)]
                                if summary["status"] != archived.status:
                                    raise ArithmeticError("The archived baseline outcome did not reproduce")
                                pd.testing.assert_frame_equal(
                                    pd.read_csv(trace_path),
                                    pd.read_csv(source(previous / str(year) / trace_path.name)),
                                )
                            elif arm == WIDER_ARM and shape == "pulse" and (year, pair["name"], seed) in recovered_keys:
                                if 4 not in summary["gradual_selections"]:
                                    raise ArithmeticError("A frozen recovery at period 4 did not reproduce")
                                recovered.append({"year": year, "pair": pair["name"], "seed": seed, **summary})
                        peers = [other for other in arms if other != arm]
                        peer = next((other for other in peers if results[other][0]["status"] == "completed"), peers[0])
                        proof = (stop_evidence(trajectory, local, case, trace, results[peer][1], results[peer][0], history)
                                 if summary["status"] == "no_admissible_candidate" else None)
                        (directory / f"{stem}_{arm}_plans.json").write_text(
                            json.dumps({"plans": plans, "branches": audit, "stop_witness": proof},
                                       default=json_value) + "\n"
                        )
                        summaries.append({"year": year, "pair": pair["name"], "shape_name": shape, "seed": seed,
                                          "arm": arm, "stop_witness": bool(proof and proof["witness"]), **summary})
                    print(year, stem, *(results[arm][0]["status"] for arm in arms), flush=True)
                for arm in arms:
                    a, b = (pair_traces[shape][arm] for shape in SHAPES)
                    cols = ["period", "status", *[key for key in ("ontario_applied_m3_s", "ottawa_executed_release_m3_s") if key in a]]
                    pd.testing.assert_frame_equal(a.loc[a.period <= 4, cols], b.loc[b.period <= 4, cols])
    table = pd.DataFrame(summaries)
    if not wider:
        heldout = heldout_common_checks(args.archive_root, args.output, config, scenario, design, table, source)
    elif args.sample == "wider-archived" and len(recovered) != len(recovered_keys):
        raise ArithmeticError("The comparison missed a recovered frozen decision")
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise ArithmeticError(f"An input changed during the study: {path}")
    table.to_csv(args.output / "runs.csv", index=False)
    if wider:
        write_wider_report(args.output, table, args.sample, recovered)
        if recovered:
            (args.output / "frozen_recoveries.json").write_text(json.dumps(recovered, default=json_value, indent=2) + "\n")
    else:
        write_report(args.output, offline, table, heldout)
    (args.output / "settings.json").write_text(json.dumps({
        "sample": args.sample, "design": design, "inputs_sha256": sources,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
