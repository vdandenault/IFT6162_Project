"""Check targeted release candidates from the two remaining frozen period-6 stops."""

from argparse import ArgumentParser
from ast import literal_eval
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from gymnasium.spaces import Box
import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import AdjustFlow
from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import (
    delay_headroom_candidates, forecast_trajectory, run_policy,
)
from bilevel_reservoirs.solvers.diagnose_replan_stop import SCENARIO, evaluate_schedules
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, FollowerTolerances, TransitState, digest, json_value, make_problem, validate_config,
)
from bilevel_reservoirs.solvers.stress_ottawa_response import offline_recoverability
from bilevel_reservoirs.solvers.study_delay_headroom import BASELINE, verify_prefix

CASES = ((1961, 102), (2019, 103))


def diagnose(archive, seed, config, sources):
    def read(name):
        path = archive / name
        sources[str(path)] = digest(path)
        return path

    settings = json.loads(read("settings.json").read_text())
    if (sources[str(CONFIG)] not in settings["input_sha256"].values()
            or settings["ordinary_replan_guards"][BASELINE] != "after_retry"
            or settings["recovery_guard_periods"][BASELINE] != 1
            or (settings["ontario_horizon"], settings["ottawa_horizon"],
                settings["align_f"], settings["candidate_family"]) != (4, 5, True, "limits")):
        raise ValueError("The archive does not match the frozen follower configuration")
    history = settings["ontario_window"]["initial_history"]
    trajectory = pd.read_csv(read("ontario_inputs.csv"))
    bank = np.load(read(f"{SCENARIO}_{seed}_forecasts.npy"))
    stem = f"{SCENARIO}_{seed}_{BASELINE}"
    archived = pd.read_csv(read(f"{stem}_trace.csv"))
    archived_offline = json.loads(read(f"{stem}_offline.json").read_text())
    config = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
    scenario = next(row for row in config["scenarios"] if row["id"] == SCENARIO)
    audit = []
    trace, _ = run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits",
        align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
        recovery_guard_periods=1, replan_guard="after_retry", **history,
    )
    failed = trace[-1]
    k = failed["period"] - 1
    if k != 5 or failed["status"] != "no_admissible_candidate":
        raise ArithmeticError("Expected the original no-candidate stop at period 6")
    replayed = pd.DataFrame(trace)
    if not replayed[["period", "status"]].equals(archived[["period", "status"]]):
        raise ArithmeticError("The archived execution statuses did not reproduce")
    verify_prefix(archived, replayed, k, config["acceptance"])
    tolerance = FollowerTolerances(**config["acceptance"])
    for key, atol in {
        "ottawa_start_storage_m3": tolerance.storage_m3,
        "ottawa_previous_release_m3_s": tolerance.flow_m3_s,
        "ontario_previous_release_m3_s": tolerance.flow_m3_s,
        "ontario_start_m": 0.500001e-6,
        "ontario_previous_ice": 0.0,
    }.items():
        np.testing.assert_allclose(failed[key], archived.iloc[-1][key], rtol=0, atol=atol)
    np.testing.assert_allclose(failed["ottawa_start_queue_m3"],
                               literal_eval(archived.iloc[-1].ottawa_start_queue_m3),
                               rtol=0, atol=tolerance.storage_m3)

    attempted = [entry for entry in audit if entry["period"] == k + 1]
    if {entry["recovery_stage"] for entry in attempted} != {"primary", "expanded", "persistence_retry"}:
        raise ArithmeticError("The expected original and retry stages did not reproduce")
    if (any("recovery_storage_maximum_m3" in entry for entry in attempted if entry["recovery_stage"] != "persistence_retry")
            or any(entry["selected"] and entry["recovery_stage"] == "persistence_retry" for entry in audit)):
        raise ArithmeticError("Expected no preceding retry and an unguarded ordinary search")

    original = bank[k, :settings["ottawa_horizon"]]
    queue = TransitState(tuple(failed["ottawa_start_queue_m3"]))
    base = make_problem(config, scenario, k, failed["ottawa_start_storage_m3"],
                        failed["ottawa_previous_release_m3_s"], queue, original)
    rows = trajectory.iloc[k:k + len(original)].to_dict("records")
    state = (failed["ontario_start_m"], failed["ontario_previous_release_m3_s"] / FLOW_SCALE,
             failed["ontario_previous_ice"])
    retry_inflow = scenario["realized_net_inflow_m3_s"][k - 1]
    realized = np.array(scenario["realized_net_inflow_m3_s"][k:k + len(original)])
    retry_upper = next(entry["recovery_storage_maximum_m3"] for entry in attempted
                       if entry["recovery_stage"] == "persistence_retry")
    guarded = replace(base, storage_maximum_m3=retry_upper,
                      net_inflow_m3_s=np.full(len(original), retry_inflow))
    problems = {"original": base, "persistence_guarded": guarded,
                "realized_offline": replace(base, net_inflow_m3_s=realized)}

    # The same external-reference and candidate-specific checks as the opt-in
    # search, applied here at a frozen stop without changing the online trigger.
    schedules = delay_headroom_candidates(attempted)
    external = base.routing_matrix @ np.full(len(original), config["reference_release_m3_s"])
    external += base.in_transit_contribution_m3_s
    mechanism = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0., 1200., (1,)))
    external_results = []
    for schedule in schedules:
        _, status, _ = forecast_trajectory(
            rows, schedule, external, base.other_downstream_m3_s,
            base.downstream_capacity_m3_s, state, mechanism, tolerance,
        )
        external_results.append({"schedule_m3_s": schedule, "status": status})
    accepted = [entry["schedule_m3_s"] for entry in external_results if entry["status"] == "admissible"]
    if not accepted:
        raise ArithmeticError("No targeted schedule survived Ontario's external reference screen")
    results = evaluate_schedules(problems, rows, state, accepted, tolerance)
    counts, witnesses = {}, {}
    for label in problems:
        checks = [result for result in results if result["case"] == label]
        counts[label] = dict(Counter(result["status"] for result in checks))
        viable = [result for result in checks if result["status"] == "admissible"]
        if viable:
            witness = viable[0]
            witness["realized_first_storage_m3_offline"] = (
                base.initial_storage_m3 + base.duration_seconds[0] * (realized[0] - witness["ottawa_release_m3_s"][0])
            )
            witnesses[label] = witness

    continuation = offline_recoverability(trajectory, config, scenario, trace[:k], **history)
    if continuation["finding"] != "joint_fixed_schedule_witness" or continuation["finding"] != archived_offline["finding"]:
        raise ArithmeticError("The independent full remaining experiment witness did not reproduce")
    lp = continuation["fixed_continuation_lp"]
    np.testing.assert_allclose(lp["storage_m3"], archived_offline["fixed_continuation_lp"]["storage_m3"],
                               rtol=0, atol=tolerance.storage_m3)
    continuation["fixed_ontario_schedule_m3_s"] = config["known_ontario_executable_schedule_m3_s"][k:]
    continuation["terminal_transit_m3"] = lp["release_m3_s"][-1] * base.duration_seconds[-1]
    summary = {
        "inherited_state": failed, "storage_headroom_m3": base.storage_maximum_m3[0] - base.initial_storage_m3,
        "targeted_schedules": len(schedules), "external_screen_survivors": len(accepted),
        "candidate_outcomes": counts, "witnesses": witnesses,
        "inflows_m3_s": {label: value.net_inflow_m3_s for label, value in problems.items()},
        "retry_storage_reserve_m3": base.storage_maximum_m3[1] - retry_upper[1],
        "full_remaining_offline": continuation,
    }
    return summary, external_results, results


def write_report(output, summaries):
    lines = [
        "# Ottawa period-6 recovery diagnostic", "",
        "The original stops and their inherited storage, releases and routing queues reproduced "
        "from the saved inputs. The comparison lowers Ontario's second-period commitment, "
        "when Ottawa's first release arrives downstream.", "",
        "| Window and seed | Targeted schedules passing the external screen | Original forecast | Guarded persistence | Realized inflows offline |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in summaries:
        counts = result["candidate_outcomes"]
        original, retry, realized = (counts[key].get("admissible", 0)
                                     for key in ("original", "persistence_guarded", "realized_offline"))
        lines.append(f"| {result['year']} seed {result['seed']} | {result['external_screen_survivors']} | "
                     f"{original} | {retry} | {realized} |")
    lines.extend([
        "", "The original forecast uses the saved causal forecast bank. Persistence uses the previous "
        "period's observed synthetic inflow and retains the original +700 m³/s first-period storage "
        "check. The realized-inflow column is an offline five-period counterfactual.", "",
        "An independent full-remaining-experiment LP also found a continuation for each stop using "
        "the fixed Ontario anchor and realized inputs. The audit checks native Ontario limits and "
        "preserves water in transit at period 12. Those witnesses show physical feasibility in the "
        "study model through the endpoint. They use perfect information and provide no claim about "
        "what a causal controller would execute or about safety beyond that endpoint.", "",
        "The JSON outputs contain every screened schedule, witness release and storage paths, "
        "downstream flows and caps, and source checksums. The Ontario inputs are historical and the "
        "Ottawa inflow, outlet, and routing model is synthetic.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    sources = {str(CONFIG): digest(CONFIG)}
    summaries = []
    args.output.mkdir(parents=True, exist_ok=False)
    for year, seed in CASES:
        summary, external, checks = diagnose(args.archive_root / f"ottawa_replan_guard_{year}",
                                             seed, config, sources)
        summaries.append({"year": year, "seed": seed, **summary})
        (args.output / f"{year}_{seed}_audit.json").write_text(
            json.dumps({"external": external, "follower": checks}, default=json_value, indent=2) + "\n"
        )
    for path, checksum in sources.items():
        if digest(Path(path)) != checksum:
            raise ArithmeticError(f"An archived input changed during diagnosis: {path}")
    (args.output / "summary.json").write_text(
        json.dumps({"cases": summaries, "input_sha256": sources}, default=json_value, indent=2) + "\n"
    )
    write_report(args.output, summaries)
    for summary in summaries:
        print(summary["year"], summary["seed"], summary["candidate_outcomes"])


if __name__ == "__main__":
    main()
