from argparse import ArgumentParser
from collections import Counter
from itertools import product
import json
from pathlib import Path

from gymnasium.spaces import Box
import numpy as np
import pandas as pd

from reservoir_control.study_config import load_settings
from reservoir_control.systems.loslr.operator import AdjustFlow
from reservoir_control.systems.loslr.physics import FLOW_SCALE
from reservoir_control.experiments.final_coupled_env import HardCoupledEnv
from reservoir_control.experiments.final_coupled_study import (
    CONFIG, NativeReference, OntarioModel, annual_cases, load_inputs, load_protocol, start_states,
)
from reservoir_control.experiments.fixed_schedule_replay import (
    FollowerTolerances, TransitState, digest, json_value, load_follower_config, make_problem,
)
from reservoir_control.experiments.report_final_coupled import playback, plt
from reservoir_control.systems.loslr.adapter import decision_time

REFERENCE = (5e9, 1)
ARMS = ("external_mpc", "stackelberg_mpc")


def variant(case, capacity, delay):
    scenario = case["scenario"]
    config = case["config"]
    dt = config["clock"]["duration_seconds"]
    scenario = {**scenario, "routing_delay_periods": delay,
                "initial_transit_volumes_m3": [dt * config["previous_executed_release_m3_s"]] * delay}
    config = {**config, "storage_maximum_m3": float(capacity), "scenarios": [scenario],
              "known_ontario_executable_schedule_m3_s": (case["trajectory"].ontFlow * FLOW_SCALE).tolist()}
    return {**case, "config": config, "scenario": scenario}


def frozen_rank(model, source, trace, design, costs):
    tol = FollowerTolerances(**source["config"]["acceptance"])
    mechanism = AdjustFlow(aid="ont", id="flow", acts_on=("ont", "flow"),
                           action_space=Box(0.0, np.inf, (1,), dtype=np.float64))
    results = []
    state = source["start"]
    records = trace.sort_values("period").to_dict("records")
    release_history = []
    for k, record in enumerate(records):
        row = source["rows"][k]
        np.testing.assert_allclose(state.level_m, record["ontario_start_m"], atol=1e-6, rtol=0)
        if k + 1 in design["snapshot_periods"]:
            for capacity, delay in product(design["storage_maximum_m3"], design["routing_delay_periods"]):
                case = variant(source, capacity, delay)
                config, scenario = case["config"], case["scenario"]
                storage = record["ottawa_start_storage_m3"]
                entry = {"year": source["year"], "period": k + 1, "storage_maximum_m3": capacity,
                         "routing_delay_periods": delay, "start_storage_m3": storage}
                if storage > capacity + tol.storage_m3:
                    results.append({**entry, "status": "inherited_storage_outside_variant"})
                    continue
                previous = release_history[-1] if release_history else config["previous_executed_release_m3_s"]
                recent = ([config["previous_executed_release_m3_s"]] * delay + release_history)[-delay:] if delay else []
                queue = TransitState(tuple(q * config["clock"]["duration_seconds"] for q in recent))
                if (capacity, delay) == REFERENCE:
                    np.testing.assert_allclose(queue.volumes_m3, json.loads(record["ottawa_start_queue_m3"]),
                                               rtol=0, atol=tol.storage_m3)
                problem = make_problem(config, scenario, k, storage, previous, queue, source["bank"][k, :min(5, 48-k)])
                env = HardCoupledEnv(model, [case], costs, recovery_mode="peak_protected")
                env.rows, env.state, env.config, env.tolerance = case["rows"], state, config, tol
                env.context = (k, problem, state.level_m, state.previous_release_m3_s / FLOW_SCALE,
                               state.previous_ice, mechanism)
                request = env.neutral_request()
                audit = []
                env.case = case
                env.guard_next = bool(records[k - 1]["retry"]) if k else False
                planned, _, stage, _ = env.plan("stackelberg_mpc", request, request, candidate_audit=audit)
                entry.update(status="ranked" if audit else "no_common_eligible_candidate",
                             stage=stage, eligible_candidates=len(audit), transit_queue_m3=queue.volumes_m3)
                if audit:
                    a = min(audit, key=lambda x: (x["external_score"], x["schedule_m3_s"]))
                    b = min(audit, key=lambda x: (x["response_score"], x["schedule_m3_s"]))
                    np.testing.assert_array_equal(planned.ontario_executable_release_m3_s, b["schedule_m3_s"])
                    if (capacity, delay) == REFERENCE:
                        np.testing.assert_allclose(a["schedule_m3_s"][0], record["ontario_applied_m3_s"],
                                                   atol=tol.flow_m3_s, rtol=0)
                    entry.update(
                        schedule_changed=a["schedule_m3_s"] != b["schedule_m3_s"],
                        first_action_changed=abs(a["schedule_m3_s"][0] - b["schedule_m3_s"][0]) > tol.flow_m3_s,
                        any_score_changed=any(abs(x["external_score"] - x["response_score"]) > 1e-9 for x in audit),
                        first_ottawa_release_span_m3_s=float(np.ptp([x["ottawa_release_m3_s"][0] for x in audit])),
                        max_routed_ottawa_span_m3_s=float(np.ptp([x["ottawa_arrival_m3_s"] for x in audit], axis=0).max()),
                        external_choice=a, response_choice=b,
                    )
                results.append(entry)
        state = model.advance_storage(state, record["ontario_applied_m3_s"], record["ontario_realized_supply_m3_s"],
                                      row["iceInd"], decision_time(row, k))
        release_history.append(record["ottawa_executed_release_m3_s"])
    return results


def closed_loop(model, cases, variants, costs):
    results = []
    for capacity, delay in variants:
        changed = [variant(case, capacity, delay) for case in cases]
        env = HardCoupledEnv(model, changed, costs, recovery_mode="peak_protected")
        for index, case in enumerate(changed):
            for arm in ARMS:
                env.reset(options={"case": index})
                done = False
                while not done:
                    _, _, done, _, _ = env.apply(arm)
                executed = [row for row in env.records if "ontario_applied_m3_s" in row]
                status = env.records[-1]["status"]
                complete = len(executed) == len(case["rows"]) and status == "executed"
                results.append({
                    "storage_maximum_m3": capacity, "routing_delay_periods": delay, "year": case["year"],
                    "arm": arm, "status": "completed" if complete else status,
                    "executed_periods": len(executed),
                    "cost": sum(row["cost"] for row in executed) if complete else None,
                    "upstream_exposure_m_periods": sum(max(0., row["ontario_end_m"] - costs["ontario_flood_cap_m"]) for row in executed),
                    "downstream_exposure_m_periods": sum(max(0., row["pointe_claire_m"] - row["downstream_target_m"]) for row in executed),
                    "max_storage_violation_m3": max((row["ottawa_storage_violation_m3"] for row in executed), default=0.),
                    "max_downstream_violation_m3_s": max((max(row["st_louis_flow_m3_s"] - row["capacity_m3_s"], 0.) for row in executed), default=0.),
                    "max_water_balance_residual_m3": max((abs(row["ottawa_system_balance_residual_m3"]) for row in executed), default=0.),
                    "terminal_transit_m3": executed[-1]["ottawa_end_transit_m3"] if complete else None,
                })
        print("closed loop", capacity, delay, flush=True)
    return results


def write_report(output, audit, closed, events):
    lines = [
        "# Ottawa storage and routing sensitivity", "",
        "External-ranked MPC with peak-protected recovery is the frozen research reference. "
        "This sensitivity changes only synthetic Ottawa usable storage and routing delay. "
        "The physical minimum stays at 1 billion m³. Frozen states retain actual level, storage, "
        "forecasts and completed releases. For other delays the transit queue is rebuilt from those "
        "releases. Zero delay routes the current action immediately.", "",
        "| Ottawa usable storage (billion m³) | Delay (periods) | Ranked states | No eligible candidate | Outside storage variant | Any Ottawa release variation | Different winning schedule | Different first Ontario release |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (cap, delay), group in audit.groupby(["storage_maximum_m3", "routing_delay_periods"]):
        ranked = group[group.status == "ranked"]
        statuses = Counter(group.status)
        lines.append(f"| {cap/1e9-1:g} | {delay} | {len(ranked)} | {statuses['no_common_eligible_candidate']} | "
                     f"{statuses['inherited_storage_outside_variant']} | "
                     f"{ranked.first_ottawa_release_span_m3_s.gt(1e-5).sum()} | "
                     f"{ranked.schedule_changed.sum()} | {ranked.first_action_changed.sum()} |")
    ranked = audit[audit.status == "ranked"]
    lines.extend(["", "The frozen check applies the same eligibility, scoring, reserve, peak check and fallback "
                  "order as the annual controller. It reproduces the archived external first action for "
                  "the reference configuration.", "",
                  f"At {ranked.any_score_changed.sum()} of {len(ranked)} ranked state–variant pairs, "
                  "using Ottawa's best response altered at least one candidate score.", ""])
    if closed:
        lines.extend(["## Separate closed-loop years", "",
                      "| Storage max / routing delay | External complete | Stackelberg complete | Common pairs | Cost delta | Ontario exposure delta | Lake St. Louis exposure delta | Realized violations |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        closed_df = pd.DataFrame(closed)
        for (cap, delay), group in closed_df.groupby(["storage_maximum_m3", "routing_delay_periods"]):
            complete = group[group.status == "completed"]
            columns = ["cost", "upstream_exposure_m_periods", "downstream_exposure_m_periods"]
            a, b = (complete[complete.arm == arm].set_index("year")[columns] for arm in ARMS)
            common = a.index.intersection(b.index)
            changes = [f"{value:+.5f}" for value in (b.loc[common] - a.loc[common]).mean()] if len(common) else ["—"] * 3
            violations = ((group.max_storage_violation_m3 > .01) | (group.max_downstream_violation_m3_s > 1e-5)).sum()
            lines.append(f"| {cap/1e9:g} billion m³ / {delay} | {len(a)} | {len(b)} | {len(common)} | "
                         + " | ".join([*changes, str(violations)]) + " |")
    else:
        lines.extend(["No variant changed the winning schedule, so the contingent closed-loop trial was unnecessary.", ""])
    lines.extend(["", "## Complete 2017 and 2019 replays with peak protection", "",
                  "| Year | Controller | Completed | Operating cost | Ontario shoreline (m·periods) | Lake St. Louis (m·periods) |",
                  "|---|---|---:|---:|---:|---:|"])
    for (year, arm), group in events.groupby(["year", "arm"], sort=False):
        complete = group[group.status == "completed"]
        metrics = complete[["full_cost", "upstream_exposure_m_periods", "downstream_exposure_m_periods"]].mean()
        lines.append(f"| {year} | {arm} | {len(complete)}/{len(group)} | "
                     + " | ".join(f"{x:.4f}" for x in metrics) + " |")
    lines.extend(["", "Ontario inputs are historical. Ottawa bounds, releases, and timing remain synthetic assumptions.", ""])
    (output / "report.md").write_text("\n".join(lines))


def plot_flood_years(traces, costs, output):
    figure, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey="row", layout="constrained")
    fields = ("ontario_end_m", "pointe_claire_m", "ontario_applied_m3_s")
    labels = ("Lake Ontario level (m)", "Pointe-Claire level (m)", "Ontario release (m³/s)")
    styles = {
        "rule_curve": ("Filtered Plan 2014", "#56677c", "-"),
        "external_mpc": ("External-ranked MPC", "#007d76", "-"),
        "stackelberg_mpc": ("Stackelberg MPC", "#142d45", ":"),
        "ppo": ("PPO (five-seed mean)", "#a3501e", "--"),
    }
    for column, year in enumerate((2017, 2019)):
        runs = traces[traces.year == year]
        for row, (field, ylabel) in enumerate(zip(fields, labels)):
            ax = axes[row, column]
            for arm, (label, color, style) in styles.items():
                series = runs[runs.arm == arm].groupby("period")[field].mean()
                ax.plot(series.index, series, style, color=color, linewidth=1.6, label=label)
            if row == 0:
                ax.set_title(str(year))
                ax.axhline(costs["ontario_flood_cap_m"], lw=1, color="#898989", linestyle="--", label="Shoreline threshold")
            if row == 1:
                target = runs[runs.arm == "rule_curve"].set_index("period").downstream_target_m
                ax.plot(target.index, target, lw=1, color="#898989", linestyle="--", label="Pointe-Claire target")
            if column == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=.17)
        axes[-1, column].set(xlabel="Quarter-month period", xlim=(1, 48))
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(handles=[axes[1, 0].lines[-1]], loc="upper left", fontsize=8)
    figure.suptitle("Ontario flood-year replays with peak-protected recovery")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = ArgumentParser(
        description="Check response-aware ranking across storage and one-step routing assumptions."
    )
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--playback", type=Path, required=True)
    args = parser.parse_args()
    design = load_settings(CONFIG, "ranking_sensitivity")
    protocol = load_protocol()
    annual = load_settings(CONFIG, "annual")
    costs = {**protocol, "incomplete_penalty": annual["incomplete_penalty"]}
    config = load_follower_config()
    reference = NativeReference()
    model = OntarioModel(reference)
    frame = load_inputs(reference)
    states = start_states(model, frame, sorted(set(design["snapshot_years"] + design["evaluation_years"])))
    frozen = annual_cases(frame, states, design["snapshot_years"], config, annual)
    root = args.archive_root
    events_dir = root / "final_recovery_peak_protected_events_v1"
    stress_dir = root / "final_recovery_peak_protected_confirmation_v1"
    events = pd.read_csv(events_dir / "runs.csv")
    traces = pd.read_csv(events_dir / "trajectories.csv")
    args.output.mkdir(parents=True, exist_ok=False)
    audits = []
    for case in frozen:
        rows = traces[(traces.year == case["year"]) & (traces.arm == "external_mpc")]
        audits.extend(frozen_rank(model, case, rows, design, costs))
    audit = pd.DataFrame(audits)
    audit.to_csv(args.output / "frozen_rankings.csv", index=False)
    winners = audit[audit.schedule_changed.fillna(False)].groupby(
        ["storage_maximum_m3", "routing_delay_periods"],
    ).size().sort_values(ascending=False, kind="stable")
    selected = list(winners.index[:design["maximum_paired_variants"]])
    closed = []
    if selected:
        evaluation = annual_cases(frame, states, design["evaluation_years"], config, annual)
        closed = closed_loop(model, evaluation, list(dict.fromkeys([REFERENCE, *selected])), costs)
        pd.DataFrame(closed).to_csv(args.output / "closed_loop.csv", index=False)
    write_report(args.output, audit, closed, events)
    plot_flood_years(traces, costs, args.output / "flood_years.png")
    playback(root, {"development": events, "stress": pd.read_csv(stress_dir / "runs.csv")},
             protocol["seeds"], args.playback,
             trajectory_paths={"development": events_dir / "trajectories.csv", "stress": stress_dir / "trajectories.csv"})
    sources = (CONFIG, events_dir / "trajectories.csv", events_dir / "runs.csv",
               stress_dir / "trajectories.csv", stress_dir / "runs.csv")
    (args.output / "settings.json").write_text(json.dumps({
        "design": design, "paired_variants": selected, "source_sha256": {str(p): digest(p) for p in sources},
    }, default=json_value, indent=2) + "\n")
    print((args.output / "report.md").read_text())


if __name__ == "__main__":
    main()
