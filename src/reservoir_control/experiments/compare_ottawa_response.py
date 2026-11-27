"""Compare Ottawa-blind and response-aware Ontario MPC on one synthetic replay."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.operator import ONTARIO_COST_COMPONENTS, ontario_stage_costs
from reservoir_control.systems.loslr.simulator import LoslrEnv, round_d
from reservoir_control.systems.loslr.physics import CONVERSION, FLOW_SCALE
from reservoir_control.experiments.fixed_schedule_replay import (
    CONFIG, FollowerTolerances, digest, json_value, replay, solve_hard_follower,
)

ONTARIO_HORIZON = 4
COMPACT_OFFSETS = (-300, -200, -100, -50, 50, 100, 200, 300)
GRID_OFFSETS = (-600, -300, -150, 0, 150, 300)


def candidates(reference, family):
    """Vary the Ontario window while holding the Ottawa tail fixed."""
    base = np.asarray(reference, dtype=float)
    n = min(ONTARIO_HORIZON, len(base))
    if family == "grid":
        return [
            tuple(float(value) for value in (*window, *base[n:]))
            for window in product(*(base[i] + np.array(GRID_OFFSETS) for i in range(n)))
            if min(window) >= 0
        ]
    options = {tuple(base)}
    for offset in COMPACT_OFFSETS:
        first = base.copy()
        first[0] += offset
        window = base.copy()
        window[:n] += offset
        options.update((tuple(first), tuple(window)))
    return [option for option in sorted(options) if min(option) >= 0]


def delay_headroom_candidates(attempted):
    """Lower the Ontario commitment where the first Ottawa release arrives."""
    existing = {entry["schedule_m3_s"] for entry in attempted}
    admissible = {entry["schedule_m3_s"] for entry in attempted if entry["status"] == "admissible"}
    return sorted({
        (schedule[0], schedule[1] - reduction, *schedule[2:])
        for schedule in admissible if len(schedule) >= 2
        for reduction in (100, 200, 250, 300, 400, 500, 600, 700)
        if schedule[1] >= reduction
    }.difference(existing))


def nearby_schedules(anchor, attempted, budget, *, headroom=None, pressure=None):
    """Choose a bounded set of perturbations to the controllable Ontario leads."""
    n = min(len(anchor), ONTARIO_HORIZON)
    base = np.asarray(anchor)
    # Native limit projections can leave roundoff in mathematically equal flows.
    seen = {tuple(round(float(flow), 6) for flow in schedule) for schedule in attempted}
    options = {}
    for offset in (100, 250, 400, 50, 150, 200, 300, 350, 450, 500, 550, 600, 650, 700):
        for lead in range(n):
            for direction in (1, -1):
                move = np.zeros(len(anchor))
                move[lead] = direction * offset
                for other in (None, *range(n)):
                    if other == lead:
                        continue
                    delta = move.copy()
                    if other is not None:
                        delta[other] = -delta[lead]
                    schedule = tuple(base + delta)
                    key = tuple(round(float(flow), 6) for flow in schedule)
                    if min(schedule) >= 0 and key not in seen:
                        options[schedule] = delta
                        seen.add(key)
    # Compare every small one-coordinate move before the equal-sized exchanges;
    # Python's stable sort preserves the balanced lead/sign enumeration.
    ordered = sorted(options, key=lambda schedule: (np.abs(options[schedule]).max(),
                                                     np.count_nonzero(options[schedule])))
    if headroom is not None:
        def headroom_gain(schedule):
            delta = options[schedule]
            relief = np.minimum(np.maximum(-delta, 0.0), pressure).sum()
            extra_conflict = np.maximum(delta - headroom, 0.0).sum()
            return float(relief - extra_conflict)

        ordered.sort(key=headroom_gain, reverse=True)
    return ordered[:budget]


def native_step(forcing, supply, flow, state, limits):
    """Use the same forecast-level and native-limit check as fixed execution."""
    level, previous_flow, previous_ice = state
    end = level + round_d((supply / FLOW_SCALE - flow) / CONVERSION, 6)
    allowed, regime = limits.constrain(
        forcing, level, end, flow, "RF",
        previous_flow=previous_flow, previous_ice=previous_ice,
    )
    return end, allowed, regime


def forecast_forcing(rows, lead, non_ontario, future_roughness=None):
    issued = rows[0]
    supply = issued[f"ontNTS_QM{lead + 1}"]
    # Only the calendar advances. Other exogenous inputs persist from issue time.
    return {**issued, **{name: rows[lead][name] for name in ("Sim", "Year", "Month", "QM")},
            "ontNTS": supply, "ontNTS_QM1": supply,
            "ptclaireR": future_roughness if lead > 0 and future_roughness is not None else issued["ptclaireR"],
            "stlouisontOut": non_ontario, "slonFlow_QM1": non_ontario}


# Generate candidates from several Ottawa arrival margins with the same rule
# for both arms. Only future releases are uncertain. The transit queue is known.
ARRIVAL_MARGINS_M3_S = (0, 150, 300, 450, 600)
NATIVE_RELEASE_BIASES_M3_S = (-600, -300, -100, 0, 100, 300, 600)


# Beginning-of-period level tiers from the pinned native Bv7 F limit.
F_LIMIT_TIERS = (
    (75.30, 22.10, 11523.848),
    (75.37, 22.20, 11885.486),
    (75.50, 22.33, 12362.610),
    (75.60, 22.40, 12622.784),
    (float("inf"), 22.48, 12922.906),
)


def forecast_f_ceiling(level, roughness, commitment):
    """Total St. Louis flow before F would lower an otherwise admissible release."""
    _, action_level, coefficient = next(tier for tier in F_LIMIT_TIERS if level < tier[0])
    # Bv7 lowers the commitment only when both the rounded stage and the
    # rounded allowed release cross their thresholds.
    stage_ceiling = 604.0 / roughness * (action_level + 0.005 - 16.57) ** (1 / 0.58)
    rounded_release_ceiling = (
        coefficient / roughness + commitment
        - FLOW_SCALE * (ceil(commitment / FLOW_SCALE) - 0.5)
    )
    # Stay inside the boundary regardless of the native half-tie rule and LP tolerance.
    return max(stage_ceiling, rounded_release_ceiling) - 1e-4


def capped_follower(problem, rows, schedule, level, future_roughness=None):
    """Apply candidate-specific forecast F ceilings to Ontario's available leads."""
    caps = np.array(problem.downstream_capacity_m3_s, dtype=float)
    for lead, commitment in enumerate(schedule[:ONTARIO_HORIZON]):
        forcing = forecast_forcing(rows, lead, 0.0, future_roughness)
        caps[lead] = min(caps[lead], forecast_f_ceiling(level, forcing["ptclaireR"], commitment))
        level += round_d((forcing["ontNTS_QM1"] / FLOW_SCALE - commitment / FLOW_SCALE) / CONVERSION, 6)
    return replace(problem, ontario_executable_release_m3_s=schedule, downstream_capacity_m3_s=caps)


def limit_candidates(rows, reference, problem, reference_ottawa, state, limits, tolerance, future_roughness=None):
    """Build full windows by projecting each lead and advancing its forecast state."""
    options = set(candidates(reference, "compact"))
    horizon = min(len(reference), ONTARIO_HORIZON)
    for margin in ARRIVAL_MARGINS_M3_S:
        arrivals = (problem.routing_matrix @ np.full(len(reference), reference_ottawa + margin)
                    + problem.in_transit_contribution_m3_s)
        for bias in NATIVE_RELEASE_BIASES_M3_S:
            current = state
            window = []
            for lead in range(horizon):
                non_ontario = problem.other_downstream_m3_s[lead] + arrivals[lead]
                forcing = forecast_forcing(rows, lead, non_ontario, future_roughness)
                supply = forcing["ontNTS_QM1"]
                headroom = (problem.downstream_capacity_m3_s[lead] - non_ontario) / FLOW_SCALE
                flow = min((reference[lead] + bias) / FLOW_SCALE, headroom)
                projected = None
                for _ in range(12):
                    if not 0 <= flow <= headroom + tolerance.flow_m3_s / FLOW_SCALE:
                        break
                    end, allowed, _ = native_step(forcing, supply, flow, current, limits)
                    if abs(allowed - flow) * FLOW_SCALE <= tolerance.flow_m3_s:
                        projected = (end, flow, forcing["iceInd"])
                        break
                    flow = allowed
                if projected is None:
                    break
                current = projected
                window.append(float(flow * FLOW_SCALE))
            if len(window) == horizon:
                options.add(tuple(window) + tuple(float(value) for value in reference[horizon:]))
    return sorted(options)


def forecast_trajectory(rows, schedule, arrivals, background, caps, state, limits, tolerance, future_roughness=None):
    """Evaluate only the four leads available to Ontario for native limits and cost."""
    level, previous_flow, previous_ice = state
    predicted = []
    combined = np.asarray(schedule) + np.asarray(background) + arrivals
    breaches = np.flatnonzero(combined > np.asarray(caps) + tolerance.flow_m3_s)
    if len(breaches):
        lead = int(breaches[0])
        return None, f"study_cap_lead_{lead + 1}", {
            "rejected_combined_m3_s": float(combined[lead]),
            "rejected_cap_m3_s": caps[lead],
        }
    for lead in range(min(len(rows), ONTARIO_HORIZON)):
        flow = schedule[lead] / FLOW_SCALE
        non_ontario = background[lead] + arrivals[lead]
        forcing = forecast_forcing(rows, lead, non_ontario, future_roughness)
        supply = forcing["ontNTS_QM1"]
        end, allowed, regime = native_step(
            forcing, supply, flow, (level, previous_flow, previous_ice), limits,
        )
        if abs(allowed - flow) * FLOW_SCALE > tolerance.flow_m3_s:
            return None, f"native_{regime}_lead_{lead + 1}", {
                "rejected_commitment_m3_s": schedule[lead],
                "native_allowed_m3_s": allowed * FLOW_SCALE,
                "rejected_forecast_non_ontario_m3_s": non_ontario,
                "rejected_forecast_ontario_supply_m3_s": supply,
            }
        routed = LoslrEnv.lrstep(level, end, flow, {**forcing, "slonFlow": non_ontario,
                                                    "jetty1R": forcing["varennesR"]})
        predicted.append({**forcing, **routed, "ontLevel": (level + end) / 2, "ontFlow": flow})
        level, previous_flow, previous_ice = end, flow, forcing["iceInd"]
    return predicted, "admissible", {}


def run_policy(trajectory, config, scenario, policy, audit, family, *,
               align_f=False, align_from=1, roughness_forecast="persistence",
               ottawa_forecast_bank=None, recovery=None, recovery_guard_periods=0,
               replan_guard=None, guard_at_periods=(), delay_headroom_search=False, delay_headroom_on_exhaustion=False,
               matched_response_ranking=False, proposal_search=None, proposal_budget=12, scenario_decisions=None,
               previous_flow=595.0, previous_ice=2.0):
    if proposal_search not in (None, "generic", "guided") or (
        proposal_search is not None and (
            policy != "external_screened" or not matched_response_ranking or recovery is None or proposal_budget <= 0
        )
    ):
        raise ValueError("Extra candidate search requires a matched external ranking and positive budget")
    if matched_response_ranking and (policy not in ("external_screened", "anticipating") or not align_f):
        raise ValueError("Matched ranking requires a screened policy and aligned F caps")
    if recovery not in (None, "expand", "persistence"):
        raise ValueError("Unknown candidate recovery mode")
    if recovery is not None and (
        (policy != "external_screened" and not matched_response_ranking) or ottawa_forecast_bank is None
    ):
        raise ValueError("Recovery requires a screened policy and an Ottawa forecast bank")
    if recovery_guard_periods not in (0, 1, 2) or (recovery_guard_periods and recovery != "persistence"):
        raise ValueError("A one- or two-period storage check requires persistence recovery")
    if replan_guard not in (None, "after_retry", "always", "tight_headroom") or (
        replan_guard is not None and recovery_guard_periods != 1
    ):
        raise ValueError("A replan check requires the one-period persistence recovery check")
    if guard_at_periods and (replan_guard != "after_retry" or min(guard_at_periods) < 1):
        raise ValueError("Forced guard periods supplement the after-retry guard and use one-based indices")
    if delay_headroom_search and (replan_guard is None or scenario["routing_delay_periods"] != 1):
        raise ValueError("The delay headroom search requires guarded replanning and a one-period delay")
    if delay_headroom_on_exhaustion and (not delay_headroom_search or recovery != "persistence"):
        raise ValueError("The broader trigger requires delay headroom search and persistence recovery")
    rows = trajectory.to_dict("records")
    tolerance = FollowerTolerances(**config["acceptance"])
    retry_at = None

    def reserve_storage(problem, periods):
        # Extra inflow in the checked intervals persists in each later boundary.
        upper = np.array(problem.storage_maximum_m3, dtype=float)
        durations = np.asarray(problem.duration_seconds)
        stress = config["forecast_noise"]["standard_deviation_m3_s"]
        upper[1:] -= stress * np.cumsum(durations * (np.arange(len(durations)) < periods))
        if np.any(upper < problem.storage_minimum_m3):
            return None
        return replace(problem, storage_maximum_m3=upper)

    def search(k, problem, level, previous_flow, previous_ice, mechanism, stage, *, guarded=False, proposals=None):
        reference = problem.ontario_executable_release_m3_s
        routing = problem.routing_matrix
        carry = problem.in_transit_contribution_m3_s
        external = routing @ np.full(len(reference), config["reference_release_m3_s"]) + carry
        admissible, forecasts = [], []
        window = rows[k:k + len(reference)]
        state = (level, previous_flow, previous_ice)
        use_f_cap = align_f and k + 1 >= align_from
        future_roughness = (
            float(np.mean([row["ptclaireR"] for row in rows[max(0, k - 3):k + 1]]))
            if roughness_forecast == "trailing_mean" else None
        )
        options = (
            limit_candidates(window, reference, problem, config["reference_release_m3_s"],
                             state, mechanism, tolerance, future_roughness)
            if family == "limits" else candidates(reference, family)
        )

        def follower_problem(schedule):
            if use_f_cap:
                return capped_follower(problem, window, schedule, level, future_roughness)
            return replace(problem, ontario_executable_release_m3_s=schedule)

        if proposals is not None:
            options = proposals
        elif stage == "expanded":
            options = sorted(set(candidates(reference, "grid")).difference(options))
        elif stage == "persistence_retry":
            options = sorted(set(options).union(candidates(reference, "grid")))
        elif stage in ("delay_headroom", "delay_headroom_retry"):
            options = delay_headroom_candidates([
                entry for entry in audit if entry["period"] == k + 1
                and entry["recovery_stage"] in ("primary", "expanded", "persistence_retry")
            ])
        for schedule in options:
            entry = {"policy": policy, "period": k + 1, "schedule_m3_s": schedule, "selected": False}
            if proposals is not None:
                entry["extra_proposal"] = proposal_search
            if recovery is not None:
                entry["recovery_stage"] = stage
            if guarded:
                entry["recovery_storage_maximum_m3"] = list(problem.storage_maximum_m3)
            audit.append(entry)
            selected_problem = follower_problem(schedule)
            if use_f_cap:
                entry["follower_capacity_m3_s"] = selected_problem.downstream_capacity_m3_s.tolist()
            if policy == "anticipating" and not matched_response_ranking:
                response = solve_hard_follower(selected_problem, tolerance)
                entry["ottawa_candidate_status"] = response.status
                if not response.success:
                    entry.update(status=f"ottawa_{response.status}", detail=response.message)
                    continue
                arrivals = routing @ response.release_m3_s + carry
            else:
                arrivals = external
            predicted, status, detail = forecast_trajectory(
                window, schedule, arrivals, problem.other_downstream_m3_s,
                problem.downstream_capacity_m3_s, state,
                mechanism, tolerance, future_roughness,
            )
            entry.update(status=status, predicted_ottawa_arrival_m3_s=arrivals.tolist(), **detail)
            if predicted is not None:
                admissible.append(entry)
                forecasts.extend(predicted)
        if not admissible:
            return None
        components = ontario_stage_costs(pd.DataFrame(forecasts).to_dict("list"))
        if not np.isfinite(components).all():
            raise ArithmeticError("Nonfinite Ontario candidate cost")
        scores = components.reshape(len(admissible), -1).sum(axis=1)
        for entry, score in zip(admissible, scores, strict=True):
            entry["score"] = float(score)
        ranked = sorted(admissible, key=lambda entry: (entry["score"], entry["schedule_m3_s"]))
        response_ranked, response_forecasts = [], []
        for rank, entry in enumerate(ranked, start=1):
            selected_problem = follower_problem(entry["schedule_m3_s"])
            if policy == "external_screened" or matched_response_ranking:
                response = solve_hard_follower(selected_problem, tolerance)
                entry["ottawa_screen_rank"] = rank
                entry["ottawa_candidate_status"] = response.status
                if response.status == "infeasible":
                    continue
                if not response.success:
                    raise RuntimeError(f"Ottawa screening failed at period {k + 1}: {response.message}")
                if matched_response_ranking or stage in ("delay_headroom", "delay_headroom_retry"):
                    planned_arrivals = routing @ response.release_m3_s + carry
                    predicted, status, _ = forecast_trajectory(
                        window, entry["schedule_m3_s"], planned_arrivals,
                        problem.other_downstream_m3_s, problem.downstream_capacity_m3_s,
                        state, mechanism, tolerance, future_roughness,
                    )
                    entry["native_status_with_ottawa_plan"] = status
                    if predicted is None:
                        continue
                    if matched_response_ranking and policy == "anticipating":
                        entry["best_response_arrival_m3_s"] = planned_arrivals.tolist()
                        response_ranked.append(entry)
                        response_forecasts.extend(predicted)
                        continue
            if proposal_search is not None and proposals is None:
                attempted = {row["schedule_m3_s"] for row in audit if row["period"] == k + 1}
                margins = {}
                if proposal_search == "guided":
                    headroom = np.maximum(selected_problem.downstream_capacity_m3_s
                                          - response.downstream_flow_m3_s, 0.0)
                    drawdown = np.maximum(
                        (response.storage_m3[1:] - selected_problem.storage_target_m3[1:])
                        / selected_problem.duration_seconds, 0.0,
                    )
                    outlet_room = np.maximum(selected_problem.release_maximum_m3_s - response.release_m3_s, 0.0)
                    pressure = np.maximum(routing @ np.minimum(drawdown, outlet_room) - headroom, 0.0)
                    margins = {"headroom": headroom, "pressure": pressure}
                extra = nearby_schedules(entry["schedule_m3_s"], attempted, proposal_budget, **margins)
                alternative = search(k, problem, *state, mechanism, stage, guarded=guarded, proposals=extra)
                if alternative is not None:
                    extra_choice = audit[-len(extra):]
                    candidate = next(row for row in extra_choice if row["selected"])
                    challenger = candidate["score"], candidate["schedule_m3_s"]
                    incumbent = entry["score"], entry["schedule_m3_s"]
                    if challenger < incumbent:
                        return alternative
                    candidate["selected"] = False
                    candidate.pop("selected_rank")
            entry["selected"] = True
            entry["selected_rank"] = rank
            return selected_problem
        if response_ranked:
            # The external score and eligibility gate are shared. Only the
            # anticipating arm replaces the score with the routed best response.
            components = ontario_stage_costs(pd.DataFrame(response_forecasts).to_dict("list"))
            if not np.isfinite(components).all():
                raise ArithmeticError("Nonfinite Ontario best-response candidate cost")
            scores = components.reshape(len(response_ranked), -1).sum(axis=1)
            for entry, score in zip(response_ranked, scores, strict=True):
                entry["best_response_score"] = float(score)
            entry = min(response_ranked, key=lambda row: (row["best_response_score"], row["schedule_m3_s"]))
            entry.update(selected=True, selected_rank=1)
            return follower_problem(entry["schedule_m3_s"])
        return None

    def choose(k, problem, level, previous_flow, previous_ice, mechanism):
        nonlocal retry_at
        state = (level, previous_flow, previous_ice, mechanism)
        check_replan = replan_guard == "always" or (
            replan_guard in ("after_retry", "tight_headroom") and retry_at is not None and k == retry_at + 1
        )
        if replan_guard == "tight_headroom":
            dt = problem.duration_seconds[0]
            forecast_fill = dt * max(problem.net_inflow_m3_s[0] - config["reference_release_m3_s"], 0.0)
            stress_volume = dt * config["forecast_noise"]["standard_deviation_m3_s"]
            room = problem.storage_maximum_m3[1] - problem.initial_storage_m3
            check_replan = check_replan or room - forecast_fill <= stress_volume
        check_replan = check_replan or k + 1 in guard_at_periods
        ordinary = reserve_storage(problem, 1) if check_replan else problem
        if ordinary is None:
            return None
        if scenario_decisions is not None and k in scenario_decisions:
            return scenario_decisions[k](k, ordinary, *state)
        result = search(k, ordinary, *state, "primary", guarded=check_replan)
        if result is not None or recovery is None:
            return result
        result = search(k, ordinary, *state, "expanded", guarded=check_replan)
        if result is not None or recovery == "expand":
            return result
        try_headroom = delay_headroom_search and len(problem.duration_seconds) >= 2
        if try_headroom and (check_replan or delay_headroom_on_exhaustion):
            result = search(k, ordinary, *state, "delay_headroom", guarded=check_replan)
            if result is not None:
                return result
        # Only the previous completed interval is visible at this decision.
        previous_inflow = (
            scenario["realized_net_inflow_m3_s"][k - 1] if k
            else config["forecast_noise"]["initial_observed_inflow_estimate_m3_s"]
        )
        retry = replace(problem, net_inflow_m3_s=np.full(len(problem.net_inflow_m3_s), previous_inflow))
        if recovery_guard_periods:
            retry = reserve_storage(retry, recovery_guard_periods)
            if retry is None:
                return None
        result = search(k, retry, *state, "persistence_retry", guarded=bool(recovery_guard_periods))
        if result is None and try_headroom and delay_headroom_on_exhaustion:
            result = search(k, retry, *state, "delay_headroom_retry", guarded=bool(recovery_guard_periods))
        if result is not None:
            retry_at = k
        return result

    return replay(trajectory, config, scenario, previous_flow=previous_flow, previous_ice=previous_ice,
                  choose_schedule=choose, ottawa_forecast_bank=ottawa_forecast_bank)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True, help="Original H4 trajectory with forecast leads")
    parser.add_argument("--fixed-schedule", type=Path, required=True)
    parser.add_argument("--candidates", choices=("compact", "grid", "limits"), default="grid")
    parser.add_argument("--align-f", action="store_true",
                        help="Constrain Ottawa arrivals by Ontario's forecast F ceiling")
    parser.add_argument("--screen-external", action="store_true",
                        help="Also run external ranking with an Ottawa feasibility screen")
    parser.add_argument("--align-from", type=int, default=1,
                        help="First period using --align-f (default: 1)")
    parser.add_argument("--roughness-forecast", choices=("persistence", "trailing_mean"),
                        default="persistence", help="Pointe-Claire forecast for leads 2–4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    trajectory = pd.read_csv(args.trajectory)
    fixed = pd.read_csv(args.fixed_schedule)
    if not trajectory[["Year", "QM"]].equals(fixed[["Year", "QM"]]):
        parser.error("The Ontario trajectory and fixed schedule must have matching dates")
    if not 1 <= args.align_from <= len(trajectory) or (args.align_from != 1 and not args.align_f):
        parser.error("--align-from requires --align-f and a period inside the trajectory")
    trajectory["ontFlow"] = fixed["ontFlow"]
    config = json.loads(CONFIG.read_text())
    scenario = next(case for case in config["scenarios"] if case["id"] == "one_period_delay_feasible")
    audit, runs, plans = [], {}, {}
    policies = ("external", "external_screened", "anticipating") if args.screen_external else (
        "external", "anticipating"
    )
    for policy in policies:
        runs[policy], plans[policy] = run_policy(
            trajectory, config, scenario, policy, audit, args.candidates,
            align_f=args.align_f, align_from=args.align_from, roughness_forecast=args.roughness_forecast,
        )
    executed = {key: [r for r in run if r["status"] == "executed"] for key, run in runs.items()}
    matched = min(map(len, executed.values()))
    summary = {
        "scenario": scenario["id"],
        "matched_executed_periods": matched,
        "decision_setting": {
            "ontario_horizon": ONTARIO_HORIZON, "ottawa_horizon": 5,
            "ottawa_fifth_release": "same reference schedule tail for every candidate",
            "external_forecast": "route reference Ottawa release of 900 m3/s plus current queue",
            "ottawa_inflow_forecast": "perfect synthetic inflows",
            "ranking": "existing unweighted sum of native Ontario indicators",
            "candidate_family": args.candidates,
            "ottawa_forecast_f_cap": args.align_f,
            "align_from_period": args.align_from if args.align_f else None,
            "pointe_claire_roughness_forecast": args.roughness_forecast,
            "candidate_offsets_m3_s": GRID_OFFSETS if args.candidates == "grid" else COMPACT_OFFSETS,
            **({"generator_arrival_margins_m3_s": ARRIVAL_MARGINS_M3_S,
                "generator_release_biases_m3_s": NATIVE_RELEASE_BIASES_M3_S}
               if args.candidates == "limits" else {}),
        },
        "input_sha256": {str(path): digest(path) for path in (args.trajectory, args.fixed_schedule, CONFIG, Path(__file__))},
        "policies": {},
    }
    for policy, records in executed.items():
        stopped = runs[policy][-1] if len(records) != len(trajectory) else None
        failed_candidates = (
            Counter(entry["status"] for entry in audit
                    if entry["policy"] == policy and entry["period"] == stopped["period"])
            if stopped else {}
        )
        summary["policies"][policy] = {
            "status": stopped["status"] if stopped else "completed",
            "executed_periods": len(records),
            "candidate_ottawa_solves": sum(
                entry["policy"] == policy and "ottawa_candidate_status" in entry for entry in audit
            ),
            "first_conflict": stopped,
            "candidate_outcomes_at_stop": failed_candidates,
            "matched_weighted_cost": sum(sum(r[name] for name in ONTARIO_COST_COMPONENTS)
                                         for r in records[:matched]) if matched else None,
            "minimum_st_louis_headroom_m3_s": min(
                (r["capacity_m3_s"] - r["st_louis_flow_m3_s"] for r in records), default=None,
            ),
            "terminal_ottawa_transit_m3": (
                records[-1]["ottawa_end_transit_m3"] if records else sum(scenario["initial_transit_volumes_m3"])
            ),
            "peak_pointe_claire_m": max((r["pointe_claire_m"] for r in records), default=None),
            "matched_cost_components": {
                name: sum(r[name] for r in records[:matched]) for name in ONTARIO_COST_COMPONENTS
            } if matched else None,
        }
    if args.screen_external:
        screened, anticipating = executed["external_screened"], executed["anticipating"]
        paired_periods = min(len(screened), len(anticipating))
        summary["screened_vs_anticipating"] = {
            "matched_periods": paired_periods,
            "different_ontario_releases": sum(
                abs(a["ontario_applied_m3_s"] - b["ontario_applied_m3_s"]) > config["acceptance"]["flow_m3_s"]
                for a, b in zip(screened[:paired_periods], anticipating[:paired_periods], strict=True)
            ),
            "cost_components": {
                policy: {name: sum(row[name] for row in executed[policy][:paired_periods])
                         for name in ONTARIO_COST_COMPONENTS}
                for policy in ("external_screened", "anticipating")
            } if paired_periods else None,
        }
    args.output.mkdir(parents=True, exist_ok=False)
    for policy, trace in runs.items():
        pd.DataFrame(trace).to_csv(args.output / f"{policy}_trace.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output / "candidates.csv", index=False)
    for filename, value in (("summary.json", summary), ("ottawa_plans.json", plans)):
        (args.output / filename).write_text(json.dumps(value, indent=2, default=json_value) + "\n")
    for policy, info in summary["policies"].items():
        print(f"{policy}: {info['status']}, {info['executed_periods']}/{len(trajectory)} periods")
    print(f"Matched periods: {matched}. Results in {args.output}")


if __name__ == "__main__":
    main()
