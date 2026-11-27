"""Finite Ontario schedule search with a shared Ottawa action across two paths."""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import ontario_stage_costs
from bilevel_reservoirs.solvers.compare_ottawa_response import (
    ONTARIO_HORIZON, candidates, capped_follower, delay_headroom_candidates, forecast_trajectory, limit_candidates,
)
from reservoir_control.systems.ottawa.hard_follower import solve_hard_follower


def first_release_interval(problem, tolerance):
    """Project feasible releases onto the first action for identity one-step routing."""
    dt = np.asarray(problem.duration_seconds)
    n = len(dt)
    if not np.array_equal(problem.routing_matrix, np.eye(n, k=-1)):
        raise ValueError("First-release intervals require unit one-period delayed routing")
    lower = np.asarray(problem.release_minimum_m3_s)
    upper = np.array(problem.release_maximum_m3_s)
    capacities = (np.asarray(problem.downstream_capacity_m3_s) - problem.ontario_executable_release_m3_s
                  - problem.other_downstream_m3_s - problem.in_transit_contribution_m3_s)
    if capacities[0] < -tolerance.flow_m3_s:
        return None
    # The final release remains in transit at the horizon boundary.
    upper[:-1] = np.minimum(upper[:-1], capacities[1:])
    if np.any(upper < lower):
        return None
    inflows = np.asarray(problem.net_inflow_m3_s)
    storage_min = np.asarray(problem.storage_minimum_m3)
    storage_max = np.asarray(problem.storage_maximum_m3)
    initial = problem.initial_storage_m3
    if not storage_min[0] - tolerance.storage_m3 <= initial <= storage_max[0] + tolerance.storage_m3:
        return None
    lo, hi = storage_min[-1], storage_max[-1]
    for j in range(n - 1, 0, -1):
        lo = max(storage_min[j], lo - dt[j] * (inflows[j] - lower[j]))
        hi = min(storage_max[j], hi - dt[j] * (inflows[j] - upper[j]))
        if lo > hi:
            return None
    lo_r = max(lower[0], inflows[0] + (initial - hi) / dt[0])
    hi_r = min(upper[0], inflows[0] + (initial - lo) / dt[0])
    return (float(lo_r), float(hi_r)) if lo_r <= hi_r else None


def fix_first_release(problem, release):
    lower = np.array(problem.release_minimum_m3_s)
    upper = np.array(problem.release_maximum_m3_s)
    lower[0] = upper[0] = release
    return replace(problem, release_minimum_m3_s=lower, release_maximum_m3_s=upper)


def gradual_schedules(problem, previous_flow):
    """Try gradual reductions toward targets derived from known arrival capacities."""
    anchor = problem.ontario_executable_release_m3_s
    n = min(len(anchor), ONTARIO_HORIZON)
    downstream_room = np.asarray(problem.downstream_capacity_m3_s) - problem.other_downstream_m3_s
    room = min(downstream_room[1:])  # Ottawa's first release arrives at the next lead.
    outlet = problem.release_maximum_m3_s[0]
    targets = {max(0., room - outlet + offset) for offset in (-500, -300, -150, 0, 200, 400, 700, 1000)}
    starts = {float(max(0, round(previous_flow) - drop)) for drop in range(0, 1451, 50)}
    return sorted({
        tuple(max(target, first - ramp * lead) for lead in range(n)) + tuple(anchor[n:])
        for first, ramp, target in product(starts, (200, 350, 500, 600, 700), targets)
        if target <= first
    })


def choose_two_paths(problem, high_forecast, rows, state, mechanism, reference_release, tolerance, audit,
                     *, extra_schedules=()):
    """Search branch schedules in the shared external Ontario ranking.

    Among admissible pairs, use the first by mean external Ontario score. The
    shared Ottawa action is the midpoint of the two unconstrained first
    best responses projected onto the common feasible interval. Each branch
    then solves the existing follower objective with the first action fixed.
    """
    tail = problem.ontario_executable_release_m3_s
    if len(tail) < 2:
        raise ValueError("Scenario adaptation requires at least two leads")
    high = np.asarray(high_forecast)[:len(tail)]
    if high[0] != problem.net_inflow_m3_s[0]:
        raise ValueError("This two-path formulation requires the same forecast at lead one")
    external = problem.routing_matrix @ np.full(len(tail), reference_release) + problem.in_transit_contribution_m3_s
    base_options = limit_candidates(rows, tail, problem, reference_release, state, mechanism, tolerance)
    grid_options = sorted(set(candidates(tail, "grid")) - set(base_options))
    attempted = []
    eligible = {}
    stages = ("primary", "expanded", "arrival_headroom")
    if extra_schedules:
        stages += ("gradual_reductions",)
    for stage in stages:
        options = (base_options if stage == "primary" else grid_options if stage == "expanded"
                   else delay_headroom_candidates(attempted) if stage == "arrival_headroom"
                   else sorted(set(extra_schedules) - {row["schedule_m3_s"] for row in attempted}))
        stage_entries, forecasts = [], []
        for schedule in options:
            predicted, status, _ = forecast_trajectory(
                rows, schedule, external, problem.other_downstream_m3_s, problem.downstream_capacity_m3_s,
                state, mechanism, tolerance,
            )
            attempted.append({"schedule_m3_s": schedule, "status": status})
            if predicted is None:
                continue
            low_problem = capped_follower(problem, rows, schedule, state[0])
            high_problem = replace(low_problem, net_inflow_m3_s=high)
            intervals = tuple(first_release_interval(branch, tolerance) for branch in (low_problem, high_problem))
            if any(intervals):
                stage_entries.append((schedule, (low_problem, high_problem), intervals))
                forecasts.extend(predicted)
        if stage_entries:
            scores = ontario_stage_costs(pd.DataFrame(forecasts).to_dict("list")).reshape(len(stage_entries), -1).sum(axis=1)
            for (schedule, path_problems, intervals), score in zip(stage_entries, scores, strict=True):
                eligible[schedule] = float(score), path_problems, intervals

        by_first = {}
        for schedule in eligible:
            by_first.setdefault(round(schedule[0], 6), []).append(schedule)
        pairs = []
        for schedules in by_first.values():
            low = [s for s in schedules if eligible[s][2][0] is not None]
            high_schedules = [s for s in schedules if eligible[s][2][1] is not None]
            for a, b in product(low, high_schedules):
                pa, pb = eligible[a], eligible[b]
                ia, ib = pa[2][0], pb[2][1]
                lo, hi = max(ia[0], ib[0]), min(ia[1], ib[1])
                if lo <= hi:
                    pairs.append((pa[0] + pb[0], a, b, lo, hi))
        audit.append({"stage": stage, "external_candidates": len(attempted),
                      "branch_eligible_candidates": len(eligible), "overlapping_pairs": len(pairs),
                      "receding_intervals": sum(item[2][0] is not None for item in eligible.values()),
                      "sustained_intervals": sum(item[2][1] is not None for item in eligible.values())})
        for score, a, b, lo, hi in sorted(pairs):
            branches = eligible[a][1][0], eligible[b][1][1]
            preferred = [solve_hard_follower(branch, tolerance) for branch in branches]
            if not all(response.success for response in preferred):
                raise RuntimeError(f"Follower solve failed after interval screening: {[r.message for r in preferred]}")
            first = float(np.clip(np.mean([response.first_release_m3_s for response in preferred]), lo, hi))
            fixed = [fix_first_release(branch, first) for branch in branches]
            responses = [solve_hard_follower(branch, tolerance) for branch in fixed]
            if not all(response.success for response in responses):
                raise RuntimeError(f"Shared-action follower solve failed: {[r.message for r in responses]}")
            validated = []
            for branch, response in zip(fixed, responses, strict=True):
                arrivals = branch.routing_matrix @ response.release_m3_s + branch.in_transit_contribution_m3_s
                _, status, _ = forecast_trajectory(
                    rows, branch.ontario_executable_release_m3_s, arrivals, branch.other_downstream_m3_s,
                    problem.downstream_capacity_m3_s, state, mechanism, tolerance,
                )
                validated.append(status)
            if all(status == "admissible" for status in validated):
                audit.append({
                    "selected": True, "stage": stage, "common_interval_m3_s": [lo, hi],
                    "ontario_schedules_m3_s": [a, b], "ottawa_first_release_m3_s": first,
                    "ottawa_schedules_m3_s": [response.release_m3_s for response in responses],
                    "storage_m3": [response.storage_m3 for response in responses],
                    "downstream_flow_m3_s": [response.downstream_flow_m3_s for response in responses],
                    "pending_transit_m3": [response.pending_planned_release_volume_m3 for response in responses],
                    "mean_external_ontario_score": score / 2,
                })
                return fixed[0]
    return None
