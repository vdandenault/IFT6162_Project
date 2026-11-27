from dataclasses import replace
from itertools import product

import numpy as np

from reservoir_control.experiments.coupled_forecast import (
    ONTARIO_HORIZON,
    capped_follower,
    forecast_forcing,
    forecast_trajectory,
    native_step,
)
from reservoir_control.systems.loslr.coupled import f_limit_target, operating_cost
from reservoir_control.systems.loslr.physics import FLOW_SCALE
from reservoir_control.systems.ottawa.hard_follower import solve_hard_follower

OFFSETS = (0, -400, 400, -800, 800, -1200, 1200, -2000, 2000)


class CoupledPlanner:
    """Generate, screen, and rank releases from the current decision context."""

    def __init__(self, case, costs, tolerance, recovery_mode):
        self.case = case
        self.rows = case["rows"]
        self.config = case["config"]
        self.costs = costs
        self.tolerance = tolerance
        self.recovery_mode = recovery_mode

    def _schedules(self, context, problem, neutral, *, profiles=False):
        k, _, level, previous, ice, mechanism = context
        horizon = len(problem.duration_seconds)
        rows = self.rows[k : k + horizon]
        arrival = (
            problem.routing_matrix @ np.full(horizon, self.config["reference_release_m3_s"])
            + problem.in_transit_contribution_m3_s
        )
        n = min(horizon, ONTARIO_HORIZON)
        requests = [(neutral + offset, *([neutral] * (n - 1))) for offset in OFFSETS]
        if profiles:
            room = np.min(
                np.subtract(problem.downstream_capacity_m3_s, problem.other_downstream_m3_s)[
                    1 : n + 1
                ],
                initial=problem.downstream_capacity_m3_s[0] - problem.other_downstream_m3_s[0],
            )
            targets = [max(0.0, room - margin) for margin in (900, 1500, 2200, 2500)]
            starts = {max(0.0, neutral + offset) for offset in OFFSETS} | {previous * FLOW_SCALE}
            requests.extend(
                tuple(
                    min(first + 700 * j, max(target, first - ramp * j)) if j else first
                    for j in range(n)
                )
                for first, target, ramp in product(sorted(starts), targets, (400, 700))
            )
        tail = tuple(problem.ontario_executable_release_m3_s[n:])
        schedules = set()
        for request in requests:
            state = level, previous, ice
            window = []
            for j, requested in enumerate(request):
                non_ontario = arrival[j] + problem.other_downstream_m3_s[j]
                forcing = forecast_forcing(rows, j, non_ontario)
                room = problem.downstream_capacity_m3_s[j] - non_ontario
                flow = min(max(requested, 0.0), room) / FLOW_SCALE
                for _ in range(8):
                    if not 0 <= flow <= room / FLOW_SCALE + 1e-7:
                        break
                    end, allowed, _ = native_step(
                        forcing, forcing["ontNTS_QM1"], flow, state, mechanism
                    )
                    if abs(allowed - flow) * FLOW_SCALE <= self.tolerance.flow_m3_s:
                        window.append(float(allowed * FLOW_SCALE))
                        state = end, allowed, forcing["iceInd"]
                        break
                    flow = allowed
                else:
                    break
                if len(window) != j + 1:
                    break
            if len(window) == n:
                schedules.add((*window, *tail))
        return rows, arrival, sorted(schedules)

    def _screen(
        self, context, problem, requested, neutral, arm, *, profiles=False, candidate_audit=None
    ):
        k, _, level, previous, ice, mechanism = context
        rows, nominal, schedules = self._schedules(context, problem, neutral, profiles=profiles)
        state = level, previous, ice

        def forecast(schedule, arrivals):
            frame, _, _ = forecast_trajectory(
                rows,
                schedule,
                arrivals,
                problem.other_downstream_m3_s,
                problem.downstream_capacity_m3_s,
                state,
                mechanism,
                self.tolerance,
            )
            return frame

        def score(frame, schedule):
            start, prior, value = level, previous * FLOW_SCALE, 0.0
            excess = []
            for j, row in enumerate(frame):
                end, _, _ = native_step(
                    row,
                    row["ontNTS"],
                    schedule[j] / FLOW_SCALE,
                    (start, prior / FLOW_SCALE, ice),
                    mechanism,
                )
                value += operating_cost(
                    start,
                    end,
                    row["ptclaireLevel"],
                    schedule[j],
                    prior,
                    self.costs,
                    terminal=k + j + 1 == len(self.rows),
                )[0]
                if candidate_audit is not None:
                    excess.append(max(0.0, row["ptclaireLevel"] - f_limit_target(start)))
                start, prior = end, schedule[j]
            return value, excess

        ranked = []
        nominal_downstream = {}
        for schedule in schedules:
            if arm in ("rule_curve", "ppo"):
                ranked.append((abs(schedule[0] - requested), schedule))
            else:
                frame = forecast(schedule, nominal)
                if frame is not None:
                    value, excess = score(frame, schedule)
                    ranked.append((value, schedule))
                    if candidate_audit is not None:
                        nominal_downstream[schedule] = (
                            [row["ptclaireLevel"] for row in frame],
                            excess,
                        )
        best = None
        for key, schedule in sorted(ranked):
            capped = capped_follower(problem, rows, schedule, level)
            response = solve_hard_follower(capped, self.tolerance)
            if response.status == "infeasible":
                continue
            if not response.success:
                raise ArithmeticError(response.message)
            arrivals = (
                capped.routing_matrix @ response.release_m3_s + capped.in_transit_contribution_m3_s
            )
            frame = forecast(schedule, arrivals)
            if frame is None:
                continue
            if arm == "ppo" and candidate_audit is not None:
                candidate_audit.append({"schedule_m3_s": tuple(schedule)})
                if best is None:
                    best = (key, schedule), capped
                continue
            if arm != "stackelberg_mpc":
                return capped
            response_score, excess = score(frame, schedule)
            if candidate_audit is not None:
                scored_leads = len(frame)
                combined = np.asarray(schedule) + capped.other_downstream_m3_s + arrivals
                nominal_stage, nominal_excess = nominal_downstream[schedule]
                candidate_audit.append(
                    {
                        "schedule_m3_s": tuple(schedule),
                        "external_score": key,
                        "response_score": response_score,
                        "ottawa_release_m3_s": response.release_m3_s.tolist(),
                        "ottawa_arrival_m3_s": arrivals.tolist(),
                        "downstream_headroom_m3_s": (capped.downstream_capacity_m3_s - combined)[
                            :scored_leads
                        ].tolist(),
                        "downstream_excess_m": excess,
                        "nominal_downstream_excess_m": nominal_excess,
                        "downstream_stage_m": [row["ptclaireLevel"] for row in frame],
                        "nominal_downstream_stage_m": nominal_stage,
                    }
                )
            rank = response_score, schedule
            if best is None or rank < best[0]:
                best = rank, capped
        return None if best is None else best[1]

    def plan(self, context, arm, requested, neutral, *, guard_next, candidate_audit=None):
        """Select an executable schedule and report any recovery attempts."""
        k, base, *_ = context

        def guard(problem):
            upper = np.array(problem.storage_maximum_m3)
            upper[1:] -= 700.0 * problem.duration_seconds[0]
            return replace(problem, storage_maximum_m3=upper)

        def select(problem, *, profiles=False):
            if self.recovery_mode == "peak_protected" and k >= 2:
                known = self.case["scenario"]["realized_net_inflow_m3_s"][:k]
                older = known[max(0, k - 4) : -1]
                if known[-1] - max(older) >= 700.0:
                    # Protect the executable action against one period at
                    # the last observed peak. Only the first boundary executes.
                    extra = max(0.0, known[-1] - problem.net_inflow_m3_s[0])
                    upper = np.array(problem.storage_maximum_m3)
                    upper[1] = min(
                        upper[1], base.storage_maximum_m3[1] - extra * problem.duration_seconds[0]
                    )
                    if upper[1] < problem.storage_minimum_m3[1]:
                        return None
                    problem = replace(problem, storage_maximum_m3=upper)
            return self._screen(
                context,
                problem,
                requested,
                neutral,
                arm,
                profiles=profiles,
                candidate_audit=candidate_audit,
            )

        ordinary = guard(base) if guard_next else base
        choice = select(ordinary, profiles=self.recovery_mode == "proactive_profiles")
        if choice is not None:
            return choice, False, "ordinary", {}

        realized = self.case["scenario"]["realized_net_inflow_m3_s"]
        past = (
            realized[k - 1]
            if k
            else self.config["forecast_noise"]["initial_observed_inflow_estimate_m3_s"]
        )
        retry = replace(base, net_inflow_m3_s=np.full(len(base.net_inflow_m3_s), past))
        guarded_retry = guard(retry)
        choice = select(guarded_retry)
        if choice is not None or self.recovery_mode == "original":
            return choice, True, "guarded_persistence", {}

        options = [("reserve_released", retry, False)]
        if self.recovery_mode != "reserve":
            options.extend(
                [
                    ("wider_ordinary", ordinary, True),
                    ("wider_guarded_persistence", guarded_retry, True),
                    ("wider_physical_persistence", retry, True),
                ]
            )
        probes = {}
        for label, problem, profiles in options:
            candidate = select(problem, profiles=profiles)
            probes[label] = candidate is not None
            if candidate is not None and self.recovery_mode != "audit":
                return candidate, True, label, probes
        return None, True, "guarded_persistence", probes
