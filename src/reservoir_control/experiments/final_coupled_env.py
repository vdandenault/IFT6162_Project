from dataclasses import replace
from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from reservoir_control.experiments.coupled_forecast import (
    ONTARIO_HORIZON,
    capped_follower,
    forecast_forcing,
    forecast_trajectory,
    native_step,
)
from reservoir_control.experiments.fixed_schedule_replay import replay_steps
from reservoir_control.systems.loslr.adapter import decision_time
from reservoir_control.systems.loslr.coupled import f_limit_target, operating_cost
from reservoir_control.systems.loslr.data import SupplyForecast
from reservoir_control.systems.loslr.model import Plan2014
from reservoir_control.systems.loslr.physics import FLOW_SCALE
from reservoir_control.systems.ottawa.hard_follower import FollowerTolerances, solve_hard_follower

OFFSETS = (0, -400, 400, -800, 800, -1200, 1200, -2000, 2000)
OBSERVATION_DIM = 77


class HardCoupledEnv(gym.Env):
    """PPO submits a rule-curve residual through the same finite execution filter."""

    def __init__(self, model, cases, costs, *, native_costs=False, recovery_mode="original"):
        self.model = model
        self.rule = Plan2014(model.reference)
        self.cases = cases
        self.costs = costs
        self.native_costs = native_costs
        if recovery_mode not in (
            "original",
            "audit",
            "reserve",
            "profiles",
            "proactive_profiles",
            "peak_protected",
        ):
            raise ValueError("Unknown shared recovery mode")
        self.recovery_mode = recovery_mode
        self.action_space = Box(-2000.0, 2000.0, (1,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, (OBSERVATION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("case")
        if index is None:
            index = int(self.np_random.integers(len(self.cases)))
        self.case = self.cases[index]
        self.rows = self.case["rows"]
        self.state = self.case["start"]
        self.config = self.case["config"]
        self.tolerance = FollowerTolerances(**self.config["acceptance"])
        self.records, self.plans = [], []
        self.guard_next = False
        self.generator = replay_steps(
            self.case["trajectory"],
            self.config,
            self.case["scenario"],
            previous_flow=self.state.previous_release_m3_s / FLOW_SCALE,
            previous_ice=self.state.previous_ice,
            ottawa_forecast_bank=self.case["bank"],
            records=self.records,
            plans=self.plans,
            scored=self.native_costs,
            expected_periods=len(self.rows),
        )
        self.context = next(self.generator)
        return self.observation(), {"case": self.case["name"]}

    def observation(self, *, neutral=None):
        k, problem, level, previous, ice, _ = self.context
        row = self.rows[k]
        s = self.state
        arrival = problem.in_transit_contribution_m3_s[0]
        supply = [row[f"ontNTS_QM{j}"] for j in range(1, 5)]
        values = [
            level - 74.8,
            previous / 1000,
            ice / 2,
            row["iceInd"] / 2,
            np.sin(2 * np.pi * (row["QM"] - 1) / 48),
            np.cos(2 * np.pi * (row["QM"] - 1) / 48),
            problem.initial_storage_m3 / 5e9,
            problem.previous_release_m3_s / 2500,
            arrival / 2500,
            row["ptclaireR"],
            row["longsaultR"],
            row["forNTS"] / 10000,
            (len(self.rows) - k) / 48,
            float(self.guard_next),
            f_limit_target(level) - 22.1,
            (self.neutral_request() if neutral is None else neutral) / 10000,
            *(value / 10000 for value in supply),
            *(
                np.pad(problem.net_inflow_m3_s, (0, 5 - len(problem.net_inflow_m3_s)), mode="edge")
                / 2500
            ),
            *(value - 74.8 for value in s.mean_level_history_m),
            s.september_level_m - 74.8 if np.isfinite(s.september_level_m) else 0.0,
            s.september_release_m3_s / 10000 if np.isfinite(s.september_release_m3_s) else 0.0,
            problem.downstream_capacity_m3_s[0] / 12000,
            problem.ontario_executable_release_m3_s[-1] / 10000,
        ]
        # 16 current features, four supply leads, five Ottawa leads, 48
        # preceding levels and four seasonal/constraint features.
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise ArithmeticError(
                f"Invalid controller observation shape/values: {observation.shape}"
            )
        return observation

    def neutral_request(self):
        k, problem, _, _, _, _ = self.context
        row = self.rows[k]
        forecast = SupplyForecast(
            tuple(row[f"ontNTS_QM{j}"] for j in range(1, 5)),
            row["forNTS"],
            problem.other_downstream_m3_s[0] + problem.in_transit_contribution_m3_s[0],
            "Past-only paired study forecast",
        )
        return self.rule.action(self.state, forecast, decision_time(row, k)).m3_s

    def schedules(self, problem, neutral, *, profiles=False):
        k, _, level, previous, ice, mechanism = self.context
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

    def screen(self, problem, requested, neutral, arm, *, profiles=False, candidate_audit=None):
        k, _, level, previous, ice, mechanism = self.context
        rows, nominal, schedules = self.schedules(problem, neutral, profiles=profiles)
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

    def plan(self, arm, requested, neutral, *, candidate_audit=None):
        k, base, *_ = self.context

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
            return self.screen(
                problem, requested, neutral, arm, profiles=profiles, candidate_audit=candidate_audit
            )

        choice = select(
            guard(base) if self.guard_next else base,
            profiles=self.recovery_mode == "proactive_profiles",
        )
        recovery = choice is None
        stage = "ordinary"
        probes = {}
        if recovery:
            realized = self.case["scenario"]["realized_net_inflow_m3_s"]
            past = (
                realized[k - 1]
                if k
                else self.config["forecast_noise"]["initial_observed_inflow_estimate_m3_s"]
            )
            retry = replace(base, net_inflow_m3_s=np.full(len(base.net_inflow_m3_s), past))
            choice = select(guard(retry))
            stage = "guarded_persistence"
            if choice is None and self.recovery_mode != "original":
                ordinary = guard(base) if self.guard_next else base
                options = [("reserve_released", retry, False)]
                if self.recovery_mode in (
                    "audit",
                    "profiles",
                    "proactive_profiles",
                    "peak_protected",
                ):
                    options += [
                        ("wider_ordinary", ordinary, True),
                        ("wider_guarded_persistence", guard(retry), True),
                        ("wider_physical_persistence", retry, True),
                    ]
                for label, problem, profiles in options:
                    candidate = select(problem, profiles=profiles)
                    probes[label] = candidate is not None
                    if candidate is not None and self.recovery_mode != "audit":
                        choice, stage = candidate, label
                        break
        return choice, recovery, stage, probes

    def apply(self, arm, residual=0.0):
        k, base, level, previous, _, _ = self.context
        neutral = self.neutral_request()
        previous_observation = self.observation(neutral=neutral)
        requested = max(0.0, neutral + residual)
        choice, recovery, stage, probes = self.plan(arm, requested, neutral)
        self.guard_next = recovery and choice is not None
        try:
            following_context = self.generator.send(choice)
        except StopIteration:
            following_context = None
        row = self.records[-1]
        executed = row["status"] == "executed"
        violation = executed and row["ottawa_storage_violation_m3"] > self.tolerance.storage_m3
        terminated = following_context is None
        if executed:
            if violation:
                row["status"] = "realized_storage_violation"
            self.state = self.model.advance_storage(
                self.state,
                row["ontario_applied_m3_s"],
                row["ontario_realized_supply_m3_s"],
                self.rows[k]["iceInd"],
                decision_time(self.rows[k], k),
            )
            cost, parts = operating_cost(
                level,
                row["ontario_end_m"],
                row["pointe_claire_m"],
                row["ontario_applied_m3_s"],
                previous * FLOW_SCALE,
                self.costs,
                terminal=k + 1 == len(self.rows) and not violation,
            )
            row.update(parts, cost=cost, downstream_target_m=f_limit_target(level))
        else:
            cost = 0.0
        penalty = (
            self.costs["incomplete_penalty"] if terminated and (violation or not executed) else 0.0
        )
        intervention = (
            abs(row["ontario_applied_m3_s"] - requested)
            if executed and arm in ("ppo", "rule_curve")
            else None
        )
        row.update(
            requested_release_m3_s=requested if arm in ("ppo", "rule_curve") else None,
            intervention_m3_s=intervention,
            retry=recovery,
            reward=-(cost + penalty),
        )
        if self.recovery_mode != "original":
            row.update(recovery_stage=stage if executed else "stopped", recovery_probes=probes)
        if following_context is not None:
            self.context = following_context
        if terminated:
            observation = previous_observation
            observation[0] = self.state.level_m - 74.8
            observation[1] = self.state.previous_release_m3_s / 10000
            observation[12] = (len(self.rows) - k - int(executed)) / 48
            if executed:
                observation[6:9] = (
                    row["ottawa_end_storage_m3"] / 5e9,
                    row["ottawa_executed_release_m3_s"] / 2500,
                    row["ottawa_end_transit_m3"] / (2500 * base.duration_seconds[0]),
                )
                observation[25:73] = np.asarray(self.state.mean_level_history_m) - 74.8
        else:
            observation = self.observation()
        return observation, row["reward"], terminated, False, row

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if (
            not self.action_space.contains(action.astype(np.float32))
            or not -2000.0 <= action[0] <= 2000.0
        ):
            raise ValueError("Expected a finite scalar residual within ±2000 m³/s")
        return self.apply("ppo", float(action[0]))
