import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from reservoir_control.experiments.coupled_planning import CoupledPlanner
from reservoir_control.experiments.fixed_schedule_replay import replay_steps
from reservoir_control.systems.loslr.adapter import decision_time
from reservoir_control.systems.loslr.coupled import f_limit_target, operating_cost
from reservoir_control.systems.loslr.data import SupplyForecast
from reservoir_control.systems.loslr.model import Plan2014
from reservoir_control.systems.loslr.physics import FLOW_SCALE
from reservoir_control.systems.ottawa.hard_follower import FollowerTolerances

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

    def plan(self, arm, requested, neutral, *, candidate_audit=None):
        planner = CoupledPlanner(self.case, self.costs, self.tolerance, self.recovery_mode)
        return planner.plan(
            self.context,
            arm,
            requested,
            neutral,
            guard_next=self.guard_next,
            candidate_audit=candidate_audit,
        )

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
