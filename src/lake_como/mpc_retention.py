"""Experimental iteration-limit candidate retention; frozen original MPC unchanged.

The planning loop intentionally mirrors the reference implementation so historical
source hashes remain valid. Tests check the unchanged planning/solver setup and
bitwise control-arm reproduction. Only candidate selection is intervened upon.
"""
import numpy as np
from scipy.optimize import minimize
from lake_como.mpc import DeterministicMPC


def select_plan(result, initial, initial_cost, objective, retain_iteration_limit):
    """Retain only bounded, finite, strictly improving status-9 iterates.

    Request bounds are not a certificate of safe lake levels or future supply.
    Recompute cost from clipped coordinates; never trust result.fun alone.
    """
    x = np.asarray(result.x)
    bounded = bool(
        x.shape == initial.shape
        and np.isfinite(x).all()
        and (x >= -1e-8).all()
        and (x <= 1 + 1e-8).all()
    )
    candidate = np.clip(x, 0, 1) if bounded else initial
    candidate_cost = float(objective(candidate))
    retained = bool(
        retain_iteration_limit
        and not result.success
        and result.status == 9
        and bounded
        and np.isfinite(candidate_cost)
        and candidate_cost < initial_cost - 1e-9
    )
    valid = bool(bounded and (result.success or retained))
    fallback = bool(
        not valid
        or not np.isfinite(candidate_cost)
        or candidate_cost > initial_cost + 1e-9
    )
    return (initial if fallback else candidate), fallback, retained, candidate_cost


class RetainingMPC(DeterministicMPC):
    """Same model/optimizer and original success flag; optional status-9 retention."""

    def __init__(self, params, forecast, config, *, retain_iteration_limit=False):
        super().__init__(params, forecast, config)
        self.retain_iteration_limit = retain_iteration_limit

    def __call__(self, observation, previous_info=None):
        # Only the shared observation is consumed. In particular, do not read
        # previous_info['storage'], live year profiles, or future demand/inflow.
        obs = np.asarray(observation)
        if obs.shape != (4,) or not np.isfinite(obs).all():
            raise ValueError("Expected default four-number finite DamEnv observation")
        phase = np.arctan2(obs[1], obs[2]) % (2 * np.pi)
        day0 = int(np.rint(phase * 365 / (2 * np.pi))) % 365
        horizon = min(self.config.horizon_days, 365 - day0)
        days = np.arange(day0 + 1, day0 + horizon + 1)
        storage = self.model.lake.level_to_storage(float(obs[0]))
        inflows = self.forecast.inflow[days - 1]
        demands = self.forecast.demand[days - 1]
        mef = np.asarray(self.model.lake.min_env_flow)[days - 1]
        span = self.model.high - self.model.low
        demand_plan = np.clip((demands + mef - self.model.low) / span, 0.0, 1.0)
        candidates = [demand_plan, np.zeros(horizon), np.ones(horizon)]
        if self.previous_day == day0 - 1 and self.previous_plan is not None:
            shifted = self.previous_plan[1:]
            candidates.append(
                np.pad(shifted, (0, max(0, horizon - len(shifted))), mode="edge")[
                    :horizon
                ]
                if len(shifted)
                else demand_plan
            )
        objective = lambda x: self.model.cost(x, storage, inflows, demands, days)
        values = [objective(x) for x in candidates]
        initial = candidates[int(np.argmin(values))].copy()
        initial_cost = min(values)
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * horizon,
            options=dict(
                maxiter=self.config.max_iterations,
                ftol=self.config.ftol,
                eps=self.config.finite_difference_step,
            ),
        )
        plan, fallback, retained, candidate_cost = select_plan(
            result, initial, initial_cost, objective, self.retain_iteration_limit
        )
        levels, releases = self.model.predict(
            storage, self.model.low + plan * span, inflows, days
        )
        self.previous_day, self.previous_plan = day0, plan.copy()
        action = np.array([self.model.low + plan[0] * span], dtype=np.float32)
        self.diagnostics.append(
            dict(
                day=day0 + 1,
                horizon_days=horizon,
                solver_success=bool(result.success),
                solver_status=int(result.status),
                solver_message=str(result.message),
                iterations=int(result.nit),
                function_evaluations=int(result.nfev),
                used_fallback=bool(fallback),
                retained_iteration_limit=bool(retained),
                candidate_cost=candidate_cost,
                selected_source=(
                    "iteration_limit"
                    if retained
                    else "initial"
                    if fallback
                    else "successful_solver"
                ),
                initial_cost=initial_cost,
                planned_cost=objective(plan),
                request_m3_s=float(action[0]),
                forecast_inflow_m3_s=float(inflows[0]),
                forecast_demand_m3_s=float(demands[0]),
                predicted_level_m=float(levels[0]),
                predicted_release_m3_s=float(releases[0]),
            )
        )
        return action
