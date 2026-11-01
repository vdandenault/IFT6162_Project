"""Deterministic daily MPC with training-only seasonal forecasts.

The planning model uses DamEnv's hourly hydraulics, but replaces binary level
penalties with squared violations to make local numerical optimization useful.
The unchanged environment reward is always used for evaluation.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from gym4real.envs.dam.lake import Lake

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT
from lake_como.splits import split_years

DEFAULT_CONFIG = PROJECT_ROOT / "configs/mpc.json"


@dataclass(frozen=True)
class MPCConfig:
    horizon_days: int = 7
    max_iterations: int = 35
    ftol: float = 1e-6
    finite_difference_step: float = 1e-4
    discount_factor: float = 0.995
    level_violation_scale_m: float = 0.05
    terminal_level_weight: float = 0.0

    def __post_init__(self):
        if not 1 <= self.horizon_days <= 365 or self.max_iterations < 1:
            raise ValueError(
                "Positive horizon (at most 365) and iteration limit required"
            )
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("MPC settings must be finite")
        if not 0 < self.discount_factor <= 1 or self.level_violation_scale_m <= 0:
            raise ValueError("Invalid discount or level-violation scale")
        if (
            self.ftol <= 0
            or self.finite_difference_step <= 0
            or self.terminal_level_weight < 0
        ):
            raise ValueError("Invalid optimizer or terminal settings")

    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SeasonalForecast:
    inflow: np.ndarray
    demand: np.ndarray
    training_years: tuple

    @classmethod
    def fit(cls, inflow, demand):
        years = split_years("train")
        if set(inflow) != set(years) or set(demand) != set(years):
            raise ValueError(
                "Forecast fitting accepts chronological training years only"
            )
        means = []
        for profiles in (inflow, demand):
            if any(len(profiles[y]) not in (365, 366) for y in years):
                raise ValueError("Forecast profiles must have 365 or 366 samples")
            matrix = np.array([profiles[y][:365] for y in years], dtype=float)
            if not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError("Forecast inputs must be finite and nonnegative")
            mean = matrix.mean(axis=0)
            mean.setflags(write=False)
            means.append(mean)
        return cls(*means, tuple(years))


class PlanningModel:
    """Side-effect-free open-loop predictions; never holds a live environment."""

    def __init__(self, params, config):
        self.lake = Lake(params["lake_params"])
        if self.lake.evaporation or params["integration"] != 24 or params["doy"] != 1:
            raise ValueError("Prototype requires default non-evaporating hourly DamEnv")
        self.config = config
        self.coefficients = dict(params["reward_coeff"])
        self.flood_level = params["flood_level"]
        self.low_level = params["starving_level"]
        self.smooth_deficit = params["smooth_daily_deficit_coeff"]
        self.low = float(params["action"]["low"])
        self.high = float(params["action"]["high"])

    def predict(self, storage, requests, inflows, days):
        if not (len(requests) == len(inflows) == len(days)):
            raise ValueError("Prediction arrays must have equal lengths")
        levels, releases = [], []
        for request, inflow, day in zip(requests, inflows, days):
            storage, release = self.lake.integration(
                24, storage, float(request), float(inflow), int(day)
            )
            levels.append(self.lake.storage_to_level(storage))
            releases.append(release)
        return np.array(levels), np.array(releases)

    def cost(self, normalized, storage, inflows, demands, days):
        requests = self.low + np.asarray(normalized) * (self.high - self.low)
        levels, releases = self.predict(storage, requests, inflows, days)
        mef = np.asarray(self.lake.min_env_flow)[days - 1]
        deficit = np.maximum(demands - np.maximum(releases - mef, 0), 0)
        if self.smooth_deficit:
            deficit *= 0.5 * (3 - np.cos((days - 1) * 2 * np.pi / 365))
        else:
            deficit *= np.where((days > 120) & (days <= 243), 2.0, 1.0)
        c = self.coefficients
        scale = self.config.level_violation_scale_m
        stage = (
            c["daily_deficit_coeff"] * deficit
            + c["wasted_water_coeff"] * np.maximum(releases - demands, 0)
            + c["clip_action_coeff"] * (requests - releases) ** 2
            + c["overflow_coeff"]
            * (np.maximum(levels - self.flood_level, 0) / scale) ** 2
            + c["starving_coeff"]
            * (np.maximum(self.low_level - levels, 0) / scale) ** 2
        )
        discounted = np.dot(self.config.discount_factor ** np.arange(len(days)), stage)
        terminal = (
            self.config.terminal_level_weight * (levels[-1] - self.lake.init_level) ** 2
        )
        # Scalar normalization changes numerical scale, not the minimizer.
        return float((discounted + terminal) / 1000.0)


class DeterministicMPC:
    """Replan from observed level/season, execute only the first bounded request."""

    def __init__(self, params, forecast, config):
        self.config = config
        self.forecast = forecast
        self.model = PlanningModel(params, config)
        self.previous_day = None
        self.previous_plan = None
        self.diagnostics = []

    @classmethod
    def from_training(cls, config=None):
        params = settings("train")
        forecast = SeasonalForecast.fit(params["inflow"], params["demand"])
        return cls(params, forecast, config or MPCConfig.load())

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
        valid = bool(
            result.success
            and np.isfinite(result.x).all()
            and (result.x >= -1e-8).all()
            and (result.x <= 1 + 1e-8).all()
        )
        candidate = np.clip(result.x, 0, 1) if valid else initial
        candidate_cost = objective(candidate)
        fallback = (
            not valid
            or not np.isfinite(candidate_cost)
            or candidate_cost > initial_cost + 1e-9
        )
        plan = initial if fallback else candidate
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


def save_diagnostics(controller, frame, output):
    import pandas as pd

    diagnostics = pd.DataFrame(controller.diagnostics)
    if len(diagnostics) != len(frame):
        raise RuntimeError("One optimizer record per daily action required")
    diagnostics["year"] = frame.year.to_numpy()
    diagnostics["actual_level_m"] = frame.level_m.to_numpy()
    diagnostics["level_error_m"] = (
        diagnostics.actual_level_m - diagnostics.predicted_level_m
    )
    diagnostics["inflow_error_m3_s"] = (
        frame.inflow_m3_s.to_numpy() - diagnostics.forecast_inflow_m3_s
    )
    diagnostics.to_csv(output / "mpc_diagnostics.csv", index=False)
    return diagnostics
