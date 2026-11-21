"""Nominal MPC with single direct shooting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from core.mechanism.base import MDPState
from core.optimizers.base import Optimizer
from scipy.optimize import minimize

from bilevel_reservoirs.controllers.operator import Ontario
from bilevel_reservoirs.simulators.loslr import LoslrEnv
from bilevel_reservoirs.simulators.native import native_reference
from bilevel_reservoirs.solvers.rule_curve import FORCING_COLUMNS, RuleCurveCfg

round_d = native_reference().utils.round_d


class MPCShootingCfg(RuleCurveCfg):
    def __init__(self) -> None:
        super().__init__()

        self.opt_class = MPCShooting
        self.prediction_horizon: int = 4
        self.weights: np.ndarray = np.ones(4, dtype=np.float64)

        # TODO: review bounds; apply to adjustments, not total flow.
        self.adjustment_min: float = -700.0
        self.adjustment_max: float = 700.0
        self.maxiter: int = 20


class MPCShooting(Optimizer):
    _env: LoslrEnv

    def __init__(self, config: MPCShootingCfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        if self.world is None:
            self.id = type(self).__name__

        self.t_0 = config.start_step
        self.conv = config.conv
        self.flow_scale = config.flow_scale
        self.prediction_horizon = config.prediction_horizon
        self.weights = np.asarray(config.weights, dtype=np.float64)
        self.adjustment_min = config.adjustment_min
        self.adjustment_max = config.adjustment_max
        self.maxiter = config.maxiter

        if config.september_rule:
            raise NotImplementedError("September rule not supported")

        if self.prediction_horizon < 1:
            raise ValueError("Invalid prediction horizon")

        if self.adjustment_min >= self.adjustment_max:
            raise ValueError("Invalid adjustment bounds")

        if self.weights.shape != (4,):
            raise ValueError("Expected four objective weights")

        if (
            not np.isfinite(self.weights).all()
            or np.any(self.weights < 0)
            or not np.any(self.weights > 0)
        ):
            raise ValueError("Invalid objective weights")

        self.data = pd.read_table(config.input_path)
        fcstNA = self.data["forNTS"].notna()
        fcstNA.iloc[: self.t_0] = True
        self.data = self.data.loc[fcstNA, :].reset_index(drop=True)

        if config.drop_last_year:
            self.data = self.data.loc[self.data["Year"] < self.data["Year"].max()].reset_index(
                drop=True
            )

        n_steps = len(self.data) - self.t_0
        self.horizon = config.env_config["horizon"]

        if not 0 < self.horizon <= n_steps:
            raise ValueError("Invalid simulation horizon")

    @property
    def env(self) -> LoslrEnv:
        return self._env

    @env.setter
    def env(self, value: LoslrEnv) -> None:
        self._env = value

    def _update(
        self, mdp: MDPState, agent: Ontario, forecasts: list[dict[str, Any]], nominal: np.ndarray
    ) -> np.ndarray:
        def rollout(adjustments: np.ndarray) -> float:
            candidate = deepcopy(mdp)
            start = candidate.t

            try:
                for u_t, data in zip(adjustments, forecasts, strict=True):
                    t = candidate.t

                    candidate.params.update(data)
                    candidate.update(
                        actions={"ont": {"adjustment": np.asarray([u_t], dtype=np.float64)}}
                    )

                    # Environment handles cubic, discharge,
                    # constraints, and routing.
                    candidate = self.env.simulate(candidate)
                    reward = agent.reward(candidate)

                    candidate.update(t=t, rewards={agent.id: reward.rewards[agent.id][0]})

                l_c = np.stack(candidate.rewards[agent.id][start : candidate.t]).sum(axis=0)
                cost = float(self.weights @ l_c)

                return cost if np.isfinite(cost) else np.inf
            except (ValueError, FloatingPointError, OverflowError):
                # Invalid hydraulic candidate is infeasible.
                return np.inf

        result = minimize(
            rollout,
            x0=nominal,
            method="Powell",
            bounds=[(self.adjustment_min, self.adjustment_max) for _ in nominal],
            options={"maxiter": self.maxiter, "xtol": 1.0},
        )
        adjustments = np.clip(result.x, self.adjustment_min, self.adjustment_max)
        cost = rollout(adjustments)
        nominal_cost = rollout(nominal)

        if np.isfinite(nominal_cost) and (not np.isfinite(cost) or nominal_cost < cost):
            adjustments = nominal.copy()
            cost = nominal_cost

        if not np.isfinite(cost):
            raise RuntimeError("MPC failed to find a finite solution")

        return adjustments

    def train(self) -> dict[str, Any]:
        agent = self.env.agents["ont"]
        mdp = self.env.reset(MDPState())
        rows = []
        nominal = None

        for i in range(self.t_0, self.t_0 + self.horizon):
            t = mdp.t
            data = {col: self.data[col].iat[i] for col in self.data.columns}
            ontLevelStart = mdp.state["ontLevelBOQ"][t]
            horizon = min(self.prediction_horizon, self.horizon - t)

            # CHANGED: each dataset row is one quarter-month.
            # No separate forecast-lead indexing.
            forecasts = [self.data.iloc[i + j].to_dict() for j in range(horizon)]

            # CHANGED: supply previous ice for each forecast step.
            for j, forecast in enumerate(forecasts):
                idx = i + j
                forecast["iceIndPrev"] = float(self.data["iceInd"].iat[idx - 1]) if idx > 0 else 2.0

            # TODO: zero adjustment is the initial guess.
            if nominal is None:
                nominal = np.zeros(horizon, dtype=np.float64)
            else:
                nominal = np.concatenate([nominal[1:], nominal[-1:]])[:horizon]

            # Optimize adjustments directly.
            nominal = self._update(mdp, agent, forecasts, nominal)

            # Receding-horizon control:
            # apply only the first adjustment.
            u_t = float(nominal[0])

            # CHANGED: pass complete current forcing row.
            # Includes foreInd, slonFlow_QM1, and NTS columns.
            mdp.params.update(data)

            # CHANGED: previous quarter-month ice.
            mdp.params["iceIndPrev"] = float(self.data["iceInd"].iat[i - 1]) if i > 0 else 2.0

            mdp.update(actions={"ont": {"adjustment": np.asarray([u_t], dtype=np.float64)}})

            mdp = self.env.simulate(mdp)
            reward = agent.reward(mdp)

            mdp.update(t=t, rewards={agent.id: reward.rewards[agent.id][0]})
            assert mdp.t == t + 1

            # Read actual discharge from the environment.
            flow = float(mdp.state["ontFlow"][mdp.t])
            regime = mdp.params["flowRegime"]
            row = data.copy()

            row.update(
                {
                    "ontFlow": flow,
                    "flowRegime": regime,
                    "controlAdjustment": u_t,
                    "ontLevelBOQ": ontLevelStart,
                    "ontLevel": (ontLevelStart + mdp.state["ontLevelBOQ"][mdp.t]) * 0.5,
                }
            )
            row.update(
                {col: mdp.state[col][mdp.t] for col in mdp.state.data if col != "ontLevelBOQ"}
            )
            rows.append(row)

        running_cost = np.stack(mdp.rewards[agent.id]).sum(axis=0)

        return {
            "policy_id": "mpc",
            "mdp": mdp,
            "data": pd.DataFrame(rows),
            "running_cost": running_cost,
        }
