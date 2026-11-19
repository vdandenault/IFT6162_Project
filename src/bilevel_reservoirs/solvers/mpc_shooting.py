"""Nominal MPC with single direct shooting"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from metamarl.mechanism.base import MDPState
from metamarl.optimizers.base import Optimizer
from scipy.optimize import minimize

from bilevel_reservoirs.controllers.operator import Ontario
from bilevel_reservoirs.simulators.loslr import LoslrEnv
from bilevel_reservoirs.solvers.rule_curve import FORCING_COLUMNS, RuleCurveCfg
from external.loslr_regulation_optimization.functions.utils import round_d


class MPCShootingCfg(RuleCurveCfg):
    def __init__(self) -> None:
        super().__init__()

        self.opt_class = MPCShooting
        self.prediction_horizon: int = 4
        self.weights: np.ndarray = np.ones(4, dtype=np.float64)
        self.flow_min: float = 0.0
        self.flow_max: float = 1150.0
        self.maxiter: int = 20


class MPCShooting(Optimizer):
    _env: LoslrEnv

    def __init__(self, config: MPCShootingCfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)

        self.t_0 = config.start_step
        self.conv = config.conv
        self.flow_scale = config.flow_scale
        self.prediction_horizon = config.prediction_horizon
        self.weights = np.asarray(config.weights, dtype=np.float64)
        self.flow_min = config.flow_min
        self.flow_max = config.flow_max
        self.maxiter = config.maxiter

        if config.september_rule:
            raise NotImplementedError("September rule not supported")

        if not 1 <= self.prediction_horizon <= 4:
            raise ValueError("Invalid prediction horizon")

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

    def _apply_constraint(self, *args, **kwargs) -> tuple[float, str]:
        return self.env.agents["ont"].mechanisms["flow"].constrain(*args, **kwargs)

    def _update(
        self, mdp: MDPState, agent: Ontario, forecasts: list[dict[str, Any]], nominal: np.ndarray
    ) -> tuple[np.ndarray, float, str]:
        def rollout(actions: np.ndarray):
            candidate = deepcopy(mdp)
            start = candidate.t
            first_flow = None
            first_regime = None

            for action, data in zip(actions, forecasts, strict=True):
                t = candidate.t
                ontLevelStart = candidate.state["ontLevelBOQ"][t]
                flow = float(action)
                dif1 = round_d((data["ontNTS_QM1"] / self.flow_scale - flow) / self.conv, 6)
                ontLevel_forecast = ontLevelStart + dif1
                flow, regime = self._apply_constraint(
                    data,
                    ontLevelStart,
                    ontLevel_forecast,
                    flow,
                    "RF",
                    previous_flow=(
                        candidate.actions["ont"]["flow"][t - 1].item() if t > 0 else 595.0
                    ),
                    previous_ice=(candidate.params["iceInd"] if t > 0 else 2.0),
                )

                if t == start:
                    first_flow = flow
                    first_regime = regime

                candidate.params.update({col: data[col] for col in FORCING_COLUMNS})
                candidate.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})

                candidate = self.env.simulate(candidate)
                reward = agent.reward(candidate)

                candidate.update(t=t, rewards={agent.id: reward.rewards[agent.id][0]})

            running_cost = np.stack(candidate.rewards[agent.id][start : candidate.t]).sum(axis=0)
            cost = float(self.weights @ running_cost)

            return cost, first_flow, first_regime

        result = minimize(
            lambda actions: rollout(actions)[0],
            x0=nominal,
            method="Powell",
            bounds=[(self.flow_min, self.flow_max) for _ in nominal],
            options={"maxiter": self.maxiter, "xtol": 1.0},
        )
        actions = np.clip(result.x, self.flow_min, self.flow_max)
        cost, flow, regime = rollout(actions)
        nominal_cost, nominal_flow, nominal_regime = rollout(nominal)

        if np.isfinite(nominal_cost) and (not np.isfinite(cost) or nominal_cost < cost):
            actions = nominal.copy()
            cost = nominal_cost
            flow = nominal_flow
            regime = nominal_regime

        if not np.isfinite(cost):
            raise RuntimeError("MPC failed to find a finite solution")

        return actions, flow, regime

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
            forecasts = []

            for k in range(horizon):
                forecast = {col: data[col] for col in FORCING_COLUMNS}

                for col in ("Sim", "Year", "Month", "QM"):
                    forecast[col] = self.data[col].iat[i + k]

                forecast.update(
                    {
                        "ontNTS": data[f"ontNTS_QM{k + 1}"],
                        "ontNTS_QM1": data[f"ontNTS_QM{k + 1}"],
                        "stlouisontOut": data["slonFlow_QM1"],
                        "slonFlow_QM1": data["slonFlow_QM1"],
                        "foreInd": data["foreInd"],
                    }
                )
                forecasts.append(forecast)

            if nominal is None:
                nominal = np.full(horizon, 595.0, dtype=np.float64)
            else:
                nominal = np.concatenate([nominal[1:], nominal[-1:]])[:horizon]

            nominal, flow, regime = self._update(mdp, agent, forecasts, nominal)
            forcing = {col: data[col] for col in FORCING_COLUMNS}

            mdp.params.update(forcing)
            mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})

            mdp = self.env.simulate(mdp)
            reward = agent.reward(mdp)

            mdp.update(t=t, rewards={agent.id: reward.rewards[agent.id][0]})
            assert mdp.t == t + 1

            row = data.copy()

            row.update(
                {
                    "ontFlow": flow,
                    "flowRegime": regime,
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
