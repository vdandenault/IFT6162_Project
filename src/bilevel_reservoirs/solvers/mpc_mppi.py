"""Nominal MPC with model predictive path integral control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from metamarl.mechanism.base import MDPState

from bilevel_reservoirs.controllers.operator import Ontario
from bilevel_reservoirs.solvers.mpc_shooting import MPCShooting, MPCShootingCfg
from bilevel_reservoirs.solvers.rule_curve import FORCING_COLUMNS
from external.loslr_regulation_optimization.functions.utils import round_d


class MPCMPPICfg(MPCShootingCfg):
    def __init__(self) -> None:
        super().__init__()

        self.opt_class = MPCMPPI
        self.num_samples: int = 64
        self.num_iterations: int = 3
        self.sigma: float = 70.0
        self.temperature: float = 1.0
        self.seed: int = 42


class MPCMPPI(MPCShooting):
    def __init__(self, config: MPCMPPICfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)

        self.num_samples = config.num_samples
        self.num_iterations = config.num_iterations
        self.sigma = config.sigma
        self.temperature = config.temperature
        self.rng = np.random.default_rng(config.seed)

        if (
            self.num_samples < 1
            or self.num_iterations < 1
            or self.sigma <= 0
            or self.temperature <= 0
        ):
            raise ValueError("Invalid MPPI configuration")

    def _update(
        self, mdp: MDPState, agent: Ontario, forecasts: list[dict[str, Any]], nominal: np.ndarray
    ) -> tuple[np.ndarray, float, str]:
        def rollout(actions: np.ndarray):
            candidate = deepcopy(mdp)
            start = candidate.t
            first_flow = None
            first_regime = None
            actions = np.clip(actions, self.flow_min, self.flow_max)

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

        reference = np.clip(nominal, self.flow_min, self.flow_max)
        nominal = reference.copy()
        nominal_cost, nominal_flow, nominal_regime = rollout(nominal)

        if not np.isfinite(nominal_cost):
            nominal_cost = np.inf

        best_actions = nominal.copy()
        best_cost = nominal_cost
        best_flow = nominal_flow
        best_regime = nominal_regime

        for _ in range(self.num_iterations):
            epsilon = self.rng.normal(
                loc=0.0, scale=self.sigma, size=(self.num_samples, len(nominal))
            )
            candidates = nominal + epsilon
            evaluations = [rollout(actions) for actions in candidates]
            costs = np.asarray([evaluation[0] for evaluation in evaluations], dtype=np.float64)
            valid = np.isfinite(costs)

            if not valid.any():
                continue

            correction = (epsilon @ (nominal - reference)) / self.sigma**2
            log_weights = -costs / self.temperature - correction

            log_weights[~valid] = -np.inf

            log_weights -= np.max(log_weights)
            probabilities = np.exp(log_weights)
            probabilities /= probabilities.sum()
            proposed = np.clip(nominal + probabilities @ epsilon, self.flow_min, self.flow_max)
            proposed_cost, proposed_flow, proposed_regime = rollout(proposed)

            if np.isfinite(proposed_cost) and proposed_cost < best_cost:
                best_actions = proposed.copy()
                best_cost = proposed_cost
                best_flow = proposed_flow
                best_regime = proposed_regime

            best_index = np.argmin(np.where(valid, costs, np.inf))

            if costs[best_index] < best_cost:
                best_actions = np.clip(candidates[best_index], self.flow_min, self.flow_max)
                best_cost = costs[best_index]
                best_flow = evaluations[best_index][1]
                best_regime = evaluations[best_index][2]

            nominal = best_actions.copy()

        if not np.isfinite(best_cost):
            raise RuntimeError("MPPI failed to find a finite solution")

        return best_actions, best_flow, best_regime
