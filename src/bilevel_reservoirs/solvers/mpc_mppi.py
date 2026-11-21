"""Model predictive path integral MPC over Lake Ontario discharge adjustments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from metamarl.mechanism.base import MDPState

from bilevel_reservoirs.controllers.operator import Ontario
from bilevel_reservoirs.solvers.mpc_shooting import MPCShooting, MPCShootingCfg


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
            or not np.isfinite(self.sigma)
            or self.sigma <= 0
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("Invalid MPPI configuration")

    def _update(
        self, mdp: MDPState, agent: Ontario, forecasts: list[dict[str, Any]], nominal: np.ndarray
    ) -> np.ndarray:
        """Return the best adjustment sequence for the receding-horizon MPC loop."""
        lower, upper = self.adjustment_min, self.adjustment_max

        def rollout(adjustments: np.ndarray) -> float:
            candidate = deepcopy(mdp)
            start = candidate.t

            try:
                for adjustment, forcing in zip(adjustments, forecasts, strict=True):
                    t = candidate.t
                    candidate.params.update(forcing)
                    candidate.update(
                        actions={"ont": {"adjustment": np.asarray([adjustment], dtype=np.float64)}}
                    )
                    candidate = self.env.simulate(candidate)
                    reward = agent.reward(candidate)
                    candidate.update(t=t, rewards={agent.id: reward.rewards[agent.id][0]})

                running_cost = np.stack(candidate.rewards[agent.id][start : candidate.t]).sum(
                    axis=0
                )
                cost = float(self.weights @ running_cost)
                return cost if np.isfinite(cost) else np.inf
            except (ValueError, FloatingPointError, OverflowError):
                return np.inf

        reference = np.clip(np.asarray(nominal, dtype=np.float64), lower, upper)
        center = reference.copy()
        best_actions = reference.copy()
        best_cost = rollout(reference)

        for _ in range(self.num_iterations):
            epsilon = self.rng.normal(0.0, self.sigma, size=(self.num_samples, len(center)))
            sampled_actions = np.clip(center + epsilon, lower, upper)
            costs = np.asarray([rollout(actions) for actions in sampled_actions], dtype=np.float64)
            valid = np.isfinite(costs)
            if not np.any(valid):
                continue

            # Stable exponential weights; invalid rollouts receive zero weight.
            correction = (epsilon @ (center - reference)) / self.sigma**2
            logits = -costs[valid] / self.temperature - correction[valid]
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()

            # Use the effective perturbation after clipping to control bounds.
            perturbations = sampled_actions[valid] - center
            proposed = np.clip(center + probabilities @ perturbations, lower, upper)
            proposed_cost = rollout(proposed)

            if proposed_cost < best_cost:
                best_cost = proposed_cost
                best_actions = proposed.copy()

            best_index = int(np.argmin(np.where(valid, costs, np.inf)))
            if costs[best_index] < best_cost:
                best_cost = float(costs[best_index])
                best_actions = sampled_actions[best_index].copy()

            center = best_actions.copy()

        if not np.isfinite(best_cost):
            raise RuntimeError("MPPI failed to find a finite solution")

        return best_actions
