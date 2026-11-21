"""Beam-search MPC over Lake Ontario discharge adjustments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from metamarl.mechanism.base import MDPState

from bilevel_reservoirs.controllers.operator import Ontario
from bilevel_reservoirs.solvers.mpc_shooting import MPCShooting, MPCShootingCfg


class MPCBeamSearchCfg(MPCShootingCfg):
    def __init__(self) -> None:
        super().__init__()
        self.opt_class = MPCBeamSearch
        self.beam_width: int = 12
        self.flow_offsets: tuple[float, ...] = (-140.0, -70.0, -35.0, 0.0, 35.0, 70.0, 140.0)


class MPCBeamSearch(MPCShooting):
    def __init__(self, config: MPCBeamSearchCfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self.beam_width = config.beam_width
        self.flow_offsets = np.asarray(config.flow_offsets, dtype=np.float64)

        if self.beam_width < 1 or not np.isfinite(self.flow_offsets).all():
            raise ValueError("Invalid beam search configuration")

    def _update(
        self, mdp: MDPState, agent: Ontario, forecasts: list[dict[str, Any]], nominal: np.ndarray
    ) -> np.ndarray:
        """Return the best adjustment sequence for the receding-horizon MPC loop."""
        start = mdp.t
        nominal = np.clip(
            np.asarray(nominal, dtype=np.float64), self.adjustment_min, self.adjustment_max
        )

        def extend(path, adjustment: float, forcing: dict[str, Any]):
            _, actions, state = path
            candidate = deepcopy(state)
            t = candidate.t

            try:
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
            except (ValueError, FloatingPointError, OverflowError):
                return None

            if not np.isfinite(cost):
                return None

            return cost, (*actions, float(adjustment)), candidate

        initial = (0.0, (), mdp)
        paths = [initial]

        for k, forcing in enumerate(forecasts):
            candidates = []
            for path in paths:
                state = path[2]
                t = state.t
                previous_adjustment = (
                    float(state.actions["ont"]["adjustment"][t - 1].item()) if t > 0 else 0.0
                )
                options = np.unique(
                    np.clip(
                        np.concatenate(
                            (
                                nominal[k] + self.flow_offsets,
                                [
                                    nominal[k],
                                    0.0,
                                    previous_adjustment,
                                    self.adjustment_min,
                                    self.adjustment_max,
                                ],
                            )
                        ),
                        self.adjustment_min,
                        self.adjustment_max,
                    )
                )

                seen_flows = set()
                for adjustment in options:
                    candidate = extend(path, float(adjustment), forcing)
                    if candidate is None:
                        continue

                    next_state = candidate[2]
                    executed_flow = float(next_state.state["ontFlow"][next_state.t])
                    if executed_flow in seen_flows:
                        continue
                    seen_flows.add(executed_flow)
                    candidates.append(candidate)

            if not candidates:
                raise RuntimeError(f"No finite beam search candidates at depth {k}")
            paths = sorted(candidates, key=lambda path: path[0])[: self.beam_width]

        best = paths[0]

        # Preserve the warm-start sequence if beam pruning discarded it.
        nominal_path = initial
        for adjustment, forcing in zip(nominal, forecasts, strict=True):
            nominal_path = extend(nominal_path, float(adjustment), forcing)
            if nominal_path is None:
                break

        if nominal_path is not None and nominal_path[0] < best[0]:
            best = nominal_path

        return np.asarray(best[1], dtype=np.float64)
