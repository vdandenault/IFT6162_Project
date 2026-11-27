from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from core.mechanism.base import MDPState

from reservoir_control.systems.loslr.data import FORCING_COLUMNS
from reservoir_control.systems.loslr.native import native_reference
from reservoir_control.systems.loslr.operator import Ontario
from reservoir_control.systems.loslr.solvers.mpc_shooting import MPCShooting, MPCShootingCfg

round_d = native_reference().utils.round_d


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

        if self.beam_width < 1:
            raise ValueError("Invalid beam width")

    def _update(
        self, mdp: MDPState, agent: Ontario, forecasts: list[dict[str, Any]], nominal: np.ndarray
    ) -> tuple[np.ndarray, float, str]:
        start = mdp.t

        def extend(path, action, data):
            _, actions, state, first_flow, first_regime = path
            candidate = deepcopy(state)
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
                previous_flow=(candidate.actions["ont"]["flow"][t - 1].item() if t > 0 else 595.0),
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

            return (cost, (*actions, float(action)), candidate, first_flow, first_regime)

        initial = (0.0, (), mdp, None, None)
        nominal = np.clip(nominal, self.flow_min, self.flow_max)
        paths = [initial]

        for k, data in enumerate(forecasts):
            candidates = []

            for path in paths:
                state = path[2]
                t = state.t
                previous_flow = state.actions["ont"]["flow"][t - 1].item() if t > 0 else 595.0
                requests = np.unique(
                    np.clip(
                        np.concatenate(
                            [
                                nominal[k] + self.flow_offsets,
                                [self.flow_min, self.flow_max, previous_flow],
                            ]
                        ),
                        self.flow_min,
                        self.flow_max,
                    )
                )
                executed = set()

                for action in requests:
                    candidate = extend(path, action, data)

                    if not np.isfinite(candidate[0]):
                        continue

                    candidate_state = candidate[2]
                    flow = candidate_state.actions["ont"]["flow"][candidate_state.t - 1].item()

                    if flow in executed:
                        continue

                    executed.add(flow)
                    candidates.append(candidate)

            if not candidates:
                raise RuntimeError(f"No finite beam search candidates at depth {k}")

            paths = sorted(candidates, key=lambda path: path[0])[: self.beam_width]

        best = paths[0]
        nominal_path = initial

        for action, data in zip(nominal, forecasts, strict=True):
            nominal_path = extend(nominal_path, action, data)

        if np.isfinite(nominal_path[0]) and nominal_path[0] < best[0]:
            best = nominal_path

        _, actions, _, flow, regime = best

        return np.asarray(actions, dtype=np.float64), flow, regime
