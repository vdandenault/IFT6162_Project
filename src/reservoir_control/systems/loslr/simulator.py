from typing import Any

import numpy as np
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState

from reservoir_control.systems.loslr.native import native_reference
from reservoir_control.systems.loslr.physics import (
    CONVERSION,
    DATUM,
    FLOW_SCALE,
    PREPROJECT_SLOPE,
    checked_lake_step,
    dlake_step_dh,
    dlake_step_du,
    lake_step,
)

round_d = native_reference().utils.round_d
stLawrenceRouting = native_reference().routing.stLawrenceRouting


class LoslrEnv(MultiAgentEnv):
    def __init__(self, *, stage_init: float = 74.55, action_mode: str = "adjustment", **kwargs: Any):
        super().__init__(**kwargs)

        if action_mode not in {"flow", "adjustment"}:
            raise ValueError("Ontario action mode must be flow or adjustment")
        self.action_mode = action_mode
        self.stage_init: float = stage_init

    @reset
    def reset_loslr(self, mdp: MDPState) -> MDPState:
        return MDPState(state={"ontLevelBOQ": self.stage_init})

    lostep = staticmethod(checked_lake_step)
    lostep_jax = staticmethod(lake_step)
    dlostep_du = staticmethod(dlake_step_du)
    dlostep_dh = staticmethod(dlake_step_dh)

    @staticmethod
    def total_flow(h_t: float, h_next: float, u_t: float) -> float:
        """Decode adjustment into total Lake Ontario discharge."""

        h_moq = (h_t + h_next) * 0.5
        H_0 = DATUM
        preproj = PREPROJECT_SLOPE * (h_moq - H_0) ** 1.5

        return float(preproj + u_t)

    @staticmethod
    def lrstep(h_t: float, h_next: float, flow: float, x) -> Any:
        """St. Lawrence routing uses total discharge."""

        h_moq = (h_next + h_t) * 0.5

        return stLawrenceRouting(h_moq, flow, x)

    @transition
    def simulate(self, mdp: MDPState) -> MDPState:
        t = mdp.t
        state = mdp.state
        params = mdp.params

        h_t = state["ontLevelBOQ"][t]
        if self.action_mode == "flow":
            # Rule curve and legacy MPC callers already apply operational limits.
            flow = mdp.actions["ont"]["flow"][t].item()
            h_next = h_t + round_d((params["ontNTS"] / FLOW_SCALE - flow) / CONVERSION, 6)
            u_applied = None
        else:
            # MPC directly controls the adjustment.
            u_t = mdp.actions["ont"]["adjustment"][t].item()

            # Current quarter-month inflow.
            # q is the quarter-month within the month, 1–4.
            qm = int(params["QM"])
            q = (qm - 1) % 4 + 1
            Q_NTS = float(params[f"ontNTS_QM{q}"])

            # Lake Ontario cubic transition
            h_next = self.lostep(h_t=h_t, u_t=u_t, Q_NTS=Q_NTS)

            # Recover total discharge including preproject flow.
            flow = self.total_flow(h_t, h_next, u_t)

            # Apply operational limits to total discharge.
            previous_flow = float(state["ontFlow"][t]) if t > 0 else 595.0

            # Use the previous quarter-month ice condition.
            previous_ice = params["iceIndPrev"]
            constrained_flow, regime = (
                self.agents["ont"]
                .mechanisms["flow"]
                .constrain(
                    params,
                    h_t,
                    h_next,
                    flow,
                    "RF",
                    previous_flow=previous_flow,
                    previous_ice=previous_ice,
                )
            )
            constrained_flow = float(constrained_flow)

            # Reconcile the transition if operational limits
            # modify the proposed discharge.
            if constrained_flow != flow:
                h_target = h_t + (Q_NTS - FLOW_SCALE * constrained_flow) / (FLOW_SCALE * CONVERSION)
                h_mean = (h_t + h_target) * 0.5
                H_0 = DATUM

                if h_mean < H_0:
                    raise ValueError("Constrained level below hydraulic datum")

                u_applied = constrained_flow - (PREPROJECT_SLOPE * (h_mean - H_0) ** 1.5)
                h_next = self.lostep(h_t=h_t, u_t=u_applied, Q_NTS=Q_NTS)
            else:
                u_applied = u_t

            flow = constrained_flow

            # Preserve physical discharge for existing reward code.
            params["flowRegime"] = regime

            mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})

        # St. Lawrence routing
        x = {**params, "slonFlow": params["stlouisontOut"], "jetty1R": params["varennesR"]}
        stLawLevels = self.lrstep(h_t, h_next, flow, x)
        stlouisFlow = stLawLevels["stlouisFlow"]
        freshetIndicator = 0.0

        if t > 0:
            freshetIndicator = state["freshetIndicator"][t]
            previous_stlouis_flow = state["stlouisFlow"][t]

            if np.isfinite(previous_stlouis_flow):
                if stlouisFlow > 1.5 * previous_stlouis_flow:
                    freshetIndicator = 1.0
                elif stlouisFlow < 0.9 * previous_stlouis_flow:
                    freshetIndicator = 0.0

        h_moq = (h_next + h_t) * 0.5

        return mdp.advance(
            state={
                "ontLevelBOQ": h_next,
                "ontLevelEOQ": round_d(h_next, 2),
                "ontLevelMOQ": round_d(h_moq, 2),
                "ontFlow": flow,
                **({"ontAdjustmentApplied": u_applied} if u_applied is not None else {}),
                "freshetIndicator": freshetIndicator,
                **stLawLevels,
            }
        )
