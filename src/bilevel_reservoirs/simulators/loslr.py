# TODO fix metamarl hatchling
# TODO constraint optimization (solvers)
# TODO testing + CI pass

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from metamarl.envs.hooks import reset, transition
from metamarl.envs.marl_regulated import MultiAgentEnv
from metamarl.mechanism.base import MDPState

from external.loslr_regulation_optimization.functions.routing.stlaw import stLawrenceRouting
from external.loslr_regulation_optimization.functions.utils import round_d


class LoslrEnv(MultiAgentEnv):
    def __init__(self, *, stage_init: float = 74.55, **kwargs: Any):
        super().__init__(**kwargs)

        self.stage_init: float = stage_init

    @reset
    def reset_loslr(self, mdp: MDPState) -> MDPState:
        return MDPState(state={"ontLevelBOQ": self.stage_init})

    @staticmethod
    def lostep(h_t: float, u_t: float, Q_NTS: float) -> float:
        slope = 55.5823
        adj = 0.0014 * (2010 - 1985)

        # Lake Ontario dynamics
        A_w = 10 * slope
        H_0 = 69.474 + adj
        c = 10 * 2970
        u_Q = 10 * u_t

        # Cubic: root**3 + a*root**2 - b = 0
        a = 2 * c / A_w
        b = (Q_NTS - u_Q + 2 * c * (h_t - H_0)) / A_w
        argument = 27 * b / (2 * a**3) - 1

        if argument < -1 - 1e-10:
            raise ValueError("No nonnegative hydraulic-head root")

        if argument <= 1:
            root = (a / 3) * (2 * np.cos(np.arccos(np.clip(argument, -1, 1)) / 3) - 1)
        else:
            root = (a / 3) * (2 * np.cosh(np.arccosh(argument / 1.0) / 3) - 1)

        # Predicted end and mean lake levels
        return 2 * (H_0 + root**2) - h_t

    @staticmethod
    @jax.jit
    def lostep_jax(h_t: float, u_t: float, Q_NTS: float):
        slope = 55.5823
        adj = 0.0014 * (2010 - 1985)
        A_w = 10 * slope
        H_0 = 69.474 + adj
        c = 10 * 2970
        u_Q = 10 * u_t
        a = 2 * c / A_w
        b = (Q_NTS - u_Q + 2 * c * (h_t - H_0)) / A_w
        argument = 27 * b / (2 * a**3) - 1

        def trig_branch(arg):
            return (a / 3) * (2 * jnp.cos(jnp.arccos(jnp.clip(arg, -1.0, 1.0)) / 3) - 1)

        def hyperbolic_branch(arg):
            return (a / 3) * (2 * jnp.cosh(jnp.arccosh(jnp.maximum(arg, 1.0)) / 3) - 1)

        root = lax.cond(argument <= 1.0, trig_branch, hyperbolic_branch, argument)
        h_next = 2 * (H_0 + root**2) - h_t

        # Compiled JAX code cannot raise Python ValueError.
        return jnp.where(argument < -1.0, jnp.nan, h_next)

    # JAX derivatives
    dlostep_du = staticmethod(jax.jit(jax.grad(lostep_jax.__func__, argnums=1)))
    dlostep_dh = staticmethod(jax.jit(jax.grad(lostep_jax.__func__, argnums=0)))

    @staticmethod
    def total_flow(h_t: float, h_next: float, u_t: float) -> float:
        """Decode adjustment into total Lake Ontario discharge."""

        h_moq = (h_t + h_next) * 0.5
        H_0 = 69.474 + 0.0014 * (2010 - 1985)
        preproj = 55.5823 * (h_moq - H_0) ** 1.5

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

        # MPC directly controls the adjustment.
        u_t = mdp.actions["ont"]["adjustment"][t].item()
        h_t = state["ontLevelBOQ"][t]

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

        # CHANGED: use previous quarter-month ice condition.
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
            h_target = h_t + (Q_NTS - 10 * constrained_flow) / 29700
            h_mean = (h_t + h_target) * 0.5
            H_0 = 69.474 + 0.0014 * (2010 - 1985)

            if h_mean < H_0:
                raise ValueError("Constrained level below hydraulic datum")

            u_applied = constrained_flow - (55.5823 * (h_mean - H_0) ** 1.5)
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
                "ontAdjustmentApplied": u_applied,
                "freshetIndicator": freshetIndicator,
                **stLawLevels,
            }
        )
