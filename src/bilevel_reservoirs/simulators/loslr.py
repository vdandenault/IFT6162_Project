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

        # lo dynamics
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
            root = (a / 3) * (2 * np.cosh(np.arccosh(argument) / 3) - 1)

        # Predicted end and mean lake levels
        return 2 * (H_0 + root**2) - h_t

    @jax.jit
    def lostep_jax(h_t: float, u_t: float, Q_NTS: float):
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

        def trig_branch(arg):
            root = (a / 3) * (2 * jnp.cos(jnp.arccos(jnp.clip(arg, -1.0, 1.0)) / 3) - 1)

            return root

        def hyperbolic_branch(arg):
            root = (a / 3) * (2 * jnp.cosh(jnp.arccosh(jnp.maximum(arg, 1.0)) / 3) - 1)

            return root

        root = lax.cond(argument <= 1.0, trig_branch, hyperbolic_branch, argument)

        # Predicted end-of-quarter lake level
        return 2 * (H_0 + root**2) - h_t

    # Gradient with respect to the control u_t
    dlostep_du = jax.jit(jax.grad(lostep, argnums=1))

    # Gradient with respect to the current level h_t
    dlostep_dh = jax.jit(jax.grad(lostep, argnums=0))

    def lrstep(h_t: float, h_next: float, u_t: float, x) -> Any:
        h_moq = (h_next + h_t) * 0.5
        H_0 = 69.474 + 0.0014 * (2010 - 1985)
        preproj = 55.5823 * (h_moq - H_0) ** 1.5
        flow = preproj + u_t
        stLawLevels = stLawrenceRouting(h_moq, flow, x)

        return flow, stLawLevels

    @transition
    def simulate(self, mdp: MDPState) -> MDPState:
        t = mdp.t
        state = mdp.state
        params = mdp.params

        # TODO : control applied with step fucntion
        k = params["QM"]
        u_t = mdp.actions["ont"]["flow"][t].item()
        h_t = state["ontLevelBOQ"][t]

        # step lo
        Q_NTS = Q_NTS = params[f"ontNTS_QM{k}"]
        h_next = self.lostep(h_t=h_t, u_t=u_t, Q_NTS=Q_NTS)

        # Step slr
        x = {**params, "slonFlow": params["stlouisontOut"], "jetty1R": params["varennesR"]}
        flow, stLawLevels = self.lrstep(h_t, h_next, u_t, x)
        stlouisFlow = stLawLevels["stlouisFlow"]
        freshetIndicator = 0.0

        if t > 0:
            freshetIndicator = state["freshetIndicator"][t]
            previous_flow = state["stlouisFlow"][t]

            if np.isfinite(previous_flow):
                if stlouisFlow > 1.5 * previous_flow:
                    freshetIndicator = 1.0
                elif stlouisFlow < 0.9 * previous_flow:
                    freshetIndicator = 0.0

        return mdp.advance(
            state={
                "ontLevelBOQ": h_next, 
                "ontLevelEOQ": round_d(h_next, 2),
                "ontLevelMOQ": round_d((h_t + h_next) * 0.5, 2),
                "ontFlow": flow,  # Actual physical discharge
                "freshetIndicator": freshetIndicator, **stLawLevels}
        )
