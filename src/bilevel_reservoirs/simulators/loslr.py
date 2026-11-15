# TODO fix metamarl hatchling
# TODO constraint optimization (solvers)
# TODO testing + CI pass

from typing import Any

import numpy as np
from metamarl.envs.hooks import reset, transition
from metamarl.envs.marl_regulated import MultiAgentEnv
from metamarl.mechanism.base import MDPState

from external.loslr_regulation_optimization.functions.routing.stlaw import (
    stLawrenceRouting,
)
from external.loslr_regulation_optimization.functions.utils import round_d


class LoslrEnv(MultiAgentEnv):
    def __init__(self, *, stage_init: float = 74.55, **kwargs: Any):
        super().__init__(**kwargs)

        self.stage_init: float = stage_init

    @reset
    def reset_loslr(self, mdp: MDPState) -> MDPState:
        return MDPState(state={"ontLevelBOQ": self.stage_init})

    @transition
    def simulate(self, mdp: MDPState) -> MDPState:
        t = mdp.t
        state = mdp.state
        params = mdp.params

        # TODO : control applied with step fucntion
        flow = mdp.actions["ont"]["flow"][t].item()

        # step lo
        ontLevelBOQ_next = state["ontLevelBOQ"][t] + round_d(  # h_next
            (params["ontNTS"] / 10 - flow) / 2970, 6
        )
        ontLevelEOQ_next = round_d(ontLevelBOQ_next, 2)
        ontLevelMOQ_raw = (ontLevelBOQ_next + state["ontLevelBOQ"][t]) * 0.5  # h_mean
        ontLevelMOQ_next = round_d(ontLevelMOQ_raw, 2)

        # Step slr
        x = {
            **params,
            "slonFlow": params["stlouisontOut"],
            "jetty1R": params["varennesR"],
        }
        stLawLevels = stLawrenceRouting(ontLevelMOQ_raw, flow, x)
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
                "ontLevelBOQ": ontLevelBOQ_next,
                "ontLevelEOQ": ontLevelEOQ_next,
                "ontLevelMOQ": ontLevelMOQ_next,
                "freshetIndicator": freshetIndicator,
                **stLawLevels,
            }
        )
