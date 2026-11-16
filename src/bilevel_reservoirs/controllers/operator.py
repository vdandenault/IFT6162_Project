from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from metamarl.agents.base import Agent
from metamarl.mechanism.base import MDPState, Mechanism

# Legacy repository root
LOSLR_ROOT = (
    Path(__file__).resolve().parents[3] / "external" / "loslr_regulation_optimization"
)

# Legacy modules load lookup tables using relative paths.
with chdir(LOSLR_ROOT):
    from external.loslr_regulation_optimization.objectiveFunctions.legacyPIs.functions import (
        commercialNavigation,
        downstreamCoastalImpacts,
        hydropowerProduction,
        meadowMarsh,
        muskratLodgeViability,
        recreationalBoating,
        upstreamCoastalImpacts,
    )

PI_MODELS = [
    upstreamCoastalImpacts,
    downstreamCoastalImpacts,
    commercialNavigation,
    hydropowerProduction,
    meadowMarsh,
    muskratLodgeViability,
    recreationalBoating,
]
original_series_getitem = pd.Series.__getitem__


def legacy_series_getitem(self, key):
    if isinstance(key, (int, np.integer)) and self.index.inferred_type not in {
        "integer",
        "mixed-integer",
        "floating",
        "complex",
    }:
        return self.iloc[key]

    return original_series_getitem(self, key)


class Ontario(Agent):
    def observation(self, mdp: MDPState) -> MDPState:
        pass

    def reward(self, mdp: MDPState) -> MDPState:
        t = mdp.t
        data = {
            **{k: np.asarray([v]) for k, v in mdp.params.items()},
            **{k: np.asarray([v[t]]) for k, v in mdp.state.data.items()},
            "ontFlow": np.asarray([mdp.actions["ont"]["flow"][t - 1].item()]),
            "ontLevel": np.asarray(
                [(mdp.state["ontLevelBOQ"][t - 1] + mdp.state["ontLevelBOQ"][t]) * 0.5]
            ),
        }
        costs = np.asarray(
            [
                upstreamCoastalImpacts.runModel(data)["upstreamCoastal"].iloc[-1],
                downstreamCoastalImpacts.runModel(data)["downstreamCoastal"].iloc[-1],
                -hydropowerProduction.runModel(data)["totalEnergyValue"].iloc[-1],
                recreationalBoating.runModel(data)["totalRecBoating"].iloc[-1],
            ],
            dtype=np.float64,
        )

        return MDPState(rewards={self.id: costs})

    def terminal_cost(self, mdp: MDPState) -> np.ndarray:
        """Ecological objectives requiring trajectory history."""

        # TODO : this is an issue for MPC !!!

        data = {k: np.asarray(v) for k, v in mdp.state.data.items()}

        with (
            pd.option_context("future.infer_string", False),
            patch.object(
                pd.Series, "values", property(lambda self: self.to_numpy(copy=True))
            ),
            patch.object(pd.Series, "__getitem__", legacy_series_getitem),
        ):
            mm = meadowMarsh.runModel(data)
            muskrat = muskratLodgeViability.runModel(data)

        return np.asarray(
            [
                -mm.loc[mm["mmLowSupply"] == 1].groupby("Year")["mmArea"].sum().mean(),
                -muskrat.groupby("Year")["muskratProbLodgeViability"].sum().mean(),
            ],
            dtype=np.float64,
        )


class AdjustCrest(Mechanism):
    # CHANGED: accept decoded action explicitly
    def apply(self, mdp: MDPState, action) -> MDPState:
        t = mdp.t

        # ADDED: action is currently a release, NOT
        # an explicit physical crest elevation.
        proposed_flow = float(np.asarray(action).item())

        # ADDED: Mechanism.apply returns a residual.
        # MDPState.add sums this contribution into
        # the existing state.
        previous_value = mdp.state["proposedFlow"][t]

        return MDPState(state={"proposedFlow": (proposed_flow - previous_value)})
