from __future__ import annotations

from collections.abc import Callable
from contextlib import chdir
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pandas as pd
from core.agents.base import Agent, AgentConfig
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig

from reservoir_control.systems.loslr.native import native_reference

planLimits = native_reference().limits.planLimits

ONTARIO_COST_COMPONENTS = (
    "upstream_coastal",
    "downstream_coastal",
    "negative_energy_value",
    "recreational_boating",
)


def load_indicator(name: str):
    reference = native_reference()
    with chdir(reference.root):
        return reference.load(f"objectiveFunctions/legacyPIs/functions/{name}.py")


downstreamCoastalImpacts = load_indicator("downstreamCoastalImpacts")
hydropowerProduction = load_indicator("hydropowerProduction")
meadowMarsh = load_indicator("meadowMarsh")
muskratLodgeViability = load_indicator("muskratLodgeViability")
recreationalBoating = load_indicator("recreationalBoating")
upstreamCoastalImpacts = load_indicator("upstreamCoastalImpacts")

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


def ontario_stage_costs(data):
    """Evaluate the four Ontario stage indicators in their native units."""
    return np.column_stack(
        (
            upstreamCoastalImpacts.runModel(data)["upstreamCoastal"].to_numpy(),
            downstreamCoastalImpacts.runModel(data)["downstreamCoastal"].to_numpy(),
            -hydropowerProduction.runModel(data)["totalEnergyValue"].to_numpy(),
            recreationalBoating.runModel(data)["totalRecBoating"].to_numpy(),
        )
    ).astype(np.float64)


class Ontario(Agent):
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
        data["Month"] = np.asarray([int(mdp.params["Month"])])
        data["QM"] = np.asarray([int(mdp.params["QM"])])

        return MDPState(rewards={self.id: ontario_stage_costs(data)[0]})

    def terminal_cost(self, mdp: MDPState) -> np.ndarray:
        """Ecological objectives requiring trajectory history."""

        data = {k: np.asarray(v) for k, v in mdp.state.data.items()}

        with (
            pd.option_context("future.infer_string", False),
            patch.object(pd.Series, "values", property(lambda self: self.to_numpy(copy=True))),
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


class AdjustFlow(Mechanism):
    def __init__(self, *, constraint_fn: Callable = planLimits, **kwargs):
        super().__init__(**kwargs)

        self.constraint_fn = constraint_fn

    def constrain(
        self,
        data: dict,
        ontLevelStart: float,
        ontLevel: float,
        flow: float,
        regime: str,
        previous_flow: float,
        previous_ice: float,
        september_rule="off",
    ) -> tuple[float, str]:
        x = {
            "QM": data["QM"],
            "iceInd": data["iceInd"],
            "iceIndPrev": previous_ice,
            "ontFlowPrev": previous_flow,
            "obsontNTS": data["ontNTS"],
            "sfSupplyNTS": data["ontNTS_QM1"],
            "ontLevelStart": ontLevelStart,
            "kingLevelStart": ontLevelStart - 0.03,
            "ptclaireR": data["ptclaireR"],
            "longsaultR": data["longsaultR"],
            "saundershwR": data["saundershwR"],
            "saunderstwR": data["saunderstwR"],
            "foreInd": data["foreInd"],
            "sfSlonFlow": data["slonFlow_QM1"],
            "slonFlow": data["stlouisontOut"],
        }
        limits = self.constraint_fn(data["QM"], ontLevel, flow, regime, x, september_rule)

        return limits["ontFlow"], limits["ontRegime"]

    def apply(self, mdp: MDPState, action, *, september_rule="off") -> MDPState:
        t = mdp.t
        data = mdp.params
        flow, regime = self.constrain(
            data=data,
            ontLevelStart=mdp.state["ontLevelBOQ"][t],
            ontLevel=data["ontLevel_forecast"],
            flow=float(np.asarray(action).item()),
            regime=data["flowRegime"],
            previous_flow=(mdp.actions["ont"]["flow"][t - 1].item() if t > 0 else 595.0),
            previous_ice=data["iceIndPrev"],
            september_rule=september_rule,
        )
        mdp.params["flowRegime"] = regime

        return mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})


@dataclass(frozen=True)
class AdjustFlowConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = AdjustFlow
    constraint_fn: Callable = planLimits

    def build(self, aid) -> AdjustFlow:
        mechanism = super().build(aid)
        mechanism.constraint_fn = self.constraint_fn

        return mechanism


class OntarioConfig(AgentConfig):
    agent_cls: ClassVar[type[Agent]] = Ontario
