# TODO fix metamarl hatchling
# TODO rolling window
# TODO constraint optimization (solvers)
# TODO testing + CI pass

from pathlib import Path
from typing import Any

import numpy as np
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState

from external.loslr_regulation_optimization.functions.limits.onlyPhysicalLimits import (
    getPlanLimitsInputs,
    planLimits,
)
from external.loslr_regulation_optimization.functions.limits.septemberRule import (
    septemberRule,
)
from external.loslr_regulation_optimization.functions.routing.stlaw import (
    getStLawrenceRoutingInputs,
    stLawrenceRouting,
)
from external.loslr_regulation_optimization.functions.utils import round_d

SIMULATED_COLUMNS = {
    "ontLevel",
    "ontFlow",
    "ontLevelBOQ",
    "ontLevelEOQ",
    "ontLevelMOQ",
}


class LoslrLeaderEnv(MultiAgentEnv):
    def __init__(self, ecology: dict, **kwargs: Any):
        super().__init__(**kwargs)

        # Dataset
        self.data = np.genfromtxt(
            Path(ecology["data_path"]),
            names=True,
            delimiter="\t",
            dtype=None,
            encoding=None,
        )

        self.data.setflags(write=False)

        self.historical_columns = tuple(
            name for name in self.data.dtype.names if name not in SIMULATED_COLUMNS
        )
        self.history_window = int(ecology.get("history_window", 48))

        # Lake Ontario dynamics
        self.conv = float(ecology.get("conv", 2970))
        self.flow_scale = float(ecology.get("flow_scale", 10.0))

        # Initial conditions
        self.initial_level = float(ecology.get("initial_level", 74.55))
        self.start_step = int(ecology.get("start_step", 48))
        self.initial_previous_flow = float(ecology.get("initial_previous_flow", 595.0))
        self.initial_previous_ice = float(ecology.get("initial_previous_ice", 2.0))

        # TODO (nadine) possibly move to operator agent
        # Operating constraints
        self.september_rule = ecology.get("september_rule", False)
        self.limit_type = ecology.get("limit_type", "onlyPhysicalLimits")

        # Simulation horizon
        # TODO (nadine) review rolling window implementation
        if not 0 <= self.start_step < len(self.data):
            raise ValueError("Invalid start_step")

        self.n_steps = len(self.data) - self.start_step

        if self.horizon is None:
            self.horizon = self.n_steps
        elif self.horizon > self.n_steps:
            raise ValueError("Horizon exceeds available data")

    def _historical_state(self, index: int) -> dict:
        """Read one historical quarter-month from memory."""

        row = self.data[index]

        return {name: row[name].item() for name in self.historical_columns}

    @reset
    def reset_loslr(self, mdp: MDPState) -> MDPState:
        state = self._historical_state(self.start_step)

        state.update(
            {
                "ontLevelBOQ": self.initial_level,
                "ontFlowPrev": self.initial_previous_flow,
                "iceIndPrev": self.initial_previous_ice,
            }
        )

        return MDPState(
            params={
                "conv": self.conv,
                "flow_scale": self.flow_scale,
                "start_step": self.start_step,
            },
            state=state,
        )

    @transition
    def simulate(self, mdp: MDPState) -> MDPState:
        t = mdp.t

        if t >= self.horizon:
            raise IndexError("Simulation horizon reached")

        qm = mdp.state["QM"][t]
        level =mdp.state["ontLevelBOQ"][t]
        nts = mdp.state["ontNTS"][t]
        forecast_nts = mdp.state["ontNTS_QM1"][t]

        # Control
        # TODO implement finite and possibly move constraints there
        release = np.asarray(mdp.actions["operator"]["release"][t]).reshape(-1)[0]

        # Preliminary predicted level
        predicted_change = round_d((forecast_nts / self.flow_scale - release) / self.conv, 6)
        predicted_level = level + predicted_change

        # Proposed release and regime
        regime = "RF"

        # September rule
        if self.september_rule:
            qm32_flow = np.nan
            qm32_level = np.nan

            if qm > 32:
                qm32_t = t - (qm - 32)

                if qm32_t < 0:
                    raise ValueError("QM32 history is unavailable for September rule")

                qm32_flow = mdp.state["ontFlow"][qm32_t]
                qm32_level = mdp.state["ontLevelBOQ"][qm32_t]

            release, regime = septemberRule(
                qm,
                level,
                release,
                regime,
                qm32_level,
                qm32_flow,
                conv=self.conv,
            )

            if regime != "RF":
                predicted_change = round_d(
                    (forecast_nts / self.flow_scale - release) / self.conv,
                    6,
                )
                predicted_level = round_d(
                    level + predicted_change,
                    6,
                )

        # Operating limits
        if self.limit_type == "onlyPhysicalLimits":
            if t == 0:
                # Previous-period boundary conditions
                limit_inputs = {
                    "QM": qm,
                    "iceInd": mdp.state["iceInd"][t],
                    "iceIndPrev": self.initial_previous_ice,
                    "obsontNTS": nts,
                    "ontLevelStart": level,
                    "kingLevelStart": level - 0.03,
                    "longsaultR": mdp.state["longsaultR"][t],
                    "saundershwR": mdp.state["saundershwR"][t],
                    "saunderstwR": mdp.state["saunderstwR"][t],
                }
            else:
                limit_inputs = getPlanLimitsInputs(mdp.state.data, t)

            limits_output = planLimits(
                qm,
                predicted_level,
                release,
                regime,
                limit_inputs,
                septemberRule if self.september_rule else "off",
                conv=self.conv,
            )
            release = limits_output["ontFlow"]
            regime = limits_output["ontRegime"]
        elif self.limit_type not in ("none", "off"):
            raise ValueError(f"Unknown limit type: {self.limit_type}")

        if not np.isfinite(release):
            raise ValueError("Final release must be finite")

        # Lake Ontario physical dynamics
        level_change = round_d(
            (nts / self.flow_scale - release) / self.conv,
            6,
        )
        next_level = level + level_change
        end_level = round_d(next_level, 2)
        mean_level = (level + next_level) / 2
        mean_level_rounded = round_d(mean_level, 2)

        # St. Lawrence routing
        routing_inputs = getStLawrenceRoutingInputs(
            mdp.state.data,
            t,
        )
        routing_outputs = stLawrenceRouting(
            mean_level,
            release,
            routing_inputs,
        )

        # Current timestep outputs
        mdp.update(
            state={
                "ontFlow": release,
                "flowRegime": regime,
                "rfFlow": release,
                "ontLevelEOQ": end_level,
                "ontLevelMOQ": mean_level_rounded,
                "ontLevel": mean_level,
                **routing_outputs,
            }
        )

        # Next timestep
        next_state = {
            "ontLevelBOQ": next_level,
            "ontFlowPrev": release,
            "iceIndPrev": mdp.state["iceInd"][t],
        }
        next_index = self.start_step + t + 1

        if next_index < len(self.data):
            next_state.update(self._historical_state(next_index))

        return mdp.advance(state=next_state)
