from __future__ import annotations

from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, override

import numpy as np
import pandas as pd
import toml
from metamarl.mechanism.base import MDPState
from metamarl.optimizers.base import Optimizer, OptimizerConfig

from bilevel_reservoirs.controllers.operator import Ontario
from external.loslr_regulation_optimization.functions.limits.Bv7 import planLimits
from external.loslr_regulation_optimization.functions.limits.septemberRule import (
    septemberRule,
)
from external.loslr_regulation_optimization.functions.release.ruleCurve import (
    formatDecisionVariables,
    releaseFunction,
)
from external.loslr_regulation_optimization.functions.utils import round_d
from src.bilevel_reservoirs.simulators.loslr import LoslrEnv

FORCING_COLUMNS = {
    "Sim",
    "Year",
    "Month",
    "QM",
    "erieOut",
    # actual forcing
    "ontNTS",
    "stlouisontOut",
    "desprairiesOut",
    "richelieuOut",
    "stfrancoisOut",
    "stmauriceOut",
    "iceInd",
    "tidalInd",
    "ogdensburgR",
    "cardinalR",
    "iroquoishwR",
    "saundershwR",
    "iroquoistwR",
    "morrisburgR",
    "longsaultR",
    "saunderstwR",
    "cornwallR",
    "summerstownR",
    "ptclaireR",
    "batiscanR",
    "threeriversR",
    "stpierreR",
    "sorelR",
    "varennesR",
}


class ForecastSkill(StrEnum):
    """forecast skill: 12-month status quo AR(1) [sqAR], linear model [sqLM], perfect [0]"""

    sqAR = "sqAR"
    sqLM = "sqLM"
    perfect = "0"


class ForecastLeadTime(StrEnum):
    """forecast lead-time: 12month, 6month, 3month, 1month"""

    annual = "12month"
    biannual = "6month"
    quarterly = "3month"
    monthly = "1month"


class Trace(StrEnum):
    """Trace to simulate"""

    Y1900_2008 = "1900_2008"
    Y1900_2020 = "1900_2020"
    Y1961_2020 = "1961_2020"


class RuleCurveCfg(OptimizerConfig):
    def __init__(self) -> None:
        super().__init__(opt_class=RuleCurve)

        self.policy_ids: list[int] = [0]
        self.data_path: Path | None = None
        self.start_step: int = 48
        self.drop_last_year: bool = False
        self.conv: float = 2970
        self.flow_scale: float = 10.0

    @override
    def training(
        self,
        *,
        data_path: str | Path = Path("data"),
        forecast_lead_time: ForecastLeadTime = ForecastLeadTime.annual,
        forecast_skill: ForecastSkill = ForecastSkill.sqAR,
        trace: Trace = Trace.Y1961_2020,
        september_rule: bool = False,
        id: int | list[int] = 0,
        drop_last_year: bool = False,
        start_step: int = 48,
        **kwargs: Any,
    ) -> Self:
        self.data_path = Path(data_path)
        self.forecast_lead_time = forecast_lead_time
        self.forecast_skill = forecast_skill
        self.trace = trace
        self.september_rule = september_rule
        self.drop_last_year = drop_last_year
        self.start_step = start_step

        if self.start_step < 48:
            raise ValueError("Rule curve requires 48 previous quarter-months")

        # Load baseline configuration
        cfg = toml.load(self.data_path / "baseline" / "config.toml")
        self.releaseFunInputs = cfg["releaseFunction"]
        self.dvName = cfg["decisionVariables"]["dvName"]

        # Dataset
        self.input_path = (
            self.data_path
            / "input"
            / cfg["experimentalDesign"]["trace"]
            / self.trace
            / f"{self.forecast_lead_time}_{self.forecast_skill}.txt"
        )

        # Policy IDs
        self.policy_ids = [id] if isinstance(id, int) else list(id)

        if not self.policy_ids:
            raise ValueError("At least one policy ID is required")

        # Policy parameters
        policies = pd.read_csv(
            self.data_path / "baseline" / "satisficingPolicies.csv", index_col="ID"
        )
        self.pars = {
            pid: formatDecisionVariables(
                policies.loc[pid, self.dvName].tolist(), **self.releaseFunInputs
            )
            for pid in self.policy_ids
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

        return super().training(episodes=1)

    @override
    def environment(self, env=LoslrEnv, **kwargs: Any) -> Self:
        return super().environment(env=env, **kwargs)


class RuleCurve(Optimizer):
    _env: LoslrEnv

    def __init__(self, config: RuleCurveCfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)

        self.t_0 = config.start_step
        self.conv = config.conv
        self.flow_scale = config.flow_scale
        self.releaseFunInputs = config.releaseFunInputs
        self.september_rule = septemberRule if config.september_rule else "off"
        self.pids = deque(config.policy_ids)  # CHANGED
        self.pars = config.pars

        # Dataset
        self.data = pd.read_table(config.input_path)
        fcstNA = self.data["forNTS"].notna()
        fcstNA.iloc[: self.t_0] = True
        self.data = self.data.loc[fcstNA, :].reset_index(drop=True)

        if config.drop_last_year:
            self.data = self.data.loc[
                self.data["Year"] < self.data["Year"].max()
            ].reset_index(drop=True)

        # Simulation horizon
        n_steps = len(self.data) - self.t_0
        self.horizon = config.env_config["horizon"]

        if not 0 < self.horizon <= n_steps:
            raise ValueError("Invalid simulation horizon")

    @property
    def env(self) -> LoslrEnv:
        return self._env

    @env.setter
    def env(self, value: LoslrEnv) -> None:
        self._env = value

    def _apply_constraint(
        self,
        data: dict[str, Any],
        ontLevelStart: float,
        ontLevel: float,
        flow: float,
        regime: str,
        previous_flow: float,
        previous_ice: float,
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
        limitsOutput = planLimits(
            data["QM"], ontLevel, flow, regime, x, self.september_rule
        )

        return limitsOutput["ontFlow"], limitsOutput["ontRegime"]

    def _update(self) -> int:
        if not self.pids:
            raise StopIteration("No remaining policies")
        return self.pids.popleft()

    def train(self) -> dict[str, Any]:
        agent = Ontario(id="ont", policy_id="ont", mechanisms={})

        p = self._update()
        pars = self.pars[p]
        mdp = self.env.reset(MDPState())
        h = deque(self.data["ontLevelMOQ"].iloc[self.t_0 - 48 : self.t_0], maxlen=48)
        rows = []

        for i in range(self.t_0, self.t_0 + self.horizon):
            t = mdp.t
            data = {col: self.data[col].iat[i] for col in self.data.columns}
            qm = data["QM"]
            ontLevelStart = mdp.state["ontLevelBOQ"][t]

            # Rule curve release
            x = {
                "forNTS": data["forNTS"],
                "ontLevelBOQ": ontLevelStart,
                "annavgLevel": np.mean(h),
                "sfSupplyNTS": [data[f"ontNTS_QM{k}"] for k in range(1, 5)],
            }
            rfOutputs = releaseFunction(x, pars, **self.releaseFunInputs)
            flow = rfOutputs["rfFlow"]
            regime = "RF"

            # Forecasted Ontario level
            dif1 = round_d((data["ontNTS_QM1"] / self.flow_scale - flow) / self.conv, 6)
            ontLevel_forecast = ontLevelStart + dif1

            # September rule
            if self.september_rule != "off":
                t32 = t - (qm - 32)
                qm32Flow = (
                    mdp.actions["ont"]["flow"][t32].item()
                    if qm > 32 and t32 >= 0
                    else np.nan
                )
                qm32Level = (
                    mdp.state["ontLevelBOQ"][t32] if qm > 32 and t32 >= 0 else np.nan
                )
                flow, regime = self.september_rule(
                    qm, ontLevelStart, flow, regime, qm32Level, qm32Flow, conv=self.conv
                )

                if regime != "RF":
                    dif1 = round_d(
                        (data["ontNTS_QM1"] / self.flow_scale - flow) / self.conv, 6
                    )
                    ontLevel_forecast = round_d(ontLevelStart + dif1, 6)

            # Operational constraints
            flow, regime = self._apply_constraint(
                data,
                ontLevelStart,
                ontLevel_forecast,
                flow,
                regime,
                previous_flow=(
                    mdp.actions["ont"]["flow"][t - 1].item() if t > 0 else 595.0
                ),
                previous_ice=self.data["iceInd"].iloc[i - 1] if t > 0 else 2.0,
            )

            # Physical forcing
            forcing = {col: data[col] for col in FORCING_COLUMNS}

            mdp.params.update(forcing)

            # Control
            mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})

            # Physical transition
            mdp = self.env.simulate(mdp)

            # Stage cost
            reward = agent.reward(mdp)
            mdp.update(t=t, rewards={agent.id: reward.rewards[agent.id][0]})

            assert mdp.t == t + 1

            # Update controller history
            h.append(mdp.state["ontLevelMOQ"][mdp.t])

            # Simulation outputs
            row = data.copy()

            row.update(rfOutputs)
            row.update(
                {
                    "ontFlow": flow,
                    "flowRegime": regime,
                    "ontLevelBOQ": ontLevelStart,
                    "ontLevel": (ontLevelStart + mdp.state["ontLevelBOQ"][mdp.t]) * 0.5,
                }
            )
            row.update(
                {
                    col: mdp.state[col][mdp.t]
                    for col in mdp.state.data
                    if col != "ontLevelBOQ"
                }
            )
            rows.append(row)

        # aggregate stage costs into running cost
        running_cost = np.stack(mdp.rewards[agent.id]).sum(axis=0)

        return {
            "policy_id": p,
            "mdp": mdp,
            "data": pd.DataFrame(rows),
            "running_cost": running_cost,
        }
