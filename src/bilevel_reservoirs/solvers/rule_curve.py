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

from bilevel_reservoirs.simulators.loslr import LoslrEnv
from external.loslr_regulation_optimization.functions.limits.septemberRule import septemberRule
from external.loslr_regulation_optimization.functions.release.ruleCurve import (
    formatDecisionVariables,
    releaseFunction,
)
from external.loslr_regulation_optimization.functions.utils import round_d

FORCING_COLUMNS = {
    "Sim",
    "Year",
    "Month",
    "QM",
    "erieOut",
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
    sqAR = "sqAR"
    sqLM = "sqLM"
    perfect = "0"


class ForecastLeadTime(StrEnum):
    annual = "12month"
    biannual = "6month"
    quarterly = "3month"
    monthly = "1month"


class Trace(StrEnum):
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

        cfg = toml.load(self.data_path / "baseline" / "config.toml")
        self.releaseFunInputs = cfg["releaseFunction"]
        self.dvName = cfg["decisionVariables"]["dvName"]
        self.input_path = (
            self.data_path
            / "input"
            / cfg["experimentalDesign"]["trace"]
            / self.trace
            / f"{self.forecast_lead_time}_{self.forecast_skill}.txt"
        )
        self.policy_ids = [id] if isinstance(id, int) else list(id)

        if not self.policy_ids:
            raise ValueError("At least one policy ID is required")

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

    @override
    def _env_creator(self, *, agents_cfgs=None, **kwargs):
        if agents_cfgs is not None:
            kwargs["agents_cfg_dict"] = agents_cfgs

        return super()._env_creator(**kwargs)


class RuleCurve(Optimizer):
    _env: LoslrEnv

    def __init__(self, config: RuleCurveCfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)

        self.t_0 = config.start_step
        self.conv = config.conv
        self.flow_scale = config.flow_scale
        self.releaseFunInputs = config.releaseFunInputs
        self.september_rule = septemberRule if config.september_rule else "off"
        self.pids = deque(config.policy_ids)
        self.pars = config.pars
        self.data = pd.read_table(config.input_path)
        fcstNA = self.data["forNTS"].notna()
        fcstNA.iloc[: self.t_0] = True
        self.data = self.data.loc[fcstNA].reset_index(drop=True)

        if config.drop_last_year:
            self.data = self.data.loc[self.data["Year"] < self.data["Year"].max()].reset_index(
                drop=True
            )

        self.horizon = config.env_config["horizon"]

        if not 0 < self.horizon <= len(self.data) - self.t_0:
            raise ValueError("Invalid simulation horizon")

    @property
    def env(self) -> LoslrEnv:
        return self._env

    @env.setter
    def env(self, value: LoslrEnv) -> None:
        self._env = value

    def _update(self) -> int:
        if not self.pids:
            raise StopIteration("No remaining policies")

        return self.pids.popleft()

    def train(self) -> dict[str, Any]:
        agent = self.env.agents["ont"]
        p = self._update()
        mdp = self.env.reset(MDPState())
        h = deque(self.data["ontLevelMOQ"].iloc[self.t_0 - 48 : self.t_0], maxlen=48)
        rows = []

        for i in range(self.t_0, self.t_0 + self.horizon):
            t = mdp.t
            data = {col: self.data[col].iat[i] for col in self.data.columns}
            qm = data["QM"]
            ontLevelStart = mdp.state["ontLevelBOQ"][t]

            # Rule curve release
            rfOutputs = releaseFunction(
                {
                    "forNTS": data["forNTS"],
                    "ontLevelBOQ": ontLevelStart,
                    "annavgLevel": np.mean(h),
                    "sfSupplyNTS": [data[f"ontNTS_QM{k}"] for k in range(1, 5)],
                },
                self.pars[p],
                **self.releaseFunInputs,
            )
            flow = rfOutputs["rfFlow"]
            regime = "RF"
            ontLevel_forecast = ontLevelStart + round_d(
                (data["ontNTS_QM1"] / self.flow_scale - flow) / self.conv, 6
            )

            # September rule
            if self.september_rule != "off":
                t32 = t - (qm - 32)
                qm32Flow = (
                    mdp.actions["ont"]["flow"][t32].item() if qm > 32 and t32 >= 0 else np.nan
                )
                qm32Level = mdp.state["ontLevelBOQ"][t32] if qm > 32 and t32 >= 0 else np.nan
                flow, regime = self.september_rule(
                    qm, ontLevelStart, flow, regime, qm32Level, qm32Flow, conv=self.conv
                )

                if regime != "RF":
                    ontLevel_forecast = round_d(
                        ontLevelStart
                        + round_d((data["ontNTS_QM1"] / self.flow_scale - flow) / self.conv, 6),
                        6,
                    )

            # Forcing and constraint inputs
            mdp.params.update(
                {
                    col: data[col]
                    for col in FORCING_COLUMNS | {"ontNTS_QM1", "slonFlow_QM1", "foreInd"}
                }
            )
            mdp.params.update(
                {
                    "ontLevel_forecast": ontLevel_forecast,
                    "flowRegime": regime,
                    "iceIndPrev": self.data["iceInd"].iloc[i - 1] if t > 0 else 2.0,
                }
            )

            # Operational constraints
            mdp = agent.mechanisms["flow"].apply(mdp, flow, september_rule=self.september_rule)

            # Physical transition
            mdp = self.env.simulate(mdp)

            # Stage cost
            mdp.update(t=t, rewards={agent.id: agent.reward(mdp).rewards[agent.id][0]})
            assert mdp.t == t + 1
            h.append(mdp.state["ontLevelMOQ"][mdp.t])
            rows.append(
                {
                    **data,
                    **rfOutputs,
                    "ontFlow": mdp.actions["ont"]["flow"][t].item(),
                    "flowRegime": mdp.params["flowRegime"],
                    "ontLevelBOQ": ontLevelStart,
                    "ontLevel": (ontLevelStart + mdp.state["ontLevelBOQ"][mdp.t]) * 0.5,
                    **{
                        col: mdp.state[col][mdp.t] for col in mdp.state.data if col != "ontLevelBOQ"
                    },
                }
            )

        return {
            "policy_id": p,
            "mdp": mdp,
            "data": pd.DataFrame(rows),
            "running_cost": np.stack(mdp.rewards[agent.id]).sum(axis=0),
        }
