from __future__ import annotations

import tomllib
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, override

import numpy as np
import pandas as pd
from core.mechanism.base import MDPState
from core.optimizers.base import Optimizer, OptimizerConfig

from reservoir_control.systems.loslr.data import read_policies, read_trace, trace_path
from reservoir_control.systems.loslr.native import native_reference
from reservoir_control.systems.loslr.simulator import LoslrEnv

round_d = native_reference().utils.round_d
septemberRule = native_reference().september.septemberRule
formatDecisionVariables = native_reference().release.formatDecisionVariables
releaseFunction = native_reference().release.releaseFunction

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
    "ontNTS_QM1",
    "ontNTS_QM2",
    "ontNTS_QM3",
    "ontNTS_QM4",
    "foreInd",
    "slonFlow_QM1",
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

        with (self.data_path / "baseline" / "config.toml").open("rb") as config:
            cfg = tomllib.load(config)
        self.releaseFunInputs = cfg["releaseFunction"]
        self.dvName = cfg["decisionVariables"]["dvName"]
        trace_name = Path(cfg["experimentalDesign"]["trace"]) / self.trace / f"{self.forecast_lead_time}_{self.forecast_skill}"
        self.input_path = trace_path(self.data_path, trace_name)
        self.policy_ids = [id] if isinstance(id, int) else list(id)

        if not self.policy_ids:
            raise ValueError("At least one policy ID is required")

        policies = read_policies(self.data_path)
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
        env_config = {"action_mode": "flow", **kwargs.pop("env_config", {})}
        if env_config["action_mode"] != "flow":
            raise ValueError("Rule curve and inherited MPC solvers require flow mode")
        return super().environment(env=env, env_config=env_config, **kwargs)

    @override
    def _env_creator(self, *, agents_cfgs=None, **kwargs):
        if agents_cfgs is not None:
            kwargs["agents_cfg_dict"] = agents_cfgs

        return super()._env_creator(**kwargs)


class RuleCurve(Optimizer):
    _env: LoslrEnv

    def __init__(self, config: RuleCurveCfg, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        if self.world is None:
            self.id = type(self).__name__

        self.t_0 = config.start_step
        self.conv = config.conv
        self.flow_scale = config.flow_scale
        self.releaseFunInputs = config.releaseFunInputs
        self.september_rule = septemberRule if config.september_rule else "off"
        self.pids = deque(config.policy_ids)
        self.pars = config.pars
        self.data = read_trace(config.input_path)
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

    def _simulate(self, mdp: MDPState) -> MDPState:
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
        x = {**params, "slonFlow": params["stlouisontOut"], "jetty1R": params["varennesR"]}
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

            # Release Function
            # extract hydrologic variables from input dictionary
            lfSupply = data["forNTS"]
            ontLevelBOQ = mdp.state["ontLevelBOQ"][t]
            annavgLevel = np.mean(h)
            sfSupplyNTS = [data[f"ontNTS_QM{k}"] for k in range(1, 5)]

            # simplified formulation to determine if wet and confident
            wet = self.pars[p]["wetIndicatorThreshold"]
            low99limit = lfSupply - self.pars[p]["wetConfidenceThreshold"]
            lfInd = np.nan
            lfCon = np.nan

            # if the supply index is greater than the wet threshold
            if lfSupply > wet:
                lfInd = 1

                # if lower confidence interval is greater than the wet threshold
                if low99limit >= wet:
                    lfCon = 3

            # else:
            #     lfInd = np.nan
            #     lfCon = np.nan

            # -------------------------------------------------------------------------
            # rule curve [short forecast and balance subroutine]
            # -------------------------------------------------------------------------

            # calculate rule curve release for each forecasted quarter-month (1 - 4)
            nforecasts = 4
            startLev = []

            startLev.append(ontLevelBOQ)

            sfFlow = []

            for k in range(nforecasts):
                # set constants and indicators
                slope = 55.5823
                adj = 0.0014 * (2010 - 1985)
                H_0 = 69.474 + adj
                lastflow = 0
                ont_blv = startLev[k]

                # The actual rule_curve policy
                for _ in range(10):
                    preproj = slope * (ont_blv - H_0) ** 1.5

                    if lfSupply >= 7011.0:
                        if lfInd == 1 and lfCon == 3:
                            c1 = self.pars[p]["C1*"]
                            regime = "RCW*"
                        else:
                            c1 = self.pars[p]["C1"]
                            regime = "RCW"

                        p1 = self.pars[p]["P1"]
                        rc = c1 * ((lfSupply - 7011.0) / (8552.0 - 7011.0)) ** p1
                        signed_rc = rc
                    else:
                        c2 = self.pars[p]["C2"]
                        p2 = self.pars[p]["P2"]
                        regime = "RCD"
                        rc = c2 * ((7011.0 - lfSupply) / (7011.0 - 5717.0)) ** p2
                        signed_rc = -rc

                    # Preserve legacy release rounding and convergence
                    release = round_d(preproj + signed_rc, 0)

                    if abs(release - lastflow) <= 0.0001:
                        break

                    endLev_new = startLev[k] + (sfSupplyNTS[k] / 10 - release) / self.conv
                    ont_blv = round_d((startLev[k] + endLev_new) * 0.5, 2)
                    lastflow = release

                # try to keep ontario level up in dry periods
                if annavgLevel <= self.pars[p]["dryLevelThreshold"]:
                    release = release + self.pars[p]["dryFlowAdjustment"]
                    regime = regime + "-"

                sfFlow.append(release)

                # compute water level change using forecasted supply and flow
                dif1 = round_d((sfSupplyNTS[k] / 10 - release) / self.conv, 6)
                endLev1 = round_d(startLev[k] + dif1, 6)

                # Update initial conditions for next forecast
                if k < list(range(nforecasts))[-1]:
                    startLev.append(endLev1)

            # Calculate average release across the four forecasts
            flow = round_d(sum(sfFlow) / nforecasts, 0)
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
            mdp = agent.mechanisms["adjustment"].apply(
                mdp, flow, september_rule=self.september_rule
            )

            # Physical transition
            mdp = self._simulate(mdp)

            # Stage cost
            mdp.update(t=t, rewards={agent.id: agent.reward(mdp).rewards[agent.id][0]})
            assert mdp.t == t + 1
            h.append(mdp.state["ontLevelMOQ"][mdp.t])
            rows.append(
                {
                    **data,
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
