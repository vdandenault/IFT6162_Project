"""State-explicit adapter to the native Ontario balance, Bv7 limits and routing."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from reservoir_control.core.contracts import DecisionTime, Transition
from .data import BALANCE_CONVERSION, Forcing, SupplyForecast, baseline_parameters
from .native import NativeReference


@dataclass(frozen=True)
class OntarioState:
    level_m: float
    previous_release_m3_s: float
    previous_ice: int
    mean_level_history_m: tuple[float, ...]
    september_level_m: float = math.nan
    september_release_m3_s: float = math.nan


@dataclass(frozen=True)
class DecisionInputs:
    """Forecasts and hydraulic estimates available before selecting a release."""

    forecast: SupplyForecast
    ice: int
    longsault_roughness: float
    pointe_claire_roughness: float


@dataclass(frozen=True)
class ReleaseRequest:
    m3_s: float
    regime: str = "RF"
    round_preliminary_level: bool = False
    rulecurve_flow_m3_s: float = math.nan
    preproject_flow_m3_s: float = math.nan
    rulecurve_adjustment_native: float = math.nan
    rulecurve_regime: str = ""


class Plan2014:
    """Rule curve + September policy; physical limits are applied by the engine."""

    def __init__(self, reference: NativeReference):
        self.reference = reference
        self.parameters = baseline_parameters(reference)

    def action(self, state: OntarioState, forecast: SupplyForecast,
               time: DecisionTime) -> ReleaseRequest:
        if len(state.mean_level_history_m) != 48:
            raise ValueError("Plan 2014 needs 48 preceding mean levels")
        x = {"forNTS": forecast.annual_nts_m3_s, "ontLevelBOQ": state.level_m,
             "annavgLevel": np.mean(state.mean_level_history_m),
             "sfSupplyNTS": forecast.nts_m3_s}
        result = self.reference.release.releaseFunction(x, self.parameters)
        release, regime = self.reference.september.septemberRule(
            time.period, state.level_m, result["rfFlow"], "RF",
            state.september_level_m, state.september_release_m3_s / 10,
            conv=BALANCE_CONVERSION,
        )
        return ReleaseRequest(float(release * 10), str(regime), regime != "RF",
                              float(result["rfFlow"] * 10), float(result["pprFlow"] * 10),
                              float(result["rfOutput"]), str(result["rfRegime"]))


class OntarioModel:
    """No Gymnasium or controller dependency; action is an outflow request in m³/s.

    With explicit decision inputs, Bv7 limits use decision-time estimates.
    The default path preserves native reference execution.
    The fixed native balance convention deliberately does not use duration_seconds.
    """

    def __init__(self, reference: NativeReference):
        self.reference = reference

    def transition(self, state: OntarioState, action: ReleaseRequest | float,
                   forcing: Forcing, time: DecisionTime, *,
                   decision: DecisionInputs | None = None) -> Transition[OntarioState]:
        limits = self.release_limits(state, action, time, decision=decision, forcing=forcing)
        if not isinstance(action, ReleaseRequest):
            action = ReleaseRequest(float(action))
        rnd = self.reference.utils.round_d
        actual_native = float(limits["ontFlow"])
        next_state = self.advance_storage(state, actual_native * 10, forcing.nts_m3_s,
                                          forcing.ice, time)
        mean_unrounded = (state.level_m + next_state.level_m) * 0.5
        route_inputs = dict(forcing.native_metadata)
        route_inputs.update({"slonFlow": forcing.slon_m3_s, "iceInd": forcing.ice})
        # Preserve the native input formatter's mappings, including its jetty1R
        # alias to varennesR, rather than silently correcting reference behavior.
        route_inputs["jetty1R"] = route_inputs["varennesR"]
        route = self.reference.routing.stLawrenceRouting(mean_unrounded, actual_native, route_inputs)
        metrics = {
            "ontario_level_start_m": state.level_m,
            "ontario_level_mean_m": mean_unrounded,
            "ontario_level_end_m": rnd(next_state.level_m, 2),
            "requested_release_m3_s": action.m3_s,
            "executed_release_m3_s": actual_native * 10,
            "lake_st_louis_flow_m3_s": float(route["stlouisFlow"] * 10),
            "pointe_claire_level_m": float(route["ptclaireLevel"]),
        }
        info = {"native_routing": route, "native_limits": limits,
                "flow_regime": limits["ontRegime"], "balance_conversion": BALANCE_CONVERSION,
                "f_limit_uses_realized_slon": decision is None and not forcing.use_supplied_forecast_for_f_limit,
                "m_limit_uses_realized_slon": decision is None,
                "reference_calendar": "48 periods/year; fixed native balance conversion"}
        return Transition(next_state, metrics, info)

    def release_limits(self, state: OntarioState, action: ReleaseRequest | float,
                       time: DecisionTime, *, decision: DecisionInputs | None = None,
                       forcing: Forcing | None = None) -> dict:
        """Project a request through the same Bv7 rules used in execution."""
        if decision is None and forcing is None:
            raise ValueError("Supply decision estimates or native replay forcing")
        if not math.isclose(time.duration_seconds, 365.0 * 86400.0 / 48,
                            rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Ontario reference duration must use its fixed nominal 365/48-day reporting convention")
        if not 1 <= time.period <= 48:
            raise ValueError("Ontario period must be in 1..48")
        if len(state.mean_level_history_m) != 48:
            raise ValueError("State must contain 48 preceding mean levels")
        if not isinstance(action, ReleaseRequest):
            action = ReleaseRequest(float(action))
        if not math.isfinite(action.m3_s) or action.m3_s < 0:
            raise ValueError("Release request must be finite and nonnegative")
        rnd = self.reference.utils.round_d
        request_native = action.m3_s / 10
        forecast = forcing.forecast if decision is None else decision.forecast
        difference = rnd((forecast.nts_m3_s[0] / 10 - request_native)
                         / BALANCE_CONVERSION, 6)
        preliminary_level = state.level_m + difference
        if action.round_preliminary_level:
            preliminary_level = rnd(preliminary_level, 6)
        if decision is None:
            limit_inputs = dict(forcing.native_metadata)
            limit_inputs.update({
                "QM": time.period, "iceInd": forcing.ice, "iceIndPrev": state.previous_ice,
                "ontFlowPrev": state.previous_release_m3_s / 10,
                "obsontNTS": forcing.nts_m3_s, "sfSupplyNTS": forcing.forecast.nts_m3_s[0],
                "ontLevelStart": state.level_m, "kingLevelStart": state.level_m - 0.03,
                "foreInd": int(forcing.use_supplied_forecast_for_f_limit),
                "sfSlonFlow": forcing.forecast.slon_m3_s, "slonFlow": forcing.slon_m3_s,
            })
        else:
            # Bv7 reads slonFlow for M and sfSlonFlow for F. Supply the
            # decision-time estimate to both and exclude realized metadata.
            limit_inputs = {
                "QM": time.period, "iceInd": decision.ice, "iceIndPrev": state.previous_ice,
                "ontFlowPrev": state.previous_release_m3_s / 10,
                "obsontNTS": forecast.nts_m3_s[0], "sfSupplyNTS": forecast.nts_m3_s[0],
                "ontLevelStart": state.level_m, "kingLevelStart": state.level_m - 0.03,
                "foreInd": 1, "sfSlonFlow": forecast.slon_m3_s, "slonFlow": forecast.slon_m3_s,
                "longsaultR": decision.longsault_roughness,
                "ptclaireR": decision.pointe_claire_roughness,
                # Required by the pinned function's input unpacking, but unused.
                "saundershwR": math.nan, "saunderstwR": math.nan,
            }
        limits = self.reference.limits.planLimits(
            time.period, preliminary_level, request_native, action.regime,
            limit_inputs, self.reference.september.septemberRule, output="all",
        )
        return limits

    def advance_storage(self, state: OntarioState, release_m3_s: float, nts_m3_s: float,
                        ice: int, time: DecisionTime) -> OntarioState:
        rnd = self.reference.utils.round_d
        actual_native = release_m3_s / 10
        delta = rnd((nts_m3_s / 10 - actual_native) / BALANCE_CONVERSION, 6)
        next_level = state.level_m + delta
        mean_unrounded = (state.level_m + next_level) * 0.5
        mean_rounded = rnd(mean_unrounded, 2)
        september_level = state.september_level_m
        september_release = state.september_release_m3_s
        if time.period <= 31:
            september_level = september_release = math.nan
        elif time.period == 32:
            september_level, september_release = state.level_m, actual_native * 10
        return OntarioState(
            next_level, release_m3_s, ice, (*state.mean_level_history_m[1:], mean_rounded),
            september_level, september_release,
        )
