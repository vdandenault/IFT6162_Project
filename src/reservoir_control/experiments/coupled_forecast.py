from dataclasses import replace
from math import ceil

import numpy as np

from reservoir_control.systems.loslr.physics import CONVERSION, FLOW_SCALE
from reservoir_control.systems.loslr.simulator import LoslrEnv, round_d

ONTARIO_HORIZON = 4


def native_step(forcing, supply, flow, state, limits):
    """Use the same forecast-level and native-limit check as fixed execution."""
    level, previous_flow, previous_ice = state
    end = level + round_d((supply / FLOW_SCALE - flow) / CONVERSION, 6)
    allowed, regime = limits.constrain(
        forcing, level, end, flow, "RF", previous_flow=previous_flow, previous_ice=previous_ice
    )
    return end, allowed, regime


def forecast_forcing(rows, lead, non_ontario, future_roughness=None):
    issued = rows[0]
    supply = issued[f"ontNTS_QM{lead + 1}"]
    return {
        **issued,
        **{name: rows[lead][name] for name in ("Sim", "Year", "Month", "QM")},
        "ontNTS": supply,
        "ontNTS_QM1": supply,
        "ptclaireR": future_roughness
        if lead > 0 and future_roughness is not None
        else issued["ptclaireR"],
        "stlouisontOut": non_ontario,
        "slonFlow_QM1": non_ontario,
    }


# Beginning-of-period level tiers from the pinned native Bv7 F limit.
F_LIMIT_TIERS = (
    (75.30, 22.10, 11523.848),
    (75.37, 22.20, 11885.486),
    (75.50, 22.33, 12362.610),
    (75.60, 22.40, 12622.784),
    (float("inf"), 22.48, 12922.906),
)


def forecast_f_ceiling(level, roughness, commitment):
    """Total St. Louis flow before F would lower an otherwise admissible release."""
    _, action_level, coefficient = next(tier for tier in F_LIMIT_TIERS if level < tier[0])
    # Bv7 lowers the commitment only when both the rounded stage and the
    # rounded allowed release cross their thresholds.
    stage_ceiling = 604.0 / roughness * (action_level + 0.005 - 16.57) ** (1 / 0.58)
    rounded_release_ceiling = (
        coefficient / roughness + commitment - FLOW_SCALE * (ceil(commitment / FLOW_SCALE) - 0.5)
    )
    # Stay inside the boundary regardless of the native half-tie rule and LP tolerance.
    return max(stage_ceiling, rounded_release_ceiling) - 1e-4


def capped_follower(problem, rows, schedule, level, future_roughness=None):
    """Apply candidate-specific forecast F ceilings to Ontario's available leads."""
    caps = np.array(problem.downstream_capacity_m3_s, dtype=float)
    for lead, commitment in enumerate(schedule[:ONTARIO_HORIZON]):
        forcing = forecast_forcing(rows, lead, 0.0, future_roughness)
        caps[lead] = min(caps[lead], forecast_f_ceiling(level, forcing["ptclaireR"], commitment))
        level += round_d(
            (forcing["ontNTS_QM1"] / FLOW_SCALE - commitment / FLOW_SCALE) / CONVERSION, 6
        )
    return replace(problem, ontario_executable_release_m3_s=schedule, downstream_capacity_m3_s=caps)


def forecast_trajectory(
    rows, schedule, arrivals, background, caps, state, limits, tolerance, future_roughness=None
):
    """Evaluate only the four leads available to Ontario for native limits and cost."""
    level, previous_flow, previous_ice = state
    predicted = []
    combined = np.asarray(schedule) + np.asarray(background) + arrivals
    breaches = np.flatnonzero(combined > np.asarray(caps) + tolerance.flow_m3_s)
    if len(breaches):
        lead = int(breaches[0])
        return (
            None,
            f"study_cap_lead_{lead + 1}",
            {"rejected_combined_m3_s": float(combined[lead]), "rejected_cap_m3_s": caps[lead]},
        )
    for lead in range(min(len(rows), ONTARIO_HORIZON)):
        flow = schedule[lead] / FLOW_SCALE
        non_ontario = background[lead] + arrivals[lead]
        forcing = forecast_forcing(rows, lead, non_ontario, future_roughness)
        supply = forcing["ontNTS_QM1"]
        end, allowed, regime = native_step(
            forcing, supply, flow, (level, previous_flow, previous_ice), limits
        )
        if abs(allowed - flow) * FLOW_SCALE > tolerance.flow_m3_s:
            return (
                None,
                f"native_{regime}_lead_{lead + 1}",
                {
                    "rejected_commitment_m3_s": schedule[lead],
                    "native_allowed_m3_s": allowed * FLOW_SCALE,
                    "rejected_forecast_non_ontario_m3_s": non_ontario,
                    "rejected_forecast_ontario_supply_m3_s": supply,
                },
            )
        routed = LoslrEnv.lrstep(
            level, end, flow, {**forcing, "slonFlow": non_ontario, "jetty1R": forcing["varennesR"]}
        )
        predicted.append({**forcing, **routed, "ontLevel": (level + end) / 2, "ontFlow": flow})
        level, previous_flow, previous_ice = end, flow, forcing["iceInd"]
    return predicted, "admissible", {}
