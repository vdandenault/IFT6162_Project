"""Attribute a candidate rejection using offline substitutions at one decision."""

from argparse import ArgumentParser
import ast
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from bilevel_reservoirs.controllers.operator import AdjustFlow
from bilevel_reservoirs.simulators.loslr import round_d
from bilevel_reservoirs.simulators.physics import CONVERSION, FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import (
    ONTARIO_HORIZON, forecast_forcing, native_step,
)

GROUPS = ("ottawa_arrival", "ontario_supply", "routing_conditions")
ROUTING_FIELDS = ("iceInd", "foreInd", "ptclaireR", "longsaultR", "saundershwR", "saunderstwR")


def substitutions(rows, fixed, arrivals, backgrounds, state, limits):
    """Keep the inherited state and commitments fixed in all substitutions."""
    horizon = min(ONTARIO_HORIZON, len(fixed))
    groups = [group for size in range(len(GROUPS) + 1) for group in combinations(GROUPS, size)]
    groups.append(("pointe_claire_roughness",))
    for group in groups:
        level, previous_flow, previous_ice = state
        for lead, actual in enumerate(rows[:horizon]):
            flow = fixed[lead]["ontario_applied_m3_s"] / FLOW_SCALE
            arrival = arrivals[lead]
            if "ottawa_arrival" in group and lead > 0:
                arrival = fixed[lead]["ottawa_arrival_m3_s"]
            non_ontario = backgrounds[lead] + arrival
            forcing = forecast_forcing(rows, lead, non_ontario)
            supply_for_balance = forcing["ontNTS"]
            if "ontario_supply" in group:
                forcing.update(ontNTS=actual["ontNTS"], ontNTS_QM1=actual["ontNTS_QM1"])
                supply_for_balance = actual["ontNTS"]
            if "routing_conditions" in group:
                forcing.update({key: actual[key] for key in ROUTING_FIELDS})
            if "pointe_claire_roughness" in group:
                forcing["ptclaireR"] = actual["ptclaireR"]
            _, allowed, regime = native_step(
                forcing, forcing["ontNTS_QM1"], flow, (level, previous_flow, previous_ice), limits,
            )
            yield {
                "substitutions": "+".join(group) or "original_forecast",
                "lead": lead + 1, "quarter_month": actual["QM"],
                "ontario_start_level_m": level, "commitment_m3_s": flow * FLOW_SCALE,
                "allowed_m3_s": allowed * FLOW_SCALE, "regime": regime,
                "accepted": abs(allowed - flow) * FLOW_SCALE <= 1e-5,
                "ottawa_arrival_m3_s": arrival, "non_ontario_flow_m3_s": non_ontario,
                "supply_used_for_limit_m3_s": forcing["ontNTS_QM1"],
                "supply_used_for_balance_m3_s": supply_for_balance,
                "pointe_claire_roughness": forcing["ptclaireR"],
            }
            # Continue the shadow trajectory at the commitment to inspect each lead.
            level += round_d((supply_for_balance / FLOW_SCALE - flow) / CONVERSION, 6)
            previous_flow, previous_ice = flow, forcing["iceInd"]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--fixed-replay", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--period", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.trajectory)
    fixed = pd.read_csv(args.fixed_replay)
    if (not fixed["status"].eq("executed").all()
            or not np.array_equal(source[["Year", "QM"]], fixed[["year", "quarter_month"]])):
        parser.error("The fixed replay must execute all periods and match the Ontario trajectory dates")
    k = args.period - 1
    if not 0 < k < len(fixed):
        parser.error("Choose a period with an executed predecessor")
    candidates = pd.read_csv(args.comparison / "candidates.csv")
    limits = AdjustFlow(aid="ont", id="flow", acts_on=("ont", "flow"),
                        action_space=Box(0.0, np.inf, shape=(1,), dtype=np.float64))
    results = []
    for policy in ("external", "anticipating"):
        trace = pd.read_csv(args.comparison / f"{policy}_trace.csv")
        prior = trace.loc[(trace.period == k) & trace.status.eq("executed")].iloc[0]
        pool = candidates.loc[(candidates.period == args.period) & candidates.policy.eq(policy)]
        reference = tuple(fixed.ontario_applied_m3_s.iloc[k:k + 5])
        entry = pool.loc[pool.schedule_m3_s.map(ast.literal_eval).map(lambda plan: plan == reference)].iloc[0]
        arrivals = ast.literal_eval(entry.predicted_ottawa_arrival_m3_s)
        state = prior.ontario_end_m, prior.ontario_applied_m3_s / FLOW_SCALE, source.iceInd.iloc[k - 1]
        results.extend(
            {"policy": policy, **record} for record in substitutions(
                source.iloc[k:k + 5].to_dict("records"), fixed.iloc[k:k + 5].to_dict("records"),
                arrivals, fixed.background_m3_s.iloc[k:k + 5].tolist(), state, limits,
            )
        )
    destination = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as output:
        destination.to_csv(output, index=False)
    print(destination.loc[destination.lead.eq(2),
                          ["policy", "substitutions", "allowed_m3_s", "regime", "accepted"]].to_string(index=False))


if __name__ == "__main__":
    main()
