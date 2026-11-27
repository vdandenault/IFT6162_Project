"""Separate storage reserve, fixed fifth commitment and candidate coverage offline."""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from bilevel_reservoirs.controllers.operator import AdjustFlow
from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.compare_ottawa_response import capped_follower, forecast_trajectory
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, FollowerTolerances, TransitState, json_value, make_problem,
)
from bilevel_reservoirs.solvers.scenario_ottawa import choose_two_paths, gradual_schedules
from reservoir_control.systems.ottawa.delay_diagnostic import explicit_storage_feasibility


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    root = args.archive_root
    audit = json.loads((root / "ottawa_plateau_recoverability_verified/prefixes.json").read_text())["cases"]
    design = json.loads((root / "ottawa_two_path_wider_fresh_v1/settings.json").read_text())["design"]
    pending = [case for case in audit if case.get("forecast_probe", {}).get("accepted_with_realized_plateau") is False]
    if len(pending) != 9:
        raise ValueError("Expected the nine stops not reopened by the realized-inflow probe")
    args.output.mkdir(parents=True, exist_ok=False)
    scenario_base = next(row for row in config["scenarios"] if row["id"] == "one_period_delay")
    tolerance = FollowerTolerances(**config["acceptance"])
    mechanism = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0., np.inf, (1,), dtype=np.float64))
    records = []
    for case in pending:
        year = case["year"]
        parent = pd.read_csv(root / f"ottawa_replan_guard_{year}/ontario_inputs.csv")
        settings = {**config, "known_ontario_executable_schedule_m3_s": (parent.ontFlow * FLOW_SCALE).tolist()}
        inflows = next(pair["plateau"] for pair in design["pairs"] if pair["name"] == case["pair"])
        scenario = {**scenario_base, "realized_net_inflow_m3_s": inflows}
        prefix = case["prefixes"][-1]
        k = prefix["executed_periods"]
        actual = inflows[k:k + 5]
        rows = parent.iloc[k:k + 5].to_dict("records")
        state = (prefix["ontario_level_m"], prefix["previous_ontario_release_m3_s"] / FLOW_SCALE,
                 prefix["previous_ice"])
        base = make_problem(settings, scenario, k, prefix["ottawa_storage_m3"],
                            prefix["previous_ottawa_release_m3_s"], TransitState(tuple(prefix["transit_queue_m3"])),
                            actual)
        witness = prefix["witness"]["ontario_schedule_m3_s"][:5]
        guarded = np.array(base.storage_maximum_m3, copy=True)
        guarded[1:] -= base.duration_seconds[0] * config["forecast_noise"]["standard_deviation_m3_s"]
        cells = {}
        for label, bounds in (("physical", base.storage_maximum_m3), ("reserve", guarded)):
            for fifth, flow in (("anchor", base.ontario_executable_release_m3_s[-1]), ("witness", witness[-1])):
                anchor = (*base.ontario_executable_release_m3_s[:-1], flow)
                p = replace(base, storage_maximum_m3=bounds, ontario_executable_release_m3_s=anchor)
                requested = [*witness[:-1], flow]
                fixed = capped_follower(p, rows, requested, state[0])
                fit = explicit_storage_feasibility(fixed, tolerance)
                if fit["status"] == "solver_failure":
                    raise ArithmeticError(fit["message"])
                native = None
                if fit["status"] == "feasible":
                    arrivals = fixed.routing_matrix @ fit["release_m3_s"] + fixed.in_transit_contribution_m3_s
                    _, native, _ = forecast_trajectory(rows, requested, arrivals, p.other_downstream_m3_s,
                                                       p.downstream_capacity_m3_s, state, mechanism, tolerance)
                extra = gradual_schedules(p, prefix["previous_ontario_release_m3_s"])
                search = []
                answer = choose_two_paths(p, actual, rows, state, mechanism, config["reference_release_m3_s"],
                                          tolerance, search, extra_schedules=extra)
                cells[f"{label}_{fifth}"] = {"witness_lp": fit["status"], "witness_native": native,
                                            "accepted": answer is not None, "audit": search}
        records.append({"year": year, "seed": case["seed"], "pair": case["pair"], "period": k + 1, "cells": cells})
        print(year, case["seed"], {key: (v["witness_native"], v["accepted"]) for key, v in cells.items()}, flush=True)
    (args.output / "summary.json").write_text(json.dumps(records, default=json_value, indent=2) + "\n")
    lines = ["# Nine-state constraint diagnostic", "",
             "Uses recorded inherited states and realized Ottawa inputs offline. The fifth-lead intervention "
             "substitutes the full-window witness commitment. It is a diagnostic scenario, with Ontario's first "
             "four causal native checks retained. The same branch search is used in all four cells, including "
             "at the one archived stop that actually used the ordinary persistence fallback.", "",
             "| Bounds | Fifth lead | Native-checked witness | Search accepted |", "|---|---|---:|---:|"]
    for key in records[0]["cells"]:
        bound, fifth = key.split("_")
        cells = [r["cells"][key] for r in records]
        lines.append(f"| {bound} | {fifth} | {sum(c['witness_native']=='admissible' for c in cells)}/9 | "
                     f"{sum(c['accepted'] for c in cells)}/9 |")
    (args.output / "report.md").write_text("\n".join([*lines, ""]))


if __name__ == "__main__":
    main()
