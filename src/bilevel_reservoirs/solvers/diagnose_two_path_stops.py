"""Factorial check of candidate coverage and storage reserve at archived branch stops."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

from gymnasium.spaces import Box
import numpy as np
import pandas as pd

from bilevel_reservoirs.controllers.operator import AdjustFlow
from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.fixed_schedule_replay import (
    CONFIG, LOOKAHEAD, FollowerTolerances, TransitState, digest, json_value, make_problem, validate_config,
)
from bilevel_reservoirs.solvers.scenario_ottawa import choose_two_paths, gradual_schedules


def write_report(output, cases):
    lines = [
        "# Two-path branch stop diagnostic", "",
        "The experiment holds the two issued forecasts, inherited state, routing, and physical limits "
        "fixed at each of the 12 held-out period-4 stops. The wider search appends gradual Ontario "
        "release profiles generated from the previous executed release, downstream capacities, and Ottawa's "
        "outlet bound. It preserves the fifth commitment and the original scoring and native eligibility checks.", "",
        "| Case | Current with reserve | Current without reserve | Wider with reserve | Wider without reserve |",
        "|---|---|---|---|---|",
    ]
    keys = ("current_guarded", "current_plain", "wider_guarded", "wider_plain")
    for case in cases:
        states = ["Feasible" if case["cells"][key]["accepted"] else "Stopped" for key in keys]
        lines.append(f"| {case['year']} {case['pair']} seed {case['seed']} | " + " | ".join(states) + " |")
    totals = [sum(case["cells"][key]["accepted"] for case in cases) for key in keys]
    unresolved = [case for case in cases if not case["cells"]["wider_plain"]["accepted"]]
    no_sustained = sum(case["cells"]["wider_plain"]["audit"][-1]["sustained_intervals"] == 0
                       for case in unresolved)
    lines.extend([
        "| **Feasible cases** | " + " | ".join(f"**{total}/{len(cases)}**" for total in totals) + " |", "",
        f"Of the {len(unresolved)} cases still stopped with the wider search and no reserve, "
        f"{no_sustained} had no feasible first-release interval for the sustained forecast alone "
        "among the tested Ontario candidates.", "",
        "The reserve is the existing +700 m³/s first-period stress, or 459.9 million m³ over one period. "
        "Both reserve settings retain the original physical constraints. Feasibility here means the "
        "actual candidate-selection algorithm accepted both branches with shared first actions. A stop "
        "can reflect the search, forecasts, or reserve. It does not establish physical infeasibility.", "",
        "No realized inflow values enter the four cells. The source study's realized-inflow witnesses "
        "are a separate reference. These are frozen-decision checks, with no closed-loop reruns or "
        "change to controller defaults.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive_root / "ottawa_two_path_recourse_v1"
    provenance = {}

    def source(path):
        provenance[str(path)] = digest(path)
        return path

    config = json.loads(source(CONFIG).read_text())
    validate_config(config)
    study = json.loads(source(root / "settings.json").read_text())
    if provenance[str(CONFIG)] not in study["inputs_sha256"].values():
        raise ValueError("Follower settings differ from the archived study")
    archive = json.loads(source(root / "heldout_stops_common_actions.json").read_text())
    if len(archive) != 12 or len({(x["year"], x["pair"], x["seed"]) for x in archive}) != 12:
        raise ValueError("Expected the 12 distinct held-out pulse stops")
    tolerance = FollowerTolerances(**config["acceptance"])
    scenario = next(row for row in config["scenarios"] if row["id"] == "one_period_delay")
    reserve_rate = config["forecast_noise"]["standard_deviation_m3_s"]
    if reserve_rate != 700:
        raise ValueError("Expected the frozen +700 m³/s reserve")
    limits = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0., np.inf, (1,), dtype=np.float64))
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in archive:
        year, pair, seed = (case[key] for key in ("year", "pair", "seed"))
        window = args.archive_root / f"ottawa_replan_guard_{year}"
        forcing = source(window / "ontario_inputs.csv")
        if provenance[str(forcing)] not in study["inputs_sha256"].values():
            raise ValueError("Ontario forcing differs from the archive")
        trajectory = pd.read_csv(forcing)
        local = {**config, "known_ontario_executable_schedule_m3_s": (trajectory.ontFlow * FLOW_SCALE).tolist()}
        stem = f"{pair}_pulse_{seed}"
        trace = pd.read_csv(source(root / str(year) / f"{stem}_two_path_trace.csv"))
        stop = trace.iloc[-1]
        if stop.period != 4 or stop.status != "no_admissible_candidate":
            raise ValueError("Expected a rejected period-4 decision")
        k = int(stop.period) - 1
        with np.load(source(root / str(year) / f"{stem}_forecasts.npz")) as bank:
            low, high = (bank[name][k, :LOOKAHEAD] for name in ("receding", "sustained"))
        queue = TransitState(tuple(json.loads(stop.ottawa_start_queue_m3)))
        base = make_problem(local, scenario, k, stop.ottawa_start_storage_m3,
                            stop.ottawa_previous_release_m3_s, queue, low)
        upper = np.array(base.storage_maximum_m3, copy=True)
        upper[1:] -= base.duration_seconds[0] * reserve_rate
        guarded = replace(base, storage_maximum_m3=upper)
        state = (stop.ontario_start_m, stop.ontario_previous_release_m3_s / FLOW_SCALE, stop.ontario_previous_ice)
        rows = trajectory.iloc[k:k + LOOKAHEAD].to_dict("records")
        extra = gradual_schedules(base, stop.ontario_previous_release_m3_s)
        archived_plan = json.loads(source(root / str(year) / f"{stem}_two_path_plans.json").read_text())
        cells = {}
        for search, schedules in (("current", ()), ("wider", extra)):
            for reserve, problem in (("guarded", guarded), ("plain", base)):
                audit = []
                result = choose_two_paths(problem, high, rows, state, limits, config["reference_release_m3_s"],
                                          tolerance, audit, extra_schedules=schedules)
                if search == "current" and reserve == "guarded":
                    original = archived_plan["branches"][str(k + 1)]
                    if result is not None or len(audit) != len(original) or any(
                        any(old[key] != new[key] for key in old)
                        for old, new in zip(original, audit, strict=True)
                    ):
                        raise ArithmeticError("The archived guarded stop did not reproduce")
                cells[f"{search}_{reserve}"] = {"accepted": result is not None, "audit": audit}
        cases.append({"year": year, "pair": pair, "seed": seed, "period": k + 1,
                      "receding_forecast_m3_s": low, "sustained_forecast_m3_s": high,
                      "inherited_state": state, "queue_m3": queue.volumes_m3,
                      "initial_storage_m3": base.initial_storage_m3, "cells": cells})
        print(year, pair, seed, {key: value["accepted"] for key, value in cells.items()}, flush=True)
    for path, original_hash in provenance.items():
        if digest(Path(path)) != original_hash:
            raise ArithmeticError(f"An archive input changed: {path}")
    (args.output / "summary.json").write_text(json.dumps(
        {"cases": cases, "input_sha256": provenance}, default=json_value, indent=2
    ) + "\n")
    write_report(args.output, cases)


if __name__ == "__main__":
    main()
