"""Compare causal Ottawa forecast recession with persistence under one fixed guard."""

from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.simulators.physics import FLOW_SCALE
from bilevel_reservoirs.solvers.diagnose_sharp_peak import CASES, HYDROLOGY
from bilevel_reservoirs.solvers.fixed_schedule_replay import CONFIG, LOOKAHEAD, digest, json_value, validate_config
from bilevel_reservoirs.solvers.stress_ottawa_response import offline_recoverability
from bilevel_reservoirs.solvers.study_storage_guard import run
from bilevel_reservoirs.solvers.validate_storage_guard import assess, stop_evidence
from reservoir_control.systems.ottawa.horizon_study import forecast_bank

ARMS = ("persistence", "recession")
ISOLATED_ARMS = (*ARMS, "isolated_jump")
ISOLATED_DESIGN = CONFIG.with_name("isolated_jump_v1.json")
FRESH_SEEDS = (125, 126, 127, 128)
HISTORY_PERIODS = 3
HALF_LIFE_PERIODS = 1.0


def recession_bank(observed, persistence, *, jump_minimum_m3_s=None):
    """Keep lead one, then let positive departures from recent history decay."""
    if len(observed) != len(persistence):
        raise ValueError("Observation series and forecast bank must share the same clock")
    bank = np.array(persistence, copy=True)
    # Lead one has no adjustment. For lead two, remove half the departure.
    decay = 1.0 - np.exp2(-np.arange(bank.shape[1]) / HALF_LIFE_PERIODS)
    for decision in range(2, len(observed)):
        last = observed[decision - 1]
        earlier = observed[max(0, decision - 1 - HISTORY_PERIODS):decision - 1]
        if jump_minimum_m3_s is not None and last - max(earlier) < jump_minimum_m3_s:
            continue
        departure = max(0.0, last - float(np.median(earlier)))
        bank[decision] -= departure * decay
    bank.flags.writeable = False
    return bank


def forecast_metrics(bank, observed):
    errors = []
    for index, issued in enumerate(bank):
        horizon = min(LOOKAHEAD, len(observed) - index)
        errors.extend(issued[1:horizon] - observed[index + 1:index + horizon])
    return {"tail_mae_m3_s": float(np.mean(np.abs(errors))), "tail_bias_m3_s": float(np.mean(errors))}


def plateau_initial_witness(trajectory, config, scenario, history):
    """Check whole-window physical feasibility independently of the forecasts."""
    previous = round(history["previous_flow"] * FLOW_SCALE)
    proposals = (
        [max(target, previous - drop - ramp * period) for period in range(len(trajectory))]
        for drop, ramp, target in product(
            range(500, 1201, 10), (500, 600, 690, 700, 800, 900, 1000), (4950, 5000, 5100, 5300, 5500, 6000),
        )
    )
    return offline_recoverability(trajectory, config, scenario, [], additional_schedules=proposals, **history)


def write_report(output, runs, forecasts, design, sample, initial_witnesses=None):
    isolated = sample == "isolated-jump"
    arms = ISOLATED_ARMS if isolated else ARMS
    lines = [
        "# Ottawa causal recession forecast", "",
        "The first lead and random perturbations are shared. Recession lets a high latest inflow "
        "decay toward the median of up to three earlier completed periods, with a one-period half-life "
        "starting after lead one. The controller uses no future observed inflow in this transformation.", "",
        f"All arms use always-on +{design['guard_stress_m3_s']} m³/s guarding, the same physical "
        "bounds, native Ontario checks, objectives, and recovery stages. The existing constant-persistence "
        "retry remains the fallback in every arm. The experiment changes the ordinary forecast only.", "",
        f"This report covers the **{sample}** noise seeds and the declared synthetic Ottawa flows "
        "paired with the same historical Ontario windows.", "",
    ]
    if isolated:
        lines.extend([
            "The isolated-jump arm applies recession only when the latest observation exceeds every "
            f"available earlier observation in the three-period lookback by at least {design['jump_minimum_m3_s']} m³/s. "
            "That threshold uses the existing guard magnitude. The sharp peak and sustained rise have "
            "identical observations through period 3, so their issued banks match through decision 4 "
            "under each forecast rule and seed.", "",
        ])
    lines.extend([
        "| Inflow pattern | Forecast | Completed | Overflow cases | Low-water violations | Downstream violations | No candidate | Stops with a continuation witness |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for shape in (*design["inflows_m3_s"], "all"):
        group = runs if shape == "all" else runs[runs.hydrology == shape]
        for arm in arms:
            rows = group[group.arm == arm]
            lines.append(f"| {shape} | {arm} | {(rows.status == 'completed').sum()}/{len(rows)} | "
                         f"{rows.upper_violation.sum()} | {rows.lower_violation.sum()} | "
                         f"{rows.downstream_violation.sum()} | {(rows.status == 'no_admissible_candidate').sum()} | "
                         f"{rows.stop_witness.sum()} |")
    pairs = runs.pivot(index=["year", "hydrology", "seed"], columns="arm", values="status")
    p_complete = pairs.persistence == "completed"
    for arm in arms[1:]:
        complete = pairs[arm] == "completed"
        lines.extend([
            "", f"{arm.replace('_', ' ').capitalize()} added {(complete & ~p_complete).sum()} completions and lost "
            f"{(p_complete & ~complete).sum()} compared with persistence.",
        ])
    lines.extend([
        "",
        "## Forecast accuracy on leads 2–5", "",
        "Scores use all forecast issue times, including those after an early stop. Scores are computed "
        "offline and counted once per flow shape and seed across the Ontario windows.", "",
        "| Inflow pattern | Forecast | Mean absolute error (m³/s) | Signed bias (m³/s) |",
        "|---|---|---:|---:|",
    ])
    for (shape, arm), group in forecasts.groupby(["hydrology", "arm"], sort=False):
        lines.append(f"| {shape} | {arm} | {group.tail_mae_m3_s.mean():.1f} | {group.tail_bias_m3_s.mean():+.1f} |")
    if sample == "archived":
        lines.extend([
            "", "## Three previously diagnosed sharp-peak stops", "",
            "| Window | Seed | Persistence | Recession |",
            "|---|---:|---|---|",
        ])
        for year, seed in CASES:
            row = pairs.loc[(year, HYDROLOGY, seed)]
            lines.append(f"| {year} | {seed} | {row.persistence} | {row.recession} |")
    if initial_witnesses:
        lines.extend([
            "", "## Sustained-rise feasibility from the initial state", "",
            "The following offline checks use realized inflows through the endpoint and verify native Ontario "
            "limits. They establish feasibility from the initial state when a witness is found. Each stopped "
            "run needs a separate check from its own inherited state.", "",
            "| Ontario window | Initial feasibility finding |", "|---|---|",
        ])
        lines.extend(f"| {year} | {witness['finding']} |" for year, witness in initial_witnesses.items())
    lines.extend([
        "", "A continuation witness uses a matching completed causal trajectory or a separate offline "
        "feasibility check with realized inflows. A stopped case without a witness remains unresolved "
        "unless the physical relaxation certifies infeasibility. All hydrographs and Ottawa outlet "
        "assumptions are synthetic. The forecast and guard defaults were preserved.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("results/loslr"))
    parser.add_argument("--sample", choices=("archived", "fresh", "isolated-jump"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = {}

    def source(path):
        provenance[str(path)] = digest(path)
        return path

    config = json.loads(source(CONFIG).read_text())
    validate_config(config)
    prior = args.archive_root / "ottawa_storage_guard_hydrology_v1"
    settings = json.loads(source(prior / "settings.json").read_text())
    design = settings["design"]
    noise = config["forecast_noise"]
    if (provenance[str(CONFIG)] not in settings["input_sha256"].values()
            or noise["standard_deviation_m3_s"] != design["guard_stress_m3_s"]):
        raise ValueError("The current follower settings differ from the archived guard")
    base = next(case for case in config["scenarios"] if case["id"] == design["base_scenario"])
    historical = pd.read_csv(source(prior / "runs.csv"))
    baseline = historical[historical.arm == "always"].set_index(["year", "hydrology", "seed"])
    isolated = args.sample == "isolated-jump"
    arms = ISOLATED_ARMS if isolated else ARMS
    seeds = design["forecast_seeds"] if args.sample == "archived" else FRESH_SEEDS
    if isolated:
        isolation = json.loads(source(ISOLATED_DESIGN).read_text())
        seeds = isolation["forecast_seeds"]
        if (isolation["jump_minimum_m3_s"] != design["guard_stress_m3_s"] or set(seeds) & set(FRESH_SEEDS)
                or isolation["plateau_inflow_m3_s"][:3] != design["inflows_m3_s"][HYDROLOGY][:3]):
            raise ValueError("The isolated-jump study must preserve its threshold, fresh seeds and paired opening")
        design = {**design, "jump_minimum_m3_s": isolation["jump_minimum_m3_s"],
                  "inflows_m3_s": {**design["inflows_m3_s"], "sharp_rise_stays_high": isolation["plateau_inflow_m3_s"]}}
    if args.sample != "archived" and set(seeds) & set(design["forecast_seeds"]):
        raise ValueError("Fresh noise seeds overlap the archived sample")
    args.output.mkdir(parents=True, exist_ok=False)
    summary, scores, initial_witnesses = [], [], {}
    for year in design["ontario_years"]:
        archive = args.archive_root / f"ottawa_replan_guard_{year}"
        forcing_path = source(archive / "ontario_inputs.csv")
        settings_path = source(archive / "settings.json")
        if any(provenance[str(path)] != settings["input_sha256"].get(str(path))
               and provenance[str(path)] not in settings["input_sha256"].values()
               for path in (forcing_path, settings_path)):
            raise ValueError(f"Inputs changed for Ontario window {year}")
        trajectory = pd.read_csv(forcing_path)
        local = {**config, "known_ontario_executable_schedule_m3_s": (FLOW_SCALE * trajectory.ontFlow).tolist()}
        history = json.loads(settings_path.read_text())["ontario_window"]["initial_history"]
        window = args.output / str(year)
        window.mkdir()
        for shape, values in design["inflows_m3_s"].items():
            observed = np.asarray(values, dtype=float)
            scenario = {**base, "realized_net_inflow_m3_s": values}
            if shape == "sharp_rise_stays_high":
                initial_witnesses[year] = plateau_initial_witness(trajectory, local, scenario, history)
            for seed in seeds:
                stem = f"{shape}_{seed}"
                origin = forecast_bank(values, "noisy", seed, noise["initial_observed_inflow_estimate_m3_s"],
                                       noise["standard_deviation_m3_s"])
                if args.sample == "archived":
                    saved = np.load(source(prior / str(year) / f"{stem}_forecasts.npy"))
                    np.testing.assert_array_equal(origin, saved)
                    origin = saved
                banks = {"persistence": origin, "recession": recession_bank(observed, origin)}
                if isolated:
                    banks["isolated_jump"] = recession_bank(
                        observed, origin, jump_minimum_m3_s=design["jump_minimum_m3_s"],
                    )
                    if shape == "sharp_rise_stays_high":
                        for arm in arms:
                            sharp = np.load(window / f"{HYDROLOGY}_{seed}_{arm}_forecasts.npy")
                            np.testing.assert_array_equal(banks[arm][:4], sharp[:4])
                results = {}
                for arm, bank in banks.items():
                    np.testing.assert_array_equal(bank[:, 0], origin[:, 0])
                    name = f"{stem}_{arm}"
                    np.save(window / f"{name}_forecasts.npy", bank)
                    if year == design["ontario_years"][0]:
                        scores.append({"hydrology": shape, "seed": seed, "arm": arm,
                                       **forecast_metrics(bank, observed)})
                    if args.sample == "archived" and arm == "persistence":
                        archived = baseline.loc[(year, shape, seed)].to_dict()
                        trace_path = source(prior / str(year) / f"{stem}_always_trace.csv")
                        frame = pd.read_csv(trace_path)
                        # Parts of the archived trace use list-valued transit fields.
                        for field in ("ottawa_start_queue_m3", "ottawa_end_queue_m3"):
                            if field in frame:
                                frame[field] = frame[field].map(lambda value: json.loads(value) if isinstance(value, str) else None)
                        trace = frame.to_dict("records")
                        outcome = {key: archived[key] for key in (
                            "status", "executed_periods", "last_decision", "max_storage_violation_m3",
                            "max_downstream_violation_m3_s", "ordinary_guard_periods", "candidate_solves",
                        )}
                        proof = (json.loads(source(prior / str(year) / f"{stem}_always_stop.json").read_text())
                                 if outcome["status"] == "no_admissible_candidate" else None)
                    else:
                        outcome, trace, plans, _ = run(trajectory, local, scenario, bank, history, "always")
                        pd.DataFrame(trace).to_csv(window / f"{name}_trace.csv", index=False)
                        (window / f"{name}_plans.json").write_text(json.dumps(plans, default=json_value) + "\n")
                        proof = None
                    results[arm] = (outcome, trace, proof)
                for arm in arms:
                    outcome, trace, proof = results[arm]
                    result = assess(outcome, trace, config)
                    if outcome["status"] == "no_admissible_candidate" and proof is None:
                        peer = next((other for other in arms if other != arm and results[other][0]["status"] == "completed"),
                                    next(other for other in arms if other != arm))
                        peer_outcome, peer_trace, _ = results[peer]
                        proof = stop_evidence(trajectory, local, scenario, trace, peer_trace, peer_outcome, history)
                        (window / f"{stem}_{arm}_stop.json").write_text(
                            json.dumps({"stop": trace[-1], **proof}, default=json_value, indent=2) + "\n"
                        )
                    summary.append({
                        "year": year, "hydrology": shape, "seed": seed, "arm": arm,
                        "upper_violation": result["above_storage_bound_m3"] > config["acceptance"]["storage_m3"],
                        "lower_violation": result["below_storage_bound_m3"] > config["acceptance"]["storage_m3"],
                        "downstream_violation": result["max_downstream_violation_m3_s"] > config["acceptance"]["flow_m3_s"],
                        "stop_witness": bool(proof and proof["witness"]),
                        "stop_finding": proof["finding"] if proof else None, **result,
                    })
                print(year, shape, seed, *(results[arm][0]["status"] for arm in arms), flush=True)

    for path, expected in provenance.items():
        if digest(Path(path)) != expected:
            raise ArithmeticError(f"A comparison input changed: {path}")
    runs, forecasts = pd.DataFrame(summary), pd.DataFrame(scores)
    runs.to_csv(args.output / "runs.csv", index=False)
    forecasts.to_csv(args.output / "forecast_metrics.csv", index=False)
    write_report(args.output, runs, forecasts, design, args.sample, initial_witnesses)
    if initial_witnesses:
        (args.output / "sustained_rise_initial_feasibility.json").write_text(
            json.dumps(initial_witnesses, default=json_value, indent=2) + "\n"
        )
    (args.output / "settings.json").write_text(json.dumps({
        "sample": args.sample, "seeds": list(seeds), "design": design,
        "history_periods": HISTORY_PERIODS, "half_life_periods": HALF_LIFE_PERIODS,
        "guard_trigger": "always", "input_sha256": provenance,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
