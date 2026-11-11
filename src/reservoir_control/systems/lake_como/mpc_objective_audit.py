"""Paired finite-grid comparison of binary and surrogate MPC level penalties.

Hindsight diagnostic only. Both objectives score every identical candidate plan
once, using identical simulated transitions and training-only demand forecasts.
No local optimizer or closed-loop rollout is involved.
"""
import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC
from reservoir_control.systems.lake_como.mpc_diagnostics import level_penalties
from reservoir_control.systems.lake_como.mpc_hindsight import digest, load_baseline
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.splits import split_years

# Nested uniform grids; the finer one-day grid resolves submillimeter crossings.
GRIDS = {1: (401, 10001), 2: (21, 41), 3: (11, 21)}


@dataclass(frozen=True)
class PlanScore:
    surrogate: float
    binary: float
    levels: np.ndarray
    releases: np.ndarray
    nonlevel_cost: np.ndarray


def score_plan(model, requests, storage, inflows, demands, days):
    """Both normalized objectives share one hourly rollout and nonlevel costs."""
    requests = np.asarray(requests, dtype=float)
    days = np.asarray(days, dtype=int)
    demands = np.asarray(demands, dtype=float)
    levels, releases = model.predict(storage, requests, inflows, days)
    binary, surrogate = level_penalties(levels, model)
    mef = np.asarray(model.lake.min_env_flow)[days - 1]
    deficit = np.maximum(demands - np.maximum(releases - mef, 0), 0)
    if model.smooth_deficit:
        deficit *= 0.5 * (3 - np.cos((days - 1) * 2 * np.pi / 365))
    else:
        deficit *= np.where((days > 120) & (days <= 243), 2.0, 1.0)
    c = model.coefficients
    nonlevel = (
        c["daily_deficit_coeff"] * deficit
        + c["wasted_water_coeff"] * np.maximum(releases - demands, 0)
        + c["clip_action_coeff"] * (requests - releases) ** 2
    )
    discounts = model.config.discount_factor ** np.arange(len(days))
    terminal = (
        model.config.terminal_level_weight * (levels[-1] - model.lake.init_level) ** 2
    )
    return PlanScore(
        float((np.dot(discounts, nonlevel + surrogate) + terminal) / 1000),
        float((np.dot(discounts, nonlevel + binary) + terminal) / 1000),
        levels,
        releases,
        nonlevel,
    )


def compare_grid(model, storage, inflows, demands, days, points):
    """Exhaustive selection on a shared grid; costs in original reward units."""
    values = np.linspace(model.low, model.high, points).astype(np.float32).astype(float)
    plans = np.array(list(product(values, repeat=len(days))))
    costs = np.empty((len(plans), 2))
    first_levels = np.empty(len(plans))
    for i, requests in enumerate(plans):
        score = score_plan(model, requests, storage, inflows, demands, days)
        costs[i] = score.surrogate, score.binary
        first_levels[i] = score.levels[0]
    indices = dict(
        surrogate=int(costs[:, 0].argmin()), binary=int(costs[:, 1].argmin())
    )
    chosen = {}
    for objective, index in indices.items():
        score = score_plan(model, plans[index], storage, inflows, demands, days)
        # Independently verify the diagnostic surrogate against the deployed model.
        normalized = (plans[index] - model.low) / (model.high - model.low)
        np.testing.assert_allclose(
            score.surrogate,
            model.cost(normalized, storage, inflows, demands, days),
            rtol=1e-10,
            atol=1e-10,
        )
        chosen[objective] = dict(requests=plans[index].copy(), score=score)
    s, b = indices["surrogate"], indices["binary"]
    binary_regret = float((costs[s, 1] - costs[b, 1]) * 1000)
    surrogate_regret = float((costs[b, 0] - costs[s, 0]) * 1000)
    tolerance = max(0.1, 0.001 * abs(costs[b, 1] * 1000))
    row = dict(
        horizon=len(days),
        grid_points=points,
        grid_evaluations=len(plans),
        surrogate_cost_of_surrogate_choice=float(costs[s, 0] * 1000),
        binary_cost_of_surrogate_choice=float(costs[s, 1] * 1000),
        surrogate_cost_of_binary_choice=float(costs[b, 0] * 1000),
        binary_cost_of_binary_choice=float(costs[b, 1] * 1000),
        binary_regret=binary_regret,
        surrogate_regret=surrogate_regret,
        material_binary_regret=binary_regret > tolerance,
        tolerance=tolerance,
        surrogate_first_request_m3_s=float(plans[s, 0]),
        binary_first_request_m3_s=float(plans[b, 0]),
        first_request_changed=bool(plans[s, 0] != plans[b, 0]),
        plan_changed=bool(not np.array_equal(plans[s], plans[b])),
        surrogate_minimum_ties=int(
            np.isclose(costs[:, 0], costs[s, 0], rtol=0, atol=1e-12).sum()
        ),
        binary_minimum_ties=int(
            np.isclose(costs[:, 1], costs[b, 1], rtol=0, atol=1e-12).sum()
        ),
    )
    for objective, value in chosen.items():
        score = value["score"]
        row[f"{objective}_flood_days"] = int((score.levels > model.flood_level).sum())
        row[f"{objective}_low_water_days"] = int((score.levels < model.low_level).sum())
        row[f"{objective}_first_level_m"] = float(score.levels[0])
    profile = (
        pd.DataFrame(
            dict(
                request_m3_s=plans[:, 0],
                surrogate_cost=costs[:, 0] * 1000,
                binary_cost=costs[:, 1] * 1000,
                first_level_m=first_levels,
            )
        )
        if len(days) == 1
        else None
    )
    return row, chosen, profile


def save_plots(results, profiles, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fine = results.groupby(["origin", "case", "horizon"], sort=False).tail(1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, origin in zip(axes, ["seasonal_state", "hindsight_state"]):
        f = fine[fine.origin == origin]
        for horizon in GRIDS:
            h = f[f.horizon == horizon]
            ax.plot(h.case, h.binary_regret, marker="o", label=f"{horizon}-day problem")
        ax.set(
            title=origin.replace("_", " "),
            ylabel="Binary planning cost saved by binary choice",
            xlabel="Fixed snapshot",
        )
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8)
    fig.suptitle(
        "HINDSIGHT ONLY · identical grids, forecasts and initial states within every paired comparison"
    )
    fig.savefig(output / "binary_regret.png", dpi=170)
    plt.close(fig)
    # All old cases remain in CSV; plot fixed low/flood persistent cases for clarity.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, (origin, case) in zip(
        axes.flat,
        product(
            ["seasonal_state", "hindsight_state"],
            ["low_persistent", "flood_persistent"],
        ),
    ):
        p = profiles[(profiles.origin == origin) & (profiles.case == case)]
        r = fine[
            (fine.origin == origin) & (fine.case == case) & (fine.horizon == 1)
        ].iloc[0]
        left = min(r.surrogate_first_request_m3_s, r.binary_first_request_m3_s) - 15
        right = max(r.surrogate_first_request_m3_s, r.binary_first_request_m3_s) + 15
        local = p[p.request_m3_s.between(max(0, left), min(491.61, right))]
        ax.plot(local.request_m3_s, local.surrogate_cost, label="Surrogate cost")
        ax.plot(
            local.request_m3_s, local.binary_cost, label="Binary level-penalty cost"
        )
        ax.axvline(
            r.surrogate_first_request_m3_s,
            color="tab:blue",
            ls=":",
            label="Surrogate choice",
        )
        ax.axvline(
            r.binary_first_request_m3_s,
            color="tab:orange",
            ls="--",
            label="Binary choice",
        )
        ax.set(
            title=f'{origin.replace("_"," ")} · {case}',
            xlabel="First request (m³/s)",
            ylabel="Positive one-day planning cost",
        )
        ax.legend(fontsize=7)
    fig.suptitle(
        "Same one-day candidates · perfect inflow, seasonal demand · local action ranges"
    )
    fig.savefig(output / "one_day_cost_profiles.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "results/comparison/mpc_validation_v1",
    )
    parser.add_argument(
        "--hindsight", type=Path, default=PROJECT_ROOT / "results/mpc/perfect_inflow_v1"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "results/mpc/diagnosis_v1/selected_cases.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/mpc/objective_grid_v1"
    )
    args = parser.parse_args()
    _, config, seasonal, _ = load_baseline(args.baseline)
    prior_plan = json.loads((args.hindsight / "plan.json").read_text())
    prior_run = json.loads((args.hindsight / "run.json").read_text())
    if (
        not prior_run["complete"]
        or prior_run["test_evaluated"]
        or prior_plan["config"] != asdict(config)
    ):
        raise ValueError("Completed matching validation hindsight diagnostic required")
    for path, expected in prior_plan["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Hindsight source changed: {path}")
    hindsight = pd.read_csv(
        args.hindsight / "trajectories.csv", float_precision="round_trip"
    )
    hindsight = hindsight[hindsight.variant == "perfect_inflow_hindsight"]
    cases = pd.read_csv(args.cases)
    if (
        not set(cases.year).issubset(split_years("validation"))
        or cases.case.duplicated().any()
    ):
        raise ValueError("Unique, validation-only diagnostic cases required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new, empty --output")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = DeterministicMPC.from_training(config)
    model = reference.model
    actual_inflows = settings("validation")["inflow"]
    origins = {"seasonal_state": seasonal, "hindsight_state": hindsight}
    input_files = [
        args.cases,
        args.baseline / "trajectories.csv",
        args.hindsight / "trajectories.csv",
        args.hindsight / "plan.json",
        args.hindsight / "run.json",
    ]
    plan = dict(
        label="HINDSIGHT-ONLY PAIRED OBJECTIVE GRID AUDIT",
        baseline_config=asdict(config),
        cases=cases.to_dict("records"),
        origins=list(origins),
        grids=GRIDS,
        changed="Only binary versus squared level costs; each candidate uses a shared rollout and other costs.",
        fixed="Observed float32 initial level, actual horizon inflows, training-only demand climatology, hydraulics, discount and terminal cost",
        solver="Exhaustive common finite Cartesian grid for both objectives; no SLSQP selection",
        action_precision="Uniform requests quantized to executable float32; ties use ascending lexicographic order",
        finest_one_day_level_resolution_m=float(
            (model.high - model.low) / 10000 * 86400 / model.lake.surface
        ),
        horizon_note="One-, two-, three-day diagnostic problems, not the deployed seven-day controller",
        interpretation="Binary planning cost still uses forecast demand; not realized annual return or a deployable result",
        new_closed_loop_evaluation=False,
        test_evaluated=False,
        deployed_controller_changed=False,
        input_sha256={str(p): digest(p) for p in input_files},
        source_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__),
                PROJECT_ROOT / "src/reservoir_control/systems/lake_como/mpc.py",
                PROJECT_ROOT / "src/reservoir_control/systems/lake_como/mpc_diagnostics.py",
            ]
        },
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    paths = []
    profiles = []
    for origin, frame in origins.items():
        if frame.duplicated(["year", "day"]).any():
            raise ValueError("Duplicate snapshot keys")
        for case in cases.itertuples(index=False):
            initial = frame[(frame.year == case.year) & (frame.day == case.day)]
            if len(initial) != 1:
                raise ValueError("Exactly one saved snapshot per case required")
            level = float(np.float32(initial.iloc[0].level_before_m))
            storage = model.lake.level_to_storage(level)
            for horizon, refinements in GRIDS.items():
                days = np.arange(case.day, case.day + horizon)
                if days[-1] > 365:
                    raise ValueError(
                        "Diagnostic cases must leave room for all short horizons"
                    )
                inflows = np.array(actual_inflows[case.year])[days - 1]
                demands = reference.forecast.demand[days - 1]
                for points in refinements:
                    row, choices, profile = compare_grid(
                        model, storage, inflows, demands, days, points
                    )
                    row.update(
                        origin=origin,
                        case=case.case,
                        year=case.year,
                        day=case.day,
                        initial_level_m=level,
                    )
                    rows.append(row)
                    for objective, choice in choices.items():
                        binary, surrogate = level_penalties(
                            choice["score"].levels, model
                        )
                        for k, day in enumerate(days):
                            paths.append(
                                dict(
                                    origin=origin,
                                    case=case.case,
                                    horizon=horizon,
                                    grid_points=points,
                                    objective=objective,
                                    step=k + 1,
                                    day=int(day),
                                    request_m3_s=choice["requests"][k],
                                    predicted_level_m=choice["score"].levels[k],
                                    predicted_release_m3_s=choice["score"].releases[k],
                                    nonlevel_cost=choice["score"].nonlevel_cost[k],
                                    binary_level_cost=binary[k],
                                    surrogate_level_cost=surrogate[k],
                                )
                            )
                    if horizon == 1 and points == refinements[-1]:
                        profile["origin"] = origin
                        profile["case"] = case.case
                        profiles.append(profile)
            print(f"Paired objective grids complete: {origin}, {case.case}", flush=True)
    results = pd.DataFrame(rows)
    winning_paths = pd.DataFrame(paths)
    one_day = pd.concat(profiles, ignore_index=True)
    for _, group in results.groupby(["origin", "case", "horizon"]):
        g = group.sort_values("grid_points")
        for column in [
            "binary_cost_of_binary_choice",
            "surrogate_cost_of_surrogate_choice",
        ]:
            if not g[column].diff().dropna().le(1e-8).all():
                raise ValueError("Nested-grid optimum worsened with refinement")
    results.to_csv(args.output / "objective_comparisons.csv", index=False)
    winning_paths.to_csv(args.output / "winning_plans.csv", index=False)
    one_day.to_csv(args.output / "one_day_profiles.csv", index=False)
    fine = results.groupby(["origin", "case", "horizon"], sort=False).tail(1)
    summary = fine.groupby(["origin", "horizon"], sort=False).agg(
        cases=("case", "size"),
        material_binary_regret_cases=("material_binary_regret", "sum"),
        mean_binary_regret=("binary_regret", "mean"),
        max_binary_regret=("binary_regret", "max"),
        changed_first_requests=("first_request_changed", "sum"),
    )
    summary.to_csv(args.output / "summary.csv")
    save_plots(results, one_day, args.output)
    record = dict(
        complete=True,
        test_evaluated=False,
        new_closed_loop_evaluation=False,
        deployed_controller_changed=False,
        fine_grid_comparisons=len(fine),
        material_binary_regret_cases=int(fine.material_binary_regret.sum()),
        changed_first_requests=int(fine.first_request_changed.sum()),
        maximum_binary_regret=float(fine.binary_regret.max()),
        minimum_binary_regret=float(fine.binary_regret.min()),
        sources_unchanged=all(
            digest(Path(p)) == v for p, v in plan["source_sha256"].items()
        ),
        inputs_unchanged=all(
            digest(Path(p)) == v for p, v in plan["input_sha256"].items()
        ),
    )
    if not record["sources_unchanged"] or not record["inputs_unchanged"]:
        raise ValueError("Diagnostic inputs changed")
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output / "report.md").write_text(
        f"""# Binary versus surrogate penalties — HINDSIGHT ONLY

Every candidate is scored by both objectives on the same hourly rollout, same
perfect inflow, same seasonal demand forecast, same state and same finite grid.
Six previously selected calendar snapshots are reused from each of the seasonal
and hindsight closed-loop trajectories. No cases are selected by this study's outcome.

```text
{summary.round(4).to_string()}
```

A positive binary regret means the surrogate-selected plan has higher binary
planning cost than the binary-selected plan on the same grid. Costs are positive
and reported in original reward units, including unchanged deficit, waste, clipping,
discounting, and terminal terms. Demand remains forecast, so binary planning cost
is not a hindsight realized reward. The finite grid is shared; no local solver is
used to select either plan. Grid refinement never increased either optimum.

The finest one-day spacing corresponds to about 0.029 mm of level change in an
unclipped day, resolving the 0.125 mm median low-water exceedance from the earlier
hindsight study. Multi-day grids remain coarser and can mask narrow optima. Ties
are counted and resolved lexicographically; action differences without meaningful
cost regret are not evidence of misalignment. These are selected 1–3-day state
problems, not new annual rollouts or a global seven-day solution. Deployed MPC,
baseline artifacts, and chronological test years are unchanged.

[Binary regret](binary_regret.png) · [One-day cost profiles](one_day_cost_profiles.png)
"""
    )
    print(summary.round(4).to_string())
    print(record)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
