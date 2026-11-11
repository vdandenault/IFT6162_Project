"""Read-only flood feasibility audit on frozen trajectories; no safety controller.

Finite action grids establish sampled witnesses, not global infeasibility.
Conservative mass-balance certificates are reported separately from grid results.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import MPCConfig, PlanningModel
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.splits import split_years


def training_error_quantile(inflows, climatology, quantile=0.99):
    if set(inflows) != set(split_years("train")):
        raise ValueError("Exact training membership required")
    matrix = np.array([inflows[y][:365] for y in split_years("train")], dtype=float)
    errors = np.c_[
        matrix[:, 0] - climatology[0], matrix[:, 1:] - matrix[:, :-1]
    ].ravel()
    return max(0.0, float(np.quantile(errors, quantile, method="higher")))


def grid_rollout(lake, storage, inflow, days, requests):
    """Hourly model across a state × action grid, including initial level in peak."""
    storage = np.asarray(storage, dtype=float)
    inflow = np.asarray(inflow, dtype=float)
    days = np.asarray(days, dtype=int)
    if not (len(storage) == len(inflow) == len(days)):
        raise ValueError("Mismatched arrays")
    states = np.broadcast_to(storage[:, None], (len(storage), len(requests))).copy()
    peaks = states / lake.surface + lake.min_level
    mef = np.asarray(lake.min_env_flow)[days - 1, None]
    for _ in range(24):
        levels = states / lake.surface + lake.min_level
        rating = lake.C_r * np.maximum(levels - lake.alpha, 0.0) ** lake.beta
        minimum = np.where(
            levels <= lake.min_level,
            0.0,
            np.where(levels <= lake.max_level, mef, rating),
        )
        maximum = np.where(
            levels <= lake.min_level,
            0.0,
            np.where(
                levels <= lake.linear_limit,
                lake.linear_slope * levels + lake.linear_intercept,
                rating,
            ),
        )
        release = np.minimum(maximum, np.maximum(minimum, requests[None, :]))
        states += 3600 * (inflow[:, None] - release)
        peaks = np.maximum(peaks, states / lake.surface + lake.min_level)
    return states / lake.surface + lake.min_level, peaks


def lower_bounds(lake, storage, inflow, days, nominal_high):
    """Necessary-condition bounds valid for every nominal request.

    No-outflow trajectory bounds storage above. Forced minimum discharge can exceed
    nominal requests, so its rating curve must enter the outflow upper bound.
    Subtracting an upper bound on outflow gives a conservative storage lower bound.
    This does not assume that the discrete full-day transition is monotone.
    """
    assert not lake.evaporation and lake.C_r >= 0 and lake.beta > 0
    storage = np.asarray(storage)
    inflow = np.asarray(inflow)
    days = np.asarray(days, dtype=int)
    lower = storage.astype(float).copy()
    for hour in range(24):
        upper_level = (storage + 3600 * hour * inflow) / lake.surface + lake.min_level
        rating_upper = lake.C_r * np.maximum(upper_level - lake.alpha, 0.0) ** lake.beta
        release_upper = np.maximum.reduce(
            [
                np.full(len(storage), nominal_high),
                np.asarray(lake.min_env_flow)[days - 1],
                rating_upper,
            ]
        )
        lower += 3600 * (inflow - release_upper)
    first_release = np.array(
        [
            lake.actual_release(nominal_high, float(s), int(d))
            for s, d in zip(storage, days)
        ]
    )
    first_level = (
        storage + 3600 * (inflow - first_release)
    ) / lake.surface + lake.min_level
    return lower / lake.surface + lake.min_level, first_level


def summarize_audit(frame):
    rows = []
    for (family, scenario), g in frame.groupby(["family", "scenario"]):
        actual_flood = g.actual_endpoint_flood.astype(bool)
        starts_safe = ~g.starts_flooded.astype(bool)
        rows.append(
            dict(
                family=family,
                scenario=scenario,
                annual_episodes=g[["training_seed", "year"]].drop_duplicates().shape[0],
                states=len(g),
                actual_flood_states=int(actual_flood.sum()),
                starts_flooded=int(g.starts_flooded.sum()),
                grid_no_endpoint_safe_action=int((~g.grid_endpoint_safe).sum()),
                grid_no_hourly_safe_action=int((~g.grid_hourly_safe).sum()),
                certified_endpoint_infeasible=int(
                    g.certified_endpoint_infeasible.sum()
                ),
                certified_first_hour_infeasible_from_safe=int(
                    (starts_safe & g.certified_first_hour_infeasible).sum()
                ),
                actual_flood_with_endpoint_witness=int(
                    (actual_flood & g.grid_endpoint_safe).sum()
                ),
                actual_crossing_with_hourly_witness=int(
                    (actual_flood & starts_safe & g.grid_hourly_safe).sum()
                ),
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/safety/flood_feasibility_v1",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    ppo_root = PROJECT_ROOT / "results/ppo/past_inflow_v1"
    mpc_root = PROJECT_ROOT / "results/mpc/iterate_retention_v1"
    for root in [ppo_root, mpc_root]:
        run = json.loads((root / "run.json").read_text())
        if not run["complete"] or run["test_evaluated"]:
            raise ValueError("Completed validation inputs required")
    files = [
        root / name
        for root in [ppo_root, mpc_root]
        for name in ["run.json", "trajectories.csv"]
    ]
    frames = []
    ppo = pd.read_csv(ppo_root / "trajectories.csv", float_precision="round_trip")
    for variant in ["four_features", "past_inflow"]:
        f = ppo[ppo.variant == variant].copy()
        f["family"] = "ppo_" + variant
        frames.append(f)
    f = (
        pd.read_csv(mpc_root / "trajectories.csv", float_precision="round_trip")
        .query('method == "retain_iterate"')
        .copy()
    )
    f["family"] = "mpc_persistence"
    f["training_seed"] = 0
    frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    assert len(data) == 40150 and sorted(data.year.unique()) == split_years(
        "validation"
    )
    params = settings("train")
    model = PlanningModel(params, MPCConfig.load())
    lake = model.lake
    climatology = np.array(
        [params["inflow"][y][:365] for y in split_years("train")]
    ).mean(axis=0)
    error_upper = training_error_quantile(params["inflow"], climatology)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        threshold_m=model.flood_level,
        actions="201 nominal float32 requests including bounds; fixed request within each day",
        timing="Endpoint safety versus all 24 hourly endpoints including the initial level. Already-flooded starts are separate.",
        scenarios=[
            "actual_inflow_ex_post",
            "persistence_plus_training_q99_error_stress",
        ],
        training_positive_error_q99_m3_s=error_upper,
        interpretation="Grid witnesses are sufficient under specified fixed inflow; no grid witness is not proof of infeasibility. Separate conservative bounds certify some cases. Stress scenario is not a robust interval guarantee.",
        scope="One-day reachability from saved states, not avoidability from annual reset, a new policy, or a realized irrigation trade-off.",
        test_evaluated=False,
        source_sha256={
            str(Path(__file__)): digest(Path(__file__)),
            **{
                str(PROJECT_ROOT / p): digest(PROJECT_ROOT / p)
                for p in [
                    "src/reservoir_control/systems/lake_como/mpc.py",
                    "src/reservoir_control/systems/lake_como/environment.py",
                    "external/gym4real/gym4real/envs/dam/lake.py",
                ]
            },
        },
        input_sha256={str(p): digest(p) for p in files},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    requests = np.linspace(model.low, model.high, 201).astype(np.float32).astype(float)
    for (family, seed, year), g in data.groupby(["family", "training_seed", "year"]):
        g = g.sort_values("day")
        assert len(g) == 365
        previous = np.r_[climatology[0], g.inflow_m3_s.to_numpy()[:-1]]
        for scenario, flows in [
            ("actual_inflow_ex_post", g.inflow_m3_s.to_numpy()),
            (
                "persistence_plus_training_q99_error_stress",
                np.maximum(0, previous + error_upper),
            ),
        ]:
            for start in range(0, len(g), 128):
                chunk = g.iloc[start : start + 128]
                q = flows[start : start + len(chunk)]
                storage = chunk.storage_before_m3.to_numpy()
                days = chunk.day.to_numpy()
                levels, peaks = grid_rollout(lake, storage, q, days, requests)
                low_bound, first_bound = lower_bounds(
                    lake, storage, q, days, model.high
                )
                for i, row in enumerate(chunk.itertuples()):
                    end_index = int(levels[i].argmin())
                    peak_index = int(peaks[i].argmin())
                    safe = levels[i] <= model.flood_level
                    hour_safe = peaks[i] <= model.flood_level
                    # Canonical scalar checks on endpoints near the decision boundary.
                    if (
                        abs(levels[i, end_index] - model.flood_level) < 1e-7
                        or row.flood
                    ):
                        s, _ = lake.integration(
                            24,
                            row.storage_before_m3,
                            requests[end_index],
                            q[i],
                            row.day,
                        )
                        np.testing.assert_allclose(
                            levels[i, end_index],
                            lake.storage_to_level(s),
                            atol=1e-10,
                            rtol=0,
                        )
                    rows.append(
                        dict(
                            family=family,
                            training_seed=int(seed),
                            year=int(year),
                            day=int(row.day),
                            scenario=scenario,
                            starting_level_m=row.level_before_m,
                            inflow_m3_s=q[i],
                            actual_endpoint_flood=bool(row.flood),
                            starts_flooded=row.level_before_m > model.flood_level,
                            grid_endpoint_safe=bool(safe.any()),
                            grid_hourly_safe=bool(hour_safe.any()),
                            minimum_sampled_end_level_m=levels[i, end_index],
                            minimum_sampled_peak_level_m=peaks[i, peak_index],
                            endpoint_witness_request_m3_s=float(
                                requests[np.flatnonzero(safe)[0]]
                            )
                            if safe.any()
                            else np.nan,
                            endpoint_best_request_m3_s=requests[end_index],
                            certified_endpoint_infeasible=bool(
                                low_bound[i] > model.flood_level + 1e-7
                            ),
                            certified_first_hour_infeasible=bool(
                                first_bound[i] > model.flood_level + 1e-7
                            ),
                            endpoint_lower_bound_m=low_bound[i],
                            first_hour_lower_bound_m=first_bound[i],
                        )
                    )
        print(family, seed, year, "audited", flush=True)
    audit = pd.DataFrame(rows)
    assert (~audit.certified_endpoint_infeasible | ~audit.grid_endpoint_safe).all()
    assert (~audit.certified_first_hour_infeasible | ~audit.grid_hourly_safe).all()
    summary = summarize_audit(audit)
    audit.to_csv(args.output / "states.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    per_year = (
        audit.query('scenario == "actual_inflow_ex_post"')
        .groupby(["family", "training_seed", "year"])
        .agg(
            actual_flood_days=("actual_endpoint_flood", "sum"),
            endpoint_no_grid_witness=("grid_endpoint_safe", lambda x: int((~x).sum())),
            certified_endpoint_infeasible=("certified_endpoint_infeasible", "sum"),
        )
    )
    per_year.to_csv(args.output / "annual.csv")
    actual = summary[summary.scenario == "actual_inflow_ex_post"].set_index("family")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    labels = ["PPO: 4 inputs", "PPO: past inflow", "MPC: persistence"]
    order = ["ppo_four_features", "ppo_past_inflow", "mpc_persistence"]
    for ax, column, title in [
        (axes[0], "actual_flood_states", "Observed flood endpoints / annual episode"),
        (
            axes[1],
            "actual_flood_with_endpoint_witness",
            "Observed flood endpoints with a grid-safe alternative / episode",
        ),
    ]:
        values = actual.loc[order, column] / actual.loc[order, "annual_episodes"]
        ax.bar(labels, values, color=["#267ead", "#e78b27", "#27856a"])
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(
        "Ex-post one-day flood audit · known actual inflow · NOT a deployed safety controller",
        fontsize=14,
    )
    fig.savefig(args.output / "flood_feasibility.png", dpi=170)
    plt.close(fig)
    unchanged = all(
        digest(Path(p)) == h
        for key in ["source_sha256", "input_sha256"]
        for p, h in plan[key].items()
    )
    assert unchanged
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                complete=True,
                test_evaluated=False,
                source_and_inputs_unchanged=True,
                audited_states=len(data),
                scenario_rows=len(audit),
                controller_changed=False,
            ),
            indent=2,
        )
        + "\n"
    )
    report = (
        "# Flood feasibility audit\n\n"
        + summary.to_string(index=False)
        + "\n\nKnown actual inflow is ex-post privileged information. Grid failure is not an infeasibility proof; certificates use independent conservative bounds. No annual counterfactual policy or absolute safety claim.\n"
    )
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
