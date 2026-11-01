"""Closed-loop one-day grid comparison under an explicit inflow forecast.

This is a separate myopic diagnostic, not the deployed seven-day MPC.
A vectorized hourly rollout enumerates every action; scalar model checks guard
threshold ambiguity and near-minimum numerical differences for both objectives.
"""
import argparse
import ast
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import settings
from lake_como.mpc import DeterministicMPC, PlanningModel
from lake_como.mpc_diagnostics import level_penalties
from lake_como.mpc_hindsight import digest, load_baseline, perfect_inflow_controller
from lake_como.mpc_objective_audit import score_plan
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import evaluate
from lake_como.splits import split_years

POINTS = 10001
METRICS = [
    "full_return",
    "objective_return",
    "flood_days",
    "low_water_days",
    "deficit_million_m3",
    "terminal_storage_m3",
]


@dataclass(frozen=True)
class ActionScores:
    costs: np.ndarray
    levels: np.ndarray
    releases: np.ndarray
    scalar_rechecks: int


def score_actions(model, requests, storage, inflow, demand, day):
    """Batched counterpart to scalar score_plan; objective columns proxy/binary."""
    requests = np.asarray(requests, dtype=float)
    lake = model.lake
    states = np.full(len(requests), storage, dtype=float)
    release_steps = []
    ambiguous = np.zeros(len(requests), dtype=bool)
    for _ in range(24):
        levels = states / lake.surface + lake.min_level
        # Switches in the upstream piecewise hydraulic model need scalar checks.
        for boundary in (lake.min_level, lake.linear_limit, lake.max_level):
            ambiguous |= np.abs(levels - boundary) < 1e-10
        rating = lake.C_r * np.maximum(levels - lake.alpha, 0.0) ** lake.beta
        minimum = np.where(
            levels <= lake.min_level,
            0.0,
            np.where(levels <= lake.max_level, lake.min_env_flow[day - 1], rating),
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
        release = np.minimum(maximum, np.maximum(minimum, requests))
        release_steps.append(release)
        states = states + 3600.0 * (inflow - release)
    levels = states / lake.surface + lake.min_level
    releases = np.mean(release_steps, axis=0)
    for boundary in (model.flood_level, model.low_level):
        ambiguous |= np.abs(levels - boundary) < 1e-9
    deficit = np.maximum(
        demand - np.maximum(releases - lake.min_env_flow[day - 1], 0.0), 0.0
    )
    seasonal = (
        0.5 * (3 - np.cos((day - 1) * 2 * np.pi / 365))
        if model.smooth_deficit
        else (2.0 if 120 < day <= 243 else 1.0)
    )
    c = model.coefficients
    other = (
        c["daily_deficit_coeff"] * deficit * seasonal
        + c["wasted_water_coeff"] * np.maximum(releases - demand, 0.0)
        + c["clip_action_coeff"] * (requests - releases) ** 2
    )
    binary, surrogate = level_penalties(levels, model)
    terminal = model.config.terminal_level_weight * (levels - lake.init_level) ** 2
    costs = np.column_stack(
        [(other + surrogate + terminal) / 1000, (other + binary + terminal) / 1000]
    )
    checked = np.zeros(len(requests), dtype=bool)

    def canonical(indices):
        for index in indices:
            score = score_plan(
                model, [requests[index]], storage, [inflow], [demand], np.array([day])
            )
            costs[index] = score.surrogate, score.binary
            levels[index] = score.levels[0]
            releases[index] = score.releases[0]
            checked[index] = True

    canonical(np.flatnonzero(ambiguous))
    # Recheck the near-best candidates for both objectives, including a new winner
    # exposed by a scalar correction. Both policies get the same corrected table.
    while True:
        near = np.zeros(len(requests), dtype=bool)
        for column in (0, 1):
            best = costs[:, column].min()
            band = 1e-10 * max(1.0, abs(best))
            near |= costs[:, column] <= best + band
        pending = np.flatnonzero(near & ~checked)
        if not len(pending):
            break
        canonical(pending)
    if not np.isfinite(costs).all():
        raise ValueError("Nonfinite grid cost")
    return ActionScores(costs, levels, releases, int(checked.sum()))


class OneDayGridPolicy:
    """Identical exhaustive search for either objective and supplied forecast."""

    def __init__(self, params, forecast, config, objective, points=POINTS):
        if objective not in ("surrogate", "binary") or points < 2:
            raise ValueError("Invalid objective or grid")
        if config.horizon_days != 1:
            raise ValueError("This diagnostic is explicitly one-day")
        self.model = PlanningModel(params, config)
        self.forecast = forecast
        self.config = config
        self.objective = objective
        self.requests = (
            np.linspace(self.model.low, self.model.high, points)
            .astype(np.float32)
            .astype(float)
        )
        self.diagnostics = []

    def __call__(self, observation, previous_info=None):
        obs = np.asarray(observation)
        if obs.shape != (4,) or not np.isfinite(obs).all():
            raise ValueError("Raw finite four-number observation required")
        day = (
            int(np.rint((np.arctan2(obs[1], obs[2]) % (2 * np.pi)) * 365 / (2 * np.pi)))
            % 365
            + 1
        )
        storage = self.model.lake.level_to_storage(float(obs[0]))
        inflow = float(self.forecast.inflow[day - 1])
        demand = float(self.forecast.demand[day - 1])
        scores = score_actions(self.model, self.requests, storage, inflow, demand, day)
        proxy = int(scores.costs[:, 0].argmin())
        binary = int(scores.costs[:, 1].argmin())
        index = proxy if self.objective == "surrogate" else binary
        self.diagnostics.append(
            dict(
                day=day,
                objective=self.objective,
                grid_points=len(self.requests),
                scalar_rechecks=scores.scalar_rechecks,
                request_m3_s=self.requests[index],
                surrogate_best_request_m3_s=self.requests[proxy],
                binary_best_request_m3_s=self.requests[binary],
                forecast_inflow_m3_s=inflow,
                forecast_demand_m3_s=demand,
                predicted_level_m=scores.levels[index],
                predicted_release_m3_s=scores.releases[index],
                selected_surrogate_cost=1000 * scores.costs[index, 0],
                selected_binary_cost=1000 * scores.costs[index, 1],
            )
        )
        return np.array([self.requests[index]], dtype=np.float32)


def level_severity(daily, flood_level=1.24, low_level=-0.45):
    """Daily-endpoint level-exceedance integrals, not flood volume or damage."""
    values = daily[["objective", "year", "level_m"]].copy()
    values["flood_excess_m"] = np.maximum(values.level_m - flood_level, 0.0)
    values["low_excess_m"] = np.maximum(low_level - values.level_m, 0.0)
    return values.groupby(["objective", "year"]).agg(
        flood_excess_m_days=("flood_excess_m", "sum"),
        low_excess_m_days=("low_excess_m", "sum"),
        max_level_m=("level_m", "max"),
        min_level_m=("level_m", "min"),
    )


def plots(annual, daily, output, inflow_source="perfect"):
    label = (
        "HINDSIGHT ONLY"
        if inflow_source == "perfect"
        else "NON-ORACLE SEASONAL FORECAST"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, metric, title in zip(
        axes.flat,
        ["full_return", "flood_days", "low_water_days", "deficit_million_m3"],
        [
            "Annual full return (higher is better)",
            "Flood days",
            "Low-water days",
            "Unmet demand (million m³)",
        ],
    ):
        for objective in ("surrogate", "binary"):
            f = annual[annual.objective == objective]
            ax.plot(
                f.year, f[metric], marker="o", label=objective.title() + " level cost"
            )
        ax.set(title=title, xlabel="Validation year")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{label} · common one-day grid · only level-penalty objective differs"
    )
    fig.savefig(output / "annual_comparison.png", dpi=170)
    plt.close(fig)
    original = pd.read_csv(
        PROJECT_ROOT / "results/comparison/mpc_validation_v1/trajectories.csv"
    )
    original = original[original.family == "mpc"]
    from lake_como.mpc_diagnostics import longest_episode

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    for ax, (column, threshold, label) in zip(
        axes, [("flood", 1.24, "Flood"), ("low_water", -0.45, "Low water")]
    ):
        year = int(original.groupby("year")[column].sum().sort_index().idxmax())
        f = original[original.year == year]
        episode = longest_episode(f, f[column].astype(bool))
        start = max(1, int(episode.day.min()) - 7)
        end = min(365, int(episode.day.max()) + 7)
        for objective in ("surrogate", "binary"):
            window = daily[
                (daily.objective == objective)
                & (daily.year == year)
                & daily.day.between(start, end)
            ]
            ax.plot(window.day, window.level_m, label=objective.title() + " level cost")
        ax.axhline(threshold, color="gray", ls="--")
        ax.set(
            title=f"{label}: {year}, original baseline failure window",
            xlabel="Day",
            ylabel="Realized level (m)",
        )
        ax.legend(fontsize=8)
    fig.suptitle(f"{label} · common one-day search · closed-loop states can differ")
    fig.savefig(output / "baseline_failure_windows.png", dpi=170)
    plt.close(fig)


def forecast_for_mode(mode, year, params, reference, config, validation_inflows=None):
    """Seasonal mode must never inspect held-out inflow profiles."""
    if year not in split_years("validation"):
        raise ValueError("This comparison is validation-only")
    if mode == "seasonal":
        if reference.forecast.training_years != tuple(split_years("train")):
            raise ValueError("Seasonal forecasts must be training-only")
        return reference.forecast
    if mode == "perfect":
        if validation_inflows is None:
            raise ValueError("Explicit validation inflows required for hindsight")
        return perfect_inflow_controller(
            year, params, reference.forecast, config, validation_inflows
        ).forecast
    raise ValueError("Unknown inflow source")


def search_signature(source):
    """Compare executable search ASTs, ignoring comments and docstrings."""

    class StripDocs(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            return node

        visit_ClassDef = visit_FunctionDef

    tree = StripDocs().visit(ast.parse(source))
    names = {"ActionScores", "score_actions", "OneDayGridPolicy"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in nodes} != names:
        raise ValueError("Missing shared search implementation")
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def check_hindsight_reference(root, config):
    plan = json.loads((root / "plan.json").read_text())
    run = json.loads((root / "run.json").read_text())
    if not run["complete"] or run["test_evaluated"] or run["annual_episodes"] != 20:
        raise ValueError("Completed validation hindsight pair required")
    if plan["diagnostic_config"] != asdict(config) or plan["grid_points"] != POINTS:
        raise ValueError("Hindsight horizon/grid/settings differ")
    snapshot = root / run["execution_source_snapshot"]
    old_hash = next(
        value
        for name, value in plan["source_sha256"].items()
        if Path(name).name == "mpc_objective_closed_loop.py"
    )
    if digest(snapshot) != old_hash:
        raise ValueError("Hindsight execution snapshot changed")
    if search_signature(snapshot.read_text()) != search_signature(
        Path(__file__).read_text()
    ):
        raise ValueError("Search code changed; do not reuse this hindsight reference")
    for name, expected in plan["source_sha256"].items():
        if (
            Path(name).name != "mpc_objective_closed_loop.py"
            and digest(Path(name)) != expected
        ):
            raise ValueError(f"Hindsight model/scoring source changed: {name}")
    files = [
        root / name
        for name in [
            "plan.json",
            "run.json",
            "annual.csv",
            "trajectories.csv",
            "level_severity.csv",
        ]
    ]
    return {str(path): digest(path) for path in files}


def compare_forecasts(annual, daily, severity, reference, output):
    """Paired 2×2 descriptive comparison, reusing the verified hindsight pair."""
    old_annual = pd.read_csv(reference / "annual.csv", float_precision="round_trip")
    old_daily = pd.read_csv(
        reference / "trajectories.csv", float_precision="round_trip"
    )
    old_severity = pd.read_csv(reference / "level_severity.csv").set_index(
        ["objective", "year"]
    )
    recomputed = level_severity(old_daily)
    pd.testing.assert_frame_equal(
        old_severity, recomputed, check_exact=False, rtol=1e-12, atol=1e-12
    )
    old_daily["inflow_source"] = "perfect"
    verify_matched_inputs(
        pd.concat([old_daily, daily], ignore_index=True), split_years("validation")
    )
    old_annual["inflow_source"] = "perfect"
    combined = pd.concat([annual, old_annual], ignore_index=True)
    sev = pd.concat(
        [
            severity.reset_index().assign(inflow_source="seasonal"),
            recomputed.reset_index().assign(inflow_source="perfect"),
        ],
        ignore_index=True,
    )
    means = combined.groupby(["inflow_source", "objective"])[METRICS].mean()
    summary = means.join(
        sev.groupby(["inflow_source", "objective"])[
            ["flood_excess_m_days", "low_excess_m_days", "max_level_m", "min_level_m"]
        ].mean()
    )
    forecast_rows = []
    for objective in ("surrogate", "binary"):
        a = combined[combined.objective == objective]
        delta = (
            a[a.inflow_source == "perfect"].set_index("year")[METRICS]
            - a[a.inflow_source == "seasonal"].set_index("year")[METRICS]
        )
        delta["objective"] = objective
        forecast_rows.append(delta.reset_index())
    combined.to_csv(output / "comparison_annual.csv", index=False)
    summary.to_csv(output / "forecast_objective_summary.csv")
    pd.concat(forecast_rows, ignore_index=True).to_csv(
        output / "perfect_minus_seasonal.csv", index=False
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    order = [
        ("seasonal", "surrogate"),
        ("seasonal", "binary"),
        ("perfect", "surrogate"),
        ("perfect", "binary"),
    ]
    labels = [
        "Seasonal\nSurrogate",
        "Seasonal\nBinary",
        "Hindsight\nSurrogate",
        "Hindsight\nBinary",
    ]
    for ax, metric, title in zip(
        axes.flat,
        [
            "full_return",
            "flood_days",
            "low_water_days",
            "deficit_million_m3",
            "flood_excess_m_days",
            "low_excess_m_days",
        ],
        [
            "Full annual return ↑",
            "Flood days ↓",
            "Low-water days ↓",
            "Unmet demand (million m³) ↓",
            "Flood exceedance (m·days) ↓",
            "Low-water exceedance (m·days) ↓",
        ],
    ):
        ax.bar(
            range(4),
            [summary.loc[key, metric] for key in order],
            color=["tab:blue", "tab:orange"] * 2,
        )
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Same one-day search · seasonal forecasts versus privileged hindsight · validation-year means"
    )
    fig.savefig(output / "forecast_objective_comparison.png", dpi=170)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "results/comparison/mpc_validation_v1",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
    )
    parser.add_argument(
        "--inflow-source", choices=["perfect", "seasonal"], default="perfect"
    )
    parser.add_argument(
        "--hindsight-reference",
        type=Path,
        default=PROJECT_ROOT / "results/mpc/objective_closed_loop_v1",
    )
    args = parser.parse_args()
    args.output = args.output or PROJECT_ROOT / (
        "results/mpc/objective_closed_loop_v1"
        if args.inflow_source == "perfect"
        else "results/mpc/objective_closed_loop_seasonal_v1"
    )
    forecast_label = (
        "HINDSIGHT" if args.inflow_source == "perfect" else "SEASONAL_NON_ORACLE"
    )
    _, baseline_config, _, _ = load_baseline(args.baseline)
    config = replace(baseline_config, horizon_days=1)
    reference_hashes = (
        check_hindsight_reference(args.hindsight_reference, config)
        if args.inflow_source == "seasonal"
        else {}
    )
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty --output")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = DeterministicMPC.from_training(baseline_config)
    params = settings("train")
    inflows = (
        settings("validation")["inflow"] if args.inflow_source == "perfect" else None
    )
    years = split_years("validation")
    files = [
        Path(__file__),
        PROJECT_ROOT / "src/lake_como/mpc_objective_audit.py",
        PROJECT_ROOT / "src/lake_como/mpc.py",
        PROJECT_ROOT / "src/lake_como/reproduce.py",
        PROJECT_ROOT / "configs/mpc.json",
    ]
    inputs = [
        args.baseline / "run.json",
        args.baseline / "trajectories.csv",
        args.baseline / "annual.csv",
        args.baseline / "mpc_diagnostics.csv",
    ]
    plan = dict(
        label=f"CLOSED-LOOP {forecast_label} OBJECTIVE ABLATION",
        inflow_source=args.inflow_source,
        reused_hindsight_sha256=reference_hashes,
        search_code_matches_hindsight=bool(reference_hashes),
        years=years,
        baseline_config=asdict(baseline_config),
        diagnostic_config=asdict(config),
        grid_points=POINTS,
        objectives=["surrogate", "binary"],
        shared=f"One-day exhaustive grid, float32 requests, {args.inflow_source} inflow, seasonal demand, hourly hydraulics, annual resets",
        contrast="Only level-penalty objective differs between the two arms",
        horizon_scope="Both diagnostic arms use one day. Neither is the deployed seven-day SLSQP MPC.",
        numerical_search="Batched hourly predictions; scalar checks at hydraulic/penalty switches and near both minima",
        terminal="Original zero terminal weight retained; no terminal storage constraint",
        deployment="Separate validation diagnostic; no deployed-controller selection",
        test_evaluated=False,
        source_sha256={str(p): digest(p) for p in files},
        baseline_sha256={str(p): digest(p) for p in inputs},
    )
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frames = []
    annuals = []
    diagnostics = []
    runtimes = []
    for objective in ("surrogate", "binary"):
        for year in years:
            forecast = forecast_for_mode(
                args.inflow_source, year, params, reference, config, inflows
            )
            policy = OneDayGridPolicy(params, forecast, config, objective)
            frame, seconds = evaluate(
                f"one_day_grid_{objective}_{forecast_label}",
                split="validation",
                year=year,
                protocol="chronological",
                correct_terminal_level=True,
                sequential_years=True,
                measure_decisions=True,
                controller_factory=lambda env: policy,
            )
            diagnostic = pd.DataFrame(policy.diagnostics)
            diagnostic["year"] = year
            diagnostic["actual_level_m"] = frame.level_m.to_numpy()
            diagnostic["level_error_m"] = (
                diagnostic.actual_level_m - diagnostic.predicted_level_m
            )
            expected_inflow = (
                frame.inflow_m3_s
                if args.inflow_source == "perfect"
                else reference.forecast.inflow
            )
            np.testing.assert_array_equal(
                diagnostic.forecast_inflow_m3_s, expected_inflow
            )
            diagnostic["inflow_error_m3_s"] = (
                frame.inflow_m3_s.to_numpy() - diagnostic.forecast_inflow_m3_s
            )
            diagnostic["inflow_source"] = args.inflow_source
            np.testing.assert_array_equal(
                diagnostic.forecast_demand_m3_s, reference.forecast.demand
            )
            annual = annual_metrics(frame)
            frame["objective"] = objective
            frame["controller"] = f"grid_{objective}_{forecast_label}"
            frame["inflow_source"] = args.inflow_source
            annual["objective"] = objective
            annual["inflow_source"] = args.inflow_source
            frames.append(frame)
            annuals.append(annual)
            diagnostics.append(diagnostic)
            runtimes.append(
                dict(
                    objective=objective,
                    year=year,
                    seconds=seconds,
                    mean_decision_ms=frame.decision_ns.mean() / 1e6,
                    p95_decision_ms=frame.decision_ns.quantile(0.95) / 1e6,
                )
            )
            print(
                f"{forecast_label} one-day {objective}: {year} complete ({seconds:.2f}s)",
                flush=True,
            )
    daily = pd.concat(frames, ignore_index=True)
    annual = pd.concat(annuals, ignore_index=True)
    trace = pd.concat(diagnostics, ignore_index=True)
    verify_matched_inputs(daily, years)
    means = annual.groupby("objective", sort=False)[METRICS].mean()
    paired = (
        annual[annual.objective == "binary"].set_index("year")[METRICS]
        - annual[annual.objective == "surrogate"].set_index("year")[METRICS]
    )
    daily.to_csv(args.output / "trajectories.csv", index=False)
    annual.to_csv(args.output / "annual.csv", index=False)
    means.to_csv(args.output / "summary.csv")
    paired.to_csv(args.output / "paired_annual.csv")
    trace.to_csv(args.output / "grid_diagnostics.csv", index=False)
    pd.DataFrame(runtimes).to_csv(args.output / "runtimes.csv", index=False)
    severity = level_severity(
        daily, reference.model.flood_level, reference.model.low_level
    )
    severity.to_csv(args.output / "level_severity.csv")
    plots(annual, daily, args.output, args.inflow_source)
    factorial = (
        compare_forecasts(
            annual, daily, severity, args.hindsight_reference, args.output
        )
        if args.inflow_source == "seasonal"
        else None
    )
    record = dict(
        complete=True,
        test_evaluated=False,
        deployed_controller_changed=False,
        annual_episodes=len(annual),
        total_decisions=len(daily),
        mean_full_return_change=float(paired.full_return.mean()),
        years_binary_better=int((paired.full_return > 0).sum()),
        source_unchanged=all(
            digest(Path(p)) == v for p, v in plan["source_sha256"].items()
        ),
        baseline_unchanged=all(
            digest(Path(p)) == v for p, v in plan["baseline_sha256"].items()
        ),
        scope=f"Binary minus surrogate on common ONE-DAY {args.inflow_source} grids; not a seven-day MPC improvement estimate",
        inflow_source=args.inflow_source,
        reference_unchanged=all(
            digest(Path(p)) == v for p, v in reference_hashes.items()
        ),
    )
    if (
        not record["source_unchanged"]
        or not record["baseline_unchanged"]
        or not record["reference_unchanged"]
    ):
        raise ValueError("Study sources changed")
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output / "report.md").write_text(
        f"""# Closed-loop objective comparison — {forecast_label}

Both arms use the same 10,001-action exhaustive one-day grid, {args.inflow_source} inflow,
seasonal demand, hourly hydraulics, float32 observations/actions, and annual resets.
Only binary versus surrogate level penalties differ. This is a common myopic
search diagnostic, not a selected seven-day MPC replacement.

```text
{means.round(3).to_string()}
```

## Level-severity means

```text
{severity.groupby('objective').mean().round(6).to_string()}
```

Fewer flood days need not mean lower cumulative exceedance. These integrals
use daily endpoints in m·days, not flood volume or calibrated damage. Keep
supply and severity visible alongside the benchmark return.

Binary minus surrogate mean full-return change: {paired.full_return.mean():,.2f}.
Binary improves full return in {int((paired.full_return>0).sum())}/10 validation years.
The result measures the objective intervention in these two closed-loop policies;
their water states diverge, so individual days are not same-state comparisons.
Demand remains imperfect and no terminal storage value is added. Do not attribute
a difference from prior seven-day SLSQP results solely to the objective: this
experiment also has a different horizon/search, shared only within the pair.

[Annual comparison](annual_comparison.png) · [Fixed failure windows](baseline_failure_windows.png)
"""
    )
    if factorial is not None:
        with (args.output / "report.md").open("a") as report:
            report.write(
                "\n## Matched forecast/objective comparison\n\n```text\n"
                + factorial.round(5).to_string()
                + "\n```\n"
            )
    print(means.round(2).to_string())
    print(record)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
