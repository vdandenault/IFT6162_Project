"""Compare pkl 2014 with RuleCurve refactor and three MPC solvers.

Numerical results are saved under experiments/; interactive plots use MetaMARL's
W&B reporting backend. No plot images are written locally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from gymnasium.spaces import Box
from metamarl.metrics.enums import ReduceProtocol
from metamarl.metrics.logger import MetricLogger
from metamarl.metrics.schemas import MetricSchema
from metamarl.reporting.query import Query
from metamarl.reporting.wandb import WandbConfig
from pydantic import Field, create_model

from bilevel_reservoirs.controllers.operator import AdjustFlowConfig, OntarioConfig
from bilevel_reservoirs.simulators.loslr import LoslrEnv
from bilevel_reservoirs.solvers.mpc_beam import MPCBeamSearchCfg
from bilevel_reservoirs.solvers.mpc_mppi import MPCMPPICfg
from bilevel_reservoirs.solvers.mpc_shooting import MPCShootingCfg
from bilevel_reservoirs.solvers.rule_curve import (
    ForecastLeadTime,
    ForecastSkill,
    RuleCurveCfg,
    Trace,
)
from external.loslr_regulation_optimization.functions.limits.Bv7 import planLimits

DATA_PATH = Path("data")
OUTPUT_PATH = Path("experiments") / "mpc_shooting"
WANDB_PROJECT = "bilevel_reservoirs"
POLICY_ID = 0
START_STEP = 48
N_STEPS = 48
PREDICTION_HORIZON = 4
SHOOTING_MAXITER = 20

DATE_COLUMNS = ["Sim", "Year", "Month", "QM"]
METHODS = {
    "pkl": "pkl baseline",
    "rule_curve": "RuleCurve",
    "mpc_shooting": "MPC shooting",
    "mpc_beam": "MPC beam search",
    "mpc_mppi": "MPC MPPI",
}
COST_COLUMNS = ("upstreamCoastal", "downstreamCoastal", "negativeEnergyValue", "totalRecBoating")
COST_KEYS = (
    "upstream_coastal",
    "downstream_coastal",
    "negative_energy_value",
    "recreational_boating",
)
COST_TITLES = (
    "Upstream coastal impacts",
    "Downstream coastal impacts",
    "Hydropower energy value (negative cost)",
    "Recreational boating",
)
SERIES = {"reduce": ReduceProtocol.SERIES}
LOG = logging.getLogger(__name__)

# A flat MetricSchema lets MetaMARL's Query select all five methods in one plot.
# Generate the repetitive Pydantic fields rather than declaring 65 by hand.
METRIC_KEYS = (
    "flow",
    "level",
    "pointe_claire",
    "weighted_stage",
    "weighted_cumulative",
    *(f"{key}_{kind}" for key in COST_KEYS for kind in ("stage", "cumulative")),
)
ComparisonMetrics = create_model(
    "ComparisonMetrics",
    __base__=MetricSchema,
    step=(int | None, Field(default=None, json_schema_extra=SERIES)),
    **{
        f"{method}_{metric}": (float | None, Field(default=None, json_schema_extra=SERIES))
        for method in METHODS
        for metric in METRIC_KEYS
    },
)


def make_agent(mechanism_id: str) -> OntarioConfig:
    return OntarioConfig(
        id="ont",
        policy_id="ont",
        mechanisms=AdjustFlowConfig(
            id=mechanism_id,
            acts_on=("ont", mechanism_id),
            action_space=Box(0.0, np.inf, shape=(1,), dtype=np.float64),
            constraint_fn=planLimits,
        ),
    )


def load_pkl() -> tuple[pd.DataFrame, np.ndarray]:
    baseline = DATA_PATH / "baseline"
    data_all = pd.read_pickle(baseline / "legacy_outSim.pkl").reset_index(drop=True)
    costs_all = pd.read_pickle(baseline / "output.pkl").reset_index(drop=True)

    if len(data_all) != len(costs_all) or len(data_all) < N_STEPS:
        raise ValueError("pkl trajectory and costs must have equal, sufficient lengths")

    data = data_all.iloc[:N_STEPS].copy()
    evaluated = costs_all.iloc[:N_STEPS]
    stage_costs = np.column_stack(
        (
            evaluated["upstreamCoastal"],
            evaluated["downstreamCoastal"],
            -evaluated["totalEnergyValue"],
            evaluated["totalRecBoating"],
        )
    ).astype(np.float64)

    if stage_costs.shape != (N_STEPS, len(COST_KEYS)) or not np.isfinite(stage_costs).all():
        raise ValueError("pkl stage costs have an invalid shape or values")

    LOG.info("Loaded %d pkl baseline steps", N_STEPS)
    return data, stage_costs


def run_solver(name: str, config: Any, pkl: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    solver = config.build_optimizer()
    expected = solver.data.iloc[START_STEP : START_STEP + N_STEPS][DATE_COLUMNS]
    pd.testing.assert_frame_equal(
        pkl[DATE_COLUMNS].reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )

    LOG.info("Starting %s", name)
    started = perf_counter()
    result = solver.train()
    elapsed = perf_counter() - started

    data = result["data"].reset_index(drop=True)
    mdp = result["mdp"]
    stage_costs = np.stack(mdp.rewards["ont"]).astype(np.float64)
    total_cost = np.asarray(result["running_cost"], dtype=np.float64)

    if mdp.t != N_STEPS or len(data) != N_STEPS or stage_costs.shape != (N_STEPS, len(COST_KEYS)):
        raise ValueError(f"{name} returned an unexpected horizon or stage-cost shape")
    if not np.isfinite(stage_costs).all():
        raise ValueError(f"{name} returned non-finite stage costs")

    np.testing.assert_array_equal(total_cost, stage_costs.sum(axis=0))
    pd.testing.assert_frame_equal(
        pkl[DATE_COLUMNS].reset_index(drop=True),
        data[DATE_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )
    LOG.info("%s finished in %.2f s; cost vector: %s", name, elapsed, total_cost)
    return data, stage_costs, elapsed


def build_trajectory(
    data: dict[str, pd.DataFrame], costs: dict[str, np.ndarray], weights: np.ndarray
) -> pd.DataFrame:
    trajectory = pd.DataFrame({"step": np.arange(1, N_STEPS + 1)})
    for name in METHODS:
        frame = data[name]
        stage = costs[name]
        weighted = stage @ weights

        # LOSLR stores releases in tens of m³/s.
        trajectory[f"{name}_flow"] = frame["ontFlow"].to_numpy() * 10.0
        trajectory[f"{name}_level"] = frame["ontLevelMOQ"].to_numpy()
        trajectory[f"{name}_pointe_claire"] = frame["ptclaireLevel"].to_numpy()
        trajectory[f"{name}_weighted_stage"] = weighted
        trajectory[f"{name}_weighted_cumulative"] = np.cumsum(weighted)

        for i, key in enumerate(COST_KEYS):
            trajectory[f"{name}_{key}_stage"] = stage[:, i]
            trajectory[f"{name}_{key}_cumulative"] = np.cumsum(stage[:, i])

    return trajectory


def report_to_wandb(trajectory: pd.DataFrame) -> None:
    """Report interactive Plotly figures with MetaMARL; save no local images."""
    metric_logger = MetricLogger.from_schema(ComparisonMetrics)
    for row in trajectory.to_dict(orient="records"):
        metric_logger.push_data(ComparisonMetrics(**row))

    config = WandbConfig(project=WANDB_PROJECT)
    config.world = "loslr"
    config.outer_iters = N_STEPS
    reporter = config.build(label=f"mpc-comparison-h{PREDICTION_HORIZON}")
    reporter.schema = ComparisonMetrics

    def add_plot(title: str, metric: str, unit: str) -> None:
        reporter.add_query(
            Query(
                title=title,
                x=("step",),
                y=tuple((f"{name}_{metric}",) for name in METHODS),
                legend_labels=tuple(METHODS.values()),
                plot_modes=("lines",) * len(METHODS),
                x_label="Quarter-month step",
                y_label=unit,
            )
        )

    add_plot("Ontario discharge", "flow", "m³/s")
    add_plot("Ontario mean level", "level", "m")
    add_plot("Pointe-Claire level", "pointe_claire", "m")
    add_plot("Weighted stage cost", "weighted_stage", "Weighted cost")
    add_plot("Weighted cumulative cost", "weighted_cumulative", "Weighted cumulative cost")

    for key, title in zip(COST_KEYS, COST_TITLES, strict=True):
        add_plot(f"{title} — stage", f"{key}_stage", "Quarter-month cost ($)")
        add_plot(f"{title} — cumulative", f"{key}_cumulative", "Cumulative cost ($)")

    try:
        reporter.report(metric_logger.peek())
        LOG.info("Reported five-method comparison to W&B project %s", WANDB_PROJECT)
    finally:
        reporter.close()


def build_summary(
    data: dict[str, pd.DataFrame],
    costs: dict[str, np.ndarray],
    runtimes: dict[str, float],
    weights: np.ndarray,
) -> pd.DataFrame:
    totals = {name: costs[name].sum(axis=0) for name in METHODS}
    summary = pd.DataFrame(totals, index=COST_COLUMNS)
    summary.loc["weightedObjective"] = {name: float(weights @ totals[name]) for name in METHODS}
    summary.loc["runtimeSeconds"] = runtimes
    summary.loc["finalOntarioLevel"] = {
        name: float(data[name]["ontLevelEOQ"].iloc[-1]) for name in METHODS
    }
    summary.loc["maxOntarioLevel"] = {
        name: float(data[name]["ontLevelEOQ"].max()) for name in METHODS
    }
    summary.loc["maxPointeClaireLevel"] = {
        name: float(data[name]["ptclaireLevel"].max()) for name in METHODS
    }
    objective_rows = [*COST_COLUMNS, "weightedObjective"]
    for name in METHODS:
        if name in ("pkl", "rule_curve"):
            continue
        for baseline in ("pkl", "rule_curve"):
            summary[f"{name}_minus_{baseline}"] = summary[name] - summary[baseline]
            baseline_cost = summary.loc[objective_rows, baseline].to_numpy(dtype=float)
            candidate_cost = summary.loc[objective_rows, name].to_numpy(dtype=float)
            summary[f"{name}_improvement_vs_{baseline}_pct"] = np.nan
            summary.loc[objective_rows, f"{name}_improvement_vs_{baseline}_pct"] = np.divide(
                100.0 * (baseline_cost - candidate_cost),
                np.abs(baseline_cost),
                out=np.full_like(baseline_cost, np.nan),
                where=baseline_cost != 0,
            )

    LOG.info("Performance summary:\n%s", summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    pkl_data, pkl_costs = load_pkl()
    weights = 1.0 / np.maximum(np.mean(np.abs(pkl_costs), axis=0), 1.0)
    LOG.info("Objective weights: %s", weights)

    training = dict(
        data_path=DATA_PATH,
        forecast_lead_time=ForecastLeadTime.annual,
        forecast_skill=ForecastSkill.sqAR,
        trace=Trace.Y1961_2020,
        september_rule=False,
        id=POLICY_ID,
        drop_last_year=False,
        start_step=START_STEP,
    )
    environment = dict(
        horizon=N_STEPS, env=LoslrEnv, env_config={"stage_init": 74.55, "mechanism_id": None}
    )
    mpc_training = dict(training, prediction_horizon=PREDICTION_HORIZON, weights=weights)

    configs = {
        "rule_curve": (
            RuleCurveCfg()
            .training(**training)
            .environment(**environment)
            .agents(make_agent("adjustment"))
        ),
        "mpc_shooting": (
            MPCShootingCfg()
            .training(**mpc_training, maxiter=SHOOTING_MAXITER)
            .environment(**environment)
            .agents(make_agent("flow"))
        ),
        "mpc_beam": (
            MPCBeamSearchCfg()
            .training(**mpc_training)
            .environment(**environment)
            .agents(make_agent("flow"))
        ),
        "mpc_mppi": (
            MPCMPPICfg()
            .training(**mpc_training)
            .environment(**environment)
            .agents(make_agent("flow"))
        ),
    }

    data = {"pkl": pkl_data}
    costs = {"pkl": pkl_costs}
    runtimes = {"pkl": np.nan}

    for key, config in configs.items():
        data[key], costs[key], runtimes[key] = run_solver(METHODS[key], config, pkl_data)

    checked = [
        "ontFlow",
        "flowRegime",
        "ontLevelBOQ",
        "ontLevelEOQ",
        "ontLevelMOQ",
        "ptclaireLevel",
    ]
    if data["rule_curve"][checked].equals(data["pkl"][checked]):
        LOG.info("RuleCurve exactly matches selected pkl baseline columns")
    else:
        LOG.warning("RuleCurve differs from the pkl baseline in selected columns")

    LOG.warning(
        "RuleCurve uses its legacy linear transition, while MPC optimizers use "
        "LoslrEnv; comparisons may reflect dynamics differences, not only controllers."
    )

    summary = build_summary(data, costs, runtimes, weights)
    trajectory = build_trajectory(data, costs, weights)
    stages = pd.DataFrame(
        {f"{name}_{key}": costs[name][:, i] for name in METHODS for i, key in enumerate(COST_KEYS)}
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        frame.to_csv(OUTPUT_PATH / f"{name}.csv", index=False)
    summary.to_csv(OUTPUT_PATH / "comparison.csv")
    trajectory.to_csv(OUTPUT_PATH / "trajectory.csv", index=False)
    stages.to_csv(OUTPUT_PATH / "stage_costs.csv", index=False)
    LOG.info("Saved numerical results to %s", OUTPUT_PATH)

    report_to_wandb(trajectory)


if __name__ == "__main__":
    main()
