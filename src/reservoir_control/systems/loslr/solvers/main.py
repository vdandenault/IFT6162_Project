"""Compare the saved Plan2014 baseline, refactored RuleCurve, and MPC shooting in W&B."""

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
from pydantic import Field

from reservoir_control.systems.loslr.operator import AdjustFlowConfig, OntarioConfig
from reservoir_control.systems.loslr.simulator import LoslrEnv
from reservoir_control.systems.loslr.solvers.mpc_shooting import MPCShootingCfg
from reservoir_control.systems.loslr.solvers.rule_curve import (
    ForecastLeadTime,
    ForecastSkill,
    RuleCurveCfg,
    Trace,
)
from external.loslr_regulation_optimization.functions.limits.Bv7 import planLimits

DATA_PATH = Path("data")
WANDB_PROJECT = "bilevel_reservoirs"
POLICY_ID = 0
START_STEP = 48
N_STEPS = 48
PREDICTION_HORIZON = 4
MAXITER = 20
DATE_COLUMNS = ["Sim", "Year", "Month", "QM"]
RUNS = ("baseline_pkl", "rule_curve", "mpc")
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


class ComparisonMetrics(MetricSchema):
    """Time-series metrics consumed by W&B query reporter."""

    step: int | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_flow: float | None = Field(default=None, json_schema_extra=SERIES)
    rule_curve_flow: float | None = Field(default=None, json_schema_extra=SERIES)
    mpc_flow: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_level: float | None = Field(default=None, json_schema_extra=SERIES)
    rule_curve_level: float | None = Field(default=None, json_schema_extra=SERIES)
    mpc_level: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_pointe_claire: float | None = Field(default=None, json_schema_extra=SERIES)
    rule_curve_pointe_claire: float | None = Field(default=None, json_schema_extra=SERIES)
    mpc_pointe_claire: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_weighted_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    rule_curve_weighted_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    mpc_weighted_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_weighted_cumulative: float | None = Field(default=None, json_schema_extra=SERIES)
    rule_curve_weighted_cumulative: float | None = Field(default=None, json_schema_extra=SERIES)
    mpc_weighted_cumulative: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_upstream_coastal_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_upstream_coastal_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    mpc_upstream_coastal_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_upstream_coastal_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_upstream_coastal_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_upstream_coastal_cumulative: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_downstream_coastal_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_downstream_coastal_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_downstream_coastal_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_downstream_coastal_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_downstream_coastal_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_downstream_coastal_cumulative: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_negative_energy_value_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_negative_energy_value_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_negative_energy_value_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_negative_energy_value_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_negative_energy_value_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_negative_energy_value_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    baseline_pkl_recreational_boating_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_recreational_boating_stage: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_recreational_boating_stage: float | None = Field(default=None, json_schema_extra=SERIES)
    baseline_pkl_recreational_boating_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    rule_curve_recreational_boating_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
    )
    mpc_recreational_boating_cumulative: float | None = Field(
        default=None, json_schema_extra=SERIES
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


def load_baseline_pkl() -> tuple[pd.DataFrame, np.ndarray]:
    baseline = DATA_PATH / "baseline"
    data_all = pd.read_pickle(baseline / "legacy_outSim.pkl").reset_index(drop=True)
    costs_all = pd.read_pickle(baseline / "output.pkl").reset_index(drop=True)

    if len(data_all) != len(costs_all) or len(data_all) < N_STEPS:
        raise ValueError("baseline_pkl trajectory and costs must have equal, sufficient lengths")

    data = data_all.iloc[:N_STEPS].copy()
    costs = costs_all.iloc[:N_STEPS]
    stage_costs = np.column_stack(
        (
            costs["upstreamCoastal"],
            costs["downstreamCoastal"],
            -costs["totalEnergyValue"],
            costs["totalRecBoating"],
        )
    ).astype(np.float64)

    if not np.isfinite(stage_costs).all():
        raise ValueError("baseline_pkl stage costs contain non-finite values")

    LOG.info("Loaded %d baseline_pkl baseline steps", N_STEPS)

    return data, stage_costs


def run_solver(
    name: str, config: Any, baseline_pkl: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, float]:
    solver = config.build_optimizer()
    expected = solver.data.iloc[START_STEP : START_STEP + N_STEPS][DATE_COLUMNS]

    pd.testing.assert_frame_equal(
        baseline_pkl[DATE_COLUMNS].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    LOG.info("Starting %s", name)

    started = perf_counter()
    result = solver.train()
    elapsed = perf_counter() - started
    data = result["data"].reset_index(drop=True)
    mdp = result["mdp"]
    costs = np.stack(mdp.rewards["ont"]).astype(np.float64)

    if mdp.t != N_STEPS or len(data) != N_STEPS or costs.shape != (N_STEPS, 4):
        raise ValueError(f"{name} returned an unexpected horizon or stage-cost shape")

    if not np.isfinite(costs).all():
        raise ValueError(f"{name} returned non-finite stage costs")

    np.testing.assert_array_equal(result["running_cost"], costs.sum(axis=0))
    pd.testing.assert_frame_equal(
        baseline_pkl[DATE_COLUMNS].reset_index(drop=True),
        data[DATE_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )
    LOG.info("%s finished in %.2f s; cost vector: %s", name, elapsed, costs.sum(axis=0))

    return data, costs, elapsed


def build_trajectory(
    data: dict[str, pd.DataFrame], costs: dict[str, np.ndarray], weights: np.ndarray
) -> pd.DataFrame:
    trajectory = pd.DataFrame({"step": np.arange(1, N_STEPS + 1)})

    for name in RUNS:
        frame = data[name]
        stage = costs[name]
        weighted = stage @ weights
        trajectory[f"{name}_flow"] = frame["ontFlow"].to_numpy() * 10
        trajectory[f"{name}_level"] = frame["ontLevelMOQ"].to_numpy()
        trajectory[f"{name}_pointe_claire"] = frame["ptclaireLevel"].to_numpy()
        trajectory[f"{name}_weighted_stage"] = weighted
        trajectory[f"{name}_weighted_cumulative"] = np.cumsum(weighted)

        for i, key in enumerate(COST_KEYS):
            trajectory[f"{name}_{key}_stage"] = stage[:, i]
            trajectory[f"{name}_{key}_cumulative"] = np.cumsum(stage[:, i])

    return trajectory


def report_to_wandb(trajectory: pd.DataFrame) -> None:
    """Publish interactive Plotly queries through MetaMARL; do not save figures."""

    metric_logger = MetricLogger.from_schema(ComparisonMetrics)

    for row in trajectory.to_dict(orient="records"):
        metric_logger.push_data(ComparisonMetrics(**row))

    config = WandbConfig(project=WANDB_PROJECT)
    config.world = "loslr"
    config.outer_iters = N_STEPS
    reporter = config.build(label=f"rulecurve-vs-shooting-h{PREDICTION_HORIZON}")
    reporter.schema = ComparisonMetrics

    def add_plot(title: str, metric: str, unit: str) -> None:
        fields = tuple((f"{name}_{metric}",) for name in RUNS)

        reporter.add_query(
            Query(
                title=title,
                x=("step",),
                y=fields,
                legend_labels=("baseline_pkl", "RuleCurve", "MPC shooting"),
                plot_modes=("lines",) * len(RUNS),
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
        add_plot(f"{title} — stage", f"{key}_stage", "Cost per quarter-month ($)")
        add_plot(f"{title} — cumulative", f"{key}_cumulative", "Cumulative cost ($)")

    try:
        reporter.report(metric_logger.peek())
        LOG.info("Published MetaMARL interactive plots to W&B project %s", WANDB_PROJECT)
    finally:
        reporter.close()


def log_comparison(
    data: dict[str, pd.DataFrame],
    costs: dict[str, np.ndarray],
    runtimes: dict[str, float],
    weights: np.ndarray,
) -> None:
    totals = {name: costs[name].sum(axis=0) for name in RUNS}
    summary = pd.DataFrame(totals, index=COST_COLUMNS)
    summary.loc["weightedObjective"] = {name: float(weights @ totals[name]) for name in RUNS}
    summary.loc["runtimeSeconds"] = runtimes
    summary.loc["finalOntarioLevel"] = {
        name: float(data[name]["ontLevelEOQ"].iloc[-1]) for name in RUNS
    }
    summary.loc["maxOntarioLevel"] = {name: float(data[name]["ontLevelEOQ"].max()) for name in RUNS}
    summary.loc["maxPointeClaireLevel"] = {
        name: float(data[name]["ptclaireLevel"].max()) for name in RUNS
    }
    summary["MPC - baseline_pkl"] = summary["mpc"] - summary["baseline_pkl"]

    LOG.info("Experiment summary:\n%s", summary.to_string(float_format=lambda x: f"{x:,.4f}"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    baseline_pkl_data, baseline_pkl_costs = load_baseline_pkl()
    weights = 1.0 / np.maximum(np.mean(np.abs(baseline_pkl_costs), axis=0), 1.0)

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
    rule_curve_config = (
        RuleCurveCfg()
        .training(**training)
        .environment(**environment)
        .agents(make_agent("adjustment"))
    )
    mpc_config = (
        MPCShootingCfg()
        .training(
            **training, prediction_horizon=PREDICTION_HORIZON, weights=weights, maxiter=MAXITER
        )
        .environment(**environment)
        .agents(make_agent("flow"))
    )
    rule_curve_data, rule_curve_costs, rule_curve_runtime = run_solver(
        "RuleCurve", rule_curve_config, baseline_pkl_data
    )
    compared_columns = [
        "ontFlow",
        "flowRegime",
        "ontLevelBOQ",
        "ontLevelEOQ",
        "ontLevelMOQ",
        "ptclaireLevel",
    ]

    if baseline_pkl_data[compared_columns].equals(rule_curve_data[compared_columns]):
        LOG.info("Refactored RuleCurve matches the saved baseline on selected columns")
    else:
        LOG.warning("RuleCurve differs from the saved baseline on selected columns")

    LOG.warning(
        "RuleCurve currently uses its legacy linear _simulate(), whereas MPC uses "
        "the cubic LoslrEnv; differences do not isolate the controller alone."
    )

    mpc_data, mpc_costs, mpc_runtime = run_solver("MPC shooting", mpc_config, baseline_pkl_data)
    data = {"baseline_pkl": baseline_pkl_data, "rule_curve": rule_curve_data, "mpc": mpc_data}
    costs = {"baseline_pkl": baseline_pkl_costs, "rule_curve": rule_curve_costs, "mpc": mpc_costs}
    runtimes = {"baseline_pkl": np.nan, "rule_curve": rule_curve_runtime, "mpc": mpc_runtime}

    log_comparison(data, costs, runtimes, weights)
    report_to_wandb(build_trajectory(data, costs, weights))


if __name__ == "__main__":
    main()
