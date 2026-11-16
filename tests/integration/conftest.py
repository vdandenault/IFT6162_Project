from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box

from bilevel_reservoirs.controllers.operator import AdjustFlowConfig
from bilevel_reservoirs.controllers.operator import OntarioConfig as Ontario
from bilevel_reservoirs.simulators.loslr import LoslrEnv
from bilevel_reservoirs.solvers.rule_curve import (
    ForecastLeadTime,
    ForecastSkill,
    RuleCurveCfg,
    Trace,
)
from external.loslr_regulation_optimization.functions.limits.Bv7 import planLimits

DATA_PATH = Path(__file__).resolve().parents[2] / "data"
POLICY_ID = 0
START_STEP = 48
N_STEPS = 2880


@pytest.fixture(scope="session")
def config():
    return (
        RuleCurveCfg()
        .training(
            data_path=DATA_PATH,
            forecast_lead_time=ForecastLeadTime.annual,
            forecast_skill=ForecastSkill.sqAR,
            trace=Trace.Y1961_2020,
            september_rule=False,
            id=POLICY_ID,
            drop_last_year=False,
            start_step=START_STEP,
        )
        .environment(
            horizon=N_STEPS,
            env=LoslrEnv,
            env_config={"stage_init": 74.55, "mechanism_id": None},
        )
        .agents(
            Ontario(
                id="ont",
                policy_id="ont",
                mechanisms=AdjustFlowConfig(
                    id="flow",
                    acts_on=("ont", "flow"),
                    action_space=Box(0.0, np.inf, shape=(1,), dtype=np.float64),
                    constraint_fn=planLimits,
                ),
            )
        )
    )


@pytest.fixture(scope="session")
def reference():
    return pd.read_pickle(DATA_PATH / "baseline" / "legacy_outSim.pkl")


@pytest.fixture(scope="session")
def reference_evaluated():
    return pd.read_pickle(DATA_PATH / "baseline" / "output.pkl")


@pytest.fixture(scope="session")
def reference_stage_costs(reference_evaluated):
    return np.column_stack(
        [
            reference_evaluated["upstreamCoastal"].to_numpy(),
            reference_evaluated["downstreamCoastal"].to_numpy(),
            -reference_evaluated["totalEnergyValue"].to_numpy(),
            reference_evaluated["totalRecBoating"].to_numpy(),
        ]
    ).astype(np.float64)


@pytest.fixture(scope="session")
def results(config):
    solver = config.build_optimizer()

    return solver.train()


@pytest.fixture(scope="session")
def refactored(results):
    return results["data"]


@pytest.fixture(scope="session")
def mdp(results):
    return results["mdp"]


@pytest.fixture(scope="session")
def stage_costs(mdp):
    return np.stack(mdp.rewards["ont"])
