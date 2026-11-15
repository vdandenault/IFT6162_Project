from pathlib import Path

import pandas as pd
import pytest

from bilevel_reservoirs.simulators.loslr import LoslrEnv
from bilevel_reservoirs.solvers.rule_curve import (
    ForecastLeadTime,
    ForecastSkill,
    RuleCurveCfg,
    Trace,
)

pytestmark = pytest.mark.integration

DATA_PATH = Path(__file__).resolve().parents[2] / "data"
POLICY_ID = 0
START_STEP = 48
N_STEPS = 2880

RULE_CURVE_COLUMNS = [
    "rfFlow",
    "rfRegime",
    "pprFlow",
    "rfOutput",
    "ontFlow",
    "flowRegime",
]

SIMULATED_COLUMNS = [
    # Lake Ontario
    "ontLevelBOQ",
    "ontLevelEOQ",
    "ontLevelMOQ",
    # St. Lawrence
    "stlouisFlow",
    "kingstonLevel",
    "alexbayLevel",
    "brockvilleLevel",
    "ogdensburgLevel",
    "cardinalLevel",
    "iroquoishwLevel",
    "iroquoistwLevel",
    "morrisburgLevel",
    "longsaultLevel",
    "saundershwLevel",
    "saunderstwLevel",
    "cornwallLevel",
    "summerstownLevel",
    "lerybeauharnoisLevel",
    "ptclaireLevel",
    "jetty1Level",
    "stlambertLevel",
    "varennesLevel",
    "sorelLevel",
    "lacstpierreLevel",
    "maskinongeLevel",
    "troisrivieresLevel",
    "batiscanLevel",
]


@pytest.fixture(scope="module")
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
            env_config={
                "stage_init": 74.55,
                "mechanism_id": None,
                "agents_cfg_dict": {},
            },
        )
    )


@pytest.fixture(scope="module")
def reference():
    return pd.read_pickle(DATA_PATH / "baseline" / "legacy_outSim.pkl")


@pytest.fixture(scope="module")
def results(config):
    solver = config.build_optimizer()
    return solver.train()


@pytest.fixture(scope="module")
def refactored(results):
    return results["data"][POLICY_ID]


@pytest.fixture(scope="module")
def mdp(results):
    return results["mdp"][POLICY_ID]


def test_rule_curve_horizon(refactored, mdp, reference):
    assert mdp.t == N_STEPS
    assert len(refactored) == N_STEPS
    assert len(reference) == N_STEPS


def test_rule_curve_policy(refactored, reference):
    pd.testing.assert_frame_equal(
        refactored[RULE_CURVE_COLUMNS].reset_index(drop=True),
        reference[RULE_CURVE_COLUMNS].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )


def test_rule_curve_dynamics(refactored, reference):
    pd.testing.assert_frame_equal(
        refactored[SIMULATED_COLUMNS].reset_index(drop=True),
        reference[SIMULATED_COLUMNS].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )
