from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import toml
from core.mechanism.base import MDPState

from external.loslr_regulation_optimization.functions.limits.Bv7 import (
    getPlanLimitsInputs,
    planLimits,
)
from external.loslr_regulation_optimization.functions.release.ruleCurve import (
    formatDecisionVariables,
    getReleaseFunctionInputs,
    releaseFunction,
)
from external.loslr_regulation_optimization.functions.routing.stlaw import (
    getStLawrenceRoutingInputs,
    stLawrenceRouting,
)
from external.loslr_regulation_optimization.optimizationSimulation import (
    simulation,
)
from src.bilevel_reservoirs.simulators.loslr import LoslrEnv

pytestmark = pytest.mark.integration

DATA_PATH = Path(__file__).resolve().parents[2] / "data"
POLICY_ID = 0
START_STEP = 48
N_STEPS = 2880


FORCING_COLUMNS = {
    # Lake Ontario
    "ontNTS",
    # St. Lawrence supplies
    "stlouisontOut",
    "desprairiesOut",
    "richelieuOut",
    "stfrancoisOut",
    "stmauriceOut",
    # Physical conditions
    "iceInd",
    "tidalInd",
    # Roughness coefficients
    "ogdensburgR",
    "cardinalR",
    "iroquoishwR",
    "saundershwR",
    "iroquoistwR",
    "morrisburgR",
    "longsaultR",
    "saunderstwR",
    "cornwallR",
    "summerstownR",
    "ptclaireR",
    "batiscanR",
    "threeriversR",
    "stpierreR",
    "sorelR",
    "varennesR",
}


ROUTED_COLUMNS = {
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
}


SIMULATED_COLUMNS = [
    "ontLevelBOQ",
    "ontLevelEOQ",
    "ontLevelMOQ",
    *sorted(ROUTED_COLUMNS),
]


@pytest.fixture(scope="module")
def config():
    return toml.load(DATA_PATH / "baseline" / "config.toml")


@pytest.fixture(scope="module")
def data(config):
    trace = config["experimentalDesign"]["trace"]
    optimized_input = config["experimentalDesign"]["inputFile"]

    data = pd.read_table(DATA_PATH / "input" / trace / f"{optimized_input}.txt")

    fcstNA = data["forNTS"].notna()
    fcstNA.iloc[:START_STEP] = True

    data = data.loc[fcstNA, :].reset_index(drop=True)
    data = data.iloc[: START_STEP + N_STEPS].copy()

    assert len(data) == START_STEP + N_STEPS

    return data


@pytest.fixture(scope="module")
def forcing(data):
    return data.iloc[START_STEP:].reset_index(drop=True).loc[:, sorted(FORCING_COLUMNS)]


@pytest.fixture(scope="module")
def legacy(config, data):
    decision_variables = config["decisionVariables"]
    release_fun_inputs = config["releaseFunction"]

    pars = formatDecisionVariables(
        pd.read_csv(
            DATA_PATH / "baseline" / "satisficingPolicies.csv",
            index_col="ID",
        )
        .loc[POLICY_ID, decision_variables["dvName"]]
        .tolist(),
        **release_fun_inputs,
    )

    with patch.object(
        pd.Series,
        "values",
        property(lambda self: self.to_numpy(copy=True)),
    ):
        legacy = simulation(
            data.copy(),
            release_fun_inputs,
            getReleaseFunctionInputs,
            releaseFunction,
            "off",
            getPlanLimitsInputs,
            planLimits,
            getStLawrenceRoutingInputs,
            stLawrenceRouting,
            pars,
        )

    return legacy.iloc[START_STEP:].reset_index(drop=True)


@pytest.fixture(scope="module")
def reference():
    return pd.read_pickle(DATA_PATH / "baseline" / "legacy_outSim.pkl")


@pytest.fixture
def env():
    return LoslrEnv(
        world=None,
        mechanism_id=None,
        agents_cfg_dict={},
        horizon=N_STEPS,
    )


def test_reference(legacy, reference):
    pd.testing.assert_frame_equal(
        legacy,
        reference,
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )


def test_loslr_one_step(env, forcing, legacy):
    mdp = env.reset(MDPState())

    assert mdp.t == 0

    np.testing.assert_equal(
        mdp.state["ontLevelBOQ"][0],
        legacy["ontLevelBOQ"].iloc[0],
    )

    mdp.params.update(forcing.iloc[0].to_dict())

    flow = legacy["ontFlow"].iloc[0]

    mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})

    mdp = env.simulate(mdp)

    assert mdp.t == 1

    np.testing.assert_equal(
        mdp.state["ontLevelBOQ"][1],
        legacy["ontLevelBOQ"].iloc[1],
        err_msg="Mismatch in next ontLevelBOQ",
    )

    for col in SIMULATED_COLUMNS[1:]:
        np.testing.assert_equal(
            mdp.state[col][1],
            legacy[col].iloc[0],
            err_msg=f"Mismatch in {col}",
        )


def test_loslr_numerical_stability(env, forcing, legacy):
    mdp = env.reset(MDPState())

    results = []

    for t in range(N_STEPS):
        flow = legacy["ontFlow"].iloc[t]

        row = {"ontLevelBOQ": mdp.state["ontLevelBOQ"][mdp.t]}

        mdp.params.update(forcing.iloc[t].to_dict())

        mdp.update(actions={"ont": {"flow": np.asarray([flow], dtype=np.float64)}})

        np.testing.assert_equal(
            mdp.state["ontLevelBOQ"][mdp.t],
            legacy["ontLevelBOQ"].iloc[t],
            err_msg=f"Mismatch in ontLevelBOQ at t={t}",
        )

        mdp = env.simulate(mdp)

        assert mdp.t == t + 1

        if t + 1 < N_STEPS:
            np.testing.assert_equal(
                mdp.state["ontLevelBOQ"][mdp.t],
                legacy["ontLevelBOQ"].iloc[t + 1],
                err_msg=f"Mismatch in next ontLevelBOQ at t={t}",
            )

        for col in SIMULATED_COLUMNS[1:]:
            value = mdp.state[col][mdp.t]

            assert np.isfinite(value), f"Non-finite value in {col} at t={t}: {value}"

            np.testing.assert_equal(
                value,
                legacy[col].iloc[t],
                err_msg=f"Mismatch in {col} at t={t}",
            )

            row[col] = value

        results.append(row)

    refactored = pd.DataFrame(
        results,
        columns=SIMULATED_COLUMNS,
    )

    assert len(refactored) == N_STEPS
    assert mdp.t == N_STEPS

    assert np.isfinite(refactored.to_numpy()).all()
    assert np.isfinite(mdp.state["ontLevelBOQ"][N_STEPS])

    pd.testing.assert_frame_equal(
        refactored,
        legacy[SIMULATED_COLUMNS].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )
