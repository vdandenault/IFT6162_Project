import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration
N_STEPS = 2880
RULE_CURVE_COLUMNS = ["ontFlow", "flowRegime"]
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


def test_rule_curve_running_cost(results, reference_stage_costs):
    np.testing.assert_array_equal(
        results["running_cost"], reference_stage_costs.sum(axis=0), strict=True
    )
