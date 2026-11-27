from copy import deepcopy

import numpy as np
import pytest

from reservoir_control.systems.ottawa.follower_problem import load_follower_config
from reservoir_control.systems.ottawa.forecasts import forecast_bank


def inputs():
    c = load_follower_config()
    case = next(x for x in c["scenarios"] if x["id"] == "one_period_delay")
    return c, case


def bank_for(c, s, mode="noisy", seed=11):
    return forecast_bank(
        s["realized_net_inflow_m3_s"],
        mode,
        seed,
        c["forecast_noise"]["initial_observed_inflow_estimate_m3_s"],
        c["forecast_noise"]["standard_deviation_m3_s"],
    )


def test_bank_is_causal_and_independent_of_unobserved_suffix():
    c, s = inputs()
    original = bank_for(c, s)
    # At decision k, input k has not been observed, including current-period input.
    for k in range(12):
        altered = deepcopy(s)
        altered["realized_net_inflow_m3_s"][k:] = [99999.0] * (12 - k)
        np.testing.assert_array_equal(bank_for(c, altered)[: k + 1], original[: k + 1])
    with pytest.raises(ValueError):
        original[0, 0] = 0


def test_seeded_forecasts_repeat_and_leave_unavailable_leads_empty():
    config, scenario = inputs()
    bank = bank_for(config, scenario)
    np.testing.assert_array_equal(bank_for(config, scenario), bank)
    assert bank_for(config, scenario, seed=22)[0, 0] != bank[0, 0]

    realized = scenario["realized_net_inflow_m3_s"]
    perfect = bank_for(config, scenario, mode="perfect")
    for step in range(len(realized)):
        remaining = len(realized) - step
        np.testing.assert_array_equal(perfect[step, :remaining], realized[step:])
        assert np.isnan(perfect[step, remaining:]).all()
        assert np.isnan(bank[step, remaining:]).all()
