from pathlib import Path

import pandas as pd
import pytest

from reservoir_control.experiments.fixed_schedule_replay import repair_offline, replay
from reservoir_control.study_config import REPO_ROOT, load_follower_config
from reservoir_control.systems.loslr.data import read_trace, trace_path

pytestmark = pytest.mark.integration


@pytest.fixture
def inputs():
    config = load_follower_config()
    path = trace_path(REPO_ROOT / "data", Path("historic/1961_2020/12month_sqAR"))
    trajectory = read_trace(path).iloc[48:60].copy()
    trajectory["ontLevelBOQ"] = 74.55
    trajectory["ontFlow"] = [623, 693, 710, 687, 672, 666, 653, 667, 737, 769, 795, 821]
    return trajectory, config, {case["id"]: case for case in config["scenarios"]}


def test_rejects_f_limit_change_without_executing_ottawa(inputs):
    trajectory, config, cases = inputs
    records, plans = replay(
        trajectory,
        config,
        cases["one_period_delay_feasible"],
        previous_flow=595.0,
        previous_ice=2.0,
    )
    executed, rejected = records
    assert executed["status"] == "executed"
    assert executed["ontario_applied_m3_s"] == executed["ontario_committed_m3_s"] == 6230
    assert executed["st_louis_flow_m3_s"] == pytest.approx(
        6230 + executed["background_m3_s"] + executed["ottawa_arrival_m3_s"]
    )
    assert abs(executed["ottawa_system_balance_residual_m3"]) < 0.01
    assert plans[0]["problem"]["ontario_executable_release_m3_s"] == [6230, 6930, 7100, 6870, 6720]
    assert rejected["status"] == "commitment_rejected"
    assert rejected["ontario_limit_regime"] == "F"
    assert rejected["ontario_allowed_m3_s"] == 6220
    assert rejected["ottawa_start_queue_m3"] == executed["ottawa_end_queue_m3"]
    assert "ottawa_executed_release_m3_s" not in rejected
    assert "ontario_applied_m3_s" not in rejected


def test_infeasible_follower_stops_before_either_action(inputs):
    trajectory, config, cases = inputs
    records, plans = replay(
        trajectory, config, cases["deliberately_infeasible"], previous_flow=595.0, previous_ice=2.0
    )
    (record,) = records
    assert record["status"] == "ottawa_solve_failed"
    assert plans[0]["status"] == "infeasible"
    assert "ontario_applied_m3_s" not in record
    assert "ottawa_executed_release_m3_s" not in record


def test_offline_repair_replays_as_one_fixed_schedule(inputs):
    trajectory, config, cases = inputs
    original = trajectory["ontFlow"].copy()
    history = {"previous_flow": 595.0, "previous_ice": 2.0}
    scenario = cases["one_period_delay_feasible"]
    schedule, edits = repair_offline(trajectory, config, scenario, **history)
    pd.testing.assert_series_equal(trajectory["ontFlow"], original)
    assert edits[0]["period"] == 2
    records, _ = replay(schedule, config, scenario, **history)
    assert len(records) == 12
    assert all(row["status"] == "executed" for row in records)
    assert [row["lookahead"] for row in records] == [5] * 8 + [4, 3, 2, 1]
    assert all(row["ontario_applied_m3_s"] == row["ontario_committed_m3_s"] for row in records)
    assert all(row["st_louis_flow_m3_s"] <= row["capacity_m3_s"] + 1e-5 for row in records)
    assert records[-1]["ottawa_end_transit_m3"] > 0


def test_ottawa_recession_forecast_preserves_lead_one_and_uses_prior_observations():
    import numpy as np

    from reservoir_control.systems.ottawa.forecasts import forecast_bank, recession_bank

    observed = np.array([900.0, 1050.0, 2350.0, 1650.0, 1050.0, 900.0, 900.0, 900.0])
    origin = forecast_bank(observed, "noisy", 119, 900.0, 700.0)
    revised = recession_bank(observed, origin)
    np.testing.assert_array_equal(revised[:, 0], origin[:, 0])
    np.testing.assert_array_equal(np.isnan(revised), np.isnan(origin))
    np.testing.assert_allclose(
        origin[3, :5] - revised[3, :5], 1375.0 * (1.0 - np.exp2(-np.arange(5)))
    )

    changed = observed.copy()
    changed[3:] += 9000.0
    changed_origin = forecast_bank(changed, "noisy", 119, 900.0, 700.0)
    np.testing.assert_array_equal(recession_bank(changed, changed_origin)[:4], revised[:4])


def test_isolated_recession_cannot_distinguish_a_future_plateau_after_the_jump():
    import numpy as np

    from reservoir_control.systems.ottawa.forecasts import forecast_bank, recession_bank

    pulse = np.array([900.0, 1050.0, 2350.0, 1650.0, 1050.0, 900.0, 900.0])
    plateau = np.array([900.0, 1050.0, 2350.0, 2350.0, 2350.0, 2150.0, 1800.0])
    pulse_origin = forecast_bank(pulse, "noisy", 129, 900.0, 700.0)
    plateau_origin = forecast_bank(plateau, "noisy", 129, 900.0, 700.0)
    pulse_forecast = recession_bank(pulse, pulse_origin, jump_minimum_m3_s=700.0)
    plateau_forecast = recession_bank(plateau, plateau_origin, jump_minimum_m3_s=700.0)

    np.testing.assert_array_equal(pulse_forecast[:4], plateau_forecast[:4])
    np.testing.assert_array_equal(plateau_forecast[:, 0], plateau_origin[:, 0])
    assert plateau_forecast[3, 1] < plateau_origin[3, 1]
    np.testing.assert_array_equal(plateau_forecast[4:], plateau_origin[4:])
    np.testing.assert_array_equal(pulse_forecast[4:], pulse_origin[4:])


def test_forecast_f_ceiling_brackets_native_limit(inputs):
    from gymnasium.spaces import Box

    from reservoir_control.experiments.coupled_forecast import (
        forecast_f_ceiling,
        forecast_forcing,
        native_step,
    )
    from reservoir_control.systems.loslr.operator import AdjustFlow

    trajectory, _, _ = inputs
    rows = trajectory.iloc[2:].to_dict("records")
    state = (74.444, 612.0, 1.0)
    commitment = 5800.0
    ceiling = forecast_f_ceiling(state[0], rows[0]["ptclaireR"], commitment)
    limits = AdjustFlow(acts_on=("ont", "flow"), action_space=Box(0.0, 1200.0, (1,)))
    for delta in (0.0, 1.0):
        forcing = forecast_forcing(rows, 0, ceiling + delta - commitment)
        _, allowed, regime = native_step(forcing, forcing["ontNTS_QM1"], 580.0, state, limits)
        if delta == 0.0:
            assert allowed == 580.0
        else:
            assert regime == "F" and allowed < 580.0
