
from pathlib import Path

import pandas as pd
import pytest

from reservoir_control.study_config import REPO_ROOT
from reservoir_control.systems.loslr.data import read_trace, trace_path

from reservoir_control.experiments.fixed_schedule_replay import load_follower_config, repair_offline, replay

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
        trajectory, config, cases["one_period_delay_feasible"],
        previous_flow=595.0, previous_ice=2.0,
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
        trajectory, config, cases["deliberately_infeasible"],
        previous_flow=595.0, previous_ice=2.0,
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


def test_matched_planners_keep_shared_fifth_commitment(inputs):
    from reservoir_control.experiments.compare_ottawa_response import run_policy

    trajectory, config, cases = inputs
    scenario = cases["one_period_delay_feasible"]
    schedule, _ = repair_offline(
        trajectory, config, scenario, previous_flow=595.0, previous_ice=2.0,
    )
    trajectory["ontFlow"] = schedule["ontFlow"].to_numpy()
    audits = {}
    for policy in ("external", "anticipating"):
        audit = []
        records, plans = run_policy(trajectory, config, scenario, policy, audit, "compact")
        first = [entry for entry in audit if entry["period"] == 1]
        audits[policy] = [entry["schedule_m3_s"] for entry in first]
        assert all(entry["schedule_m3_s"][4] == 10 * trajectory["ontFlow"].iloc[4] for entry in first)
        assert all(entry["predicted_ottawa_arrival_m3_s"][0] == pytest.approx(900)
                   for entry in first if "predicted_ottawa_arrival_m3_s" in entry)
        chosen = next(entry for entry in first if entry["selected"])
        assert plans[0]["problem"]["ontario_executable_release_m3_s"] == chosen["schedule_m3_s"]
        assert records[0]["ontario_applied_m3_s"] == chosen["schedule_m3_s"][0]
    assert audits["external"] == audits["anticipating"]


def test_matched_response_ranking_uses_one_eligibility_gate(inputs):
    from reservoir_control.experiments.compare_ottawa_response import run_policy
    from reservoir_control.systems.ottawa.horizon_study import forecast_bank

    trajectory, config, cases = inputs
    scenario = cases["one_period_delay_feasible"]
    fixed, _ = repair_offline(
        trajectory, config, scenario, previous_flow=595.0, previous_ice=2.0,
    )
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    bank = forecast_bank(scenario["realized_net_inflow_m3_s"], "noisy", 107, 900., 700.)
    first = {}
    for policy in ("external_screened", "anticipating"):
        audit = []
        run_policy(
            trajectory, config, scenario, policy, audit, "limits",
            align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
            recovery_guard_periods=1, replan_guard="after_retry", delay_headroom_search=True,
            delay_headroom_on_exhaustion=True, matched_response_ranking=True,
        )
        first[policy] = [row for row in audit if row["period"] == 1]
    external = first["external_screened"]
    nested = first["anticipating"]
    assert [(row["schedule_m3_s"], row["status"]) for row in external] == [
        (row["schedule_m3_s"], row["status"]) for row in nested
    ]
    eligible = [row for row in nested if "best_response_score" in row]
    assert len(eligible) > 1
    assert all(row["native_status_with_ottawa_plan"] == "admissible" for row in eligible)
    assert next(row for row in external if row["selected"])["schedule_m3_s"] == min(
        eligible, key=lambda row: (row["score"], row["schedule_m3_s"])
    )["schedule_m3_s"]
    assert next(row for row in nested if row["selected"]) == min(
        eligible, key=lambda row: (row["best_response_score"], row["schedule_m3_s"])
    )


def test_candidate_proposals_match_budgets_and_free_capacity_at_the_arrival():
    import numpy as np

    from reservoir_control.experiments.compare_ottawa_response import nearby_schedules

    anchor = (6000., 6000., 6000., 6000., 6700.)
    attempted = {anchor, (6050., *anchor[1:])}
    generic = nearby_schedules(anchor, attempted, 12)
    guided = nearby_schedules(
        anchor, attempted, 12,
        headroom=np.array([1000., 0., 1000., 1000., 0.]),
        pressure=np.array([0., 400., 0., 0., 0.]),
    )
    for proposals in (generic, guided):
        assert len(set(proposals)) == 12
        assert not attempted.intersection(proposals)
        assert all(schedule[4] == anchor[4] for schedule in proposals)
    assert guided != generic
    assert all(schedule[1] <= 5600. for schedule in guided)


def test_ottawa_recession_forecast_preserves_lead_one_and_uses_prior_observations():
    import numpy as np

    from reservoir_control.systems.ottawa.horizon_study import forecast_bank, recession_bank

    observed = np.array([900., 1050., 2350., 1650., 1050., 900., 900., 900.])
    origin = forecast_bank(observed, "noisy", 119, 900., 700.)
    revised = recession_bank(observed, origin)
    np.testing.assert_array_equal(revised[:, 0], origin[:, 0])
    np.testing.assert_array_equal(np.isnan(revised), np.isnan(origin))
    np.testing.assert_allclose(origin[3, :5] - revised[3, :5],
                               1375. * (1. - np.exp2(-np.arange(5))))

    changed = observed.copy()
    changed[3:] += 9000.
    changed_origin = forecast_bank(changed, "noisy", 119, 900., 700.)
    np.testing.assert_array_equal(recession_bank(changed, changed_origin)[:4], revised[:4])


def test_isolated_recession_cannot_distinguish_a_future_plateau_after_the_jump():
    import numpy as np

    from reservoir_control.systems.ottawa.horizon_study import forecast_bank, recession_bank

    pulse = np.array([900., 1050., 2350., 1650., 1050., 900., 900.])
    plateau = np.array([900., 1050., 2350., 2350., 2350., 2150., 1800.])
    pulse_origin = forecast_bank(pulse, "noisy", 129, 900., 700.)
    plateau_origin = forecast_bank(plateau, "noisy", 129, 900., 700.)
    pulse_forecast = recession_bank(pulse, pulse_origin, jump_minimum_m3_s=700.)
    plateau_forecast = recession_bank(plateau, plateau_origin, jump_minimum_m3_s=700.)

    np.testing.assert_array_equal(pulse_forecast[:4], plateau_forecast[:4])
    np.testing.assert_array_equal(plateau_forecast[:, 0], plateau_origin[:, 0])
    assert plateau_forecast[3, 1] < plateau_origin[3, 1]
    np.testing.assert_array_equal(plateau_forecast[4:], plateau_origin[4:])
    np.testing.assert_array_equal(pulse_forecast[4:], pulse_origin[4:])


def test_native_limit_candidates_extend_external_replay(inputs):
    from reservoir_control.experiments.compare_ottawa_response import run_policy

    trajectory, config, cases = inputs
    scenario = cases["one_period_delay_feasible"]
    fixed, _ = repair_offline(trajectory, config, scenario, previous_flow=595.0, previous_ice=2.0)
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    external_audit, anticipating_audit = [], []
    external, _ = run_policy(trajectory, config, scenario, "external", external_audit, "limits")
    anticipating, _ = run_policy(trajectory, config, scenario, "anticipating", anticipating_audit, "limits")
    assert len(external) == 12 and all(row["status"] == "executed" for row in external)
    assert all(row["ontario_applied_m3_s"] == row["ontario_committed_m3_s"] for row in external)
    assert anticipating[-1]["period"] == 3
    assert anticipating[-1]["status"] == "no_admissible_candidate"
    assert all(entry["status"].startswith("native_F") for entry in anticipating_audit if entry["period"] == 3)
    for entry in external_audit + anticipating_audit:
        plan = entry["schedule_m3_s"]
        if len(plan) == 5:
            assert plan[4] == 10 * fixed["ontFlow"].iloc[entry["period"] + 3]


def test_forecast_f_ceiling_brackets_native_limit(inputs):
    from gymnasium.spaces import Box

    from reservoir_control.systems.loslr.operator import AdjustFlow
    from reservoir_control.experiments.compare_ottawa_response import (
        forecast_f_ceiling, forecast_forcing, native_step,
    )

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


def test_f_alignment_clears_the_original_anticipating_stop(inputs):
    from reservoir_control.experiments.compare_ottawa_response import run_policy

    trajectory, config, cases = inputs
    scenario = cases["one_period_delay_feasible"]
    fixed, _ = repair_offline(trajectory, config, scenario, previous_flow=595.0, previous_ice=2.0)
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    baseline, _ = run_policy(trajectory, config, scenario, "anticipating", [], "limits")
    aligned, plans = run_policy(
        trajectory, config, scenario, "anticipating", [], "limits", align_f=True, align_from=3,
    )
    assert baseline[:2] == aligned[:2]
    assert len(aligned) == 12 and all(row["status"] == "executed" for row in aligned)
    assert plans[2]["problem"]["downstream_capacity_m3_s"][4] == scenario["downstream_capacity_m3_s"][6]
    assert plans[2]["problem"]["ontario_executable_release_m3_s"][4] == 10 * fixed["ontFlow"].iloc[6]
    assert aligned[2]["st_louis_flow_m3_s"] <= aligned[2]["follower_capacity_m3_s"] + 1e-5


def test_external_screen_keeps_ranking_and_skips_infeasible_schedules(inputs):
    from reservoir_control.experiments.compare_ottawa_response import run_policy

    trajectory, config, cases = inputs
    scenario = cases["one_period_delay_feasible"]
    fixed, _ = repair_offline(trajectory, config, scenario, previous_flow=595.0, previous_ice=2.0)
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    baseline_audit, screened_audit = [], []
    baseline, _ = run_policy(
        trajectory, config, scenario, "external", baseline_audit, "limits", align_f=True,
    )
    screened, plans = run_policy(
        trajectory, config, scenario, "external_screened", screened_audit, "limits", align_f=True,
    )
    assert baseline[-1]["period"] == 5 and baseline[-1]["ottawa_status"] == "infeasible"
    assert screened[:4] == baseline[:4]
    assert len(screened) == 12 and all(row["status"] == "executed" for row in screened)

    baseline_scores = {
        row["schedule_m3_s"]: row.get("score") for row in baseline_audit if row["period"] == 5
    }
    choices = [row for row in screened_audit if row["period"] == 5]
    assert {row["schedule_m3_s"]: row.get("score") for row in choices} == baseline_scores
    ranked = sorted(
        (row for row in choices if row["status"] == "admissible"),
        key=lambda row: (row["score"], row["schedule_m3_s"]),
    )
    assert [row.get("ottawa_candidate_status") for row in ranked[:3]] == ["infeasible"] * 3
    chosen = ranked[3]
    assert chosen["selected"] and chosen["selected_rank"] == 4
    assert all("ottawa_screen_rank" not in row for row in ranked[4:])
    assert plans[4]["problem"]["ontario_executable_release_m3_s"] == chosen["schedule_m3_s"]
    assert screened[4]["ontario_applied_m3_s"] == chosen["schedule_m3_s"][0]


def test_noisy_policy_uses_bank_and_updates_storage_with_realized_inflow(inputs):
    import numpy as np

    from reservoir_control.experiments.compare_ottawa_response import run_policy
    from reservoir_control.systems.ottawa.horizon_study import forecast_bank

    trajectory, config, cases = inputs
    reference = cases["one_period_delay_feasible"]
    fixed, _ = repair_offline(trajectory, config, reference, previous_flow=595.0, previous_ice=2.0)
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    scenario = cases["one_period_delay"]
    noise = config["forecast_noise"]
    bank = forecast_bank(
        scenario["realized_net_inflow_m3_s"], "noisy", 11,
        noise["initial_observed_inflow_estimate_m3_s"], noise["standard_deviation_m3_s"],
    )
    for policy in ("external_screened", "anticipating"):
        trace, plans = run_policy(
            trajectory, config, scenario, policy, [], "limits", align_f=True,
            ottawa_forecast_bank=bank,
        )
        assert trace[-1]["status"] == "no_admissible_candidate" and trace[-1]["period"] == 7
        for k, (row, plan) in enumerate(zip(trace[:-1], plans, strict=True)):
            np.testing.assert_array_equal(plan["problem"]["net_inflow_m3_s"], bank[k, :5])
            assert row["ottawa_end_storage_m3"] - row["ottawa_predicted_storage_m3"] == pytest.approx(
                config["clock"]["duration_seconds"] * (row["ottawa_net_inflow_m3_s"] - bank[k, 0]),
                abs=0.01,
            )


@pytest.mark.parametrize("scenario_name,last_period,violates", [
    ("one_period_delay_feasible", 12, False),
    ("one_period_delay", 6, True),
])
def test_recovery_uses_past_inflow_and_checks_realized_storage(
    inputs, scenario_name, last_period, violates,
):
    import numpy as np

    from reservoir_control.experiments.compare_ottawa_response import run_policy
    from reservoir_control.systems.ottawa.horizon_study import forecast_bank

    trajectory, config, cases = inputs
    fixed, _ = repair_offline(
        trajectory, config, cases["one_period_delay_feasible"],
        previous_flow=595.0, previous_ice=2.0,
    )
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    scenario = cases[scenario_name]
    noise = config["forecast_noise"]
    inputs_m3_s = scenario["realized_net_inflow_m3_s"]
    bank = forecast_bank(
        inputs_m3_s, "noisy", 22, noise["initial_observed_inflow_estimate_m3_s"],
        noise["standard_deviation_m3_s"],
    )
    audit = []
    trace, plans = run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits",
        align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
    )
    selected = {row["period"]: row["recovery_stage"] for row in audit if row["selected"]}
    assert selected[5] == "persistence_retry"
    assert {row["recovery_stage"] for row in audit if row["period"] == 5} == {
        "primary", "expanded", "persistence_retry",
    }
    for index, (row, plan) in enumerate(zip(trace, plans, strict=True)):
        actual_forecast = plan["problem"]["net_inflow_m3_s"]
        original = bank[index, :len(actual_forecast)]
        expected = ([inputs_m3_s[index - 1]] * len(original)
                    if selected[index + 1] == "persistence_retry" else original)
        np.testing.assert_array_equal(actual_forecast, expected)
        assert row["ottawa_original_forecast_inflow_m3_s"] == original[0]
    assert trace[-1]["period"] == last_period
    assert (trace[-1]["ottawa_storage_violation_m3"] > 0) == violates
    if violates:
        assert selected[last_period] == "primary"
        assert trace[-1]["reason"] == "Realized Ottawa storage left its bounds"


@pytest.mark.parametrize("guard_periods", [1, 2])
def test_guard_checks_excess_inflow_on_retry_only(inputs, guard_periods):
    import numpy as np

    from reservoir_control.experiments.compare_ottawa_response import run_policy
    from reservoir_control.systems.ottawa.horizon_study import forecast_bank

    trajectory, config, cases = inputs
    fixed, _ = repair_offline(
        trajectory, config, cases["one_period_delay_feasible"],
        previous_flow=595.0, previous_ice=2.0,
    )
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    scenario = cases["one_period_delay"]
    noise = config["forecast_noise"]
    bank = forecast_bank(
        scenario["realized_net_inflow_m3_s"], "noisy", 22,
        noise["initial_observed_inflow_estimate_m3_s"], noise["standard_deviation_m3_s"],
    )
    audit = []
    trace, plans = run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits", align_f=True,
        ottawa_forecast_bank=bank, recovery="persistence", recovery_guard_periods=guard_periods,
    )
    assert len(trace) == 12 and all(row["status"] == "executed" for row in trace)
    assert max(row["ottawa_storage_violation_m3"] for row in trace) == 0
    selected = next(row for row in audit if row["period"] == 5 and row["selected"])
    assert selected["recovery_stage"] == "persistence_retry"
    maximum = config["storage_maximum_m3"]
    excess = noise["standard_deviation_m3_s"] * config["clock"]["duration_seconds"]
    expected = maximum - excess * np.minimum(np.arange(6), guard_periods)
    np.testing.assert_array_equal(plans[4]["problem"]["storage_maximum_m3"], expected)
    assert plans[5]["problem"]["storage_maximum_m3"].tolist() == [maximum] * 6
    assert trace[4]["ottawa_predicted_storage_m3"] + excess <= maximum + 0.01


def test_after_retry_guard_covers_the_next_ordinary_decision(inputs):
    from reservoir_control.experiments.compare_ottawa_response import run_policy
    from reservoir_control.systems.ottawa.horizon_study import forecast_bank

    trajectory, config, cases = inputs
    fixed, _ = repair_offline(
        trajectory, config, cases["one_period_delay_feasible"],
        previous_flow=595.0, previous_ice=2.0,
    )
    trajectory["ontFlow"] = fixed["ontFlow"].to_numpy()
    scenario = cases["one_period_delay"]
    noise = config["forecast_noise"]
    bank = forecast_bank(
        scenario["realized_net_inflow_m3_s"], "noisy", 22,
        noise["initial_observed_inflow_estimate_m3_s"], noise["standard_deviation_m3_s"],
    )
    audit = []
    run_policy(
        trajectory, config, scenario, "external_screened", audit, "limits",
        align_f=True, ottawa_forecast_bank=bank, recovery="persistence",
        recovery_guard_periods=1, replan_guard="after_retry",
    )
    retries = {row["period"] for row in audit if row["selected"] and row["recovery_stage"] == "persistence_retry"}
    assert 5 in retries
    ordinary = [row for row in audit if row["recovery_stage"] != "persistence_retry"]
    assert any(row["period"] == 6 for row in ordinary)
    for row in ordinary:
        guarded = "recovery_storage_maximum_m3" in row
        assert guarded == (row["period"] - 1 in retries)
        if guarded:
            maximum = config["storage_maximum_m3"]
            assert row["recovery_storage_maximum_m3"][0] == maximum
            assert row["recovery_storage_maximum_m3"][1] == maximum - 700 * 657000
