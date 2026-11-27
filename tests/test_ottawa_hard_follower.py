from dataclasses import replace

import numpy as np
import pytest

from reservoir_control.systems.ottawa.follower_problem import load_follower_config, make_problem
from reservoir_control.systems.ottawa.hard_follower import (
    FollowerWeights,
    HardFollowerProblem,
    InvalidFollowerInput,
    solve_hard_follower,
)
from reservoir_control.systems.ottawa.routing import TransitState


def problem(**kwargs):
    p = HardFollowerProblem(
        10.0,
        1.0,
        [2.0],
        [3.0],
        [1.0],
        [[1.0]],
        [0.0],
        [10.0],
        [0.0, 0.0],
        [20.0, 20.0],
        [0.0],
        [10.0],
        [10.0, 10.0],
        [0.0],
        [1.0],
        1.0,
        1.0,
        FollowerWeights(1.0, 1.0, 1.0, 1.0),
    )
    return replace(p, **kwargs)


def test_analytic_one_period_and_hard_cap():
    r = solve_hard_follower(problem())
    assert r.success
    assert r.first_release_m3_s == pytest.approx(1.25)
    assert r.storage_m3[-1] == pytest.approx(10.75)
    assert r.objective == pytest.approx(2.75)
    capped = solve_hard_follower(problem(downstream_capacity_m3_s=[5.0]))
    assert capped.success and capped.first_release_m3_s == pytest.approx(1.0)
    assert capped.downstream_flow_m3_s[0] <= 5.0 + 1e-5


def test_infeasible_no_action_and_initial_state():
    for p in [problem(downstream_capacity_m3_s=[3.5]), problem(initial_storage_m3=30.0)]:
        r = solve_hard_follower(p)
        assert r.status == "infeasible" and r.first_release_m3_s is None
        assert r.storage_m3 is None


@pytest.mark.parametrize(
    "changes",
    [
        dict(flow_scale_m3_s=0),
        dict(weights=FollowerWeights(release=0)),
        dict(duration_seconds=[-1]),
        dict(net_inflow_m3_s=[1, 2]),
        dict(routing_matrix=[[2.0]]),
    ],
)
def test_invalid(changes):
    with pytest.raises(InvalidFollowerInput):
        solve_hard_follower(problem(**changes))


def test_previous_executed_and_seconds():
    p = problem(previous_release_m3_s=8.0, weights=FollowerWeights(0.0, 1.0, 3.0, 0.0))
    assert solve_hard_follower(p).first_release_m3_s == pytest.approx(6.0)
    assert solve_hard_follower(
        replace(p, previous_release_m3_s=0.0)
    ).first_release_m3_s == pytest.approx(0.0, abs=1e-7)
    p = problem(
        initial_storage_m3=1e9,
        net_inflow_m3_s=[10.0],
        duration_seconds=[657000.0],
        storage_minimum_m3=[0.0, 0.0],
        storage_maximum_m3=[2e9, 2e9],
        storage_target_m3=[1e9, 1e9],
        release_target_m3_s=[9.0],
        downstream_capacity_m3_s=[100.0],
        storage_scale_m3=1e9,
        weights=FollowerWeights(0.0, 1.0, 0.0, 0.0),
    )
    result = solve_hard_follower(p)
    assert result.storage_m3[-1] - 1e9 == pytest.approx(657000.0)
    assert result.residuals["mass_balance_error_m3"] < 0.01


def test_delayed_volume_and_horizon_queue():
    state = TransitState((6.0,))
    a, c = state.forecast([2.0, 4.0])
    np.testing.assert_allclose(a, [[0, 0], [0.5, 0]])
    np.testing.assert_allclose(c, [3, 0])
    final = state.after_plan([4.0, 5.0], [2.0, 4.0])
    assert final.total_m3 == 20.0
    assert 6.0 + 4 * 2 + 5 * 4 - np.sum((a @ np.array([4.0, 5.0]) + c) * [2, 4]) == final.total_m3
    assert TransitState((1.0, 2.0, 3.0)).after_plan([4.0], [2.0]).volumes_m3 == (2.0, 3.0, 8.0)
    p = problem(routing_matrix=[[0.0]], in_transit_contribution_m3_s=[7.0])
    assert solve_hard_follower(p).status == "infeasible"
    r = solve_hard_follower(replace(p, downstream_capacity_m3_s=[20.0]))
    assert r.pending_planned_release_volume_m3 == pytest.approx(r.first_release_m3_s)


def test_solver_failure_and_residual_gate(monkeypatch):
    import reservoir_control.systems.ottawa.hard_follower as module

    def fail(*args):
        raise RuntimeError("test failure")

    original = module.solve_response
    monkeypatch.setattr(module, "solve_response", fail)
    assert solve_hard_follower(problem()).status == "solver_failure"

    def bad(*args):
        fit = original(*args)
        return replace(fit, x=np.array([100.0]))

    monkeypatch.setattr(module, "solve_response", bad)
    result = solve_hard_follower(problem())
    assert result.status == "solver_failure" and result.first_release_m3_s is None


def test_approximate_status_requires_physical_checks(monkeypatch):
    import reservoir_control.systems.ottawa.hard_follower as module

    original = module.solve_response

    def approximate(*args):
        return replace(original(*args), solver_success=False)

    monkeypatch.setattr(module, "solve_response", approximate)
    result = solve_hard_follower(problem())
    assert result.success and result.status == "optimal_kkt_checked"


def test_noncausal_routing_rejected():
    p = problem(
        net_inflow_m3_s=[2.0, 2.0],
        ontario_executable_release_m3_s=[3.0, 3.0],
        other_downstream_m3_s=[1.0, 1.0],
        routing_matrix=[[0.0, 1.0], [0.0, 0.0]],
        in_transit_contribution_m3_s=[0.0, 0.0],
        downstream_capacity_m3_s=[10.0, 10.0],
        storage_minimum_m3=[0.0, 0.0, 0.0],
        storage_maximum_m3=[20.0, 20.0, 20.0],
        release_minimum_m3_s=[0.0, 0.0],
        release_maximum_m3_s=[10.0, 10.0],
        storage_target_m3=[10.0, 10.0, 10.0],
        release_target_m3_s=[0.0, 0.0],
        duration_seconds=[1.0, 1.0],
    )
    with pytest.raises(InvalidFollowerInput, match="causal"):
        solve_hard_follower(p)


def test_problem_builder_uses_current_state_and_slices_the_forecast_window():
    config = load_follower_config()
    scenario = next(case for case in config["scenarios"] if case["id"] == "one_period_delay")
    duration = config["clock"]["duration_seconds"]
    transit = TransitState((500 * duration,))
    built = make_problem(config, scenario, 2, 3e9, 700, transit, [800, 900])

    assert built.initial_storage_m3 == 3e9
    assert built.previous_release_m3_s == 700
    assert built.net_inflow_m3_s == [800, 900]
    assert (
        built.ontario_executable_release_m3_s
        == config["known_ontario_executable_schedule_m3_s"][2:4]
    )
    assert built.downstream_capacity_m3_s == scenario["downstream_capacity_m3_s"][2:4]
    np.testing.assert_array_equal(built.duration_seconds, [duration, duration])
    np.testing.assert_array_equal(built.in_transit_contribution_m3_s, [500, 0])
    np.testing.assert_array_equal(built.routing_matrix, [[0, 0], [1, 0]])
