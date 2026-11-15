from dataclasses import replace
import json
import numpy as np
import pytest
from reservoir_control.systems.ottawa.hard_follower import (
    HardFollowerProblem, FollowerWeights, solve_hard_follower, InvalidFollowerInput)
from reservoir_control.systems.ottawa.routing import TransitState
from reservoir_control.systems.ottawa.hard_follower_study import CONFIG, rollout, inflow_forecast


def problem(**kwargs):
    p = HardFollowerProblem(10., 1., [2.], [3.], [1.], [[1.]], [0.], [10.],
        [0.,0.], [20.,20.], [0.], [10.], [10.,10.], [0.], [1.], 1., 1.,
        FollowerWeights(1.,1.,1.,1.))
    return replace(p, **kwargs)


def test_analytic_one_period_and_hard_cap():
    r=solve_hard_follower(problem())
    assert r.success
    assert r.first_release_m3_s == pytest.approx(1.25)
    assert r.storage_m3[-1] == pytest.approx(10.75)
    assert r.objective == pytest.approx(2.75)
    capped=solve_hard_follower(problem(downstream_capacity_m3_s=[5.]))
    assert capped.success and capped.first_release_m3_s == pytest.approx(1.)
    assert capped.downstream_flow_m3_s[0] <= 5.+1e-5


def test_infeasible_no_action_and_initial_state():
    for p in [problem(downstream_capacity_m3_s=[3.5]), problem(initial_storage_m3=30.)]:
        r=solve_hard_follower(p)
        assert r.status=='infeasible' and r.first_release_m3_s is None
        assert r.storage_m3 is None


@pytest.mark.parametrize('changes', [dict(flow_scale_m3_s=0), dict(weights=FollowerWeights(release=0)),
    dict(duration_seconds=[-1]), dict(net_inflow_m3_s=[1,2]), dict(routing_matrix=[[2.]])])
def test_invalid(changes):
    with pytest.raises(InvalidFollowerInput): solve_hard_follower(problem(**changes))


def test_previous_executed_and_seconds():
    p=problem(previous_release_m3_s=8., weights=FollowerWeights(0.,1.,3.,0.))
    assert solve_hard_follower(p).first_release_m3_s == pytest.approx(6.)
    assert solve_hard_follower(replace(p,previous_release_m3_s=0.)).first_release_m3_s == pytest.approx(0., abs=1e-7)
    p=problem(initial_storage_m3=1e9, net_inflow_m3_s=[10.], duration_seconds=[657000.],
        storage_minimum_m3=[0.,0.],storage_maximum_m3=[2e9,2e9],
        storage_target_m3=[1e9,1e9], release_target_m3_s=[9.],
        downstream_capacity_m3_s=[100.], storage_scale_m3=1e9,weights=FollowerWeights(0.,1.,0.,0.))
    result=solve_hard_follower(p)
    assert result.storage_m3[-1]-1e9 == pytest.approx(657000.)
    assert result.residuals['mass_balance_error_m3'] < .01


def test_delayed_volume_and_horizon_queue():
    state=TransitState((6.,))
    a,c=state.forecast([2.,4.])
    np.testing.assert_allclose(a,[[0,0],[.5,0]])
    np.testing.assert_allclose(c,[3,0])
    final=state.after_plan([4.,5.],[2.,4.])
    assert final.total_m3==20.
    assert 6.+4*2+5*4-np.sum((a@np.array([4.,5.])+c)*[2,4])==final.total_m3
    assert TransitState((1.,2.,3.)).after_plan([4.],[2.]).volumes_m3 == (2.,3.,8.)
    p=problem(routing_matrix=[[0.]], in_transit_contribution_m3_s=[7.])
    assert solve_hard_follower(p).status=='infeasible'
    r=solve_hard_follower(replace(p,downstream_capacity_m3_s=[20.]))
    assert r.pending_planned_release_volume_m3==pytest.approx(r.first_release_m3_s)


def test_solver_failure_and_residual_gate(monkeypatch):
    import reservoir_control.systems.ottawa.hard_follower as module
    def fail(*args): raise RuntimeError('test failure')
    original=module.solve_response
    monkeypatch.setattr(module,'solve_response',fail)
    assert solve_hard_follower(problem()).status=='solver_failure'
    def bad(*args):
        fit=original(*args)
        return replace(fit,x=np.array([100.]))
    monkeypatch.setattr(module,'solve_response',bad)
    result=solve_hard_follower(problem())
    assert result.status=='solver_failure' and result.first_release_m3_s is None


def config(): return json.loads(CONFIG.read_text())


def test_rollout_noise_reproducible_and_causal():
    c=config();s=c['scenarios'][0]
    a,_,summary=rollout(c,s,'mpc','noisy',11)
    b,_,_=rollout(c,s,'mpc','noisy',11)
    np.testing.assert_array_equal(a.forecast_net_inflow_m3_s,b.forecast_net_inflow_m3_s)
    np.testing.assert_allclose(a.executed_release_m3_s,b.executed_release_m3_s)
    assert abs(summary['cumulative_system_balance_error_m3'])<.01
    x=inflow_forecast('noisy',[123.],900,3,np.random.default_rng(1),0)
    np.testing.assert_array_equal(x,[123]*3)
    assert (a.planned_first_release_m3_s==a.executed_release_m3_s).all()


def test_first_action_only_replans_from_realized_state():
    c=config(); case=c['scenarios'][0];calls=[]
    def spy(p,t):
        calls.append(p)
        return solve_hard_follower(p,t)
    trace,plans,summary=rollout(c,case,'mpc','noisy',11,solver=spy)
    for k in range(1,len(calls)):
        assert calls[k].initial_storage_m3==trace.storage_end_m3.iloc[k-1]
        assert calls[k].previous_release_m3_s==trace.executed_release_m3_s.iloc[k-1]
    for k,row in trace.iterrows():
        assert row.executed_release_m3_s==plans[k]['planned_release_m3_s'][0]
        assert row.storage_end_m3-row.storage_start_m3==pytest.approx(657000*(row.realized_net_inflow_m3_s-row.executed_release_m3_s),abs=1e-5)


def test_delayed_rollout_and_stopping_without_clipping():
    c=config();case=c['scenarios'][-1]
    trace,_,s=rollout(c,case,'mpc','perfect',0)
    assert s['complete'] and s['final_transit_volume_m3']>0
    assert np.max(abs(trace.system_balance_error_m3))<.01
    trace,_,s=rollout(c,c['scenarios'][1],'reference','perfect',0)
    assert s['status']=='physical_storage_violation'
    assert trace.storage_end_m3.iloc[-1]>c['storage_maximum_m3']
    trace,plans,s=rollout(c,c['scenarios'][3],'mpc','perfect',0)
    assert len(trace)==0 and plans[0]['action'] is None and s['status']=='infeasible'


def test_approximate_status_requires_physical_checks(monkeypatch):
    import reservoir_control.systems.ottawa.hard_follower as module
    original=module.solve_response
    def approximate(*args): return replace(original(*args),solver_success=False)
    monkeypatch.setattr(module,'solve_response',approximate)
    result=solve_hard_follower(problem())
    assert result.success and result.status=='optimal_kkt_checked'


def test_noncausal_routing_rejected():
    p=problem(net_inflow_m3_s=[2.,2.],ontario_executable_release_m3_s=[3.,3.],
        other_downstream_m3_s=[1.,1.],routing_matrix=[[0.,1.],[0.,0.]],
        in_transit_contribution_m3_s=[0.,0.],downstream_capacity_m3_s=[10.,10.],
        storage_minimum_m3=[0.,0.,0.],storage_maximum_m3=[20.,20.,20.],
        release_minimum_m3_s=[0.,0.],release_maximum_m3_s=[10.,10.],
        storage_target_m3=[10.,10.,10.],release_target_m3_s=[0.,0.],duration_seconds=[1.,1.])
    with pytest.raises(InvalidFollowerInput, match='causal'): solve_hard_follower(p)


def test_realized_forecast_departure_is_recorded_not_clipped():
    c=config();c['forecast_noise']['standard_deviation_m3_s']=0
    case=dict(c['scenarios'][0]);case['realized_net_inflow_m3_s']=[10000.]*12
    trace,_,summary=rollout(c,case,'mpc','noisy',11)
    assert summary['status']=='physical_storage_violation' and len(trace)==1
    assert trace.predicted_first_step_feasible.iloc[0]
    assert trace.storage_end_m3.iloc[0]>c['storage_maximum_m3']
    assert summary['predicted_feasible_realized_violation_steps']==1
