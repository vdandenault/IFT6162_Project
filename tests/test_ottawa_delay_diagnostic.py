from copy import deepcopy
import json
import numpy as np
import pytest
from reservoir_control.systems.ottawa.delay_diagnostic import (
    explicit_storage_feasibility, boundary_state, witness_frame, bottleneck)
from reservoir_control.systems.ottawa.hard_follower import solve_hard_follower
from reservoir_control.systems.ottawa.hard_follower_study import CONFIG, make_problem, rollout
from reservoir_control.systems.ottawa.routing import TransitState


def setup():
    c=json.loads(CONFIG.read_text())
    return c,next(x for x in c['scenarios'] if x['id']=='one_period_delay')


def test_initial_feasible_explicit_storage_lp_and_transit():
    c,s=setup()
    p=make_problem(c,s,0,c['initial_storage_m3'],c['previous_executed_release_m3_s'],
                   TransitState(tuple(s['initial_transit_volumes_m3'])),s['realized_net_inflow_m3_s'])
    result=explicit_storage_feasibility(p)
    assert result['status']=='feasible'
    trace=witness_frame(c,s,result)
    assert abs(trace.system_balance_error_m3).max()<.01
    np.testing.assert_allclose(trace.downstream_m3_s,result['downstream_m3_s'],atol=1e-5)
    assert trace.transit_end_m3.iloc[-1]==pytest.approx(trace.release_m3_s.iloc[-1]*657000.)
    assert trace.transit_end_m3.iloc[-1]>0
    assert trace.storage_end_m3.min()>=c['storage_minimum_m3']-.01
    assert trace.storage_end_m3.max()<=c['storage_maximum_m3']+.01


def test_completion_lost_after_second_action_not_initially():
    c,s=setup();trace,plans,summary=rollout(c,s,'mpc','perfect',0)
    assert summary['status']=='infeasible' and len(trace)==3
    for k,expected in enumerate(['feasible','feasible','infeasible','infeasible']):
        storage,previous,queue=boundary_state(c,s,trace,k)
        p=make_problem(c,s,k,storage,previous,queue,s['realized_net_inflow_m3_s'][k:])
        assert explicit_storage_feasibility(p)['status']==expected
        certificate=bottleneck(c,s,k,storage)
        if k>=2:
            assert certificate['end_period']==6
            assert certificate['unavoidable_excess_m3']==pytest.approx(141065207.5426,abs=.01)
            assert certificate['unavoidable_storage_m3']>c['storage_maximum_m3']
        else: assert certificate is None
    storage,previous,queue=boundary_state(c,s,trace,3)
    p=make_problem(c,s,3,storage,previous,queue,s['realized_net_inflow_m3_s'][3:7])
    assert explicit_storage_feasibility(p)['status']=='infeasible'


@pytest.mark.parametrize('horizon',[5,6,8,12])
def test_longer_horizons_unchanged_physics_and_endpoint(horizon):
    c,s=setup(); original=deepcopy(c);c['horizon']=horizon
    trace,plans,summary=rollout(c,s,'mpc','perfect',0)
    assert summary['complete'] and len(trace)==12
    assert len(plans[-1]['planned_release_m3_s'])==1
    assert summary['final_transit_volume_m3']==pytest.approx(trace.executed_release_m3_s.iloc[-1]*657000)
    assert abs(trace.system_balance_error_m3).max()<.01
    assert trace.downstream_violation_m3_s.max()<1e-5
    c['horizon']=original['horizon'];assert c==original
    if horizon==12:
        initial=make_problem(c,s,0,c['initial_storage_m3'],c['previous_executed_release_m3_s'],
            TransitState(tuple(s['initial_transit_volumes_m3'])),s['realized_net_inflow_m3_s'])
        fit=solve_hard_follower(initial)
        np.testing.assert_allclose(fit.release_m3_s,trace.executed_release_m3_s,atol=.01,rtol=0)
