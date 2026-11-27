from copy import deepcopy
import numpy as np
import pytest
from reservoir_control.systems.ottawa.forecasts import forecast_bank
from reservoir_control.systems.ottawa.hard_follower_study import load_follower_config, rollout


HORIZONS = (4, 5, 6, 8, 12)


def inputs():
    c=load_follower_config();case=next(x for x in c['scenarios'] if x['id']=='one_period_delay')
    return c,case


def bank_for(c,s,mode='noisy',seed=11):
    return forecast_bank(s['realized_net_inflow_m3_s'],mode,seed,
        c['forecast_noise']['initial_observed_inflow_estimate_m3_s'],c['forecast_noise']['standard_deviation_m3_s'])


def test_overlaps_match_every_time_and_horizon():
    c,s=inputs();bank=bank_for(c,s)
    for k in range(12):
        for a in HORIZONS:
            for b in HORIZONS:
                overlap=min(a,b,12-k)
                np.testing.assert_array_equal(bank[k,:min(a,12-k)][:overlap],bank[k,:min(b,12-k)][:overlap])
    for h in HORIZONS:
        changed=deepcopy(c);changed['horizon']=h
        _,plans,_=rollout(changed,s,'mpc','noisy',11,forecast_bank=bank)
        for p in plans:
            actual=p['supplied_problem']['net_inflow_m3_s']
            np.testing.assert_array_equal(actual,bank[p['step'],:len(actual)])


def test_bank_is_causal_and_independent_of_unobserved_suffix():
    c,s=inputs();original=bank_for(c,s)
    # At decision k, input k has not been observed, including current-period input.
    for k in range(12):
        altered=deepcopy(s);altered['realized_net_inflow_m3_s'][k:]=[99999.]*(12-k)
        np.testing.assert_array_equal(bank_for(c,altered)[:k+1],original[:k+1])
    with pytest.raises(ValueError): original[0,0]=0


def test_early_stop_and_run_order_cannot_change_bank():
    c,s=inputs();bank=bank_for(c,s);before=bank.copy()
    for h in [12,4,8,5,6]:
        changed=deepcopy(c);changed['horizon']=h
        rollout(changed,s,'mpc','noisy',11,forecast_bank=bank)
    np.testing.assert_array_equal(bank,before)
    np.testing.assert_array_equal(bank_for(c,s),before)


@pytest.mark.parametrize('horizon',HORIZONS)
def test_perfect_delay_reproduces_with_unchanged_endpoint(horizon):
    c,s=inputs();bank=bank_for(c,s,'perfect',0);c['horizon']=horizon
    trace,plans,summary=rollout(c,s,'mpc','perfect',0,forecast_bank=bank)
    old,_,original=rollout(c,s,'mpc','perfect',0)
    np.testing.assert_allclose(trace.executed_release_m3_s,old.executed_release_m3_s,atol=1e-8)
    assert summary['complete']==(horizon!=4)
    assert summary['status']==('infeasible' if horizon==4 else 'completed')
    assert summary['final_transit_volume_m3']==pytest.approx(trace.executed_release_m3_s.iloc[-1]*657000.)
    assert abs(trace.system_balance_error_m3).max()<.01
    assert trace.downstream_violation_m3_s.max()<1e-5
    assert trace.storage_violation_m3.max()<.01
    if horizon!=4: assert len(plans[-1]['planned_release_m3_s'])==1
