from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from reservoir_control.systems.ottawa.noisy_failure_diagnostic import (
    ARCHIVE, analyze_case, inherited_states, classify)


def load(scenario,seed):
    c=json.loads((ARCHIVE/'configuration.json').read_text())['original_configuration'];c['horizon']=5
    case=next(v for v in c['scenarios'] if v['id']==scenario)
    with np.load(ARCHIVE/'forecast_bank.npz',allow_pickle=False) as saved:
        bank=saved[f'{scenario}_noisy_{seed}'].copy()
    return c,case,bank


@pytest.mark.parametrize('scenario,seed,lost,truth',[
    ('tight_downstream',11,3,'infeasible'),('tight_downstream',22,3,'infeasible'),
    ('tight_downstream',33,3,'infeasible'),('one_period_delay',11,2,'infeasible'),
    ('one_period_delay',22,1,'infeasible'),('one_period_delay',33,2,'infeasible'),
    ('one_period_delay_feasible',22,None,'feasible')])
def test_archived_replay_independent_checks_and_witnesses(tmp_path,monkeypatch,scenario,seed,lost,truth):
    # If replay accidentally reverts to forecast generation, fail immediately.
    import reservoir_control.systems.ottawa.hard_follower_study as study
    def forbidden(*args,**kwargs): raise AssertionError('Forecast regeneration forbidden')
    monkeypatch.setattr(study,'inflow_forecast',forbidden)
    c,case,bank=load(scenario,seed);before=bank.copy();folder=tmp_path/'case'
    row,checks,abc,comparison=analyze_case(c,case,seed,ARCHIVE,bank,folder)
    np.testing.assert_array_equal(bank,before)
    assert row['initial_full_feasibility']=='feasible'
    assert row['A_noisy_same_window']=='infeasible'
    assert row['B_true_same_window']==row['C_true_full_remaining']==truth
    assert row['first_loss_action']==lost
    assert row['max_actual_balance_residual_m3']<.01
    assert not row['both_mechanisms_demonstrated']
    for k,result in enumerate(checks):
        if result['independent_lp']['status']=='feasible':
            w=pd.read_csv(folder/f'offline_true_witness_after_{k}_actions.csv')
            assert abs(w.system_balance_residual_m3).max()<.01
            np.testing.assert_allclose(w.storage_start_m3.iloc[0],result['state']['storage_m3'],atol=.01,rtol=0)
            expected_queue=case['routing_delay_periods']*w.release_m3_s.iloc[-1]*657000.
            assert w.transit_end_m3.iloc[-1]==pytest.approx(expected_queue)
        else:
            cert=result['volume_certificate'];assert cert and cert['unavoidable_excess_m3']>0
    if lost:
        assert comparison['action_period']==lost
        assert checks[lost-1]['independent_lp']['status']=='feasible'
        assert checks[lost]['independent_lp']['status']=='infeasible'
    else:
        assert comparison is None
        assert (folder/f'offline_true_witness_after_{row["executed_periods"]}_actions.csv').exists()


def test_corrupt_inherited_transit_detected():
    c,case,bank=load('one_period_delay',11)
    trace=pd.read_csv(ARCHIVE/'runs/one_period_delay_noisy_11_h5.csv')
    broken=trace.copy();broken.loc[0,'transit_queue_end_m3']='[0.0]'
    with pytest.raises(AssertionError): inherited_states(c,case,broken)


def test_changed_forecast_bank_detected(tmp_path):
    c,case,bank=load('tight_downstream',11);bank[0,0]=1e8
    with pytest.raises((AssertionError,RuntimeError)):
        analyze_case(c,case,11,ARCHIVE,bank,tmp_path/'changed')


def test_logical_inconsistency_not_forced_into_classification():
    with pytest.raises(RuntimeError): classify('infeasible','infeasible','feasible','feasible')
    assert classify('infeasible','feasible','infeasible','feasible')=='loss_of_recoverability_and_local_forecast_rejection'
    assert classify('infeasible','infeasible','infeasible','infeasible')=='initial_physical_infeasibility'
