import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from reservoir_control.systems.lake_como.simulation import mapped_ppo_duration as duration
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo_study import train as reference_train
from reservoir_control.systems.lake_como.simulation.ppo_controlled import same_state
from reservoir_control.systems.lake_como.simulation.target_year_check import validation_pairs


def make_reference(tmp_path,seed,prefix):
    reference=tmp_path/'reference';folder=reference/f'seed{seed}'
    status=reference_train(folder,seed,prefix)
    assert status['complete']
    return reference,status


def test_partial_episode_prefix_and_live_continuation(tmp_path):
    reference,before=make_reference(tmp_path,9001,4096);digest=duration.sha256(reference/'seed9001/final.pt')
    folder=tmp_path/'continued'
    result=duration.train_seed(folder,9001,reference,prefix_steps=4096,total_steps=4160)
    assert result['complete'] and result['prefix_verified'] and result['steps']==4160
    assert result['unfinished_length']==before['unfinished_length']+64
    assert result['episodes_started']==len(before['episodes'])+1
    assert result['final_episode_elapsed_hours']==result['unfinished_length']*24
    prefix=json.loads((folder/'prefix_verification.json').read_text())
    assert prefix['live_state_and_rng_unchanged'] and prefix['episode_accounting_and_update_log_exact']
    assert duration.sha256(reference/'seed9001/final.pt')==digest
    original=torch.load(reference/'seed9001/final.pt',weights_only=True)
    captured=torch.load(folder/'prefix_4096.pt',weights_only=True)
    assert same_state(original,captured)
    assert torch.load(folder/'final.pt',weights_only=True)['updates']>captured['updates']


def test_prefix_exactly_at_terminal_defers_normal_reset_until_after_verification(tmp_path):
    reference,_=make_reference(tmp_path,777,730);folder=tmp_path/'continued'
    result=duration.train_seed(folder,777,reference,prefix_steps=730,total_steps=740)
    prefix=json.loads((folder/'prefix_verification.json').read_text())
    assert result['complete'] and prefix['current_episode_elapsed_hours']==730*24
    assert prefix['episodes_started']==1 and result['episodes_started']==2
    assert result['unfinished_length']==10 and result['final_episode_elapsed_hours']==240


def test_mismatching_prefix_stops_before_extension(tmp_path):
    reference,_=make_reference(tmp_path,777,128)
    path=reference/'seed777/final.pt';state=torch.load(path,weights_only=True)
    state['model']['actor.0.bias'][0]+=.1;torch.save(state,path)
    result=duration.train_seed(tmp_path/'continued',777,reference,prefix_steps=128,total_steps=256)
    assert not result['complete'] and not result['prefix_verified'] and result['steps']==128
    assert result['failure']['type']=='PrefixMismatchError'


def annual(value):
    return pd.DataFrame([dict(seed=seed,block_start=a,year=year,complete=True,**{m:value for m in duration.COMPARISON_METRICS})
        for seed in duration.SEEDS for a,b in validation_pairs() for year in [a,b]])


def test_required_cohort_scoring_and_duplicate_guards():
    before=annual(1.);after=annual(.8)
    statuses=[dict(seed=s,complete=True,prefix_verified=True,steps=400000) for s in duration.SEEDS]
    summary,paired=duration.comparison(before,after,statuses)
    assert summary['all_required_complete'] and len(paired)==50
    assert abs(summary['mean_cost_difference']+.2)<1e-12
    for broken in [after.iloc[:-1],pd.concat([after.iloc[:-1],after.iloc[[0]]],ignore_index=True)]:
        result,comparison=duration.comparison(before,broken,statuses)
        assert not result['all_required_complete'] and comparison.empty
        assert all(v is None for metrics in result['means'].values() for v in metrics.values())
    missing=after.copy();missing.loc[0,'annual_cost']=np.nan
    assert not duration.comparison(before,missing,statuses)[0]['all_required_complete']
    statuses[0]['prefix_verified']=False
    assert not duration.comparison(before,after,statuses)[0]['all_required_complete']


def test_prepare_freezes_plan_without_training(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Training must not run in prepare')
    monkeypatch.setattr(duration,'train_seed',forbidden)
    reference = tmp_path/'declared_reference'
    reference.mkdir()
    config = duration.asdict(duration.PPOConfig())
    plan = dict(config=config, steps_per_seed=duration.PREFIX_STEPS, seeds=list(duration.SEEDS),
                action_contract=duration.CONTRACT, source_sha256={}, quartiles=[])
    (reference/'plan.json').write_text(json.dumps(plan))
    annual(1.).assign(arm='treatment').to_csv(reference/'annual.csv', index=False)
    for seed in duration.SEEDS:
        folder = reference/f'seed{seed}'
        folder.mkdir()
        (folder/'training.json').write_text(json.dumps(dict(complete=True, steps=duration.PREFIX_STEPS)))
        torch.save(dict(seed=seed, steps=duration.PREFIX_STEPS, config=config), folder/'final.pt')
        (folder/'final.action.json').write_text(json.dumps(duration.CONTRACT))
        (folder/'updates.csv').write_text('step\n')
    root=tmp_path/'study'
    assert duration.main(['--stage','prepare','--reference',str(reference),'--root',str(root)])==0
    plan=json.loads((root/'plan.json').read_text())
    assert plan['prefix_steps']==200000 and plan['total_steps']==400000
    assert plan['seeds']==[123,124,125,126,127] and not plan['test_evaluated']
    assert not list(root.glob('seed*'))
    with pytest.raises(FileExistsError):duration.prepare(root)


def test_completed_seed_stage_does_not_retrain_and_interrupted_seed_is_refused(tmp_path,monkeypatch):
    root=tmp_path/'study';root.mkdir()
    for seed in duration.SEEDS:
        folder=root/f'seed{seed}';folder.mkdir();(folder/'training.json').write_text('{}')
    monkeypatch.setattr(duration,'training_status',lambda *args:dict(complete=True))
    assert duration.run_training(root,dict(reference=str(tmp_path/'ref')),workers=1)
    (root/'seed123/training.json').unlink()
    with pytest.raises(FileExistsError,match='Interrupted'):duration.run_training(root,dict(reference=str(tmp_path/'ref')),workers=1)


def test_invalid_worker_count_has_no_side_effects(tmp_path):
    root=tmp_path/'study'
    with pytest.raises(SystemExit) as error:duration.main(['--workers','0','--root',str(root)])
    assert error.value.code==2 and not root.exists()


def test_endpoint_evaluation_cache_and_frozen_checkpoint_guard(tmp_path,monkeypatch):
    root=tmp_path/'study';root.mkdir()
    for seed in duration.SEEDS:(root/f'seed{seed}').mkdir()
    calls=[]
    monkeypatch.setattr(duration,'training_status',lambda *args:dict(complete=True,final_checkpoint_sha256='frozen'))
    def evaluate(root,seed,plan):
        calls.append(seed)
        return annual(.8).query('seed==@seed').to_dict('records')
    monkeypatch.setattr(duration,'evaluate_endpoint',evaluate)
    plan=dict(reference=str(tmp_path/'ref'))
    assert duration.run_evaluation(root,plan) and len(calls)==5
    assert duration.run_evaluation(root,plan) and len(calls)==5
    (root/'seed123/evaluation_annual.csv').write_text('tampered')
    with pytest.raises(ValueError,match='Cached evaluation'):duration.run_evaluation(root,plan)


def test_seed_cohort_must_contain_all_five_distinct_learners():
    statuses=[dict(seed=123,complete=True,prefix_verified=True,steps=400000) for _ in range(5)]
    assert not duration.comparison(annual(1),annual(.8),statuses)[0]['all_required_complete']
