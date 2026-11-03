"""Fixed-actor target ablation: identical states/batches, bootstrap vs rollout targets.

uv run --group reproduction python -m lake_como.sac_critic_targets
Fresh rollout targets come only from fitting training years. Checking returns are
reused for evaluation, never fitting. No actor or original controller is changed.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from lake_como.environment import make_env
from lake_como.inflow_observation import PastInflowObservation
from lake_como.mpc_hindsight import digest
from lake_como.paths import PROJECT_ROOT
from lake_como.sac import SACAgent,checked_optimizer_step
from lake_como.sac_critic_fit import (assert_frozen,freeze_noncritics,frozen_state,score_pairs,training_partitions)
from lake_como.sac_critic_sampling import point_predictions,broad_score
from lake_como.sac_critic_reference import reference_errors
from lake_como.sac_diagnostics import rollout_branch,critic_values
from lake_como.sac_study import verify_plan

STUDY=PROJECT_ROOT/'results/sac/baseline_200k_v1'
POOL=PROJECT_ROOT/'results/sac/critic_sampling_v1'
PAIRS=PROJECT_ROOT/'results/sac/critic_fit_v1'
REFERENCES=PROJECT_ROOT/'results/sac/critic_reference_v1'


@dataclass(frozen=True)
class TargetConfig:
    anchor_days: tuple[int,...] = (45,135,225,315)
    rollout_replicates: int = 8
    budgets: tuple[int,...] = (100,1000)
    batch_size: int = 128
    learning_rate: float = .0003


def check_fit_scope(years):
    fitting,checking=training_partitions()
    if not len(years) or not np.isin(years,fitting).all() or np.isin(years,checking).any():
        raise ValueError('Only the declared fitting training years may enter target fitting')


def target_batches(years,config,seed):
    check_fit_scope(years)
    return np.random.default_rng(81000+seed).integers(len(years),size=(max(config.budgets),config.batch_size))


def assert_identical_non_targets(left,right):
    for name in ['observations','requests','years','days']:
        np.testing.assert_array_equal(left[name],right[name])


def as_point_data(records):
    return dict(observations=np.asarray([row['observation'] for row in records],np.float32),
                requests=np.asarray([row['request_m3_s'] for row in records],np.float32),
                years=np.asarray([row['year'] for row in records],np.int32),
                days=np.asarray([row['day'] for row in records],np.int32))


def collect_fit_targets(base,seed,plan,config,source,bootstrap):
    fitting,_=training_partitions()
    lookup={(int(y),int(d)):i for i,(y,d) in enumerate(zip(source['years'][:15330],source['days'][:15330]))}
    records=[];replicates=[];steps=0
    for year in fitting:
        env=PastInflowObservation(make_env('train',year),plan['fallback'])
        obs,_=env.reset()
        try:
            for day in range(1,max(config.anchor_days)+1):
                request=float(base.physical_action(torch.from_numpy(base.act(obs,deterministic=True)))[0])
                if day in config.anchor_days:
                    index=lookup[(year,day)]
                    np.testing.assert_array_equal(obs,source['observations'][index])
                    assert request==source['requests'][index]
                    values=[]
                    for replicate in range(config.rollout_replicates):
                        result=rollout_branch(env,base,request,
                            noise_seed=seed*10000000+year*1000+day*10+replicate)
                        steps+=result['continuation_steps'];values.append(result['soft_return'])
                        replicates.append(dict(seed=seed,year=year,day=day,replicate=replicate,**result))
                    records.append(dict(seed=seed,year=year,day=day,observation=obs.tolist(),request_m3_s=request,
                                        bootstrap_target=float(bootstrap[index]),rollout_target=float(np.mean(values)),
                                        rollout_sd=float(np.std(values,ddof=1))))
                obs,_,terminated,truncated,_=env.step(np.array([request],np.float32));steps+=1
                assert not terminated and not truncated
        finally:
            env.close()
    check_fit_scope([r['year'] for r in records])
    return records,pd.DataFrame(replicates),steps


@torch.no_grad()
def evaluate_points(agent,points,returns,metadata,arm,budget,partition):
    q1,q2=point_predictions(agent,points,np.arange(len(points['years'])))
    p=np.minimum(q1.numpy(),q2.numpy())
    metrics=reference_errors(p,returns)
    summary=dict(seed=int(metadata[0]['seed']),arm=arm,steps=budget,partition=partition,anchors=len(p),**metrics)
    rows=[]
    for i,meta in enumerate(metadata):
        rows.append(dict(seed=meta['seed'],year=meta['year'],day=meta['day'],partition=partition,
                         arm=arm,steps=budget,q1=float(q1[i]),q2=float(q2[i]),min_q=float(p[i]),
                         rollout_mean=float(returns[i].mean()),rollout_sd=float(returns[i].std(ddof=1)),
                         conditional_error=float(p[i]-returns[i].mean())))
    return summary,pd.DataFrame(rows)


def ordered_returns(records,replicates):
    result=[]
    for record in records:
        frame=replicates[(replicates.seed==record['seed'])&(replicates.year==record['year'])&(replicates.day==record['day'])]
        if sorted(frame.replicate.tolist())!=list(range(8)):
            raise ValueError('Missing or duplicate continuation replicates')
        result.append(frame.sort_values('replicate').soft_return.to_numpy())
    return np.asarray(result,np.float64)


def run_seed(seed,root,plan,config,checking_anchors,checking_returns):
    output=root/f'seed_{seed}';output.mkdir()
    base=SACAgent.load(STUDY/f'seed_{seed}/final.pt');freeze_noncritics(base)
    frozen=frozen_state(base);rng=base.generator.get_state().clone()
    pool={k:v.copy() for k,v in np.load(POOL/f'seed_{seed}/common_pool.npz').items()}
    cached=np.load(POOL/f'seed_{seed}/fixed_targets.npy')
    fit_records,fit_returns,cost=collect_fit_targets(base,seed,plan,config,pool,cached)
    fit_returns.to_csv(output/'fitting_rollout_replicates.csv',index=False)
    (output/'fitting_anchors.json').write_text(json.dumps(fit_records,indent=2)+'\n')
    fit_data=as_point_data(fit_records);fit_matrix=ordered_returns(fit_records,fit_returns)
    check_records=[r for r in checking_anchors if r['seed']==seed]
    check_data=as_point_data(check_records);check_matrix=ordered_returns(check_records,checking_returns)
    _,checking=training_partitions()
    assert set(check_data['years'])==set(checking)
    assert not np.isin(check_data['years'],fit_data['years']).any()
    targets=dict(bootstrap=torch.tensor([r['bootstrap_target'] for r in fit_records],dtype=torch.float32),
                 rollout=torch.tensor([r['rollout_target'] for r in fit_records],dtype=torch.float32))
    np.savez_compressed(output/'fit_states_targets.npz',**fit_data,
                        bootstrap_targets=targets['bootstrap'].numpy(),rollout_targets=targets['rollout'].numpy())
    batches=target_batches(fit_data['years'],config,seed)
    assert np.isin(fit_data['years'][batches],training_partitions()[0]).all()
    pairs={k:v.copy() for k,v in np.load(PAIRS/f'seed_{seed}/pairs.npz').items()}
    pair_rows=pd.read_csv(PAIRS/f'seed_{seed}/pairs.csv',float_precision='round_trip')
    pair_targets=torch.from_numpy(cached[pool['pair_indices']])
    ordinary_check=np.flatnonzero(np.isin(pool['years'][:15330],checking))
    summaries=[];predictions=[];traces=[];broad=[]
    pair_scores=[score_pairs(base,pairs,pair_rows,pair_targets,'baseline',0)]
    for partition,data,matrix,records in [('fit',fit_data,fit_matrix,fit_records),('check',check_data,check_matrix,check_records)]:
        summary,rows=evaluate_points(base,data,matrix,records,'baseline',0,partition)
        summaries.append(summary);predictions.append(rows)
    broad.append(dict(seed=seed,arm='baseline',steps=0,frozen_target_mse=broad_score(base,pool,torch.from_numpy(cached),ordinary_check)))
    for arm,target in targets.items():
        # Same observations/actions and exact same minibatch sequence in both arms.
        data=fit_data
        assert_identical_non_targets(data,fit_data)
        agent=deepcopy(base)
        optimizer=torch.optim.Adam(list(agent.q1.parameters())+list(agent.q2.parameters()),lr=config.learning_rate)
        for step,indices in enumerate(batches,1):
            q1,q2=point_predictions(agent,data,indices)
            loss=F.mse_loss(q1,target[indices])+F.mse_loss(q2,target[indices])
            grad=checked_optimizer_step(loss,optimizer,list(agent.q1.parameters())+list(agent.q2.parameters()))
            if step%100==0:
                traces.append(dict(arm=arm,step=step,loss=loss.item(),gradient_norm=grad))
            if step in config.budgets:
                for partition,points,matrix,records in [('fit',fit_data,fit_matrix,fit_records),('check',check_data,check_matrix,check_records)]:
                    summary,rows=evaluate_points(agent,points,matrix,records,arm,step,partition)
                    summaries.append(summary);predictions.append(rows)
                pair_scores.append(score_pairs(agent,pairs,pair_rows,pair_targets,arm,step))
                broad.append(dict(seed=seed,arm=arm,steps=step,frozen_target_mse=broad_score(agent,pool,torch.from_numpy(cached),ordinary_check)))
                torch.save(dict(q1=agent.q1.state_dict(),q2=agent.q2.state_dict(),seed=seed,arm=arm,steps=step,
                                scope='Frozen-actor target diagnostic, critics only'),output/f'{arm}_{step}_critics.pt')
        assert_frozen(agent,frozen,rng)
        print('Target fit complete',seed,arm,flush=True)
    assert_frozen(base,frozen,rng)
    summary=pd.DataFrame(summaries);predictions=pd.concat(predictions,ignore_index=True);pair_scores=pd.concat(pair_scores,ignore_index=True)
    summary.to_csv(output/'metrics.csv',index=False);predictions.to_csv(output/'predictions.csv',index=False)
    pair_scores.to_csv(output/'pair_scores.csv',index=False);pd.DataFrame(traces).to_csv(output/'fitting_trace.csv',index=False)
    pd.DataFrame(broad).to_csv(output/'cached_target_scores.csv',index=False)
    (output/'run.json').write_text(json.dumps(dict(complete=True,seed=seed,fit_anchors=len(fit_records),check_anchors=len(check_records),
        simulator_transitions=cost,actor_and_all_noncritic_state_unchanged=True,identical_batches=True,
        checking_returns_excluded_from_gradients=True,fit_dataset_sha256=digest(output/'fit_states_targets.npz')),indent=2)+'\n')
    return summary,predictions,pair_scores,pd.DataFrame(broad),cost


def plot(root,summary,pair_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5.5),layout='constrained')
    baseline=summary.query('arm=="baseline" and partition=="check"').iloc[0]
    for arm,label,color in [('bootstrap','Bootstrap targets','#267ead'),('rollout','Rollout-return targets','#27856a')]:
        fit=summary.query('arm==@arm and partition=="fit"').sort_values('steps')
        check=summary.query('arm==@arm and partition=="check"').sort_values('steps')
        original_fit=summary.query('arm=="baseline" and partition=="fit"').iloc[0]
        axes[0].plot([0,*fit.steps],[original_fit.mse,*fit.mse],marker='o',color=color,label=label)
        axes[1].plot([0,*check.steps],[baseline.mse,*check.mse],marker='o',color=color,label=label)
        pairs=pair_summary.query('arm==@arm and partition=="check"').sort_values('steps')
        original=pair_summary.query('arm=="baseline" and partition=="check"').iloc[0]
        axes[2].plot([0,*pairs.steps],[original.correct_fraction*100,*(pairs.correct_fraction*100)],marker='o',color=color)
    axes[0].set(title='Fitting-year conditional return error',ylabel='min-Q return MSE')
    axes[1].set(title='Separate checking-year return error',ylabel='min-Q return MSE')
    axes[2].set(title='Checking clipping-pair rankings',ylabel='Correct preferences (%)',ylim=(0,100))
    for ax in axes:
        ax.set(xlabel='Critic-only updates',xticks=[0,100,1000]);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=10)
    fig.suptitle('Bootstrap versus rollout targets · same states, initial weights, and batches\nFive seeds · 420 fitting / 420 checking anchors in historical training years · actor frozen',fontsize=15,weight='bold')
    fig.savefig(root/'target_comparison.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/critic_targets_v1')
    args=parser.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):
        parser.error('Choose a new empty output directory')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    plan=verify_plan(STUDY);config=TargetConfig();fit,check=training_partitions()
    cached_config=json.loads((REFERENCES/'plan.json').read_text())['config']
    assert list(config.anchor_days)==cached_config['anchor_days'] and config.rollout_replicates==cached_config['rollout_replicates']
    sources=[Path(__file__),PROJECT_ROOT/'src/lake_como/sac_critic_reference.py',PROJECT_ROOT/'src/lake_como/sac_critic_fit.py',
             PROJECT_ROOT/'src/lake_como/sac_critic_sampling.py',PROJECT_ROOT/'src/lake_como/sac_diagnostics.py',
             REFERENCES/'plan.json',REFERENCES/'reference_anchors.json',REFERENCES/'rollout_replicates.csv',REFERENCES/'reference_predictions_initial.csv']
    for seed in plan['config']['learner_seeds']:
        sources += [STUDY/f'seed_{seed}/final.pt',POOL/f'seed_{seed}/common_pool.npz',POOL/f'seed_{seed}/fixed_targets.npy',
                    PAIRS/f'seed_{seed}/pairs.npz',PAIRS/f'seed_{seed}/pairs.csv']
    hashes={str(p.relative_to(PROJECT_ROOT)):digest(p) for p in sources}
    declaration=dict(config=asdict(config),learner_seeds=plan['config']['learner_seeds'],fit_years=fit,checking_years=check,
        state_action_design='All fitting years at days 45/135/225/315; original actor mean first request; no selection by clipping, reward, error, or future return',
        only_intervention='Bootstrap vs rollout target scalar. Same actor/alpha/target critics, observation transform, initial critics, data rows, minibatch sequence, loss, fresh Adam .0003, and budgets.',
        bootstrap='Original frozen target-critic Bellman targets cached in critic_sampling_v1, 16 next-action samples; no moving target update',
        rollout='Mean of eight new full-year-boundary continuations of original frozen stochastic actor in fitting years; no first-action entropy or terminal bootstrap',
        checking='Reuse prior eight-continuation references in separate checking training years; never optimize on these returns. These years were used in original SAC training and earlier diagnostics, not an independent final test.',
        primary_endpoint=1000,primary_metrics='Checking conditional-return MSE/MAE versus bootstrap and unchanged critics; all five seeds. Secondary: fitting error, cached-target residual, and clipping rankings.',
        limits='Historical-year conditional returns, not unknown-inflow population truth. Fits only one actor-request action per observed state; no implication of correct off-action gradients or better reservoir decisions.',
        checkpoint_selection='Report 100 and 1000, all seeds; no best-budget promotion and no actor update',
        source_sha256=hashes,validation_evaluated=False,test_evaluated=False)
    (root/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    start=time.perf_counter()
    anchors=json.loads((REFERENCES/'reference_anchors.json').read_text())
    returns=pd.read_csv(REFERENCES/'rollout_replicates.csv',float_precision='round_trip')
    assert set(a['year'] for a in anchors)==set(check) and not set(fit)&set(check)
    summaries=[];predictions=[];pairs=[];broad=[];transitions=0
    for seed in plan['config']['learner_seeds']:
        s,p,c,b,cost=run_seed(seed,root,plan,config,anchors,returns)
        summaries.append(s);predictions.append(p);pairs.append(c);broad.append(b);transitions+=cost
    metrics=pd.concat(summaries,ignore_index=True);pred=pd.concat(predictions,ignore_index=True)
    paired=pd.concat(pairs,ignore_index=True);cached=pd.concat(broad,ignore_index=True)
    summary=metrics.groupby(['arm','steps','partition'])[['mse','mae','bias','noise_adjusted_mse','mean_mc_standard_error']].mean().reset_index()
    rankings=[]
    for (arm,steps,partition),g in paired.groupby(['arm','fit_step','partition']):
        rankings.append(dict(arm=arm,steps=steps,partition=partition,pairs=len(g),correct=int(g.min_q_correct_ranking.sum()),
                              correct_fraction=float(g.min_q_correct_ranking.mean()),gain_mae=float(g.min_q_gain_error.abs().mean())))
    rankings=pd.DataFrame(rankings)
    for name,table in [('metrics_by_seed',metrics),('summary',summary),('predictions',pred),('pair_scores',paired),('pair_summary',rankings),('cached_target_scores',cached)]:
        table.to_csv(root/f'{name}.csv',index=False)
    # Original checking references and predictions must line up exactly with baseline.
    old=pd.read_csv(REFERENCES/'reference_predictions_initial.csv',float_precision='round_trip').query('model=="baseline"')
    joined=pred.query('arm=="baseline" and partition=="check"').merge(old,on=['seed','year','day'],suffixes=('_new','_old'),validate='one_to_one')
    np.testing.assert_allclose(joined.min_q_new,joined.min_q_old,rtol=1e-6,atol=1e-5)
    assert len(joined)==420
    plot(root,summary,rankings)
    assert all(digest(PROJECT_ROOT/path)==value for path,value in hashes.items());verify_plan(STUDY)
    (root/'run.json').write_text(json.dumps(dict(complete=True,actor_and_all_noncritic_state_unchanged=True,
        identical_states_batches_optimizer_budgets=True,checking_returns_excluded_from_fitting=True,
        source_and_baseline_hashes_unchanged=True,fit_anchors=420,check_anchors=420,new_fit_rollout_replicates=3360,
        simulator_transitions=transitions,optimizer_steps=10000,seconds=time.perf_counter()-start,
        validation_evaluated=False,test_evaluated=False),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(rankings.query('partition=="check"').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
