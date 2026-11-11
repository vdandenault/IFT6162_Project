"""Fresh conditional-return checks and matched 10%-focused frozen-actor critic fits.

uv run --group reproduction python -m reservoir_control.systems.lake_como.sac_critic_reference
No actor training, validation evaluation, test evaluation, or checkpoint promotion.
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

from reservoir_control.systems.lake_como.environment import make_env
from reservoir_control.systems.lake_como.inflow_observation import PastInflowObservation
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import checked_optimizer_step
from reservoir_control.systems.lake_como.sac_critic_fit import (assert_frozen,freeze_noncritics,frozen_state,score_pairs,training_partitions)
from reservoir_control.systems.lake_como.sac_critic_sampling import (SamplingConfig,matched_minibatches,point_predictions,broad_score)
from reservoir_control.systems.lake_como.sac_diagnostics import rollout_branch,critic_values
from reservoir_control.systems.lake_como.sac_study import verify_plan

STUDY=PROJECT_ROOT/'results/sac/baseline_200k_v1'
PAIRS=PROJECT_ROOT/'results/sac/critic_fit_v1'
SAMPLING=PROJECT_ROOT/'results/sac/critic_sampling_v1'


@dataclass(frozen=True)
class ReferenceConfig:
    anchor_days: tuple[int,...] = (45,135,225,315)
    rollout_replicates: int = 8
    budgets: tuple[int,...] = (100,1000)
    batch_size: int = 128
    learning_rate: float = .0003
    # 6/6/7/6/7 complete pairs over five updates = exactly 10% of 128-row batches.
    focus_cycle_pairs: tuple[int,...] = (6,6,7,6,7)


def gentle_batches(fit_indices,fit_pairs,config,seed):
    """Reuse identical uniform rows and paired draws from the previous 50% study."""
    previous=matched_minibatches(fit_indices,fit_pairs,
        SamplingConfig(budgets=config.budgets,batch_size=config.batch_size),seed)
    uniform=previous['uniform'];focused=uniform.copy()
    counts=[]
    start=config.batch_size//2
    for step in range(len(uniform)):
        count=2*config.focus_cycle_pairs[step%len(config.focus_cycle_pairs)]
        if count<=0 or count>config.batch_size//2:
            raise ValueError('Gentle focus count outside original paired half-batch')
        focused[step,-count:]=previous['focused'][step,start:start+count]
        counts.append(count)
    counts=np.asarray(counts)
    for budget in config.budgets:
        if sum(counts[:budget])/(budget*config.batch_size)!=.1:
            raise ValueError('Budget must preserve the exact 10% focus allocation')
    return dict(uniform=uniform,focus10=focused),counts


def pair_targets(pool,target):
    return target[pool['pair_indices']]


def critic_copy(base,path):
    agent=deepcopy(base)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    agent.q1.load_state_dict(saved['q1']);agent.q2.load_state_dict(saved['q2'])
    return agent


def verify_critic_reproduction(agent,path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    for name in ['q1','q2']:
        actual=getattr(agent,name).state_dict()
        if not all(torch.equal(value,actual[key]) for key,value in saved[name].items()):
            raise AssertionError('Uniform control no longer reproduces the fixed-pool study')


def reference_errors(predictions,returns):
    """Conditional return error and MC-noise correction, not population calibration."""
    returns=np.asarray(returns,dtype=float)
    if returns.ndim!=2 or returns.shape[1]<2:
        raise ValueError('At least two independent action-noise replicates per state required')
    predictions=np.asarray(predictions,dtype=float)
    if predictions.shape!=(len(returns),):
        raise ValueError('One prediction per state required')
    means=returns.mean(1)
    variance=returns.var(1,ddof=1)/returns.shape[1]
    error=predictions-means
    return dict(bias=float(error.mean()),mae=float(abs(error).mean()),mse=float(np.mean(error**2)),
                rmse=float(np.sqrt(np.mean(error**2))),mean_mc_standard_error=float(np.sqrt(variance).mean()),
                noise_adjusted_mse=float(np.mean(error**2-variance)))


def collect_references(root,plan,config):
    """Ordinary states in checking training years; no selection on critic error."""
    _,checking=training_partitions()
    anchors=[];replicates=[];predictions=[];transitions=0
    for seed in plan['config']['learner_seeds']:
        base=SACAgent.load(STUDY/f'seed_{seed}/final.pt');freeze_noncritics(base)
        frozen=frozen_state(base);rng=base.generator.get_state().clone()
        path=SAMPLING/f'seed_{seed}'
        data={k:v.copy() for k,v in np.load(path/'common_pool.npz').items()}
        target=np.load(path/'fixed_targets.npy')
        lookup={(int(year),int(day)):i for i,(year,day) in enumerate(zip(data['years'][:15330],data['days'][:15330]))}
        models={'baseline':base}
        for arm,source in [('uniform','uniform'),('focus50','focused')]:
            for budget in config.budgets:
                models[f'{arm}_{budget}']=critic_copy(base,path/f'{source}_{budget}_critics.pt')
        for year in checking:
            env=PastInflowObservation(make_env('train',year),plan['fallback'])
            obs,_=env.reset()
            try:
                for day in range(1,max(config.anchor_days)+1):
                    request=float(base.physical_action(torch.from_numpy(base.act(obs,deterministic=True)))[0])
                    if day in config.anchor_days:
                        index=lookup[(year,day)]
                        np.testing.assert_array_equal(obs,data['observations'][index])
                        assert request==data['requests'][index]
                        anchor=len(anchors)
                        record=dict(anchor_id=anchor,seed=seed,year=year,day=day,request_m3_s=request,
                                    frozen_target=float(target[index]),observation=obs.tolist())
                        anchors.append(record)
                        for replicate in range(config.rollout_replicates):
                            result=rollout_branch(env,base,request,
                                noise_seed=seed*10000000+year*1000+day*10+replicate)
                            transitions+=result['continuation_steps']
                            replicates.append(dict(anchor_id=anchor,seed=seed,year=year,day=day,
                                                   replicate=replicate,**result))
                        for label,model in models.items():
                            q1,q2=critic_values(model,obs,[request])
                            predictions.append(dict(anchor_id=anchor,seed=seed,year=year,day=day,
                                model=label,q1=float(q1[0]),q2=float(q2[0]),min_q=float(min(q1[0],q2[0]))))
                    obs,_,terminated,truncated,_=env.step(np.array([request],np.float32));transitions+=1
                    assert not terminated and not truncated
            finally:
                env.close()
        assert_frozen(base,frozen,rng)
        print('Fresh ordinary-state returns complete',seed,flush=True)
    (root/'reference_anchors.json').write_text(json.dumps(anchors,indent=2)+'\n')
    pd.DataFrame(replicates).to_csv(root/'rollout_replicates.csv',index=False)
    pd.DataFrame(predictions).to_csv(root/'reference_predictions_initial.csv',index=False)
    return anchors,pd.DataFrame(replicates),predictions,transitions


def run_gentle_seed(seed,root,plan,config,anchors):
    output=root/f'seed_{seed}';output.mkdir()
    old=SAMPLING/f'seed_{seed}'
    data={k:v.copy() for k,v in np.load(old/'common_pool.npz').items()}
    target=torch.from_numpy(np.load(old/'fixed_targets.npy'))
    pairs={k:v.copy() for k,v in np.load(PAIRS/f'seed_{seed}/pairs.npz').items()}
    rows=pd.read_csv(PAIRS/f'seed_{seed}/pairs.csv',float_precision='round_trip')
    fitting,checking=training_partitions()
    fit=np.flatnonzero(np.isin(data['years'],fitting))
    chosen=data['pair_indices'][np.isin(rows.year,fitting)]
    batches,counts=gentle_batches(fit,chosen,config,seed)
    assert all(np.isin(data['years'][batch],fitting).all() for batch in batches.values())
    fit_actor=np.flatnonzero(np.isin(data['years'][:15330],fitting))
    check_actor=np.flatnonzero(np.isin(data['years'][:15330],checking))
    base=SACAgent.load(STUDY/f'seed_{seed}/final.pt');freeze_noncritics(base)
    frozen=frozen_state(base);rng=base.generator.get_state().clone()
    scores=[score_pairs(base,pairs,rows,pair_targets(data,target),'baseline',0)]
    broad=[];predictions=[];sampling=[];traces=[]
    for partition,indices in [('fit',fit_actor),('check',check_actor)]:
        broad.append(dict(seed=seed,arm='baseline',step=0,partition=partition,
                          frozen_target_mse=broad_score(base,data,target,indices)))
    for arm,batch in batches.items():
        model=deepcopy(base)
        optimizer=torch.optim.Adam(list(model.q1.parameters())+list(model.q2.parameters()),lr=config.learning_rate)
        for step,selected in enumerate(batch,1):
            q1,q2=point_predictions(model,data,selected)
            loss=F.mse_loss(q1,target[selected])+F.mse_loss(q2,target[selected])
            grad=checked_optimizer_step(loss,optimizer,list(model.q1.parameters())+list(model.q2.parameters()))
            if step%100==0:
                traces.append(dict(arm=arm,step=step,loss=loss.item(),gradient_norm=grad))
            if step in config.budgets:
                if arm=='uniform':
                    verify_critic_reproduction(model,old/f'uniform_{step}_critics.pt')
                scores.append(score_pairs(model,pairs,rows,pair_targets(data,target),arm,step))
                for partition,indices in [('fit',fit_actor),('check',check_actor)]:
                    broad.append(dict(seed=seed,arm=arm,step=step,partition=partition,
                                      frozen_target_mse=broad_score(model,data,target,indices)))
                for anchor in anchors:
                    if anchor['seed']==seed:
                        a,b=critic_values(model,anchor['observation'],[anchor['request_m3_s']])
                        predictions.append(dict(anchor_id=anchor['anchor_id'],seed=seed,year=anchor['year'],day=anchor['day'],
                            model=f'{arm}_{step}',q1=float(a[0]),q2=float(b[0]),min_q=float(min(a[0],b[0]))))
                sampling.append(dict(arm=arm,updates=step,transition_presentations=step*config.batch_size,
                                     explicit_pair_rows=int(counts[:step].sum()) if arm=='focus10' else 0,
                                     unique_rows=len(np.unique(batch[:step])),
                                     counterfactual_presentations=int((batch[:step]>=15330).sum())))
                torch.save(dict(q1=model.q1.state_dict(),q2=model.q2.state_dict(),seed=seed,arm=arm,steps=step,
                    scope='Frozen-actor diagnostic critics only; not a full SAC checkpoint'),output/f'{arm}_{step}_critics.pt')
        assert_frozen(model,frozen,rng)
        print('Gentle fit complete',seed,arm,flush=True)
    assert_frozen(base,frozen,rng)
    scores=pd.concat(scores,ignore_index=True);broad=pd.DataFrame(broad)
    scores.to_csv(output/'scores.csv',index=False);broad.to_csv(output/'broad_scores.csv',index=False)
    pd.DataFrame(sampling).to_csv(output/'sampling_counts.csv',index=False)
    pd.DataFrame(traces).to_csv(output/'fitting_trace.csv',index=False)
    (output/'run.json').write_text(json.dumps(dict(complete=True,seed=seed,all_noncritic_state_frozen=True,
          exact_uniform_reproduction=True,checking_rows_excluded_from_gradients=True),indent=2)+'\n')
    return scores,broad,predictions


def evaluate_references(anchors,replicates,predictions):
    anchor=pd.DataFrame(anchors).drop(columns=['observation'])
    # Uniform reproductions must agree exactly, then retain one copy of each prediction.
    pred=pd.DataFrame(predictions)
    for _,frame in pred.groupby(['anchor_id','model']):
        assert frame[['q1','q2','min_q']].nunique().max()==1
    pred=pred.drop_duplicates(['anchor_id','model'])
    means=replicates.groupby('anchor_id').soft_return.agg(mc_mean='mean',mc_sd='std',replicates='size').reset_index()
    states=pred.merge(anchor,on=['anchor_id','seed','year','day'],validate='many_to_one').merge(means,on='anchor_id',validate='many_to_one')
    states['conditional_error']=states.min_q-states.mc_mean
    states['same_state_proxy_error']=states.min_q-states.frozen_target
    states['mc_standard_error']=states.mc_sd/np.sqrt(states.replicates)
    summaries=[]
    for (seed,label),frame in states.groupby(['seed','model']):
        matrix=np.array([replicates[replicates.anchor_id==i].sort_values('replicate').soft_return.to_numpy() for i in frame.anchor_id])
        metrics=reference_errors(frame.min_q.to_numpy(),matrix)
        summaries.append(dict(seed=seed,model=label,anchors=len(frame),
                              same_state_proxy_mse=float(np.mean(frame.same_state_proxy_error**2)),**metrics))
    return states,pd.DataFrame(summaries)


def pooled_pairs(scores):
    rows=[]
    for (arm,steps,partition),frame in scores.groupby(['arm','fit_step','partition']):
        rows.append(dict(arm=arm,steps=steps,partition=partition,pairs=len(frame),
                         correct=int(frame.min_q_correct_ranking.sum()),correct_fraction=float(frame.min_q_correct_ranking.mean()),
                         gain_mae=float(frame.min_q_gain_error.abs().mean())))
    return pd.DataFrame(rows)


def plot(root,metrics,pairs,broad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5.6),layout='constrained')
    variants=['baseline','uniform_1000','focus10_1000','focus50_1000']
    labels=['Unchanged','Uniform','10% focus','50% focus']
    colors=['#53687b','#267ead','#27856a','#b78219']
    means=metrics.groupby('model').mean(numeric_only=True)
    axes[0].bar(labels,means.loc[variants,'same_state_proxy_mse'],color=colors)
    axes[0].set(title='Frozen-target error on ordinary anchors',ylabel='min-Q mean squared error')
    axes[1].bar(labels,means.loc[variants,'mse'],color=colors)
    axes[1].set(title='Fresh historical-rollout return error',ylabel='min-Q conditional return MSE')
    ranking=[pairs.query('arm=="baseline" and partition=="check"').iloc[0].correct_fraction,
             pairs.query('arm=="uniform" and steps==1000 and partition=="check"').iloc[0].correct_fraction,
             pairs.query('arm=="focus10" and steps==1000 and partition=="check"').iloc[0].correct_fraction]
    old=pd.read_csv(SAMPLING/'pooled_summary.csv').query('arm=="focused" and steps==1000 and partition=="check"').iloc[0].correct_fraction
    axes[2].bar(labels,np.array(ranking+[old])*100,color=colors)
    axes[2].set(title='Clipping-pair preference',ylabel='Correct checking-pair rankings (%)',ylim=(0,100))
    for ax in axes:
        ax.tick_params(axis='x',labelrotation=20);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Checking the measurement and a gentler sampling change\nFixed 1,000 critic updates · actor frozen · 420 ordinary training-year anchors · eight continuations each',fontsize=15,weight='bold')
    fig.savefig(root/'reference_and_gentle_focus.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/critic_reference_v1')
    args=parser.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):
        parser.error('Choose a new empty directory')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    plan=verify_plan(STUDY);config=ReferenceConfig();fit,check=training_partitions()
    paths=[Path(__file__),PROJECT_ROOT/'src/reservoir_control/systems/lake_como/sac_critic_sampling.py',
           PROJECT_ROOT/'src/reservoir_control/systems/lake_como/sac_critic_fit.py',PROJECT_ROOT/'src/reservoir_control/systems/lake_como/sac_diagnostics.py',
           SAMPLING/'plan.json',SAMPLING/'run.json']
    for seed in plan['config']['learner_seeds']:
        old=SAMPLING/f'seed_{seed}'
        paths += [STUDY/f'seed_{seed}/final.pt',old/'common_pool.npz',old/'fixed_targets.npy',
                  PAIRS/f'seed_{seed}/pairs.npz',PAIRS/f'seed_{seed}/pairs.csv']
        for arm in ['uniform','focused']:
            paths += [old/f'{arm}_{budget}_critics.pt' for budget in config.budgets]
    hashes={str(path.relative_to(PROJECT_ROOT)):digest(path) for path in paths}
    declaration=dict(config=asdict(config),learner_seeds=plan['config']['learner_seeds'],fit_years=fit,checking_years=check,
        ordinary_anchors='Days 45/135/225/315, all 21 checking training years, all five seeds; no reward/error/pair-based selection',
        return_reference='Fixed actor-mean first request, then original stochastic actor until annual boundary; eight independent noise seeds; exclude first-action entropy, include discounted later entropy; no critic bootstrap',
        reference_scope='Conditional historical-year expected action-noise return. Not the unknown-inflow population value; checking years were in original SAC training.',
        gentle_sampling='Exact 10% paired rows per 100/1000 updates via 6/6/7/6/7 pairs in 128-row batches; all remaining rows shared with uniform; original 50% paired draw prefixes reused',
        fit_control='Same cached pool/targets as previous study, ordinary Bellman loss, fresh Adam .0003; reproduce uniform critic weights exactly at both budgets',
        primary_endpoint=1000,acceptance='Descriptive: checking-pair ranking above baseline AND whole-ordinary-state frozen-target MSE <= baseline AND fresh conditional-return MSE/MAE <= baseline (1e-6 numerical clearance). No automatic actor promotion even if all pass.',
        no_selection='Report both fixed budgets and all seeds, no tuning from rollout/checking results',
        source_sha256=hashes,validation_evaluated=False,test_evaluated=False)
    (root/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    start=time.perf_counter()
    anchors,replicates,predictions,transitions=collect_references(root,plan,config)
    all_scores=[];all_broad=[]
    for seed in plan['config']['learner_seeds']:
        score,broad,new=run_gentle_seed(seed,root,plan,config,anchors)
        all_scores.append(score);all_broad.append(broad);predictions+=new
    scores=pd.concat(all_scores,ignore_index=True);broad=pd.concat(all_broad,ignore_index=True)
    pairs=pooled_pairs(scores);states,metrics=evaluate_references(anchors,replicates,predictions)
    for name,table in [('scores',scores),('paired_summary',pairs),('broad_scores',broad),('ordinary_anchor_predictions',states),('reference_metrics_by_seed',metrics)]:
        table.to_csv(root/f'{name}.csv',index=False)
    summary=metrics.groupby('model').mean(numeric_only=True).reset_index();summary.to_csv(root/'reference_summary.csv',index=False)
    broad_means=broad.query('partition=="check"').groupby(['arm','step']).frozen_target_mse.mean()
    original=summary.set_index('model').loc['baseline'];candidate=summary.set_index('model').loc['focus10_1000']
    ranked=pairs.query('partition=="check"').set_index(['arm','steps'])
    criteria=dict(pair_ranking_improved=bool(ranked.loc[('focus10',1000),'correct_fraction']>ranked.loc[('baseline',0),'correct_fraction']),
                  whole_ordinary_proxy_preserved=bool(broad_means.loc[('focus10',1000)]<=broad_means.loc[('baseline',0)]+1e-6),
                  conditional_return_mse_preserved=bool(candidate.mse<=original.mse+1e-6),
                  conditional_return_mae_preserved=bool(candidate.mae<=original.mae+1e-6))
    plot(root,metrics,pairs,broad)
    assert all(digest(PROJECT_ROOT/path)==expected for path,expected in hashes.items());verify_plan(STUDY)
    result=dict(complete=True,criteria=criteria,all_acceptance_criteria_passed=all(criteria.values()),
                actor_and_baseline_unchanged=True,uniform_reproduction_exact=True,sources_unchanged=True,
                fresh_reference_anchors=len(anchors),continuation_replicates=len(replicates),simulator_transitions=transitions,
                new_fit_simulator_steps=0,optimizer_steps=10000,seconds=time.perf_counter()-start,
                validation_evaluated=False,test_evaluated=False)
    (root/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
