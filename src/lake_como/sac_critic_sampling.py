"""Frozen-actor factorial: uniform vs focused sampling at 100 vs 1000 updates.

uv run --group reproduction python -m lake_como.sac_critic_sampling
Uses one common fixed transition pool; no actor, target, temperature, or policy updates.
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
from lake_como.sac_critic_fit import (assert_frozen,freeze_noncritics,frozen_state,
                                     score_pairs,training_partitions)
from lake_como.sac_study import verify_plan

STUDY=PROJECT_ROOT/'results/sac/baseline_200k_v1'
PAIRS=PROJECT_ROOT/'results/sac/critic_fit_v1'


@dataclass(frozen=True)
class SamplingConfig:
    budgets: tuple[int,...] = (100,1000)
    batch_size: int = 128
    focused_pairs_per_batch: int = 32
    learning_rate: float = .0003
    future_action_samples: int = 16
    trace_every: int = 100


def matched_minibatches(fit_indices,fit_pairs,config,seed):
    """Equal 128-row batches: focused = 64 shared uniform rows + 32 whole pairs."""
    fitting=np.asarray(fit_indices,dtype=np.int64)
    pairs=np.asarray(fit_pairs,dtype=np.int64)
    paired_rows=2*config.focused_pairs_per_batch
    if fitting.size==0 or pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)==0:
        raise ValueError('Nonempty fitting indices and paired row indices required')
    if paired_rows<=0 or paired_rows>=config.batch_size or not np.isin(pairs,fitting).all():
        raise ValueError('Invalid focus fraction or non-fitting pair')
    steps=max(config.budgets)
    uniform=np.random.default_rng(71000+seed).choice(fitting,size=(steps,config.batch_size),replace=True)
    chosen=np.random.default_rng(72000+seed).integers(len(pairs),size=(steps,config.focused_pairs_per_batch))
    focused=np.concatenate([uniform[:,:config.batch_size-paired_rows],pairs[chosen].reshape(steps,paired_rows)],axis=1)
    assert uniform.shape==focused.shape
    assert np.isin(focused,fitting).all()
    return dict(uniform=uniform,focused=focused)


def merge_counterfactuals(base,pairs,rows):
    """Append only cheaper rows; actor rows already exist in the daily pool."""
    lookup={(int(y),int(d)):i for i,(y,d) in enumerate(zip(base['years'],base['days']))}
    if len(lookup)!=len(base['years']):
        raise ValueError('Duplicate base year/day')
    actor_indices=[]
    for j,row in enumerate(rows.itertuples(index=False)):
        i=lookup[(int(row.year),int(row.day))]
        for key in ['observations','next_observations']:
            np.testing.assert_array_equal(base[key][i],pairs[key][j])
        if not (base['requests'][i]==pairs['requests'][j,0]
                and base['rewards'][i]==pairs['rewards'][j,0]
                and base['ends'][i]==pairs['ends'][j]):
            raise ValueError('Counterfactual actor row differs from common-pool transition')
        actor_indices.append(i)
    extra=dict(observations=pairs['observations'],next_observations=pairs['next_observations'],
               requests=pairs['requests'][:,1],rewards=pairs['rewards'][:,1],ends=pairs['ends'],
               years=pairs['years'],days=rows.day.to_numpy(dtype=np.int32))
    merged={key:np.concatenate([value,extra[key]],axis=0) for key,value in base.items()}
    pair_indices=np.column_stack([actor_indices,np.arange(len(base['years']),len(merged['years']))])
    assert np.all(merged['years'][pair_indices[:,0]]==merged['years'][pair_indices[:,1]])
    return merged,pair_indices


def collect_pool(agent,fallback,pairs,rows):
    fit,check=training_partitions()
    observations=[];following=[];requests=[];rewards=[];ends=[];years=[];days=[]
    for year in sorted(fit+check):
        env=PastInflowObservation(make_env('train',year),fallback)
        obs,_=env.reset()
        try:
            for day in range(1,366):
                request=float(agent.physical_action(torch.from_numpy(agent.act(obs,deterministic=True)))[0])
                next_obs,reward,terminated,truncated,_=env.step(np.array([request],np.float32))
                assert not terminated and truncated==(day==365)
                observations.append(obs.copy());following.append(next_obs.copy());requests.append(request)
                rewards.append(reward);ends.append(terminated or truncated);years.append(year);days.append(day)
                obs=next_obs
        finally:
            env.close()
    base=dict(observations=np.asarray(observations,np.float32),next_observations=np.asarray(following,np.float32),
              requests=np.asarray(requests,np.float32),rewards=np.asarray(rewards,np.float64),ends=np.asarray(ends,np.float32),
              years=np.asarray(years,np.int32),days=np.asarray(days,np.int32))
    merged,indices=merge_counterfactuals(base,pairs,rows)
    return merged,indices,len(base['years'])


@torch.no_grad()
def common_targets(agent,data,pair_indices,samples,seed):
    """Cache original frozen soft-Bellman targets once, with shared continuation in pairs."""
    base_size=int(pair_indices[:,1].min())
    rng=torch.Generator().manual_seed(73000+seed)
    future=[]
    for start in range(0,base_size,512):
        obs=torch.from_numpy(data['next_observations'][start:min(start+512,base_size)])
        repeated=agent.normalize(obs.repeat_interleave(samples,dim=0))
        action,logp=agent.actor(repeated,generator=rng)
        q=torch.minimum(agent.target_q1(repeated,action),agent.target_q2(repeated,action))-agent.alpha*logp
        future.append(q.reshape(len(obs),samples).mean(1))
    base_future=torch.cat(future)
    shared=torch.cat([base_future,base_future[pair_indices[:,0]]])
    targets=torch.as_tensor(agent.config.reward_scale*data['rewards'],dtype=torch.float32)
    targets+=agent.config.gamma*(1-torch.from_numpy(data['ends']))*shared
    return targets


def point_predictions(agent,data,indices):
    obs=agent.normalize(torch.from_numpy(data['observations'][indices]))
    physical=torch.from_numpy(data['requests'][indices,None])
    actions=2*(physical-agent.action_low)/(agent.action_high-agent.action_low)-1
    return agent.q1(obs,actions).squeeze(1),agent.q2(obs,actions).squeeze(1)


@torch.no_grad()
def broad_score(agent,data,targets,indices):
    errors=[]
    for start in range(0,len(indices),1024):
        selected=indices[start:start+1024]
        q1,q2=point_predictions(agent,data,selected)
        errors.append(((q1-targets[selected]).square()+(q2-targets[selected]).square()).cpu().numpy())
    return float(np.concatenate(errors).mean())


def run_seed(seed,root,plan,config):
    output=root/f'seed_{seed}';output.mkdir()
    old=PAIRS/f'seed_{seed}'
    metadata=json.loads((old/'run.json').read_text())
    assert digest(old/'pairs.npz')==metadata['dataset_sha256']
    pairs={k:v.copy() for k,v in np.load(old/'pairs.npz').items()}
    rows=pd.read_csv(old/'pairs.csv',float_precision='round_trip')
    base=SACAgent.load(STUDY/f'seed_{seed}/final.pt');freeze_noncritics(base)
    frozen=frozen_state(base);rng=base.generator.get_state().clone()
    data,pair_indices,steps=collect_pool(base,plan['fallback'],pairs,rows)
    target=common_targets(base,data,pair_indices,config.future_action_samples,seed)
    pair_targets=target[pair_indices]
    np.testing.assert_allclose((pair_targets[:,1]-pair_targets[:,0]).numpy(),
                               base.config.reward_scale*(pairs['rewards'][:,1]-pairs['rewards'][:,0]),atol=3e-6,rtol=1e-4)
    fitting,checking=training_partitions()
    fit_indices=np.flatnonzero(np.isin(data['years'],fitting))
    fit_pairs=pair_indices[np.isin(rows.year,fitting)]
    fit_actor=np.flatnonzero(np.isin(data['years'][:steps],fitting))
    check_actor=np.flatnonzero(np.isin(data['years'][:steps],checking))
    batches=matched_minibatches(fit_indices,fit_pairs,config,seed)
    assert all(np.isin(data['years'][x],fitting).all() for x in batches.values())
    np.savez_compressed(output/'common_pool.npz',**data,pair_indices=pair_indices)
    np.save(output/'fixed_targets.npy',target.numpy())
    scores=[score_pairs(base,pairs,rows,pair_targets,'baseline',0)];broad=[];traces=[];sampling=[]
    for partition,indices in [('fit',fit_actor),('check',check_actor)]:
        broad.append(dict(seed=seed,arm='baseline',step=0,partition=partition,actor_transitions=len(indices),
                          frozen_target_mse=broad_score(base,data,target,indices)))
    for arm,indices in batches.items():
        agent=deepcopy(base)
        optimizer=torch.optim.Adam(list(agent.q1.parameters())+list(agent.q2.parameters()),lr=config.learning_rate)
        for step,selected in enumerate(indices,1):
            q1,q2=point_predictions(agent,data,selected)
            loss=F.mse_loss(q1,target[selected])+F.mse_loss(q2,target[selected])
            grad=checked_optimizer_step(loss,optimizer,list(agent.q1.parameters())+list(agent.q2.parameters()))
            if step%config.trace_every==0:
                traces.append(dict(arm=arm,step=step,loss=loss.item(),gradient_norm=grad))
            if step in config.budgets:
                scores.append(score_pairs(agent,pairs,rows,pair_targets,arm,step))
                for partition,controls in [('fit',fit_actor),('check',check_actor)]:
                    broad.append(dict(seed=seed,arm=arm,step=step,partition=partition,actor_transitions=len(controls),
                                      frozen_target_mse=broad_score(agent,data,target,controls)))
                paired_members=np.unique(fit_pairs)
                visited=indices[:step].ravel()
                sampling.append(dict(seed=seed,arm=arm,steps=step,transition_presentations=len(visited),
                                     unique_transitions_seen=len(np.unique(visited)),
                                     paired_member_presentations=int(np.isin(visited,paired_members).sum()),
                                     counterfactual_presentations=int((visited>=steps).sum())))
                torch.save(dict(q1=agent.q1.state_dict(),q2=agent.q2.state_dict(),seed=seed,arm=arm,steps=step,
                                scope='Critic-only sampling ablation; frozen actor, not a full SAC checkpoint'),output/f'{arm}_{step}_critics.pt')
        assert_frozen(agent,frozen,rng)
        print('Sampling arm complete',seed,arm,flush=True)
    assert_frozen(base,frozen,rng)
    scores=pd.concat(scores,ignore_index=True);broad=pd.DataFrame(broad)
    scores.to_csv(output/'scores.csv',index=False);broad.to_csv(output/'broad_scores.csv',index=False)
    pd.DataFrame(traces).to_csv(output/'fitting_trace.csv',index=False)
    pd.DataFrame(sampling).to_csv(output/'sampling_counts.csv',index=False)
    (output/'run.json').write_text(json.dumps(dict(complete=True,seed=seed,simulator_transitions=steps,
        fit_pool_transitions=len(fit_indices),check_actor_transitions=len(check_actor),
        shared_pool_sha256=digest(output/'common_pool.npz'),all_noncritic_state_frozen=True,
        checking_years_excluded_from_gradients=True),indent=2)+'\n')
    return scores,broad,pd.DataFrame(sampling)


def summarize(scores):
    result=[]
    for (seed,arm,steps,partition),frame in scores.groupby(['seed','arm','fit_step','partition']):
        result.append(dict(seed=seed,arm=arm,steps=steps,partition=partition,pairs=len(frame),
                           correct=int(frame.min_q_correct_ranking.sum()),
                           correct_fraction=float(frame.min_q_correct_ranking.mean()),
                           gain_mae=float(frame.min_q_gain_error.abs().mean()),
                           pair_target_mse=float(frame.point_td_squared_error.mean())))
    return pd.DataFrame(result)


def plot(root,pooled,broad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
    checked=pooled.query('partition=="check"')
    baseline=checked.query('arm=="baseline"').iloc[0]
    for arm,label,color in [('uniform','Uniform sampling','#267ead'),('focused','50% paired sampling','#27856a')]:
        frame=checked.query('arm==@arm').sort_values('steps')
        axes[0].plot([0,*frame.steps],[baseline.correct_fraction*100,*(frame.correct_fraction*100)],
                      marker='o',color=color,label=label)
        axes[1].plot([0,*frame.steps],[baseline.gain_mae,*frame.gain_mae],marker='o',color=color)
        values=broad.query('partition=="check" and arm==@arm').groupby('step').frozen_target_mse.mean()
        original=broad.query('partition=="check" and arm=="baseline"').frozen_target_mse.mean()
        axes[2].plot([0,*values.index],[original,*values.to_numpy()],marker='o',color=color)
    axes[0].set(title='Cheaper-request ranking',ylabel='Correct checking pairs (%)',ylim=(0,100))
    axes[1].set(title='Reward-difference error',ylabel='Checking-pair MAE (scaled Q)')
    axes[2].set(title='Broader ordinary-state fit',ylabel='Checking actor-transition target MSE')
    for ax in axes:
        ax.set_xlabel('Critic-only updates');ax.set_xticks([0,100,1000]);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=10)
    fig.suptitle('Data sampling versus update count · same fixed pool and Bellman loss\nActor/targets/alpha frozen · five seeds · separate checking years within the training split',weight='bold',fontsize=15)
    fig.savefig(root/'sampling_vs_updates.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/critic_sampling_v1')
    args=parser.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):
        parser.error('Choose an empty new output directory')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    plan=verify_plan(STUDY);config=SamplingConfig();fitting,checking=training_partitions()
    sources=[Path(__file__),PROJECT_ROOT/'src/lake_como/sac_critic_fit.py',PAIRS/'plan.json',PAIRS/'run.json']
    for seed in plan['config']['learner_seeds']:
        sources += [STUDY/f'seed_{seed}/final.pt',PAIRS/f'seed_{seed}/pairs.npz',PAIRS/f'seed_{seed}/pairs.csv']
    hashes={str(path.relative_to(PROJECT_ROOT)):digest(path) for path in sources}
    declaration=dict(config=asdict(config),learner_seeds=plan['config']['learner_seeds'],fit_years=fitting,checking_years=checking,
        factors='Sampling uniform vs 50%-paired, at predeclared 100 and 1000 critic updates; both ordinary Bellman MSE, no pair loss',
        common_pool='All daily deterministic frozen-actor transitions in training years plus reused cheaper counterfactual rows from critic_fit_v1; identical pool available to both arms',
        match='128 transition rows per batch, fresh Adam .0003, same initial critics/frozen targets; 64 uniform rows are shared across arms each update',
        focus='Other 64 focused rows are 32 complete sampled fitting pairs; uniform arm samples all 128 uniformly with replacement from same fitting pool',
        evaluation='Paired checking rankings/MAE and broader checking actor-transition frozen-target MSE, all seeds and both fixed budgets; no best-budget selection',
        limits='Checking years were in original SAC training but not this fit. Conditional same-state pairs selected under realized inflow. Frozen-target residuals are not exact-return calibration. No actor/control improvement claim.',
        source_sha256=hashes,validation_evaluated=False,test_evaluated=False)
    (root/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    started=time.perf_counter();scores=[];broads=[];samplings=[]
    for seed in plan['config']['learner_seeds']:
        score,broad,sampling=run_seed(seed,root,plan,config);scores.append(score);broads.append(broad);samplings.append(sampling)
    scores=pd.concat(scores,ignore_index=True);broad=pd.concat(broads,ignore_index=True)
    summary=summarize(scores)
    scores.to_csv(root/'scores.csv',index=False);summary.to_csv(root/'summary_by_seed.csv',index=False)
    broad.to_csv(root/'broad_scores.csv',index=False)
    pd.concat(samplings,ignore_index=True).to_csv(root/'sampling_counts.csv',index=False)
    pooled=[]
    for (arm,steps,partition),g in scores.groupby(['arm','fit_step','partition']):
        pooled.append(dict(arm=arm,steps=steps,partition=partition,pairs=len(g),correct=int(g.min_q_correct_ranking.sum()),
                           correct_fraction=float(g.min_q_correct_ranking.mean()),gain_mae=float(g.min_q_gain_error.abs().mean()),
                           pair_target_mse=float(g.point_td_squared_error.mean())))
    pooled=pd.DataFrame(pooled);pooled.to_csv(root/'pooled_summary.csv',index=False)
    plot(root,pooled,broad)
    assert all(digest(PROJECT_ROOT/path)==value for path,value in hashes.items());verify_plan(STUDY)
    (root/'run.json').write_text(json.dumps(dict(complete=True,actor_and_baseline_unchanged=True,all_source_hashes_unchanged=True,
          validation_evaluated=False,test_evaluated=False,simulator_transitions=5*42*365,
          reused_counterfactual_transitions=sum(pd.read_csv(PAIRS/f'seed_{seed}/pairs.csv').shape[0] for seed in plan['config']['learner_seeds']),
          optimizer_steps=5*2*max(config.budgets),seconds=time.perf_counter()-started),indent=2)+'\n')
    print(pooled.query('partition=="check"').to_string(index=False),flush=True)
    print(broad.query('partition=="check"').groupby(['arm','step']).frozen_target_mse.mean().to_string(),flush=True)


if __name__=='__main__':
    main()
