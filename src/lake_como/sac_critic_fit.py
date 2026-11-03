"""Bounded frozen-actor critic-fit probe; no policy or held-out-split evaluation.

uv run --group reproduction python -m lake_como.sac_critic_fit
Only historical training years are used, split by alternating chronological index.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
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
from lake_como.sac import SACAgent, checked_optimizer_step
from lake_como.sac_diagnostics import critic_values, one_step, same_transition
from lake_como.sac_study import verify_plan
from lake_como.splits import split_years

STUDY=PROJECT_ROOT/'results/sac/baseline_200k_v1'


@dataclass(frozen=True)
class CriticFitConfig:
    steps: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.0003
    pair_loss_weight: float = 10.
    future_action_samples: int = 16
    action_grid_points: int = 21
    anchor_days: tuple[int,...] = (90,180,270,365)
    trace_every: int = 100


def training_partitions():
    years=split_years('train')
    return years[::2],years[1::2]


def critic_pair_losses(q1,q2,targets,reward_difference,pair_weight):
    """Each [batch,2] row is [actor request, cheaper identical-transition request]."""
    if q1.shape!=q2.shape or q1.shape!=targets.shape or q1.ndim!=2 or q1.shape[1]!=2:
        raise ValueError('Expected paired [batch,2] Q predictions and targets')
    if reward_difference.shape!=(q1.shape[0],):
        raise ValueError('Expected one scaled reward difference per pair')
    td=F.mse_loss(q1,targets)+F.mse_loss(q2,targets)
    pair=F.mse_loss(q1[:,1]-q1[:,0],reward_difference)+F.mse_loss(q2[:,1]-q2[:,0],reward_difference)
    return td+pair_weight*pair,td,pair


@torch.no_grad()
def fixed_targets(agent,next_observations,rewards,ends,samples,seed):
    """Same sampled continuation for both actions; original target networks frozen."""
    obs=torch.from_numpy(next_observations)
    repeated=agent.normalize(obs.repeat_interleave(samples,dim=0))
    generator=torch.Generator().manual_seed(seed)
    action,logp=agent.actor(repeated,generator=generator)
    future=torch.minimum(agent.target_q1(repeated,action),agent.target_q2(repeated,action))-agent.alpha*logp
    future=future.reshape(len(obs),samples).mean(1)
    continuation=agent.config.gamma*(1-torch.from_numpy(ends))*future
    targets=torch.as_tensor(agent.config.reward_scale*rewards,dtype=torch.float32)+continuation[:,None]
    np.testing.assert_allclose((targets[:,1]-targets[:,0]).numpy(),
                               agent.config.reward_scale*(rewards[:,1]-rewards[:,0]),rtol=1e-4,atol=3e-6)
    return targets


def freeze_noncritics(agent):
    agent.actor.requires_grad_(False)
    agent.target_q1.requires_grad_(False)
    agent.target_q2.requires_grad_(False)
    agent.log_alpha.requires_grad_(False)


def frozen_state(agent):
    return {name:value.clone() for name,value in agent.state_dict().items()
            if not name.startswith(('q1.','q2.'))}


def assert_frozen(agent,snapshot,rng):
    assert all(torch.equal(value,agent.state_dict()[name]) for name,value in snapshot.items())
    assert torch.equal(rng,agent.generator.get_state())
    assert all(p.grad is None for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.target_q1.parameters())
    assert all(p.grad is None for p in agent.target_q2.parameters())
    assert agent.log_alpha.grad is None


def collect_pairs(agent,seed,fallback,config):
    fitting,checking=training_partitions()
    rows=[];states=[];actions=[];rewards=[];following=[];ends=[];years=[];controls=[]
    count=0
    for year in split_years('train'):
        env=PastInflowObservation(make_env('train',year),fallback)
        obs,_=env.reset()
        try:
            for day in range(1,366):
                request=float(agent.physical_action(torch.from_numpy(agent.act(obs,deterministic=True)))[0])
                if day in config.anchor_days:
                    actor_result=one_step(env,request);count+=1
                    grid=np.unique(np.r_[np.linspace(0,float(agent.action_high[0]),config.action_grid_points),request]).astype(np.float32)
                    outcomes=[one_step(env,float(a)) for a in grid];count+=len(grid)
                    identical=np.array([same_transition(result,actor_result) for result in outcomes])
                    rs=np.array([result[1] for result in outcomes])
                    eligible=identical & (rs>actor_result[1]+1e-8)
                    which=int(np.argmax(np.where(eligible,rs,-np.inf))) if eligible.any() else None
                    control=dict(seed=seed,year=year,day=day,partition='fit' if year in fitting else 'check',
                                 pair_found=bool(eligible.any()),actor_request_m3_s=request,
                                 level_before_m=float(obs[0]))
                    controls.append((obs.copy(),request,control))
                    if which is not None:
                        cheap=float(grid[which])
                        rows.append(dict(**control,cheaper_request_m3_s=cheap,
                                         realized_inflow_m3_s=float(env.unwrapped.inflow[day-1]),
                                         actual_release_m3_s=actor_result[2]['release'],
                                         reward_gain=outcomes[which][1]-actor_result[1]))
                        states.append(obs.copy());actions.append([request,cheap])
                        rewards.append([actor_result[1],outcomes[which][1]])
                        following.append(actor_result[0]);ends.append(actor_result[3]);years.append(year)
                obs,_,terminated,truncated,_=env.step(np.array([request],np.float32));count+=1
                assert not terminated and truncated==(day==365)
        finally:
            env.close()
    data=dict(observations=np.asarray(states,np.float32),requests=np.asarray(actions,np.float32),
              rewards=np.asarray(rewards,np.float64),next_observations=np.asarray(following,np.float32),
              ends=np.asarray(ends,np.float32),years=np.asarray(years,np.int32))
    if len(data['years'])==0 or not np.isin(fitting,data['years']).any() or not np.isin(checking,data['years']).any():
        raise ValueError('Paired transitions missing from a partition')
    return data,pd.DataFrame(rows),controls,count


def predicted_pairs(agent,data):
    observations=np.repeat(data['observations'],2,axis=0)
    raw=torch.from_numpy(observations)
    requests=torch.from_numpy(data['requests'].reshape(-1,1))
    actions=2*(requests-agent.action_low)/(agent.action_high-agent.action_low)-1
    obs=agent.normalize(raw)
    return agent.q1(obs,actions).reshape(-1,2),agent.q2(obs,actions).reshape(-1,2)


@torch.no_grad()
def score_pairs(agent,data,rows,targets,arm,step):
    q1,q2=predicted_pairs(agent,data)
    difference=agent.config.reward_scale*(data['rewards'][:,1]-data['rewards'][:,0])
    gains=dict(q1=(q1[:,1]-q1[:,0]).numpy(),q2=(q2[:,1]-q2[:,0]).numpy(),
               min_q=(torch.minimum(q1,q2)[:,1]-torch.minimum(q1,q2)[:,0]).numpy())
    result=rows.copy()
    result['arm'],result['fit_step']=arm,step
    result['scaled_true_gain']=difference
    result['target_actor_q']=targets[:,0].numpy()
    result['target_cheaper_q']=targets[:,1].numpy()
    for label,gain in gains.items():
        result[f'{label}_predicted_gain']=gain
        result[f'{label}_correct_ranking']=gain>0
        result[f'{label}_gain_error']=gain-difference
    result['point_td_squared_error']=((q1-targets).square()+(q2-targets).square()).mean(1).numpy()
    return result


def summarize(scores):
    rows=[]
    for (seed,arm,step,partition),frame in scores.groupby(['seed','arm','fit_step','partition']):
        rows.append(dict(seed=seed,arm=arm,fit_step=step,partition=partition,pairs=len(frame),
                         correct_rankings=int(frame.min_q_correct_ranking.sum()),
                         correct_fraction=float(frame.min_q_correct_ranking.mean()),
                         gain_mae=float(frame.min_q_gain_error.abs().mean()),
                         gain_rmse=float(np.sqrt(np.mean(frame.min_q_gain_error**2))),
                         td_mse=float(frame.point_td_squared_error.mean())))
    return pd.DataFrame(rows)


def run_seed(seed,root,plan,config):
    output=root/f'seed_{seed}';output.mkdir()
    base=SACAgent.load(STUDY/f'seed_{seed}/final.pt')
    freeze_noncritics(base)
    noncritics=frozen_state(base);rng=base.generator.get_state().clone()
    data,rows,controls,transitions=collect_pairs(base,seed,plan['fallback'],config)
    np.savez_compressed(output/'pairs.npz',**data)
    rows.to_csv(output/'pairs.csv',index=False)
    target=fixed_targets(base,data['next_observations'],data['rewards'],data['ends'],
                         config.future_action_samples,50000+seed)
    np.save(output/'fixed_targets.npy',target.numpy())
    fit=np.flatnonzero(rows.partition.to_numpy()=='fit')
    checking=np.flatnonzero(rows.partition.to_numpy()=='check')
    assert len(fit)>0 and len(checking)>0
    # Checking rows cannot enter any batch or gradient computation.
    minibatches=np.random.default_rng(60000+seed).choice(fit,size=(config.steps,config.batch_size),replace=True)
    assert not np.isin(minibatches,checking).any()
    baseline=score_pairs(base,data,rows,target,'baseline',0)
    all_scores=[baseline];traces=[];control_frames=[]
    q_control_before=np.array([min(critic_values(base,obs,[request])[0][0],critic_values(base,obs,[request])[1][0])
                               for obs,request,_ in controls])
    for arm,weight in [('bellman_only',0.),('bellman_plus_pair',config.pair_loss_weight)]:
        agent=deepcopy(base)
        optimizer=torch.optim.Adam(list(agent.q1.parameters())+list(agent.q2.parameters()),lr=config.learning_rate)
        fit_data={k:v[fit] for k,v in data.items()}
        fit_targets=target[fit]
        inverse={int(index):local for local,index in enumerate(fit)}
        local_batches=np.asarray([[inverse[int(i)] for i in indices] for indices in minibatches])
        for step,indices in enumerate(local_batches,1):
            batch={k:v[indices] for k,v in fit_data.items()}
            q1,q2=predicted_pairs(agent,batch)
            reward_difference=torch.as_tensor(agent.config.reward_scale*(batch['rewards'][:,1]-batch['rewards'][:,0]),dtype=torch.float32)
            loss,td,pair=critic_pair_losses(q1,q2,fit_targets[indices],reward_difference,weight)
            grad=checked_optimizer_step(loss,optimizer,list(agent.q1.parameters())+list(agent.q2.parameters()))
            if step%config.trace_every==0 or step==config.steps:
                traces.append(dict(arm=arm,step=step,loss=loss.item(),td_loss=td.item(),pair_loss=pair.item(),gradient_norm=grad))
                all_scores.append(score_pairs(agent,data,rows,target,arm,step))
        assert_frozen(agent,noncritics,rng)
        torch.save(dict(q1=agent.q1.state_dict(),q2=agent.q2.state_dict(),arm=arm,seed=seed,
                        steps=config.steps,base_checkpoint_sha256=digest(STUDY/f'seed_{seed}/final.pt'),
                        scope='Critic-only feasibility fit; not an updated actor or resumable SAC run'),output/f'{arm}_critics.pt')
        after=np.array([min(critic_values(agent,obs,[request])[0][0],critic_values(agent,obs,[request])[1][0])
                        for obs,request,_ in controls])
        frame=pd.DataFrame([meta for _,_,meta in controls])
        frame['arm'],frame['baseline_q'],frame['fitted_q']=arm,q_control_before,after
        frame['absolute_q_shift']=abs(after-q_control_before);control_frames.append(frame)
        print('Fit complete',seed,arm,'fit pairs',len(fit),'check pairs',len(checking),flush=True)
    assert_frozen(base,noncritics,rng)
    scores=pd.concat(all_scores,ignore_index=True)
    scores.to_csv(output/'scores.csv',index=False)
    pd.DataFrame(traces).to_csv(output/'fitting_trace.csv',index=False)
    controls=pd.concat(control_frames,ignore_index=True);controls.to_csv(output/'control_value_shift.csv',index=False)
    (output/'run.json').write_text(json.dumps(dict(complete=True,seed=seed,fit_pairs=len(fit),check_pairs=len(checking),
          simulator_transitions=transitions,steps_per_arm=config.steps,
          actor_temperature_targets_normalizer_rng_unchanged=True,check_rows_excluded_from_gradients=True,
          dataset_sha256=digest(output/'pairs.npz')),indent=2)+'\n')
    return scores,controls


def plot_scores(root,summary,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    colors={'bellman_only':'#267ead','bellman_plus_pair':'#27856a'}
    for partition,ls in [('fit','-'),('check','--')]:
        baseline=summary.query('arm=="baseline" and partition==@partition')
        for arm,color in colors.items():
            trace=summary.query('arm==@arm and partition==@partition').groupby('fit_step')[['correct_fraction','gain_mae']].mean()
            x=np.r_[0,trace.index.to_numpy()]
            for ax,column in zip(axes,['correct_fraction','gain_mae']):
                y=np.r_[baseline[column].mean(),trace[column].to_numpy()]
                ax.plot(x,y,color=color,ls=ls,label=f'{arm.replace("bellman_", "")} · {partition}')
    axes[0].set(title='Correct cheaper-request ranking (min-Q)',ylabel='Mean fraction across five seeds',ylim=(0,1.02))
    axes[1].set(title='Error in predicted reward differences',ylabel='Mean absolute error, scaled Q units')
    for ax in axes:
        ax.set_xlabel('Critic-only gradient steps');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=9)
    fig.suptitle('Frozen-actor critic fit · training years only\nSolid: fit years · dashed: separate checking years · conditional on realized inflow',weight='bold')
    fig.savefig(root/'critic_fit.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/sac/critic_fit_v1')
    args=parser.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):
        parser.error('Choose a new empty directory')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    plan=verify_plan(STUDY);config=CriticFitConfig();fit,checking=training_partitions()
    paths=[Path(__file__),PROJECT_ROOT/'src/lake_como/sac_diagnostics.py',STUDY/'plan.json']
    paths+=[STUDY/f'seed_{seed}/final.pt' for seed in plan['config']['learner_seeds']]
    hashes={str(p.relative_to(PROJECT_ROOT)):digest(p) for p in paths}
    declaration=dict(config=asdict(config),learner_seeds=plan['config']['learner_seeds'],fit_years=fit,
        checking_years=checking,source_sha256=hashes,validation_evaluated=False,test_evaluated=False,
        target='Frozen original target critics; 16 final-actor samples per next observation, shared within pairs; no target-network/alpha/actor update',
        arms=['baseline','bellman_only','bellman_plus_pair'],checkpoint_selection='Fixed final 1000 steps, all seeds; no setting selection from check years',
        scope='Conditional fitting feasibility on newly collected training-only pairs. Same next state is established only under realized inflow, not every possible inflow. No expected-return improvement claim.',
        optimizer='Fresh Adam for online critics only, LR .0003; not a resumed SAC optimizer',
        control='Checking years never enter minibatches; actor and temperature/target critics/normalization/RNG remain frozen. Actor trajectories therefore cannot improve.')
    (root/'plan.json').write_text(json.dumps(declaration,indent=2)+'\n')
    started=time.perf_counter();scores=[];controls=[]
    for seed in plan['config']['learner_seeds']:
        score,control=run_seed(seed,root,plan,config);scores.append(score);controls.append(control)
    scores=pd.concat(scores,ignore_index=True);control=pd.concat(controls,ignore_index=True)
    summary=summarize(scores)
    scores.to_csv(root/'scores.csv',index=False);summary.to_csv(root/'summary_by_seed.csv',index=False)
    control.to_csv(root/'control_value_shift.csv',index=False)
    final=summary[(summary.fit_step==config.steps)|(summary.arm=='baseline')]
    pooled=final.groupby(['arm','partition']).agg(pairs=('pairs','sum'),correct_rankings=('correct_rankings','sum'),
             mean_seed_correct_fraction=('correct_fraction','mean'),mean_seed_gain_mae=('gain_mae','mean'),
             mean_seed_gain_rmse=('gain_rmse','mean'),mean_seed_td_mse=('td_mse','mean')).reset_index()
    pooled['pooled_correct_fraction']=pooled.correct_rankings/pooled.pairs
    pooled.to_csv(root/'final_summary.csv',index=False)
    plot_scores(root,summary,config)
    assert all(digest(PROJECT_ROOT/p)==h for p,h in hashes.items());verify_plan(STUDY)
    (root/'run.json').write_text(json.dumps(dict(complete=True,validation_evaluated=False,test_evaluated=False,
        actor_and_baseline_unchanged=True,source_hashes_unchanged=True,critic_steps=2*5*config.steps,
        simulator_transitions=sum(json.loads((root/f'seed_{s}/run.json').read_text())['simulator_transitions'] for s in plan['config']['learner_seeds']),
        seconds=time.perf_counter()-started),indent=2)+'\n')
    print(pooled.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
