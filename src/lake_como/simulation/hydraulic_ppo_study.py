"""One-factor full-range hydraulic action transform, original PPO otherwise."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from lake_como.paths import PROJECT_ROOT
from lake_como.environment import settings
from .ppo_agent import PPOAgent,PPOConfig
from .ppo_protocol import MatchedTrainingEpisodes,ObservationHistory
from .ppo_study import ROOT as BASELINE,SEEDS,METRICS,save,verify
from .ppo_identification import ROOT as IDENTIFICATION
from .hydraulic_ppo import HydraulicPPOAgent,CONTRACT
from .ppo_diagnosis import make_env
from .carryover_check import take_step,FIELDS
from .precision_validation import annual_records
from .target_year_check import validation_pairs
from .outlet import OutletDomainError

ROOT=PROJECT_ROOT/'results/simulator/hydraulic_ppo_v1'


def train(folder,seed,steps,agent_type=HydraulicPPOAgent):
    torch.set_num_threads(1);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    env=MatchedTrainingEpisodes();memory=ObservationHistory()
    agent=agent_type(PPOConfig(),action_low=env.action_space.low,action_high=env.action_space.high,seed=seed)
    obs,info=env.reset(seed=seed);encoded=memory.encode(obs,info)
    updates=[];episodes=[];failure=None;boundaries=0;total_reward=0.;episode_reward=0.;length=0
    action_min=float('inf');action_max=-float('inf');reward_min=float('inf');reward_max=-float('inf');balance=0.;started=time.monotonic()
    try:
        while agent.steps<steps:
            rollout={key:[] for key in ('observation','latent','log_probability','reward','value','next_value','terminated','truncated')}
            for _ in range(min(agent.config.rollout,steps-agent.steps)):
                action,z,logp,value=agent.act(encoded)
                if not env.action_space.contains(action):raise ValueError('Unbounded action')
                following,reward,term,trunc,next_info=env.step(action)
                next_encoded=memory.encode(following,next_info)
                if not np.isfinite(reward):raise ValueError('Nonfinite reward')
                if term and next_info['elapsed_hours']!=730*24:raise ValueError('Premature task termination')
                if next_info['year_boundary'] and next_info['elapsed_hours']==365*24:
                    if term or trunc:raise ValueError('Annual boundary reset')
                    boundaries+=1
                next_value=agent.value(next_encoded)
                for key,item in dict(observation=encoded,latent=z,log_probability=logp,reward=reward*agent.config.reward_scale,
                    value=value,next_value=next_value,terminated=term,truncated=trunc).items():rollout[key].append(item)
                action_min=min(action_min,float(action[0]));action_max=max(action_max,float(action[0]))
                reward_min=min(reward_min,reward);reward_max=max(reward_max,reward)
                balance=max(balance,max(abs(row['lake_balance_error_m3']) for row in next_info['hourly']))
                agent.steps+=1;total_reward+=reward;episode_reward+=reward;length+=1
                obs,info,encoded=following,next_info,next_encoded
                if term or trunc:
                    episodes.append(dict(years=list(env.current.years),length=length,physical_return=episode_reward,terminal_level=float(obs[0])))
                    length=0;episode_reward=0.
                    if agent.steps<steps:
                        obs,info=env.reset();memory.reset();encoded=memory.encode(obs,info)
            diagnostics=agent.update(rollout);updates.append(dict(step=agent.steps,**diagnostics))
            pd.DataFrame(updates).to_csv(folder/'updates.csv',index=False)
            save(folder/'progress.json',dict(seed=seed,steps=agent.steps,budget=steps,episodes=len(episodes),elapsed_seconds=time.monotonic()-started))
    except (OutletDomainError,ValueError,FloatingPointError) as error:
        failure=dict(type=type(error).__name__,message=str(error),successful_steps=agent.steps,
            years=list(env.current.years),elapsed_hours=int(info['elapsed_hours']),observation=obs.tolist(),request=action.tolist())
    agent.save(folder/'final.pt')
    result=dict(seed=seed,steps=agent.steps,budget=steps,complete=agent.steps==steps and failure is None,failure=failure,
        elapsed_seconds=time.monotonic()-started,physical_reward=total_reward,episodes=episodes,unfinished_length=length,
        annual_boundaries=boundaries,action_range=[action_min,action_max],reward_range=[reward_min,reward_max],
        max_hourly_balance_error_m3=balance,optimizer_updates=agent.updates)
    save(folder/'training.json',result);env.close();return result


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    prior=json.loads((IDENTIFICATION/'plan.json').read_text());verify(prior)
    selection=json.loads((IDENTIFICATION/'summary.json').read_text())
    if not selection['all_required_complete'] or selection['selected_experiment']!='action_transform':raise ValueError('Frozen diagnostic rule did not select this experiment')
    paths=[Path(__file__),Path(__file__).with_name('hydraulic_ppo.py'),IDENTIFICATION/'plan.json',IDENTIFICATION/'summary.json',IDENTIFICATION/'saturation.csv']
    hashes=dict(prior['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    base=json.loads((BASELINE/'plan.json').read_text())
    save(root/'plan.json',dict(source_sha256=hashes,seeds=list(SEEDS),steps_per_seed=200000,config=asdict(PPOConfig()),quartiles=base['quartiles'],action_contract=CONTRACT,
        intervention='Only deterministic state-aware full-range request warp: middle 80% spans the current instantaneous hydraulic interval, 10% tails retain every original nominal command. Same latent initialization/distribution, PPO settings, costs, physics, observations and 200k budget. New initial physical action distribution is part of the intervention.',
        information='Only observed level/season and fixed known outlet/MEF/reserve formulas; no current/future inflow or demand used to map actions. Request held constant over all 24 hours; bounds not recomputed into hourly commands.',
        likelihood='Keep pre-warp stored latent actions/log probabilities for PPO ratios. State-only fixed actuator map cancels in the importance ratio; entropy coefficient stays zero. Aliasing/atoms are allowed as latent-policy actions.',
        selection='Chosen solely by frozen fit-block saturation threshold. All five seeds and final checkpoints; no validation selection or default promotion.',
        failure='Stop seed on model-domain/numerical failure with no invented penalty, extrapolation, year exclusion or state clamp. Suppress required-study means if any training seed or validation trajectory is incomplete.',
        test_evaluated=False,costs_changed=False,physics_changed=False,default_changed=False))


def evaluate(root,seed,plan):
    folder=root/f'seed{seed}';training=json.loads((folder/'training.json').read_text())
    if not training['complete']:return []
    torch.set_num_threads(1);agent=HydraulicPPOAgent.load(folder/'final.pt');annual=[]
    for a,b in validation_pairs():
        env,obs,info=make_env('validation',(a,b));memory=ObservationHistory();rows=[];failure=None
        def controller(raw,metadata):return agent.act(memory.encode(raw,metadata),deterministic=True)[0]
        try:
            for _ in range(730):
                obs,info,row,term,trunc=take_step(env,controller,obs,info);rows.append(row)
        except (OutletDomainError,ValueError,FloatingPointError) as error:failure=dict(type=type(error).__name__,message=str(error))
        frame=pd.DataFrame(rows,columns=FIELDS);frame.to_csv(folder/f'{a}_{b}.csv',index=False)
        spec=dict(name=f'warped_seed{seed}_{a}_{b}',years=[a,b],arm='treatment',seed=seed)
        annual.extend(annual_records(frame,spec,plan['quartiles'],settings('train')['lake_params']['surface']))
        save(folder/f'{a}_{b}.json',dict(**spec,complete=len(frame)==730,failure=failure,observed_days=len(frame),two_year_cost=float(-frame.reward.sum()) if len(frame)==730 else None))
        env.close()
    return annual


def aggregate(root,rows):
    treatment=pd.DataFrame(rows);baseline=pd.read_csv(BASELINE/'annual.csv',float_precision='round_trip');baseline=baseline[baseline.arm.eq('ppo')].copy();baseline['arm']='baseline'
    combined=pd.concat([baseline,treatment],ignore_index=True);combined.to_csv(root/'annual.csv',index=False)
    statuses=[json.loads((root/f'seed{s}/training.json').read_text()) for s in SEEDS]
    complete=bool(all(r['complete'] for r in statuses) and len(treatment)==50 and treatment.complete.all())
    result=dict(all_required_complete=complete,complete_training_seeds=sum(r['complete'] for r in statuses),expected_treatment_annual_records=50,
        observed_treatment_annual_records=len(treatment),failures=[dict(seed=r['seed'],**r['failure']) for r in statuses if r['failure']],
        means={arm:{m:float(g[m].mean()) if complete else None for m in METRICS} for arm,g in combined.groupby('arm')},test_evaluated=False,default_changed=False)
    if complete:
        before=baseline.set_index(['seed','block_start','year']);after=treatment.set_index(['seed','block_start','year']).reindex(before.index)
        paired=pd.DataFrame({m:after[m]-before[m] for m in METRICS});paired.to_csv(root/'paired_seed_years.csv')
        combined.groupby(['arm','seed'])[list(METRICS)].mean().to_csv(root/'seed_means.csv')
        result.update(mean_cost_difference=float(paired.annual_cost.mean()),annual_seed_wins=int(paired.annual_cost.lt(0).sum()),paired_seed_cost_differences={str(k):float(v) for k,v in paired.groupby(level='seed').annual_cost.mean().items()})
    save(root/'summary.json',result)


def worker(args):return train(*args)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['smoke','all'],default='all');args=parser.parse_args()
    if args.stage=='smoke':print(json.dumps(train(args.root,9001,4096)));return
    prepare(args.root);plan=json.loads((args.root/'plan.json').read_text());verify(plan)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for record in pool.map(worker,[(args.root/f'seed{s}',s,200000) for s in SEEDS]):print(f"seed {record['seed']}: {record['steps']}, failure={record['failure']}",flush=True)
    rows=[]
    for seed in SEEDS:rows.extend(evaluate(args.root,seed,plan))
    verify(plan);aggregate(args.root,rows);print((args.root/'summary.json').read_text())

if __name__=='__main__':main()
