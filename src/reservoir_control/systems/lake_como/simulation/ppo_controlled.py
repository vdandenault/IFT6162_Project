"""One declared PPO training factor, preserving all baseline source and outputs.

The collector matches ppo_study.train. A duration experiment additionally splits
its rollout at the baseline budget, allowing exact 200k checkpoint reproduction.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.simulation.ppo_agent import PPOAgent,PPOConfig
from reservoir_control.systems.lake_como.simulation.ppo_protocol import MatchedTrainingEpisodes,ObservationHistory
from reservoir_control.systems.lake_como.simulation.ppo_study import ROOT as BASELINE,SEEDS,METRICS,save,verify,evaluate
from reservoir_control.systems.lake_como.simulation.ppo_diagnosis import ROOT as DIAGNOSIS
from reservoir_control.systems.lake_como.simulation.outlet import OutletDomainError

ROOT=PROJECT_ROOT/'results/simulator/ppo_controlled_v1'


def same_state(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(same_state(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(same_state(a,b) for a,b in zip(left,right))
    return left==right


def train(folder,seed,steps,config,baseline_cut=None):
    torch.set_num_threads(1);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    env=MatchedTrainingEpisodes();memory=ObservationHistory()
    agent=PPOAgent(config,action_low=env.action_space.low,action_high=env.action_space.high,seed=seed)
    obs,info=env.reset(seed=seed);encoded=memory.encode(obs,info)
    updates=[];episodes=[];failure=None;boundaries=0;total_reward=0.;episode_reward=0.;length=0
    action_min=float('inf');action_max=-float('inf');reward_min=float('inf');reward_max=-float('inf');balance=0.;started=time.monotonic();prefix_verified=None
    try:
        while agent.steps<steps:
            rollout={key:[] for key in ('observation','latent','log_probability','reward','value','next_value','terminated','truncated')}
            cut=min(steps,baseline_cut) if baseline_cut is not None and agent.steps<baseline_cut else steps
            for _ in range(min(config.rollout,cut-agent.steps)):
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
                for key,item in dict(observation=encoded,latent=z,log_probability=logp,reward=reward*config.reward_scale,
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
            if baseline_cut is not None and agent.steps==baseline_cut:
                agent.save(folder/'baseline_cut.pt')
                reference=torch.load(BASELINE/f'seed{seed}/final.pt',map_location='cpu',weights_only=True)
                captured=torch.load(folder/'baseline_cut.pt',map_location='cpu',weights_only=True)
                prefix_verified=same_state(reference,captured)
                if not prefix_verified:raise ValueError('Baseline checkpoint did not reproduce exactly; no duration attribution allowed')
    except (OutletDomainError,ValueError,FloatingPointError) as error:
        failure=dict(type=type(error).__name__,message=str(error),successful_steps=agent.steps,
            years=list(env.current.years),elapsed_hours=int(info['elapsed_hours']),observation=obs.tolist(),request=action.tolist())
    agent.save(folder/'final.pt')
    result=dict(seed=seed,steps=agent.steps,budget=steps,complete=agent.steps==steps and failure is None,failure=failure,
        elapsed_seconds=time.monotonic()-started,physical_reward=total_reward,episodes=episodes,unfinished_length=length,
        annual_boundaries=boundaries,action_range=[action_min,action_max],reward_range=[reward_min,reward_max],
        max_hourly_balance_error_m3=balance,optimizer_updates=agent.updates,baseline_prefix_verified=prefix_verified)
    save(folder/'training.json',result);env.close();return result


def prepare(root,factor,rationale):
    root.mkdir(parents=True,exist_ok=False)
    diagnostic=json.loads((DIAGNOSIS/'plan.json').read_text());verify(diagnostic)
    if not (DIAGNOSIS/'summary.json').exists():raise ValueError('Finish diagnostics before choosing the experiment')
    config=replace(PPOConfig(),gae_lambda=.99) if factor=='gae' else PPOConfig()
    budget=200000 if factor=='gae' else 400000
    paths=[Path(__file__),DIAGNOSIS/'summary.json',DIAGNOSIS/'curve_slopes.csv',DIAGNOSIS/'storage_values.csv',DIAGNOSIS/'derived_summary.json',DIAGNOSIS/'positive_storage_values.csv',Path(__file__).with_name('ppo_diagnosis_report.py')]
    hashes=dict(diagnostic['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    original=json.loads((BASELINE/'plan.json').read_text())
    save(root/'plan.json',dict(source_sha256=hashes,seeds=list(SEEDS),config=asdict(config),steps_per_seed=budget,
        baseline= str(BASELINE),factor=factor,rationale=rationale,quartiles=original['quartiles'],
        intervention='Only GAE lambda .95 -> .99; gamma .995, budget 200k, architecture, LR, reward, physics, observations, episode sampling and all other settings unchanged.' if factor=='gae' else 'Only duration 200k -> 400k. Reproduce and verify the full 200k model+optimizer checkpoint exactly, then continue without reset or RNG reseeding.',
        selection='Diagnostic training evidence only; final checkpoints, all five seeds, no intermediate validation selection. Baseline is frozen, no new defaults.',
        evaluation='All five two-year validation blocks, deterministic actions. Pair baseline and treatment by learning seed and year; compare MPC after own-seed averaging.',
        failure='Stop on domain/numerical failure; no outside-domain penalty or hard-year exclusion. Suppress required-study means if any seed/block incomplete.',
        test_evaluated=False,costs_changed=False,default_changed=False))


def aggregate(root,plan,rows):
    treatment=pd.DataFrame(rows)
    baseline=pd.read_csv(BASELINE/'annual.csv',float_precision='round_trip');baseline=baseline[baseline.arm.eq('ppo')].copy()
    baseline['arm']='baseline';treatment['arm']='treatment'
    combined=pd.concat([baseline,treatment],ignore_index=True);combined.to_csv(root/'annual.csv',index=False)
    complete=len(treatment)==50 and bool(treatment.complete.all())
    summary=dict(factor=plan['factor'],all_required_complete=complete,complete_annual_records=int(treatment.complete.sum()) if len(treatment) else 0,
        means={arm:{metric:float(group[metric].mean()) if complete else None for metric in METRICS} for arm,group in combined.groupby('arm')},
        test_evaluated=False,default_changed=False)
    if complete:
        before=baseline.set_index(['seed','block_start','year']);after=treatment.set_index(['seed','block_start','year']).reindex(before.index)
        paired=pd.DataFrame({m:after[m]-before[m] for m in METRICS});paired.to_csv(root/'paired_seed_years.csv')
        paired.groupby(level='seed').mean().to_csv(root/'paired_seed_means.csv')
        combined.groupby(['arm','seed'])[list(METRICS)].mean().to_csv(root/'seed_means.csv')
        summary.update(mean_cost_difference=float(paired.annual_cost.mean()),annual_seed_wins=int(paired.annual_cost.lt(0).sum()),
            paired_seed_cost_differences={str(k):float(v) for k,v in paired.groupby(level='seed').annual_cost.mean().items()})
    save(root/'summary.json',summary)


def worker(args):return train(*args)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--factor',choices=['gae','duration'],required=True);parser.add_argument('--rationale',required=True);args=parser.parse_args()
    prepare(args.root,args.factor,args.rationale);plan=json.loads((args.root/'plan.json').read_text());verify(plan)
    config=PPOConfig(**plan['config']);cut=200000 if args.factor=='duration' else None
    with ProcessPoolExecutor(max_workers=2) as pool:
        for record in pool.map(worker,[(args.root/f'seed{s}',s,plan['steps_per_seed'],config,cut) for s in SEEDS]):print(f"seed {record['seed']}: {record['steps']}, failure={record['failure']}",flush=True)
    rows=[]
    for seed in SEEDS:rows.extend(evaluate(args.root,seed,plan))
    verify(plan);aggregate(args.root,plan,rows);print((args.root/'summary.json').read_text())

if __name__=='__main__':main()
