"""Frozen matched PPO experiment; final checkpoints, no final-test access."""
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
from .ppo_protocol import MatchedTrainingEpisodes,ObservationHistory,OBSERVATION_NAMES,SCALES
from .carryover_check import take_step,FIELDS
from .low_water import make_protected_outlet_env
from .precision_validation import annual_records
from .target_year_check import validation_pairs
from .outlet import OutletDomainError

ROOT=PROJECT_ROOT/'results/simulator/ppo_comparison_v1'
REFERENCE=PROJECT_ROOT/'results/simulator/weighted_validation_v1'
SEEDS=(123,124,125,126,127)
METRICS=('annual_cost','unmet_target_million_m3','flood_excess_m_hours','low_water_excess_m_hours','flood_days','low_water_days')

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def train(folder,seed,steps):
    torch.set_num_threads(1)
    folder.mkdir(parents=True,exist_ok=False)
    env=MatchedTrainingEpisodes();history=ObservationHistory()
    agent=PPOAgent(action_low=env.action_space.low,action_high=env.action_space.high,seed=seed)
    obs,info=env.reset(seed=seed);encoded=history.encode(obs,info)
    updates=[];episodes=[];failure=None;boundaries=0;total_reward=0.;episode_reward=0.;length=0
    action_min=float('inf');action_max=-float('inf');reward_min=float('inf');reward_max=-float('inf');balance=0.;started=time.monotonic()
    try:
        while agent.steps<steps:
            rollout={key:[] for key in ('observation','latent','log_probability','reward','value','next_value','terminated','truncated')}
            for _ in range(min(agent.config.rollout,steps-agent.steps)):
                action,z,logp,value=agent.act(encoded)
                if not env.action_space.contains(action):raise ValueError('Unbounded action')
                following,reward,term,trunc,next_info=env.step(action)
                next_encoded=history.encode(following,next_info)
                if not np.isfinite(reward):raise ValueError('Nonfinite reward')
                if term and next_info['elapsed_hours']!=730*24:raise ValueError('Premature task termination')
                if next_info['year_boundary'] and next_info['elapsed_hours']==365*24:
                    if term or trunc:raise ValueError('Annual boundary reset')
                    boundaries+=1
                # Use the stored physical terminal observation, never the reset observation.
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
                        obs,info=env.reset();history.reset();encoded=history.encode(obs,info)
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

def evaluate(root,seed,plan):
    folder=root/f'seed{seed}';training=json.loads((folder/'training.json').read_text())
    if not training['complete']:return []
    torch.set_num_threads(1);agent=PPOAgent.load(folder/'final.pt');annual=[]
    prior=MatchedTrainingEpisodes().prior_last_flow
    area=settings('train')['lake_params']['surface']
    for a,b in validation_pairs():
        env=make_protected_outlet_env('validation',(a,b));env.fallback=prior
        obs,info=env.reset(seed=901);history=ObservationHistory();rows=[];failure=None
        def controller(raw,metadata):return agent.act(history.encode(raw,metadata),deterministic=True)[0]
        try:
            for _ in range(730):
                obs,info,row,term,trunc=take_step(env,controller,obs,info)
                rows.append(row)
        except (OutletDomainError,ValueError,FloatingPointError) as error:
            failure=dict(type=type(error).__name__,message=str(error))
        frame=pd.DataFrame(rows,columns=FIELDS)
        frame.to_csv(folder/f'{a}_{b}.csv',index=False)
        spec=dict(name=f'ppo_seed{seed}_{a}_{b}',years=[a,b],arm='ppo',seed=seed)
        annual.extend(annual_records(frame,spec,plan['quartiles'],area))
        save(folder/f'{a}_{b}.json',dict(**spec,complete=len(frame)==730,failure=failure,observed_days=len(frame),
            two_year_cost=float(-frame.reward.sum()) if len(frame)==730 else None))
        env.close()
    return annual

def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    reference=json.loads((REFERENCE/'plan.json').read_text())
    paths=list(Path(__file__).parent.glob('*.py'))+[REFERENCE/'plan.json',REFERENCE/'annual.csv',PROJECT_ROOT/'uv.lock']
    # Include every frozen simulator, forcing and reference dependency.
    hashes=dict(reference['source_sha256'])
    hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan=dict(seeds=list(SEEDS),steps_per_seed=200000,config=asdict(PPOConfig()),observations=list(OBSERVATION_NAMES),scales=list(SCALES),
        quartiles=reference['training_inflow_quartiles_m3_s'],source_sha256=hashes,torch_version=torch.__version__,
        training_years=[1946,1987],validation_pairs=[list(p) for p in validation_pairs()],
        protocol='All 41 contiguous training pairs; reset .35m, zero target EMA, common climatological prior; 730-day finite task, no burn-in. Seven causal observed inflows, no future forcing. Fixed physical reward and protected outlet constraints. Final checkpoints only, deterministic evaluation.',
        reward='Physical cost unchanged; fixed .01 reward scaling for learning only; gamma .995. No terminal reserve penalty.',
        failures='Stop training on domain exit without invented penalty; no complete-budget score for failed seeds. Suppress required-study means unless all five seeds and all trajectories complete.',
        comparison='Legacy history MPC from weighted_validation_v1; average own seeds within historical year before differences. Forecast seeds and PPO training seeds are different randomizations. No p-values or default promotion.',
        test_evaluated=False,default_changed=False)
    save(root/'plan.json',plan);return plan

def verify(plan):
    for name,digest in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+name)

def aggregate(root,plan,rows):
    ppo=pd.DataFrame(rows)
    mpc=pd.read_csv(REFERENCE/'annual.csv',float_precision='round_trip');mpc=mpc[mpc.arm.eq('legacy')].copy();mpc['arm']='mpc'
    combined=pd.concat([ppo,mpc],ignore_index=True);combined.to_csv(root/'annual.csv',index=False)
    complete=len(ppo)==50 and bool(ppo.complete.all())
    summary=dict(all_required_complete=complete,expected_ppo_annual_records=50,observed_ppo_annual_records=len(ppo),
        complete_training_seeds=sum(json.loads((root/f'seed{s}/training.json').read_text())['complete'] for s in SEEDS),
        means={arm:{metric:float(group[metric].mean()) if complete else None for metric in METRICS}
               for arm,group in combined.groupby('arm')},test_evaluated=False,default_changed=False)
    if complete:
        paired=combined.groupby(['arm','year'])[list(METRICS)].mean().unstack('arm')
        pd.DataFrame({m:paired[m]['ppo']-paired[m]['mpc'] for m in METRICS}).to_csv(root/'paired_years.csv')
        combined.groupby(['arm','seed'])[list(METRICS)].mean().to_csv(root/'seed_means.csv')
    save(root/'summary.json',summary)

def worker(args):return train(*args)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['smoke','all'],default='all');args=parser.parse_args()
    if args.stage=='smoke':
        print(json.dumps(train(args.root,9001,4096)));return
    plan=prepare(args.root);verify(plan)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for record in pool.map(worker,[(args.root/f'seed{s}',s,plan['steps_per_seed']) for s in SEEDS]):
            print(f"Seed {record['seed']}: {record['steps']} steps, failure={record['failure']}",flush=True)
    rows=[]
    for seed in SEEDS:rows.extend(evaluate(args.root,seed,plan))
    verify(plan);aggregate(args.root,plan,rows)
    print((args.root/'summary.json').read_text())

if __name__=='__main__':main()
