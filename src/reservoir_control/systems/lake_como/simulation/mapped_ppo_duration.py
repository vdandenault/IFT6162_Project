"""Duration-only mapped PPO: reproduce 200k exactly, continue in memory to 400k.

No inference-only loader is used to resume training: optimizer, RNG streams,
current physical trajectory and causal history all remain in the live collector.
Only the fixed 400k endpoint is evaluated; the reference is frozen 200k mapped PPO.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.simulation.ppo_agent import PPOConfig
from reservoir_control.systems.lake_como.simulation.ppo_protocol import MatchedTrainingEpisodes,ObservationHistory
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo import HydraulicPPOAgent,CONTRACT
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo_study import ROOT as REFERENCE,evaluate as evaluate_endpoint
from reservoir_control.systems.lake_como.simulation.ppo_controlled import same_state
from reservoir_control.systems.lake_como.simulation.ppo_study import SEEDS,METRICS,save,verify
from reservoir_control.systems.lake_como.simulation.target_year_check import validation_pairs
from reservoir_control.systems.lake_como.simulation.outlet import OutletDomainError

DEFAULT=PROJECT_ROOT/'results/simulator/mapped_ppo_duration_v1'
PREFIX_STEPS=200000
TOTAL_STEPS=400000
COMPARISON_METRICS=METRICS+('environmental_shortage_million_m3','end_storage_million_m3')


class PrefixMismatchError(ValueError):
    """No extension is allowed when the frozen prefix does not reproduce."""


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_digest(agent,env,memory,obs):
    state=dict(shuffle=agent.shuffle.bit_generator.state,environment_rng=env.np_random.bit_generator.state,
        years=list(env.current.years),hour=env.current.hour,state=asdict(env.current.state),
        demand_ema=env.current.demand_ema,last_inflow=env.current.last_inflow,
        observed_history=memory.history,last_hour=memory.last_hour,last_raw=memory.last_raw.tolist(),
        encoded=memory.last_encoded.tolist(),observation=obs.tolist(),fallback=memory.fallback)
    return dict(torch_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        trajectory_history_and_numpy_rng=hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest())


def verify_prefix(folder,reference,seed,agent,env,memory,obs,prefix_steps,statistics):
    """Pure inspection: torch.load, never Agent.load (which reseeds Torch)."""
    before=live_digest(agent,env,memory,obs)
    path=folder/f'prefix_{prefix_steps}.pt';agent.save(path)
    original=reference/f'seed{seed}'
    recorded=torch.load(original/'final.pt',map_location='cpu',weights_only=True)
    reproduced=torch.load(path,map_location='cpu',weights_only=True)
    if not same_state(recorded,reproduced):raise PrefixMismatchError('Model/optimizer/configuration/counters differ from frozen mapped prefix')
    if json.loads(path.with_suffix('.action.json').read_text())!=json.loads((original/'final.action.json').read_text()):
        raise PrefixMismatchError('Action-transform contract differs')
    status=json.loads((original/'training.json').read_text())
    if not status['complete'] or status['steps']!=prefix_steps:raise PrefixMismatchError('Reference budget or completion differs')
    for key,value in statistics.items():
        if status[key]!=value:raise PrefixMismatchError('Training prefix accounting differs: '+key)
    left=pd.read_csv(folder/'updates.csv',float_precision='round_trip')
    right=pd.read_csv(original/'updates.csv',float_precision='round_trip')
    if not left.equals(right):raise PrefixMismatchError('Optimizer update diagnostics differ from frozen prefix')
    after=live_digest(agent,env,memory,obs)
    if after!=before:raise PrefixMismatchError('Verification mutated trajectory, history or RNG state')
    record=dict(verified=True,steps=prefix_steps,model_optimizer_config_counters_exact=True,
        action_contract_exact=True,episode_accounting_and_update_log_exact=True,live_state_and_rng_unchanged=True,
        live_digest=after,current_episode_years=list(env.current.years),current_episode_elapsed_hours=env.current.hour,
        episodes_started=env.episodes,reference_checkpoint_sha256=sha256(original/'final.pt'),prefix_checkpoint_sha256=sha256(path))
    save(folder/'prefix_verification.json',record)
    return record


def train_seed(folder,seed,reference=REFERENCE,prefix_steps=PREFIX_STEPS,total_steps=TOTAL_STEPS):
    if not 0<prefix_steps<total_steps:raise ValueError('Positive prefix smaller than final budget required')
    steps=total_steps;reference=Path(reference)
    torch.set_num_threads(1);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    env=MatchedTrainingEpisodes();memory=ObservationHistory()
    agent=HydraulicPPOAgent(PPOConfig(),action_low=env.action_space.low,action_high=env.action_space.high,seed=seed)
    obs,info=env.reset(seed=seed);encoded=memory.encode(obs,info)
    updates=[];episodes=[];failure=None;boundaries=0;total_reward=0.;episode_reward=0.;length=0
    action_min=float('inf');action_max=-float('inf');reward_min=float('inf');reward_max=-float('inf');balance=0.;started=time.monotonic()
    prefix_verified=False;pending_reset=False;action=None
    try:
        while agent.steps<steps:
            if pending_reset:
                obs,info=env.reset();memory.reset();encoded=memory.encode(obs,info);pending_reset=False
            cut=prefix_steps if agent.steps<prefix_steps else steps
            rollout={key:[] for key in ('observation','latent','log_probability','reward','value','next_value','terminated','truncated')}
            for _ in range(min(agent.config.rollout,cut-agent.steps)):
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
                    if agent.steps<cut:
                        obs,info=env.reset();memory.reset();encoded=memory.encode(obs,info)
                    else:pending_reset=True
            diagnostics=agent.update(rollout);updates.append(dict(step=agent.steps,**diagnostics))
            pd.DataFrame(updates).to_csv(folder/'updates.csv',index=False)
            save(folder/'progress.json',dict(seed=seed,steps=agent.steps,budget=steps,episodes=len(episodes),elapsed_seconds=time.monotonic()-started))
            if agent.steps==prefix_steps:
                statistics=dict(physical_reward=total_reward,episodes=episodes,unfinished_length=length,
                    annual_boundaries=boundaries,action_range=[action_min,action_max],reward_range=[reward_min,reward_max],
                    max_hourly_balance_error_m3=balance,optimizer_updates=agent.updates)
                verify_prefix(folder,reference,seed,agent,env,memory,obs,prefix_steps,statistics)
                prefix_verified=True
                print(f'seed {seed}: {prefix_steps}-step mapped prefix verified; continuing in memory',flush=True)

    except (OutletDomainError,ValueError,FloatingPointError) as error:
        failure=dict(type=type(error).__name__,message=str(error),successful_steps=agent.steps,
            years=list(env.current.years),elapsed_hours=int(info['elapsed_hours']),observation=obs.tolist(),request=None if action is None else action.tolist())
    agent.save(folder/'final.pt')
    result=dict(seed=seed,steps=agent.steps,budget=steps,complete=agent.steps==steps and failure is None and prefix_verified,failure=failure,
        elapsed_seconds=time.monotonic()-started,physical_reward=total_reward,episodes=episodes,unfinished_length=length,
        annual_boundaries=boundaries,action_range=[action_min,action_max] if agent.steps else None,reward_range=[reward_min,reward_max] if agent.steps else None,
        max_hourly_balance_error_m3=balance,optimizer_updates=agent.updates,prefix_verified=prefix_verified,
        prefix_steps=prefix_steps,episodes_started=env.episodes,final_episode_elapsed_hours=env.current.hour,
        final_checkpoint_sha256=sha256(folder/'final.pt'),action_contract_sha256=sha256(folder/'final.action.json'))
    save(folder/'training.json',result);env.close();return result


def prepare(root,reference=REFERENCE):
    root=Path(root).resolve();reference=Path(reference).resolve()
    if root==reference or root.is_relative_to(reference):raise ValueError('Output must be outside the frozen reference study')
    if root.exists():raise FileExistsError('Output already exists; use a fresh directory or a declared stage')
    plan=json.loads((reference/'plan.json').read_text());verify(plan)
    if plan['config']!=asdict(PPOConfig()) or plan['steps_per_seed']!=PREFIX_STEPS or plan['seeds']!=list(SEEDS) or plan['action_contract']!=CONTRACT:
        raise ValueError('Reference must be the declared five-seed 200k mapped-PPO study')
    paths=[Path(__file__),reference/'plan.json',reference/'annual.csv']
    for seed in SEEDS:
        folder=reference/f'seed{seed}'
        status=json.loads((folder/'training.json').read_text());state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        if not status['complete'] or status['steps']!=PREFIX_STEPS or state['steps']!=PREFIX_STEPS or state['seed']!=seed or state['config']!=plan['config']:
            raise ValueError(f'Incomplete/incompatible reference seed {seed}')
        if json.loads((folder/'final.action.json').read_text())!=CONTRACT:raise ValueError('Reference checkpoint action contract differs')
        paths += [folder/name for name in ['training.json','updates.csv','final.pt','final.action.json']]
    reference_annual=pd.read_csv(reference/'annual.csv')
    reference_annual=reference_annual[reference_annual.arm.eq('treatment')]
    if not full_annual(reference_annual):raise ValueError('All reference mapped validation years must be complete')
    hashes=dict(plan['source_sha256']);hashes.update({str(p.resolve()):sha256(p) for p in paths})
    frozen=dict(source_sha256=hashes,reference=str(reference),seeds=list(SEEDS),prefix_steps=PREFIX_STEPS,total_steps=TOTAL_STEPS,
        config=plan['config'],quartiles=plan['quartiles'],action_contract=CONTRACT,torch_version=str(torch.__version__),python_version=sys.version,
        intervention='Duration only: all five original seeds freshly replay to 200k, including the original shortened last rollout, then continue to 400k in the same live collector. No resets/reseeding at the prefix except a normal task reset if exactly at a terminal boundary.',
        prefix='Exact model/optimizer/config/counter, action-contract, episode-accounting and update-log comparison; prefix-checkpoint inspection may not change Torch/shuffle/environment RNG or physical/history state. Stop a mismatching seed before extension.',
        endpoints='Frozen reference 200k and fixed new 400k only. Deterministic evaluation across all five 1988–1997 two-year validation pairs. No intermediate validation, checkpoint selection, default promotion or final-test evaluation.',
        budget='400k TOTAL per seed: 200k reproduction plus 200k extension. No inference-only checkpoint loader used for training continuation.',
        failure='No outside-domain reward, extrapolation or hard-year exclusion. Incomplete or mismatching seeds have no required-cohort comparison score. CLI returns nonzero for incomplete stages.',
        restart='Recorded seeds are not retrained. An interrupted seed without training.json is not resumable: use a fresh output root. Prefix model checkpoints do not contain a complete restartable environment/RNG snapshot.',
        test_evaluated=False,default_changed=False)
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',frozen)
    return frozen


def full_annual(frame):
    required={(seed,a,y) for seed in SEEDS for a,b in validation_pairs() for y in [a,b]}
    columns={'seed','block_start','year','complete',*COMPARISON_METRICS}
    return bool(len(frame)==50 and columns<=set(frame.columns)
        and not frame.duplicated(['seed','block_start','year']).any()
        and set(zip(frame.seed,frame.block_start,frame.year))==required and frame.complete.eq(True).all()
        and np.isfinite(frame[list(COMPARISON_METRICS)].to_numpy(float)).all())


def training_status(folder,reference):
    status=json.loads((folder/'training.json').read_text())
    if sha256(folder/'final.pt')!=status['final_checkpoint_sha256'] or sha256(folder/'final.action.json')!=status['action_contract_sha256']:
        raise ValueError('Recorded output checkpoint changed: '+str(folder))
    if status['complete']:
        if status['steps']!=TOTAL_STEPS or status['budget']!=TOTAL_STEPS or not status['prefix_verified']:raise ValueError('Invalid completed duration budget/prefix')
        prefix=json.loads((folder/'prefix_verification.json').read_text())
        original=reference/f"seed{status['seed']}/final.pt"
        if sha256(original)!=prefix['reference_checkpoint_sha256'] or sha256(folder/f'prefix_{PREFIX_STEPS}.pt')!=prefix['prefix_checkpoint_sha256']:
            raise ValueError('Verified prefix checkpoint changed')
        if not same_state(torch.load(original,map_location='cpu',weights_only=True),torch.load(folder/f'prefix_{PREFIX_STEPS}.pt',map_location='cpu',weights_only=True)):
            raise ValueError('Stored prefix no longer matches reference')
    return status


def worker(args):return train_seed(*args)


def run_training(root,plan,workers):
    reference=Path(plan['reference']);pending=[]
    for seed in SEEDS:
        folder=root/f'seed{seed}'
        if folder.exists():
            if not (folder/'training.json').exists():raise FileExistsError(f'Interrupted seed directory {folder}; use a fresh root. No mid-seed resume is supported.')
            status=training_status(folder,reference)
            print(f"seed {seed}: already recorded, complete={status['complete']}; not retraining",flush=True)
        else:pending.append((folder,seed,reference))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(worker,pending):print(f"seed {record['seed']}: {record['steps']}/{TOTAL_STEPS}, prefix={record['prefix_verified']}, failure={record['failure']}",flush=True)
    return all(training_status(root/f'seed{s}',reference)['complete'] for s in SEEDS)


def run_evaluation(root,plan):
    complete=True
    for seed in SEEDS:
        folder=root/f'seed{seed}';status=training_status(folder,Path(plan['reference']))
        if not status['complete']:
            complete=False;print(f'seed {seed}: incomplete training, withholding evaluation',flush=True);continue
        cached=folder/'evaluation_annual.csv';manifest=folder/'evaluation.json'
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved['final_checkpoint_sha256']!=status['final_checkpoint_sha256'] or saved['annual_sha256']!=sha256(cached):raise ValueError('Cached evaluation changed')
            done=saved['complete'];print(f'seed {seed}: using recorded endpoint evaluation',flush=True)
        else:
            # The frozen helper is evaluation-only; it loads the mapped contract
            # and deterministically evaluates every declared validation block.
            rows=evaluate_endpoint(root,seed,plan);frame=pd.DataFrame(rows);frame.to_csv(cached,index=False)
            done=bool(len(frame)==10 and frame.complete.eq(True).all())
            save(manifest,dict(complete=done,annual_sha256=sha256(cached),final_checkpoint_sha256=status['final_checkpoint_sha256']))
            print(f'seed {seed}: 400k endpoint evaluated, complete={done}',flush=True)
        complete=complete and done
    return complete


def comparison(reference,endpoint,statuses):
    eligible=bool(len(statuses)==5 and {r['seed'] for r in statuses}==set(SEEDS) and all(r['complete'] and r['prefix_verified'] and r['steps']==TOTAL_STEPS for r in statuses)
        and full_annual(reference) and full_annual(endpoint))
    result=dict(all_required_complete=eligible,complete_training_seeds=sum(bool(r['complete']) for r in statuses),
        verified_prefix_seeds=sum(bool(r['prefix_verified']) for r in statuses),required_endpoint_annual_records=50,
        observed_endpoint_annual_records=len(endpoint),means={},test_evaluated=False,default_changed=False)
    for label,frame in [('200k',reference),('400k',endpoint)]:
        result['means'][label]={m:float(frame[m].mean()) if eligible else None for m in COMPARISON_METRICS}
    paired=pd.DataFrame()
    if eligible:
        before=reference.set_index(['seed','block_start','year']);after=endpoint.set_index(['seed','block_start','year']).reindex(before.index)
        paired=pd.DataFrame({m:after[m]-before[m] for m in COMPARISON_METRICS})
        result.update(mean_cost_difference=float(paired.annual_cost.mean()),annual_seed_wins=int(paired.annual_cost.lt(0).sum()),
            paired_seed_cost_differences={str(k):float(v) for k,v in paired.groupby(level='seed').annual_cost.mean().items()})
    return result,paired


def aggregate(root,plan):
    reference_root=Path(plan['reference']);reference=pd.read_csv(reference_root/'annual.csv',float_precision='round_trip')
    reference=reference[reference.arm.eq('treatment')].copy();reference['arm']='200k'
    endpoints=[];statuses=[]
    for seed in SEEDS:
        folder=root/f'seed{seed}';status=training_status(folder,reference_root);statuses.append(status)
        if status['complete']:
            manifest=json.loads((folder/'evaluation.json').read_text());path=folder/'evaluation_annual.csv'
            if manifest['annual_sha256']!=sha256(path):raise ValueError('Endpoint annual file changed')
            endpoints.append(pd.read_csv(path,float_precision='round_trip'))
    endpoint=pd.concat(endpoints,ignore_index=True) if endpoints else pd.DataFrame()
    if len(endpoint):endpoint['arm']='400k'
    result,paired=comparison(reference,endpoint,statuses)
    result['failures']=[dict(seed=r['seed'],failure=r['failure']) for r in statuses if r['failure'] is not None]
    pd.concat([reference,endpoint],ignore_index=True).to_csv(root/'annual.csv',index=False)
    if len(paired):
        paired.to_csv(root/'paired_seed_years.csv');paired.groupby(level='seed').mean().to_csv(root/'paired_seed_means.csv')
        pd.concat([reference,endpoint],ignore_index=True).groupby(['arm','seed'])[list(COMPARISON_METRICS)].mean().to_csv(root/'seed_means.csv')
    save(root/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return result['all_required_complete']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT,help='Fresh study directory (or an existing directory for a later stage).')
    parser.add_argument('--reference',type=Path,default=REFERENCE,help='Frozen five-seed 200k mapped-PPO reference (used when preparing a new plan).')
    parser.add_argument('--workers',type=int,default=2,help='Concurrent learner processes, each with one Torch thread (1–5).')
    parser.add_argument('--stage',choices=['prepare','run','evaluate','aggregate','all'],default='all')
    args=parser.parse_args(argv)
    if not 1<=args.workers<=5:parser.error('--workers must be between 1 and 5')
    root=args.root.resolve()
    if args.stage in ['prepare','all']:
        plan=prepare(root,args.reference)
        print(f'Frozen plan: {root/"plan.json"}; five seeds, 200k prefix -> 400k total, final-only validation',flush=True)
    else:plan=json.loads((root/'plan.json').read_text())
    verify(plan)
    ok=True
    if args.stage in ['run','all']:ok=run_training(root,plan,args.workers) and ok
    if args.stage in ['evaluate','all']:ok=run_evaluation(root,plan) and ok
    if args.stage in ['aggregate','all']:ok=aggregate(root,plan) and ok
    verify(plan)
    return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
