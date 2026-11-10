"""Frozen-actor critic fitting and physical action-saturation identification.

No validation/test trajectories are collected. Disjoint whole TRAIN-year blocks
separate critic fitting and checking; fixed recorded weather is not an oracle
policy input and action-noise repetitions are not independent hydrological data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from lake_como.paths import PROJECT_ROOT
from .ppo_agent import PPOAgent
from .ppo_protocol import ObservationHistory
from .ppo_study import ROOT as BASELINE,SEEDS,save,verify
from .ppo_controlled import ROOT as PREVIOUS
from .ppo_diagnosis import make_env,storage_branch,discounted_returns,stochastic_return
from .model import WaterState
from .outlet import OutletDomainError

ROOT=PROJECT_ROOT/'results/simulator/ppo_identification_v1'
FIT_PAIRS=((1946,1947),(1950,1951),(1954,1955),(1958,1959),(1962,1963),(1966,1967))
CHECK_PAIRS=((1970,1971),(1974,1975),(1978,1979),(1982,1983))
SATURATION_ANCHORS=(30,120,240,395,485,665)
STOCK_ANCHORS=(60,180,425)
REPS=4
STOCK_REPS=8
DRAWS=64
FIT_UPDATES=3000


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    previous=json.loads((PREVIOUS/'plan.json').read_text());verify(previous)
    fit={y for pair in FIT_PAIRS for y in pair};check={y for pair in CHECK_PAIRS for y in pair}
    if fit&check or not (fit|check)<=set(range(1946,1988)):
        raise ValueError('Disjoint training-only fit/check blocks required')
    paths=[Path(__file__),PREVIOUS/'plan.json',PREVIOUS/'summary.json',BASELINE/'plan.json']
    paths += [BASELINE/f'seed{s}/final.pt' for s in SEEDS]
    hashes=dict(previous['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(root/'plan.json',dict(source_sha256=hashes,seeds=list(SEEDS),fit_pairs=[list(p) for p in FIT_PAIRS],check_pairs=[list(p) for p in CHECK_PAIRS],
        trajectory_repetitions=REPS,stock_repetitions=STOCK_REPS,stock_anchors=list(STOCK_ANCHORS),saturation_anchors=list(SATURATION_ANCHORS),draws=DRAWS,
        collection='Frozen original PPO actor. Four full stochastic 730-day trajectories per block, all suffix Monte Carlo returns at gamma .995 and reward scale .01. Include actual terminal observation with value target zero. No bootstrap. Collection seed = 1000000*policy_seed + 100*block_start + replicate.',
        critic='Fit only existing critic weights, Adam lr .0003, 3000 minibatches of 256 sampled uniformly with replacement, gradient norm .5. Fresh fitting optimizer. Actor, log_std and input scales remain bitwise frozen. Final fit only; no checking-based early stopping or checkpoint selection.',
        checking='Disjoint whole training-year blocks, not an untouched actor-training holdout. Evaluate broad full-return MSE and base -> +.15m storage slopes at fixed origins with eight common-noise stochastic continuations. Below-floor perturbations excluded by design.',
        saturation='At six fixed rep0 states per block draw 64 stochastic requests without changing actor. Compare instantaneous effective release and full-day release volume with the nominal endpoint commands. Daily upper plateau is within 1e-5 m3/s of maximum-command daily mean. Exclude collapsed full-day ranges from the plateau-rate mean.',
        selection='Use FIT-block saturation only: if mean daily upper-plateau probability >=.50 among noncollapsed fit states, prioritize one full-range state-aware action-transform experiment. Otherwise prioritize one critic-training experiment if checking MSE and positive-stock slope MAE each improve >=20%; otherwise retain baseline and record no qualified policy change. Never use validation metrics for this choice.',
        uncertainty='Fixed recorded future weather, unknown to policy. Monte Carlo action noise is not hydrological uncertainty; temporal observations share trajectories. No p-values or calibration claim.',
        failures='No full-return targets from incomplete trajectories. Required checking averages and selection suppressed on any missing case or failed required branch. Never clamp/extrapolate model states.',
        test_evaluated=False,validation_evaluated=False,default_changed=False))


def snapshot(env,memory):
    return dict(hour=env.hour,state=asdict(env.state),demand_ema=env.demand_ema,last_inflow=env.last_inflow,
        raw=env._observation().tolist(),encoded=memory.last_encoded.tolist(),history=list(memory.history),fallback=memory.fallback)


def restore(years,record):
    env,_,_=make_env('train',years);env.hour=int(record['hour']);env.state=WaterState(**record['state'])
    env.demand_ema=record['demand_ema'];env.last_inflow=record['last_inflow']
    memory=ObservationHistory();memory.history=list(record['history']);memory.fallback=record['fallback']
    memory.last_hour=env.hour;memory.last_raw=np.array(record['raw'],np.float32);memory.last_encoded=np.array(record['encoded'],np.float32)
    np.testing.assert_array_equal(env._observation(),memory.last_raw)
    return env,memory


def digest_actor(agent):
    h=hashlib.sha256()
    for key,value in agent.model.state_dict().items():
        if not key.startswith('critic.'):
            h.update(key.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def daily_mean(env,request):
    year=env.years[env.hour//8760];day=env.hour//24%365+1
    state,rows=env.model.rollout(env.state,np.repeat(float(request),24),env.forcing[year][day-1],
        np.repeat(env.demands[year][day-1],24),np.repeat(day,24))
    return float(np.mean([r['release_m3_s'] for r in rows]))


def sample_saturation(agent,env,memory,seed):
    torch.manual_seed(seed);x=torch.tensor(memory.last_encoded).reshape(1,-1).expand(DRAWS,-1)
    with torch.no_grad():
        latent=agent.model.distribution(x).sample();requests=agent.model.physical(latent).reshape(-1).numpy()
    day=env.hour//24%365+1;lake=env.model.lake
    instant=np.array([lake.actual_release(float(q),env.state.storage,day) for q in requests])
    low=float(env.action_space.low[0]);high=float(env.action_space.high[0])
    lo=lake.actual_release(low,env.state.storage,day);hi=lake.actual_release(high,env.state.storage,day)
    try:
        daily=np.array([daily_mean(env,q) for q in requests]);daily_lo=daily_mean(env,low);daily_hi=daily_mean(env,high)
        daily_complete=True;failure=None
    except OutletDomainError as error:
        daily=np.array([]);daily_lo=daily_hi=None;daily_complete=False;failure=str(error)
    result=dict(level=float(memory.last_raw[0]),requested_std=float(requests.std()),instantaneous_std=float(instant.std()),
        instantaneous_upper_mass=float(np.mean(abs(instant-hi)<=1e-5)),instantaneous_lower_mass=float(np.mean(abs(instant-lo)<=1e-5)),
        instantaneous_span=float(hi-lo),daily_complete=daily_complete,failure=failure,
        daily_upper_mass=float(np.mean(abs(daily-daily_hi)<=1e-5)) if daily_complete else None,
        daily_lower_mass=float(np.mean(abs(daily-daily_lo)<=1e-5)) if daily_complete else None,
        daily_span=float(daily_hi-daily_lo) if daily_complete else None,daily_std=float(daily.std()) if daily_complete else None)
    return result


def collect_case(args):
    root,seed,years,split=args;root=Path(root);folder=root/f'seed{seed}';folder.mkdir(exist_ok=True)
    torch.set_num_threads(1);agent=PPOAgent.load(BASELINE/f'seed{seed}/final.pt')
    observations=[];targets=[];episodes=[];snapshots={};failure=None;started=time.monotonic()
    try:
        for rep in range(REPS):
            env,obs,info=make_env('train',years);memory=ObservationHistory();xs=[];rewards=[]
            torch.manual_seed(1000000*seed+100*years[0]+rep)
            for ordinal in range(730):
                encoded=memory.encode(obs,info);xs.append(encoded.copy())
                if rep==0 and ordinal in set(SATURATION_ANCHORS)|set(STOCK_ANCHORS):snapshots[str(ordinal)]=snapshot(env,memory)
                action=agent.act(encoded)[0];obs,reward,term,trunc,info=env.step(action)
                if term!=(ordinal==729) or trunc:raise ValueError('Finite task boundary mismatch')
                rewards.append(reward*agent.config.reward_scale)
            xs.append(memory.encode(obs,info));observations.extend(xs)
            targets.extend(np.r_[discounted_returns(rewards,agent.config.gamma),0.])
            episodes.append(dict(replicate=rep,days=730,physical_cost=float(-sum(rewards)/agent.config.reward_scale)))
            env.close()
    except (OutletDomainError,ValueError) as error:failure=str(error)
    complete=failure is None and len(episodes)==REPS
    stem=folder/f'{split}_{years[0]}'
    save(stem.with_suffix('.json'),dict(seed=seed,years=list(years),split=split,complete=complete,failure=failure,episodes=episodes,
        snapshots=snapshots,elapsed_seconds=time.monotonic()-started))
    if not complete:return f'{stem.name}: incomplete {failure}'
    np.savez_compressed(stem.with_suffix('.npz'),observation=np.array(observations,np.float32),target=np.array(targets,np.float32))
    saturation=[]
    for ordinal in SATURATION_ANCHORS:
        env,memory=restore(years,snapshots[str(ordinal)])
        result=sample_saturation(agent,env,memory,700000+seed*10000+years[0]*10+ordinal)
        saturation.append(dict(seed=seed,block_start=years[0],split=split,ordinal=ordinal,**result));env.close()
    pd.DataFrame(saturation).to_csv(folder/f'saturation_{years[0]}.csv',index=False)
    if split=='check':
        stocks=[]
        for ordinal in STOCK_ANCHORS:
            env,memory=restore(years,snapshots[str(ordinal)])
            for delta in [0.,.15]:
                altered,changed=storage_branch(env,memory,delta)
                encoded=changed.encode(altered._observation(),altered._info());value=agent.value(encoded)
                altered.close()
                for rep in range(STOCK_REPS):
                    result=stochastic_return(env,memory,agent,delta,years[0]*100000+ordinal*100+seed+rep)
                    stocks.append(dict(seed=seed,block_start=years[0],ordinal=ordinal,delta=delta,replicate=rep,value=value,
                        original_level=float(memory.last_raw[0]),**result))
            env.close()
        pd.DataFrame(stocks).to_csv(folder/f'stock_{years[0]}.csv',index=False)
    return f'{stem.name}: {len(observations)} observations and {len(saturation)} saturation states complete'


def dataset(root,seed,split):
    pairs=FIT_PAIRS if split=='fit' else CHECK_PAIRS;xs=[];ys=[]
    for years in pairs:
        stem=root/f'seed{seed}/{split}_{years[0]}'
        status=json.loads(stem.with_suffix('.json').read_text())
        if not status['complete']:raise ValueError('Required full-return dataset incomplete')
        with np.load(stem.with_suffix('.npz')) as arrays:
            xs.append(arrays['observation']);ys.append(arrays['target'])
    return torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys))


def fitted_metrics(agent,x,y):
    with torch.no_grad():error=agent.model.value(x)-y
    return dict(mse=float(error.square().mean()),bias=float(error.mean()),mae=float(error.abs().mean()))


def fit_seed(args):
    root,seed=args;root=Path(root);torch.set_num_threads(1);agent=PPOAgent.load(BASELINE/f'seed{seed}/final.pt')
    before=digest_actor(agent);train_x,train_y=dataset(root,seed,'fit');check_x,check_y=dataset(root,seed,'check')
    original_fit=fitted_metrics(agent,train_x,train_y);original_check=fitted_metrics(agent,check_x,check_y)
    optimizer=torch.optim.Adam(agent.model.critic.parameters(),lr=3e-4,eps=1e-5);rng=np.random.default_rng(91000+seed);losses=[]
    for step in range(FIT_UPDATES):
        indices=rng.integers(len(train_x),size=256);loss=(agent.model.value(train_x[indices])-train_y[indices]).square().mean()
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(agent.model.critic.parameters(),.5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite critic fit')
        optimizer.step()
        if (step+1)%100==0:losses.append(dict(update=step+1,minibatch_loss=float(loss.item())))
    if before!=digest_actor(agent):raise ValueError('Frozen actor or scales changed')
    folder=root/f'seed{seed}';agent.save(folder/'fitted_critic.pt')
    pd.DataFrame(losses).to_csv(folder/'critic_updates.csv',index=False)
    stocks=[]
    for years in CHECK_PAIRS:
        stem=folder/f'check_{years[0]}.json';record=json.loads(stem.read_text())
        raw=pd.read_csv(folder/f'stock_{years[0]}.csv',float_precision='round_trip')
        for ordinal in STOCK_ANCHORS:
            env,memory=restore(years,record['snapshots'][str(ordinal)])
            group=raw[raw.ordinal.eq(ordinal)];complete=bool(group.complete.all() and len(group)==2*STOCK_REPS)
            values={}
            for delta in [0.,.15]:
                branch,changed=storage_branch(env,memory,delta);values[delta]=agent.value(changed.encode(branch._observation(),branch._info()));branch.close()
            original=group.groupby('delta').value.first();mean=group.groupby('delta').return_995.mean()
            noise=group.pivot(index='replicate',columns='delta',values='return_995')
            stocks.append(dict(seed=seed,block_start=years[0],ordinal=ordinal,complete=complete,
                original_per_m=float((original[.15]-original[0])/.15),fitted_per_m=(values[.15]-values[0])/.15,
                measured_per_m=float((mean[.15]-mean[0])/.15) if complete else None,
                paired_se=float(((noise[.15]-noise[0])/.15).std(ddof=1)/np.sqrt(STOCK_REPS)) if complete else None))
            env.close()
    pd.DataFrame(stocks).to_csv(folder/'storage_values.csv',index=False)
    result=dict(seed=seed,original_fit=original_fit,original_check=original_check,fitted_fit=fitted_metrics(agent,train_x,train_y),
        fitted_check=fitted_metrics(agent,check_x,check_y),actor_digest_before=before,actor_digest_after=digest_actor(agent),
        fitting_updates=FIT_UPDATES,fit_observations=len(train_x),checking_observations=len(check_x),
        physical_policy_changed=False)
    save(folder/'critic_fit.json',result);return result


def choose(complete,plateau,mse_before,mse_after,slope_before,slope_after):
    if not complete:return 'incomplete'
    if plateau is not None and plateau>=.50:return 'action_transform'
    if mse_after<=.8*mse_before and slope_after<=.8*slope_before:return 'critic_training'
    return 'none_qualified'


def summarize(root):
    saturation=pd.concat([pd.read_csv(p) for p in sorted(root.glob('seed*/saturation_*.csv'))],ignore_index=True)
    saturation.to_csv(root/'saturation.csv',index=False)
    records=[json.loads((root/f'seed{s}/critic_fit.json').read_text()) for s in SEEDS]
    stocks=pd.concat([pd.read_csv(root/f'seed{s}/storage_values.csv') for s in SEEDS],ignore_index=True)
    stocks.to_csv(root/'storage_values.csv',index=False)
    fit=saturation[saturation.split.eq('fit')&saturation.daily_span.gt(1e-5)]
    complete=bool(len(saturation)==len(SEEDS)*(len(FIT_PAIRS)+len(CHECK_PAIRS))*len(SATURATION_ANCHORS) and saturation.daily_complete.all() and stocks.complete.all())
    mse_before=float(np.mean([r['original_check']['mse'] for r in records]));mse_after=float(np.mean([r['fitted_check']['mse'] for r in records]))
    slope_before=float((stocks.original_per_m-stocks.measured_per_m).abs().mean());slope_after=float((stocks.fitted_per_m-stocks.measured_per_m).abs().mean())
    plateau=float(fit.daily_upper_mass.mean()) if complete and len(fit) else None
    selected=choose(complete,plateau,mse_before,mse_after,slope_before,slope_after)
    result=dict(all_required_complete=complete,actor_frozen_all=all(r['actor_digest_before']==r['actor_digest_after'] for r in records),
        critic_checking_mse_before=mse_before if complete else None,critic_checking_mse_after=mse_after if complete else None,
        storage_slope_mae_before=slope_before if complete else None,storage_slope_mae_after=slope_after if complete else None,
        checking_sign_agreement_before=float((np.sign(stocks.original_per_m)==np.sign(stocks.measured_per_m)).mean()) if complete else None,
        checking_sign_agreement_after=float((np.sign(stocks.fitted_per_m)==np.sign(stocks.measured_per_m)).mean()) if complete else None,
        saturation_states=len(saturation),noncollapsed_fit_states=len(fit),mean_fit_daily_upper_plateau_mass=plateau,selected_experiment=selected,
        test_evaluated=False,validation_evaluated=False,default_changed=False)
    save(root/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['prepare','collect','fit','summary','all'],default='all');args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))
    if args.stage in ['collect','all']:
        cases=[(str(args.root),s,p,label) for s in SEEDS for label,pairs in [('fit',FIT_PAIRS),('check',CHECK_PAIRS)] for p in pairs]
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(collect_case,cases):print(result,flush=True)
    if args.stage in ['fit','all']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(fit_seed,[(str(args.root),s) for s in SEEDS]):print('critic fitted',result['seed'],flush=True)
    if args.stage in ['summary','all']:summarize(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))

if __name__=='__main__':main()
