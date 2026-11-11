"""Matched frozen-actor critic refits: weather diversity versus action repeats.

Whole checking blocks are excluded from both refits. Original baseline training
already saw these years; this is a within-training diagnostic, not final testing.
No changes to prior experiment sources or artifacts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.simulation.ppo_agent import PPOAgent
from reservoir_control.systems.lake_como.simulation.ppo_protocol import ObservationHistory,SCALES
from reservoir_control.systems.lake_como.simulation.ppo_study import ROOT as BASELINE,SEEDS,save,verify
from reservoir_control.systems.lake_como.simulation.ppo_identification import ROOT as IDENTIFICATION,FIT_PAIRS,CHECK_PAIRS,STOCK_ANCHORS,STOCK_REPS,dataset,digest_actor,restore
from reservoir_control.systems.lake_como.simulation.ppo_diagnosis import make_env,discounted_returns,storage_branch
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo_study import ROOT as MAPPED
from reservoir_control.systems.lake_como.simulation.ppo_controlled import same_state

ROOT=PROJECT_ROOT/'results/simulator/ppo_coverage_v1'
ADDED_PAIRS=((1968,1969),(1972,1973),(1976,1977),(1980,1981),(1984,1985),(1986,1987))
DIVERSE_PAIRS=FIT_PAIRS+ADDED_PAIRS
ARMS=('concentrated','diverse')
UPDATES=3000
BATCH=256


def allocation(arm):
    if arm=='concentrated':return [(pair,4) for pair in FIT_PAIRS]
    if arm=='diverse':return [(pair,2) for pair in DIVERSE_PAIRS]
    raise ValueError('Declared arm required')


def first_replicates(x,y,count):
    if x.ndim!=2 or x.shape[1]!=15 or y.shape!=(len(x),) or len(x)%731 or count<1 or 731*count>len(x):
        raise ValueError('Complete 731-observation finite-task trajectories required')
    return x[:731*count].copy(),y[:731*count].copy()


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    reference=json.loads((MAPPED/'plan.json').read_text());verify(reference)
    checking={y for pair in CHECK_PAIRS for y in pair}
    for arm in ARMS:
        cases=allocation(arm)
        if sum(reps for _,reps in cases)!=24 or checking&{y for pair,_ in cases for y in pair}:raise ValueError('Matched counts and disjoint whole checking years required')
    paths=[Path(__file__),Path(__file__).with_name('ppo_seed_gain.py'),MAPPED/'plan.json',MAPPED/'summary.json',IDENTIFICATION/'plan.json',IDENTIFICATION/'summary.json',MAPPED/'annual.csv',MAPPED/'mechanism.csv']
    for seed in SEEDS:
        paths += list((IDENTIFICATION/f'seed{seed}').glob('*.npz'))+list((IDENTIFICATION/f'seed{seed}').glob('*.json'))
        paths += [IDENTIFICATION/f'seed{seed}/fitted_critic.pt',BASELINE/f'seed{seed}/final.pt',MAPPED/f'seed{seed}/final.pt',MAPPED/f'seed{seed}/final.action.json']
        paths += list((IDENTIFICATION/f'seed{seed}').glob('stock_*.csv'))
        paths += list((MAPPED/f'seed{seed}').glob('19*.csv'))+list((BASELINE/f'seed{seed}').glob('19*.csv'))
    hashes=dict(reference['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(root/'plan.json',dict(source_sha256=hashes,seeds=list(SEEDS),arms=list(ARMS),added_pairs=[list(p) for p in ADDED_PAIRS],
        allocation={arm:[dict(years=list(pair),replicates=n) for pair,n in allocation(arm)] for arm in ARMS},checking_pairs=[list(p) for p in CHECK_PAIRS],
        sample_count_per_arm_seed=24*731,updates=UPDATES,batch=BATCH,learning_rate=.0003,gradient_norm=.5,shuffle_seed_offset=91000,
        controls='Same original frozen PPO actor/critic initialization, observation scales, gamma .995, reward scale .01, samples and gradient steps. Same minibatch index stream. Concentrated must reproduce prior fitted checkpoint bitwise.',
        intervention='Allocate 24 trajectories across 12 whole years x4 action repeats versus 24 years x2 repeats. Wider weather coverage trades off action-noise repeats; it is not a pure weather effect with repetitions separately held fixed. Added years include later training years, so the internal check is not forward-in-time.',
        targets='Full Monte Carlo suffix targets through true finite-task termination; include terminal observation with value zero. Reuse existing first two replicates for shared blocks; collect only six added blocks per seed.',
        coverage='Nearest fit-observation distance in fixed scaled 15-feature space for the same checking states; median/p90, plus checking errors by observed level and checking block. Distance/error association is descriptive, not causal attribution.',
        metrics='Paired broad checking MSE/bias/MAE and absolute error in base-to-+.15m stock-value slope using the same existing frozen-actor stochastic-return labels. No checking-based checkpoint selection.',
        gate='Eligible for a FUTURE end-to-end critic experiment only if diverse mean checking MSE and mean stock-slope absolute error are each strictly lower than both concentrated and original critic, with MSE improving in at least four of five seeds versus concentrated. No policy-training follow-up is automatic.',
        seed_audit='Seed124 singled out post hoc because of previous validation gains. Retain all five seeds. Compare annual physical cost components, common MPC901 validation states at eight fixed origins, and own stochastic-policy critic values on common training states. Never rank different policies by raw critic value alone.',
        failures='No full target from an incomplete rollout; no required-group score on missing/failed cases. No outside-domain penalty, state clamp, physical changes or hidden year exclusions.',
        test_evaluated=False,policy_training=False,default_changed=False))


def collect_added(args):
    root,seed,years=args;root=Path(root);folder=root/f'seed{seed}';folder.mkdir(exist_ok=True)
    torch.set_num_threads(1);agent=PPOAgent.load(BASELINE/f'seed{seed}/final.pt');xs=[];ys=[];failure=None
    try:
        for rep in range(2):
            env,obs,info=make_env('train',years);memory=ObservationHistory();states=[];rewards=[]
            torch.manual_seed(1000000*seed+100*years[0]+rep)
            for day in range(730):
                encoded=memory.encode(obs,info);states.append(encoded.copy());obs,reward,term,trunc,info=env.step(agent.act(encoded)[0])
                if term!=(day==729) or trunc:raise ValueError('Finite task boundary mismatch')
                rewards.append(reward*.01)
            states.append(memory.encode(obs,info));xs.extend(states);ys.extend(np.r_[discounted_returns(rewards,.995),0.]);env.close()
    except ValueError as error:failure=str(error)
    complete=failure is None and len(xs)==1462
    stem=folder/f'added_{years[0]}'
    save(stem.with_suffix('.json'),dict(seed=seed,years=list(years),complete=complete,failure=failure,observations=len(xs)))
    if complete:np.savez_compressed(stem.with_suffix('.npz'),observation=np.array(xs,np.float32),target=np.array(ys,np.float32))
    return f'seed{seed}/{years[0]}: {len(xs)}/1462 observations, failure={failure}'


def seal_data(root):
    paths=[]
    for seed in SEEDS:
        for years in ADDED_PAIRS:
            stem=root/f'seed{seed}/added_{years[0]}'
            if not json.loads(stem.with_suffix('.json').read_text())['complete']:raise ValueError('Added dataset incomplete')
            paths.extend([stem.with_suffix('.json'),stem.with_suffix('.npz')])
    save(root/'dataset_manifest.json',{'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})


def fit_dataset(root,seed,arm):
    xs=[];ys=[]
    for pair,reps in allocation(arm):
        stem=(IDENTIFICATION/f'seed{seed}/fit_{pair[0]}' if pair in FIT_PAIRS else root/f'seed{seed}/added_{pair[0]}')
        if not json.loads(stem.with_suffix('.json').read_text())['complete']:raise ValueError('Required data incomplete')
        with np.load(stem.with_suffix('.npz')) as arrays:
            x,y=first_replicates(arrays['observation'],arrays['target'],reps);xs.append(x);ys.append(y)
    return torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys))


def errors(agent,x,y):
    with torch.no_grad():return (agent.model.value(x)-y).numpy()


def refit(agent,x,y,seed,updates=UPDATES):
    actor=digest_actor(agent);optimizer=torch.optim.Adam(agent.model.critic.parameters(),lr=3e-4,eps=1e-5);rng=np.random.default_rng(91000+seed);logs=[]
    for step in range(updates):
        indices=rng.integers(len(x),size=BATCH);loss=(agent.model.value(x[indices])-y[indices]).square().mean()
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(agent.model.critic.parameters(),.5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite critic fit')
        optimizer.step()
        if (step+1)%100==0:logs.append(dict(update=step+1,minibatch_loss=float(loss.item())))
    if actor!=digest_actor(agent):raise ValueError('Actor or fixed scales changed')
    return logs


def fit_seed(args):
    root,seed=args;root=Path(root);folder=root/f'seed{seed}';torch.set_num_threads(1)
    verify(json.loads((root/'dataset_manifest.json').read_text()))
    check_x,check_y=dataset(IDENTIFICATION,seed,'check');rows=[];broad=[];stock=[];predictions={}
    original=PPOAgent.load(BASELINE/f'seed{seed}/final.pt');original_actor=digest_actor(original)
    predictions['original']=errors(original,check_x,check_y)
    agents={'original':original}
    for arm in ARMS:
        x,y=fit_dataset(root,seed,arm)
        if len(x)!=17544:raise ValueError('Data budgets differ')
        agent=PPOAgent.load(BASELINE/f'seed{seed}/final.pt');logs=refit(agent,x,y,seed)
        path=folder/f'{arm}.pt';agent.save(path);pd.DataFrame(logs).to_csv(folder/f'{arm}_updates.csv',index=False)
        if arm=='concentrated':
            before=torch.load(IDENTIFICATION/f'seed{seed}/fitted_critic.pt',weights_only=True);after=torch.load(path,weights_only=True)
            if not same_state(before,after):raise ValueError('Concentrated control failed exact prior-checkpoint reproduction')
        agents[arm]=agent;error=errors(agent,check_x,check_y);predictions[arm]=error
        nearest=cKDTree(x.numpy()/np.array(SCALES)).query(check_x.numpy()/np.array(SCALES),workers=1)[0]
        np.savez_compressed(folder/f'{arm}_checking.npz',error=error,nearest=nearest)
        rows.append(dict(seed=seed,arm=arm,samples=len(x),updates=UPDATES,fit_mse=float(np.mean(errors(agent,x,y)**2)),checking_mse=float(np.mean(error**2)),
            checking_bias=float(error.mean()),checking_mae=float(abs(error).mean()),nearest_median=float(np.median(nearest)),nearest_p90=float(np.quantile(nearest,.9)),actor_frozen=digest_actor(agent)==original_actor))
    for arm,error in predictions.items():
        broad.append(dict(seed=seed,arm=arm,mse=float(np.mean(error**2)),bias=float(error.mean()),mae=float(abs(error).mean())))
        offset=0
        for years in CHECK_PAIRS:
            n=4*731;level=check_x[offset:offset+n,0].numpy();part=error[offset:offset+n];offset+=n
            for label,mask in [('all',np.ones(n,bool)),('low',level<-.45),('lower', (level>=-.45)&(level<.35)),('retained',(level>=.35)&(level<=1.24)),('flood',level>1.24)]:
                if mask.any():broad.append(dict(seed=seed,arm=arm,block_start=years[0],level_group=label,count=int(mask.sum()),mse=float(np.mean(part[mask]**2)),bias=float(part[mask].mean()),mae=float(abs(part[mask]).mean())))
    for years in CHECK_PAIRS:
        record=json.loads((IDENTIFICATION/f'seed{seed}/check_{years[0]}.json').read_text());truth=pd.read_csv(IDENTIFICATION/f'seed{seed}/stock_{years[0]}.csv',float_precision='round_trip')
        if not truth.complete.all():raise ValueError('Stock checking target incomplete')
        for ordinal in STOCK_ANCHORS:
            env,memory=restore(years,record['snapshots'][str(ordinal)]);group=truth[truth.ordinal.eq(ordinal)];returns=group.groupby('delta').return_995.mean();measured=float((returns[.15]-returns[0])/.15)
            for arm,agent in agents.items():
                values=[]
                for delta in [0.,.15]:
                    branch,changed=storage_branch(env,memory,delta);values.append(agent.value(changed.encode(branch._observation(),branch._info())));branch.close()
                stock.append(dict(seed=seed,arm=arm,block_start=years[0],ordinal=ordinal,predicted_per_m=(values[1]-values[0])/.15,measured_per_m=measured))
            env.close()
    pd.DataFrame(rows).to_csv(folder/'fit_metrics.csv',index=False);pd.DataFrame(broad).to_csv(folder/'checking_groups.csv',index=False);pd.DataFrame(stock).to_csv(folder/'stock.csv',index=False)
    save(folder/'fit_record.json',dict(seed=seed,complete=True,actor_frozen=True,concentrated_control_bitwise_equal=True))
    return f'seed {seed}: both refits complete, concentrated control exact'


def eligible(mse,slopes,wins):
    return bool(mse['diverse']<min(mse['concentrated'],mse['original']) and slopes['diverse']<min(slopes['concentrated'],slopes['original']) and wins>=4)


def summarize(root):
    fit=pd.concat([pd.read_csv(root/f'seed{s}/fit_metrics.csv') for s in SEEDS]);fit.to_csv(root/'fit_metrics.csv',index=False)
    groups=pd.concat([pd.read_csv(root/f'seed{s}/checking_groups.csv') for s in SEEDS]);groups.to_csv(root/'checking_groups.csv',index=False)
    stock=pd.concat([pd.read_csv(root/f'seed{s}/stock.csv') for s in SEEDS]);stock['absolute_error']=abs(stock.predicted_per_m-stock.measured_per_m);stock.to_csv(root/'stock.csv',index=False)
    if len(fit)!=10 or len(stock)!=180 or not fit.actor_frozen.all():raise ValueError('Required paired cohort incomplete')
    metrics=groups[groups.block_start.isna()].groupby('arm')[['mse','bias','mae']].mean()
    slope=stock.groupby('arm').absolute_error.mean();paired=fit.pivot(index='seed',columns='arm',values='checking_mse')
    gate=eligible(metrics.mse.to_dict(),slope.to_dict(),int((paired.diverse<paired.concentrated).sum()))
    result=dict(all_required_complete=True,actor_frozen_all=bool(fit.actor_frozen.all()),concentrated_controls_exact=True,
        checking={arm:{key:float(value) for key,value in row.items()} for arm,row in metrics.iterrows()},stock_slope_mae={arm:float(value) for arm,value in slope.items()},
        nearest={arm:{key:float(value) for key,value in row.items()} for arm,row in fit.groupby('arm')[['nearest_median','nearest_p90']].mean().iterrows()},
        seeds_with_lower_diverse_mse=int((paired.diverse<paired.concentrated).sum()),eligible_for_future_policy_experiment=gate,
        test_evaluated=False,policy_training=False,default_changed=False)
    save(root/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['prepare','collect','fit','summary','all'],default='all');args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))
    if args.stage in ['collect','all']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(collect_added,[(str(args.root),s,p) for s in SEEDS for p in ADDED_PAIRS]):print(result,flush=True)
    if args.stage in ['fit','all']:
        if not (args.root/'dataset_manifest.json').exists():seal_data(args.root)
        verify(json.loads((args.root/'dataset_manifest.json').read_text()))
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(fit_seed,[(str(args.root),s) for s in SEEDS]):print(result,flush=True)
    if args.stage in ['summary','all']:summarize(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))

if __name__=='__main__':main()
