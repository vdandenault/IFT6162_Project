"""Matched-state actions, weather-adjusted learning curves, stochastic critic audit.

Frozen policies only. Future forcing is used exclusively to score retrospective
branches, never passed to either controller. Original experiment files are read-only.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT
from .ppo_study import ROOT as BASELINE,REFERENCE,SEEDS,save,verify
from .ppo_agent import PPOAgent
from .ppo_protocol import MatchedTrainingEpisodes,ObservationHistory
from .low_water import make_protected_outlet_env
from .model import WaterState
from .outlet import OutletDomainError
from .weighted_study import make_controller
from .target_year_check import validation_pairs

ROOT=PROJECT_ROOT/'results/simulator/ppo_diagnosis_v1'
ANCHORS=(30,90,150,300,395,455,515,665)
TRAIN_PAIRS=((1960,1961),(1976,1977),(1984,1985))
CRITIC_ANCHORS=(30,120,395,485)
REPLICATES=8


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    base=json.loads((BASELINE/'plan.json').read_text());verify(base)
    paths=[Path(__file__),BASELINE/'plan.json',BASELINE/'summary.json']
    for seed in SEEDS:
        paths += list((BASELINE/f'seed{seed}').glob('*'))
    for a,b in validation_pairs():
        path=REFERENCE/f'{a}_{b}__seed901__legacy/run.json';paths.append(path)
        folder=Path(json.loads(path.read_text())['data_directory'])
        paths += [folder/'daily.csv',folder/'decisions.jsonl']
    hashes=dict(base['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
    save(root/'plan.json',dict(source_sha256=hashes,seeds=list(SEEDS),validation_pairs=[list(p) for p in validation_pairs()],
        action_anchors=list(ANCHORS),training_pairs=[list(p) for p in TRAIN_PAIRS],critic_anchors=list(CRITIC_ANCHORS),replicates=REPLICATES,
        actions='Replay exact saved commands from reset. From each PPO own state and MPC901 state, query the same PPO and history MPC901. Prime seven observed flows. Use common reference previous MPC plan; also query a cold-plan control. No policy receives future forcing.',
        branches='PPO mean versus MPC first action, then the SAME deterministic PPO continuation for 30 days. Branch from identical full state, history and clock. Diagnostic only, not a full-controller ranking.',
        critic='Training pairs only. Reconstruct each frozen PPO stochastic trajectory with collection seed 20000+policy seed. At fixed anchors perturb storage by +/-0.15 m only. Eight full-remaining-episode stochastic rollouts per stock, paired random draws. Compare discounted scaled returns at gamma .995 and .999 with V and dV/dh. No terminal bootstrap.',
        curves='Within-window fixed-effect linear slopes on completed stochastic training episode costs; late half 100k..200k. Weather-composition adjusted, still on-policy and not a fixed-policy learning curve. No earlier checkpoints exist.',
        selection='Choose one controlled training experiment only after these diagnostics; lock its plan before any new training/evaluation. No automatic best-seed or checkpoint selection.',
        failures='Incomplete branches have no full-window return. Required critic groups suppress their mean if any replicate incomplete. Report failures; do not extrapolate model domain.',
        test_evaluated=False,costs_changed=False,default_changed=False))


def curve_summary(root):
    rows=[];windows=[]
    for seed in SEEDS:
        record=json.loads((BASELINE/f'seed{seed}/training.json').read_text())
        frame=pd.DataFrame([dict(step=(i+1)*730,cost=-episode['physical_return'],window='_'.join(map(str,episode['years']))) for i,episode in enumerate(record['episodes'])])
        frame.to_csv(root/f'episodes_seed{seed}.csv',index=False)
        updates=pd.read_csv(BASELINE/f'seed{seed}/updates.csv')
        for low,high,label in [(0,100000,'early'),(100000,200000,'late')]:
            group=frame[frame.step.gt(low)&frame.step.le(high)].copy()
            centered=group[['step','cost']]-group.groupby('window')[['step','cost']].transform('mean')
            denominator=float((centered.step**2).sum())
            slope=float((centered.step*centered.cost).sum()/denominator) if denominator>0 else None
            rows.append(dict(seed=seed,period=label,episodes=len(group),windows=group.window.nunique(),mean_cost=float(group.cost.mean()),
                weather_adjusted_change_per_100k=None if slope is None else slope*100000,
                adjusted_fraction_per_100k=None if slope is None else slope*100000/group.cost.mean()))
        for low,high in [(0,50000),(50000,100000),(100000,150000),(150000,200000)]:
            group=updates[updates.step.gt(low)&updates.step.le(high)]
            windows.append(dict(seed=seed,step_start=low,step_end=high,**{key:float(group[key].mean()) for key in
                ['value_loss','approx_kl','actor_gradient_norm','critic_gradient_norm','clip_fraction','log_std']}))
    pd.DataFrame(rows).to_csv(root/'curve_slopes.csv',index=False)
    pd.DataFrame(windows).to_csv(root/'update_windows.csv',index=False)


def make_env(split,years):
    env=make_protected_outlet_env(split,tuple(years));env.fallback=MatchedTrainingEpisodes().prior_last_flow
    obs,info=env.reset(seed=901);return env,obs,info


def restore_mpc(history,hour,previous_plan):
    controller,_=make_controller(settings('train'),'legacy',901)
    controller.prime_history(history.history,int(hour))
    if previous_plan is not None:
        controller.previous_hour=hour-24
        controller.previous_plan=np.asarray(previous_plan,np.float32)
    return controller


def first_action_branch(env,history,agent,request,days=30):
    branch=deepcopy(env);memory=deepcopy(history);obs=branch._observation();info=branch._info()
    rows=[];failure=None;expected=min(days,(branch.total_hours-branch.hour)//24)
    try:
        for index in range(expected):
            action=np.array([request],np.float32) if index==0 else agent.act(memory.encode(obs,info),deterministic=True)[0]
            # Encode the first observation too: branch memory must advance on every day.
            memory.encode(obs,info)
            obs,reward,term,trunc,info=branch.step(action)
            hourly=info['hourly']
            rows.append(dict(cost=-reward,unmet=sum(r['unmet_demand_m3'] for r in hourly)/1e6,
                flood=sum(r['flood_excess_m_hours'] for r in hourly),low=sum(r['low_water_excess_m_hours'] for r in hourly)))
    except OutletDomainError as error:failure=str(error)
    complete=len(rows)==expected and failure is None
    result=dict(complete=complete,days=len(rows),failure=failure,end_level=float(obs[0]))
    for key in ['cost','unmet','flood','low']:result[key]=float(sum(row[key] for row in rows)) if complete else None
    branch.close();return result


def audit_case(args):
    root,seed,years=args;root=Path(root);torch.set_num_threads(1)
    agent=PPOAgent.load(BASELINE/f'seed{seed}/final.pt');a,b=years
    reference=json.loads((REFERENCE/f'{a}_{b}__seed901__legacy/run.json').read_text())
    directory=Path(reference['data_directory']);mpc_frame=pd.read_csv(directory/'daily.csv',float_precision='round_trip')
    decisions=[json.loads(line) for line in (directory/'decisions.jsonl').read_text().splitlines()]
    rows=[]
    for origin in ['ppo','mpc']:
        factual=pd.read_csv(BASELINE/f'seed{seed}/{a}_{b}.csv',float_precision='round_trip') if origin=='ppo' else mpc_frame
        env,obs,info=make_env('validation',years);history=ObservationHistory()
        for day in range(max(ANCHORS)+1):
            encoded=history.encode(obs,info)
            if day in ANCHORS:
                previous=decisions[day-1]['selected_plan']
                warm=restore_mpc(history,env.hour,previous);mpc=float(warm(obs,info)[0])
                cold=restore_mpc(history,env.hour,None);cold_request=float(cold(obs,info)[0])
                ppo=float(agent.act(encoded,deterministic=True)[0][0])
                ppo_branch=first_action_branch(env,history,agent,ppo)
                mpc_branch=first_action_branch(env,history,agent,mpc)
                past=factual.inflow_m3_s.iloc[max(0,day-30):day].mean();future=factual.inflow_m3_s.iloc[day:day+30].mean()
                row=dict(seed=seed,block_start=a,origin=origin,ordinal=day,year=years[day//365],model_day=day%365+1,
                    level=float(obs[0]),past_target=float(obs[3]),last_inflow=float(obs[4]),past_30day_inflow=float(past),
                    retrospective_refill_label=bool(future>1.5*past),ppo_request=ppo,mpc_request=mpc,cold_mpc_request=cold_request,
                    request_difference=ppo-mpc,critic_value=agent.value(encoded))
                row.update({f'{arm}_{key}':value for arm,result in [('ppo_first',ppo_branch),('mpc_first',mpc_branch)] for key,value in result.items()})
                row['delta_30day_cost']=ppo_branch['cost']-mpc_branch['cost'] if ppo_branch['complete'] and mpc_branch['complete'] else None
                rows.append(row)
            if day<max(ANCHORS):
                obs,_,_,_,info=env.step(np.array([factual.request_m3_s.iloc[day]],np.float32))
                np.testing.assert_allclose(env.model.lake.storage_to_level(env.state.storage),factual.level_m.iloc[day],rtol=0,atol=1e-14)
        env.close()
    path=root/f'actions_seed{seed}_{a}.csv';pd.DataFrame(rows).to_csv(path,index=False)
    return str(path)


def discounted_returns(rewards,gamma):
    values=np.empty(len(rewards),float);running=0.
    for index in range(len(rewards)-1,-1,-1):
        running=float(rewards[index])+gamma*running;values[index]=running
    return values


def storage_branch(env,history,delta):
    branch=deepcopy(env);memory=deepcopy(history)
    level=branch.model.lake.storage_to_level(branch.state.storage)+delta
    branch.model.lake.check_domain(level)
    branch.state=WaterState(branch.model.lake.level_to_storage(level),branch.state.command,branch.state.river_storage)
    # Replace current cached observation only; preserve its measured history.
    memory.last_raw=branch._observation().copy();memory.last_encoded=memory.last_encoded.copy()
    memory.last_encoded[0]=memory.last_raw[0]
    return branch,memory


def stochastic_return(env,history,agent,delta,random_seed):
    branch,memory=storage_branch(env,history,delta);torch.manual_seed(random_seed)
    obs=branch._observation();info=branch._info();rewards=[];failure=None
    expected=(branch.total_hours-branch.hour)//24
    try:
        while branch.hour<branch.total_hours:
            encoded=memory.encode(obs,info);action=agent.act(encoded)[0]
            obs,reward,_,_,info=branch.step(action);rewards.append(reward*agent.config.reward_scale)
    except OutletDomainError as error:failure=str(error)
    complete=len(rewards)==expected and failure is None
    result=dict(complete=complete,days=len(rewards),failure=failure)
    for gamma,label in [(.995,'return_995'),(.999,'return_999')]:
        result[label]=float(discounted_returns(rewards,gamma)[0]) if complete else None
    branch.close();return result


def critic_case(args):
    root,seed,years=args;root=Path(root);torch.set_num_threads(1)
    agent=PPOAgent.load(BASELINE/f'seed{seed}/final.pt')
    env,obs,info=make_env('train',years);history=ObservationHistory();torch.manual_seed(20000+seed)
    rows=[]
    for day in range(max(CRITIC_ANCHORS)+1):
        encoded=history.encode(obs,info)
        if day in CRITIC_ANCHORS:
            collection_rng=torch.get_rng_state()
            for delta in [-.15,0.,.15]:
                perturbed,memory=storage_branch(env,history,delta)
                value=agent.value(memory.encode(perturbed._observation(),perturbed._info()));perturbed.close()
                for rep in range(REPLICATES):
                    result=stochastic_return(env,history,agent,delta,30000+seed*1000+day*10+rep)
                    rows.append(dict(seed=seed,block_start=years[0],ordinal=day,replicate=rep,delta_level=delta,
                        original_level=float(obs[0]),value=value,**result))
            torch.set_rng_state(collection_rng)
        if day<max(CRITIC_ANCHORS):
            action=agent.act(encoded)[0];obs,_,_,_,info=env.step(action)
    env.close();path=root/f'critic_seed{seed}_{years[0]}.csv';pd.DataFrame(rows).to_csv(path,index=False);return str(path)


def summarize(root):
    actions=pd.concat([pd.read_csv(p) for p in sorted(root.glob('actions_seed*.csv'))],ignore_index=True)
    actions.to_csv(root/'actions.csv',index=False)
    critic=pd.concat([pd.read_csv(p) for p in sorted(root.glob('critic_seed*.csv'))],ignore_index=True)
    critic.to_csv(root/'critic_rollouts.csv',index=False)
    grouped=critic.groupby(['seed','block_start','ordinal','delta_level'])
    means=grouped.agg(complete=('complete','all'),count=('complete','size'),value=('value','first'),
        return_995=('return_995','mean'),return_999=('return_999','mean'),mc_sd=('return_995','std')).reset_index()
    for name in ['return_995','return_999','mc_sd']:
        means.loc[~means.complete|means['count'].ne(REPLICATES),name]=np.nan
    means['error']=means.value-means.return_995;means.to_csv(root/'critic_means.csv',index=False)
    slopes=[]
    for keys,group in means.groupby(['seed','block_start','ordinal']):
        group=group.set_index('delta_level');complete=bool(group.complete.all() and group['count'].eq(REPLICATES).all())
        row=dict(zip(['seed','block_start','ordinal'],map(int,keys)));row['complete']=complete
        for field in ['value','return_995','return_999']:
            row[field+'_per_m']=(float(group.loc[.15,field])-float(group.loc[-.15,field]))/.3 if complete else None
        slopes.append(row)
    slopes=pd.DataFrame(slopes);slopes.to_csv(root/'storage_values.csv',index=False)
    valid=means[means.delta_level.eq(0)&means.complete]
    action_complete=actions.ppo_first_complete.all() and actions.mpc_first_complete.all()
    result=dict(action_rows=len(actions),complete_action_pairs=int((actions.ppo_first_complete&actions.mpc_first_complete).sum()),
        mean_request_difference=float(actions.request_difference.mean()),
        mean_30day_cost_delta=float(actions.delta_30day_cost.mean()) if action_complete else None,
        critic_complete=bool(critic.complete.all()),critic_rollouts=len(critic),
        critic_bias=float(valid.error.mean()) if len(valid)==60 else None,
        critic_rmse=float(np.sqrt((valid.error**2).mean())) if len(valid)==60 else None,
        storage_slope_sign_agreement=float((np.sign(slopes.value_per_m)==np.sign(slopes.return_995_per_m)).mean()) if slopes.complete.all() else None,
        median_predicted_storage_value=float(slopes.value_per_m.median()) if slopes.complete.all() else None,
        median_measured_storage_value_995=float(slopes.return_995_per_m.median()) if slopes.complete.all() else None,
        median_measured_storage_value_999=float(slopes.return_999_per_m.median()) if slopes.complete.all() else None,
        test_evaluated=False,default_changed=False)
    save(root/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['prepare','actions','critic','summary','all'],default='all');args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare(args.root);curve_summary(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))
    if args.stage in ['actions','all']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for path in pool.map(audit_case,[(str(args.root),s,p) for s in SEEDS for p in validation_pairs()]):print(path,flush=True)
    if args.stage in ['critic','all']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for path in pool.map(critic_case,[(str(args.root),s,p) for s in SEEDS for p in TRAIN_PAIRS]):print(path,flush=True)
    if args.stage in ['summary','all']:summarize(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))

if __name__=='__main__':main()
