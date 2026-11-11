"""Post hoc seed124 explanation with all five frozen mapped policies retained.

Common validation states isolate decision behavior from inherited storage.
Critics are checked only against their own stochastic-policy returns on common
TRAIN states, never against another policy's value or deterministic returns.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.simulation.ppo_coverage import ROOT,IDENTIFICATION,MAPPED,SEEDS,CHECK_PAIRS,STOCK_ANCHORS,save,verify
from reservoir_control.systems.lake_como.simulation.ppo_study import REFERENCE
from reservoir_control.systems.lake_como.simulation.ppo_protocol import ObservationHistory
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo import HydraulicPPOAgent
from reservoir_control.systems.lake_como.simulation.ppo_identification import restore
from reservoir_control.systems.lake_como.simulation.ppo_diagnosis import make_env,storage_branch,stochastic_return
from reservoir_control.systems.lake_como.simulation.target_year_check import validation_pairs
from reservoir_control.systems.lake_como.simulation.outlet import OutletDomainError

ANCHORS=(30,90,150,300,395,455,515,665)
REPS=4


def policy_branch(env,memory,agent,days=30):
    branch=deepcopy(env);past=deepcopy(memory);obs=branch._observation();info=branch._info();rows=[];failure=None
    try:
        for _ in range(days):
            action=agent.act(past.encode(obs,info),deterministic=True)[0]
            obs,reward,term,trunc,info=branch.step(action);hourly=info['hourly']
            rows.append(dict(cost=-reward,unmet=sum(r['unmet_demand_m3'] for r in hourly)/1e6,
                flood=sum(r['flood_excess_m_hours'] for r in hourly),low=sum(r['low_water_excess_m_hours'] for r in hourly)))
            if (term or trunc) and len(rows)!=days:raise ValueError('Unexpected branch cutoff')
    except (OutletDomainError,ValueError) as error:failure=str(error)
    complete=len(rows)==days and failure is None
    result=dict(complete=complete,observed_days=len(rows),failure=failure,end_level=float(obs[0]) if complete else None)
    result.update({key:float(sum(row[key] for row in rows)) if complete else None for key in ['cost','unmet','flood','low']})
    branch.close();return result


def validation_actions(root):
    torch.set_num_threads(1);agents={s:HydraulicPPOAgent.load(MAPPED/f'seed{s}/final.pt') for s in SEEDS};rows=[]
    for years in validation_pairs():
        a,b=years;reference=json.loads((REFERENCE/f'{a}_{b}__seed901__legacy/run.json').read_text())
        frame=pd.read_csv(Path(reference['data_directory'])/'daily.csv',float_precision='round_trip')
        env,obs,info=make_env('validation',years);memory=ObservationHistory()
        for day in range(max(ANCHORS)+1):
            encoded=memory.encode(obs,info)
            if day in ANCHORS:
                for seed,agent in agents.items():
                    with torch.no_grad():
                        latent=agent.model.distribution(torch.tensor(encoded).reshape(1,-1)).mean
                        nominal=float(agent.model.physical(latent).item())
                    request=float(agent.act(encoded,deterministic=True)[0][0])
                    branch=deepcopy(env);following,reward,_,_,following_info=branch.step(np.array([request],np.float32));branch.close()
                    high,changed=storage_branch(env,memory,.15);slope=(agent.value(changed.encode(high._observation(),high._info()))-agent.value(encoded))/.15;high.close()
                    result=policy_branch(env,memory,agent)
                    rows.append(dict(seed=seed,block_start=a,ordinal=day,level=float(obs[0]),nominal_fraction=nominal/float(env.action_space.high[0]),request=request,
                        first_release=float(np.mean([r['release_m3_s'] for r in following_info['hourly']])),first_cost=-reward,critic_value=agent.value(encoded),
                        predicted_storage_slope=slope,**{f'branch_{k}':v for k,v in result.items()}))
            if day<max(ANCHORS):
                obs,_,_,_,info=env.step(np.array([frame.request_m3_s.iloc[day]],np.float32))
                np.testing.assert_allclose(env.model.lake.storage_to_level(env.state.storage),frame.level_m.iloc[day],rtol=0,atol=1e-14)
        env.close()
    pd.DataFrame(rows).to_csv(root/'seed_common_actions.csv',index=False)


def critic_case(args):
    root,seed,years=args;root=Path(root);torch.set_num_threads(1);agent=HydraulicPPOAgent.load(MAPPED/f'seed{seed}/final.pt')
    source=json.loads((IDENTIFICATION/f'seed123/check_{years[0]}.json').read_text());rows=[]
    for ordinal in STOCK_ANCHORS:
        env,memory=restore(years,source['snapshots'][str(ordinal)])
        for delta in [0.,.15]:
            branch,past=storage_branch(env,memory,delta);value=agent.value(past.encode(branch._observation(),branch._info()));branch.close()
            for rep in range(REPS):
                result=stochastic_return(env,memory,agent,delta,years[0]*100000+ordinal*100+7000+rep)
                rows.append(dict(seed=seed,block_start=years[0],ordinal=ordinal,delta=delta,replicate=rep,value=value,**result))
        env.close()
    pd.DataFrame(rows).to_csv(root/f'seed_critic_{seed}_{years[0]}.csv',index=False)
    return f'mapped critic seed{seed}, block{years[0]} complete'


def summarize(root):
    actions=pd.read_csv(root/'seed_common_actions.csv',float_precision='round_trip')
    rollouts=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted(root.glob('seed_critic_*.csv'))]);rollouts.to_csv(root/'seed_critic_rollouts.csv',index=False)
    contrasts=[]
    for keys,group in rollouts.groupby(['seed','block_start','ordinal']):
        means=group.groupby('delta')[['value','return_995']].mean();complete=bool(len(group)==2*REPS and group.complete.all())
        contrasts.append(dict(seed=int(keys[0]),block_start=int(keys[1]),ordinal=int(keys[2]),complete=complete,
            predicted_slope=float((means.loc[.15,'value']-means.loc[0,'value'])/.15),
            measured_slope=float((means.loc[.15,'return_995']-means.loc[0,'return_995'])/.15) if complete else None,
            base_error=float(means.loc[0,'value']-means.loc[0,'return_995']) if complete else None))
    slopes=pd.DataFrame(contrasts);slopes['slope_absolute_error']=abs(slopes.predicted_slope-slopes.measured_slope);slopes.to_csv(root/'seed_critic_slopes.csv',index=False)
    annual=pd.read_csv(MAPPED/'annual.csv',float_precision='round_trip');annual.to_csv(root/'seed_factual_annual.csv',index=False)
    before=annual[annual.arm.eq('baseline')].set_index(['seed','block_start','year']);after=annual[annual.arm.eq('treatment')].set_index(['seed','block_start','year']).reindex(before.index)
    fields=['annual_cost','unmet_target_million_m3','flood_excess_m_hours','low_water_excess_m_hours','end_storage_million_m3','low_water_days','remaining_river_million_m3']
    delta=pd.DataFrame({key:after[key]-before[key] for key in fields});delta.to_csv(root/'seed_factual_deltas.csv')
    complete=bool(len(actions)==200 and actions.branch_complete.all() and len(rollouts)==480 and rollouts.complete.all())
    result=dict(all_required_complete=complete,common_action_rows=len(actions),stochastic_rollouts=len(rollouts),
        factual_delta_by_seed={str(seed):{key:float(value) for key,value in row.items()} for seed,row in delta.groupby(level='seed').mean().iterrows()},
        common_state_by_seed={str(seed):{key:float(group[key].mean()) if complete else None for key in ['request','first_release','branch_cost','branch_end_level','predicted_storage_slope']} for seed,group in actions.groupby('seed')},
        own_policy_critic_by_seed={str(seed):{key:float(group[key].mean()) if complete else None for key in ['base_error','slope_absolute_error','predicted_slope','measured_slope']} for seed,group in slopes.groupby('seed')},
        limitations='Seed124 was selected post hoc for diagnosis, not promoted. Branches share factual MPC901 stock and history and are conditional 30-day tests, not full-episode controller scores. Critic error includes realized-weather uncertainty and off-policy common-state coverage; raw V values cannot rank different policies.',
        test_evaluated=False,default_changed=False)
    save(root/'seed_gain_summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['actions','critic','summary','all'],default='all');args=parser.parse_args()
    verify(json.loads((args.root/'plan.json').read_text()))
    if args.stage in ['actions','all']:validation_actions(args.root);print('Common-state actions complete',flush=True)
    if args.stage in ['critic','all']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(critic_case,[(str(args.root),s,p) for s in SEEDS for p in CHECK_PAIRS]):print(result,flush=True)
    if args.stage in ['summary','all']:summarize(args.root)
    verify(json.loads((args.root/'plan.json').read_text()))

if __name__=='__main__':main()
