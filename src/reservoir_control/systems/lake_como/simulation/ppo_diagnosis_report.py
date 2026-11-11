"""Derived physical-action and paired-noise summaries; never modifies a policy."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.simulation.ppo_diagnosis import ROOT,ANCHORS,BASELINE,REFERENCE,SEEDS,make_env
from reservoir_control.systems.lake_como.simulation.target_year_check import validation_pairs
from reservoir_control.systems.lake_como.simulation.ppo_study import save


def first_day_physics(root):
    rows=[]
    for seed in SEEDS:
        for a,b in validation_pairs():
            audit=pd.read_csv(root/f'actions_seed{seed}_{a}.csv')
            reference=json.loads((REFERENCE/f'{a}_{b}__seed901__legacy/run.json').read_text())
            for origin in ['ppo','mpc']:
                path=BASELINE/f'seed{seed}/{a}_{b}.csv' if origin=='ppo' else Path(reference['data_directory'])/'daily.csv'
                frame=pd.read_csv(path,float_precision='round_trip');env,obs,info=make_env('validation',(a,b))
                for day in range(max(ANCHORS)+1):
                    if day in ANCHORS:
                        decision=audit[audit.origin.eq(origin)&audit.ordinal.eq(day)].iloc[0]
                        row=dict(seed=seed,block_start=a,origin=origin,ordinal=day,level=float(obs[0]))
                        for arm in ['ppo','mpc']:
                            branch=deepcopy(env);request=float(decision[arm+'_request'])
                            following,reward,_,_,metadata=branch.step(np.array([request],np.float32))
                            hourly=metadata['hourly']
                            row.update({arm+'_'+key:value for key,value in dict(request=request,release=float(np.mean([r['release_m3_s'] for r in hourly])),
                                cost=-reward,unmet=sum(r['unmet_demand_m3'] for r in hourly)/1e6,
                                remaining_river=sum(r['river_out_m3'] for r in hourly)/1e6,end_level=float(following[0])).items()})
                            branch.close()
                        row.update(actual_release_difference=row['ppo_release']-row['mpc_release'],
                            first_day_cost_difference=row['ppo_cost']-row['mpc_cost'],
                            requests_aliased=abs(row['ppo_release']-row['mpc_release'])<1e-5)
                        rows.append(row)
                    if day<max(ANCHORS):obs,_,_,_,info=env.step(np.array([frame.request_m3_s.iloc[day]],np.float32))
                env.close()
    frame=pd.DataFrame(rows);frame.to_csv(root/'first_day_physics.csv',index=False)
    return frame


def report(root):
    physics=first_day_physics(root);critic=pd.read_csv(root/'critic_rollouts.csv')
    slopes=pd.read_csv(root/'storage_values.csv')
    pairs=critic.pivot(index=['seed','block_start','ordinal','replicate'],columns='delta_level',values=['return_995','return_999'])
    paired=[]
    for keys,group in pairs.groupby(level=[0,1,2]):
        row=dict(zip(['seed','block_start','ordinal'],map(int,keys)))
        for field in ['return_995','return_999']:
            differences=(group[field][.15]-group[field][-.15])/.3
            row[field+'_paired_se']=float(differences.std(ddof=1)/np.sqrt(len(differences))) if differences.notna().all() and len(differences)==8 else None
        paired.append(row)
    joined=slopes.merge(pd.DataFrame(paired),on=['seed','block_start','ordinal'],validate='one_to_one')
    joined['resolved_action_noise_sign']=joined.return_995_per_m.abs()>2*joined.return_995_paired_se
    joined.to_csv(root/'storage_value_uncertainty.csv',index=False)
    means=pd.read_csv(root/'critic_means.csv');baseline=means[means.delta_level.eq(0)]
    positive=[]
    for keys,group in critic.groupby(['seed','block_start','ordinal']):
        averages=group.groupby('delta_level')[['value','return_995','return_999']].mean()
        row=dict(zip(['seed','block_start','ordinal'],map(int,keys)))
        row['original_level']=float(group.original_level.iloc[0])
        row['minus_branch_below_protected_floor']=bool(row['original_level']-.15<-.5)
        for field in ['value','return_995','return_999']:
            row[field+'_per_m']=float((averages.loc[.15,field]-averages.loc[0,field])/.15)
        noise=group.pivot(index='replicate',columns='delta_level',values='return_995')
        row['paired_se_per_m']=float(((noise[.15]-noise[0])/.15).std(ddof=1)/np.sqrt(len(noise)))
        positive.append(row)
    positive=pd.DataFrame(positive)
    positive.to_csv(root/'positive_storage_values.csv',index=False)

    summary=dict(physical_action_rows=len(physics),aliased_requests=int(physics.requests_aliased.sum()),
        physical_by_origin={str(origin):{key:float(group[key].mean()) for key in ['actual_release_difference','first_day_cost_difference','ppo_remaining_river','mpc_remaining_river']} for origin,group in physics.groupby('origin')},
        critic_conditional_weather_warning='Rollouts share a fixed recorded future; noise estimates cover action draws only. They do not measure unknown-weather uncertainty or independent hydrological evidence.',
        median_action_noise_se=float(baseline.mc_sd.median()/np.sqrt(8)),
        resolved_slope_rows=int(joined.resolved_action_noise_sign.sum()),
        resolved_slope_sign_agreement=float((np.sign(joined[joined.resolved_action_noise_sign].value_per_m)==np.sign(joined[joined.resolved_action_noise_sign].return_995_per_m)).mean()) if joined.resolved_action_noise_sign.any() else None,
        positive_stock_cases=len(positive),
        minus_branch_below_protected_floor=int(positive.minus_branch_below_protected_floor.sum()),
        median_positive_predicted_storage_value=float(positive.value_per_m.median()),
        median_positive_measured_storage_value_995=float(positive.return_995_per_m.median()),
        median_positive_measured_storage_value_999=float(positive.return_999_per_m.median()),
        positive_slope_sign_agreement=float((np.sign(positive.value_per_m)==np.sign(positive.return_995_per_m)).mean()),
        interpretation='Use base -> +.15m for reachable-stock interpretation. The minus-stock central differences include states below the protected floor and remain labeled out-of-support sensitivities.',
        test_evaluated=False)
    save(root/'derived_summary.json',summary)
    save(root/'derived_sources.json',{'source_sha256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();report(args.root)
