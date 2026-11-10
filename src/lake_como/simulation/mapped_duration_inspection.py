"""Read-only inspection of completed mapped-PPO duration logs and trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT
from .mapped_ppo_duration import DEFAULT as SOURCE,training_status,full_annual
from .ppo_study import SEEDS,save,verify
from .ppo_curve_audit import fixed_effect_slope
from .target_year_check import validation_pairs

DEFAULT=PROJECT_ROOT/'results/simulator/mapped_duration_inspection_v1'
PERIODS={'before_extension':(100000,200000),'extension':(200000,400000),'recent100k':(300000,400000),'last50k':(350000,400000)}


def episodes(record):
    if not record['complete'] or not record['prefix_verified'] or record['steps']!=400000:raise ValueError('Complete verified 400k run required')
    rows=[]
    for index,item in enumerate(record['episodes']):
        if item['length']!=730:raise ValueError('Complete finite-task episodes required')
        rows.append(dict(start=index*730,step=(index+1)*730,cost=-item['physical_return'],window='_'.join(map(str,item['years']))))
    if len(rows)*730+record['unfinished_length']!=400000:raise ValueError('Episode budget mismatch')
    frame=pd.DataFrame(rows);frame['adjusted_cost']=frame.cost-frame.groupby('window').cost.transform('mean')+frame.cost.mean()
    return frame


def select_period(frame,lo,hi,contained=True):
    return frame[(frame.start.ge(lo) if contained else frame.step.gt(lo))&frame.step.le(hi)]


def trend_rows(frame,seed):
    rows=[]
    for period,(lo,hi) in PERIODS.items():
        for convention in ['whole_episodes','ending_in_period']:
            group=select_period(frame,lo,hi,contained=convention=='whole_episodes');slope=fixed_effect_slope(group)
            rows.append(dict(seed=seed,period=period,convention=convention,episodes=len(group),windows=group.window.nunique(),
                mean_episode_cost=float(group.cost.mean()),change_per_100k=None if slope is None else slope*100000,
                fraction_per_100k=None if slope is None else slope*100000/group.cost.mean()))
    return rows


def prepare(source,output):
    source=source.resolve();output=output.resolve()
    plan=json.loads((source/'plan.json').read_text());verify(plan)
    if output==source or output.is_relative_to(source):raise ValueError('Inspection outputs must stay outside the frozen study')
    paths=[Path(__file__),source/'plan.json',source/'summary.json',source/'annual.csv',source/'paired_seed_years.csv']
    for seed in SEEDS:paths += [p for p in (source/f'seed{seed}').iterdir() if p.is_file()]
    hashes=dict(plan['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output.mkdir(parents=True,exist_ok=False)
    save(output/'plan.json',dict(source_sha256=hashes,source=str(source),periods={k:list(v) for k,v in PERIODS.items()},
        analysis='Read-only completed logs. Primary fixed-effect trends use whole episodes contained in each interval. Ending-in-period convention is reported as a sensitivity for consistency with earlier audits.',
        limitations='Changing stochastic on-policy returns, correlated episodes and shrinking exploration; no intermediate saved-policy validation curve or convergence claim.',
        no_800k_extension=True,new_training=False,test_evaluated=False,default_changed=False))
    return plan


def inspect(source,output,plan):
    reference=Path(plan['reference']);slopes=[];updates=[];counts=[];verified_days=0
    for seed in SEEDS:
        folder=source/f'seed{seed}';record=training_status(folder,reference)
        frame=episodes(record);frame.to_csv(output/f'episodes_seed{seed}.csv',index=False);slopes.extend(trend_rows(frame,seed))
        counts.append(dict(seed=seed,completed_episodes=len(frame),unfinished_steps=record['unfinished_length'],
            extension_whole_episodes=len(select_period(frame,200000,400000)),
            extension_straddling_episode_days=int(((frame.start<200000)&(frame.step>200000)).sum())*730))
        log=pd.read_csv(folder/'updates.csv',float_precision='round_trip')
        if log.step.iloc[-1]!=400000 or not np.isfinite(log.select_dtypes('number')).all().all():raise ValueError('Finite complete optimizer logs required')
        for lo,hi in [(100000,200000),(200000,300000),(300000,400000),(350000,400000)]:
            group=log[log.step.gt(lo)&log.step.le(hi)]
            updates.append(dict(seed=seed,from_step=lo,to_step=hi,**{key:float(group[key].mean()) for key in
                ['value_loss','approx_kl','actor_gradient_norm','critic_gradient_norm','clip_fraction']},
                gaussian_std=float(np.exp(group.log_std.clip(-5,1)).mean()),kl_stop_fraction=float(group.kl_early_stop.mean()),
                minibatch_updates=int(group.minibatch_updates.sum())))
        area=settings('train')['lake_params']['surface']
        for a,b in validation_pairs():
            daily=pd.read_csv(folder/f'{a}_{b}.csv',float_precision='round_trip');status=json.loads((folder/f'{a}_{b}.json').read_text())
            if len(daily)!=730 or not status['complete']:raise ValueError('Complete endpoint validation required')
            np.testing.assert_allclose(-daily.reward,daily.unmet_target_m3/1e6+daily.flood_excess_m_hours+daily.low_water_excess_m_hours,rtol=0,atol=1e-12)
            np.testing.assert_allclose(area*(daily.level_m-daily.level_before_m),(daily.inflow_m3_s-daily.release_m3_s)*86400,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(daily.level_before_m.iloc[1:].to_numpy(),daily.level_m.iloc[:-1].to_numpy())
            verified_days+=len(daily)
    slopes=pd.DataFrame(slopes);slopes.to_csv(output/'trends.csv',index=False);pd.DataFrame(updates).to_csv(output/'updates.csv',index=False);pd.DataFrame(counts).to_csv(output/'episode_accounting.csv',index=False)
    annual=pd.read_csv(source/'annual.csv',float_precision='round_trip')
    for arm in ['200k','400k']:
        if not full_annual(annual[annual.arm.eq(arm)]):raise ValueError('Full paired endpoints required')
    metrics=annual.groupby('arm')[['annual_cost','unmet_target_million_m3','flood_excess_m_hours','low_water_excess_m_hours','flood_days','low_water_days','end_storage_million_m3']].mean()
    metrics.to_csv(output/'endpoint_metrics.csv')
    pd.DataFrame([dict(arm=arm,episode_end_storage_million_m3=float(group[group.position_in_pair.eq(2)].end_storage_million_m3.mean())) for arm,group in annual.groupby('arm')]).to_csv(output/'episode_end_storage.csv',index=False)
    primary=slopes[slopes.convention.eq('whole_episodes')]
    result=dict(source=str(source),saved_optimizer_rows_per_seed=196,primary_trend_convention='whole_episodes',
        whole_episode_fraction_per_100k={period:{str(int(row.seed)):float(row.fraction_per_100k) for row in group.itertuples()} for period,group in primary.groupby('period')},
        endtime_convention_fraction_per_100k={period:{str(int(row.seed)):float(row.fraction_per_100k) for row in group.itertuples()} for period,group in slopes[slopes.convention.eq('ending_in_period')].groupby('period')},
        extension_completed_episode_counts={str(row['seed']):row['extension_whole_episodes'] for row in counts},
        verified_endpoint_daily_records=verified_days,metrics={str(arm):{k:float(v) for k,v in row.items()} for arm,row in metrics.iterrows()},
        decision='Record the 400k experimental endpoint and stop the duration sweep; no 800k run. Do not infer convergence or select intermediate checkpoints from changing-policy logs.',
        new_training=False,test_evaluated=False,default_changed=False)
    save(output/'summary.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args()
    plan=prepare(args.source,args.output);inspect(args.source,args.output,plan);verify(json.loads((args.output/'plan.json').read_text()))

if __name__=='__main__':main()
