"""Read-only weather-adjusted on-policy progress, not checkpoint evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .ppo_study import ROOT as BASELINE,SEEDS,save
from .hydraulic_ppo_study import ROOT as MAPPED

SOURCES={'original':BASELINE,'mapped':MAPPED}


def fixed_effect_slope(frame):
    if len(frame)<2:return None
    centered=frame[['step','cost']]-frame.groupby('window')[['step','cost']].transform('mean')
    denominator=float((centered.step**2).sum())
    return float((centered.step*centered.cost).sum()/denominator) if denominator>0 else None


def episode_frame(record):
    if not record['complete'] or record['steps']!=200000:raise ValueError('Complete declared 200k run required')
    rows=[]
    for index,episode in enumerate(record['episodes']):
        if episode['length']!=730:raise ValueError('Full finite two-year episodes required')
        rows.append(dict(step=(index+1)*730,cost=-episode['physical_return'],window='_'.join(map(str,episode['years']))))
    if len(rows)*730+record['unfinished_length']!=record['steps']:raise ValueError('Episode interaction accounting mismatch')
    frame=pd.DataFrame(rows)
    frame['adjusted_cost']=frame.cost-frame.groupby('window').cost.transform('mean')+frame.cost.mean()
    return frame


def duration_signal(slopes):
    late=slopes[slopes.period.eq('late')];last=slopes[slopes.period.eq('last50k')]
    if len(late)!=5 or len(last)!=5:raise ValueError('All five learner seeds required')
    improved=int(late.fraction_per_100k.le(-.02).sum());still_decreasing=int(last.change_per_100k.lt(0).sum())
    return dict(late_seeds_improving_at_least_2pct_per_100k=improved,last50k_seeds_decreasing=still_decreasing,
        duration_probe_supported=bool(improved>=3 and still_decreasing>=3))


def run(root):
    records=[];updates=[]
    for model,source in SOURCES.items():
        for seed in SEEDS:
            status=json.loads((source/f'seed{seed}/training.json').read_text());frame=episode_frame(status)
            frame['model']=model;frame['seed']=seed;frame.to_csv(root/f'curve_episodes_{model}_{seed}.csv',index=False)
            log=pd.read_csv(source/f'seed{seed}/updates.csv',float_precision='round_trip')
            if log.step.iloc[-1]!=200000 or not np.isfinite(log.select_dtypes('number')).all().all():raise ValueError('Finite complete update log required')
            for lo,hi,period in [(0,100000,'early'),(100000,200000,'late'),(150000,200000,'last50k')]:
                group=frame[frame.step.gt(lo)&frame.step.le(hi)];slope=fixed_effect_slope(group)
                records.append(dict(model=model,seed=seed,period=period,episodes=len(group),windows=group.window.nunique(),mean_cost=float(group.cost.mean()),
                    change_per_100k=None if slope is None else slope*100000,fraction_per_100k=None if slope is None else slope*100000/group.cost.mean(),
                    full_episode_steps=len(frame)*730,excluded_unfinished_steps=status['unfinished_length']))
            for lo,hi in [(0,50000),(50000,100000),(100000,150000),(150000,200000)]:
                group=log[log.step.gt(lo)&log.step.le(hi)]
                row=dict(model=model,seed=seed,step_start=lo,step_end=hi)
                row.update({key:float(group[key].mean()) for key in ['value_loss','approx_kl','actor_gradient_norm','critic_gradient_norm','clip_fraction','log_std']})
                row.update(kl_stop_fraction=float(group.kl_early_stop.mean()),mean_gaussian_std=float(np.exp(group.log_std.clip(-5,1)).mean()),
                    minibatch_updates=int(group.minibatch_updates.sum()))
                updates.append(row)
    slopes=pd.DataFrame(records);slopes.to_csv(root/'curve_slopes.csv',index=False);pd.DataFrame(updates).to_csv(root/'curve_updates.csv',index=False)
    mapped=slopes[slopes.model.eq('mapped')];signal=duration_signal(mapped)
    result=dict(**signal,
        mapped_late_fraction_by_seed={str(int(r.seed)):float(r.fraction_per_100k) for r in mapped[mapped.period.eq('late')].itertuples()},
        mapped_last50k_fraction_by_seed={str(int(r.seed)):float(r.fraction_per_100k) for r in mapped[mapped.period.eq('last50k')].itertuples()},
        interpretation='Within-window linear trends in changing stochastic policies. Fixed effects adjust sampled weather-window composition, not exploration/optimization noise. No intermediate checkpoints exist; no fixed-policy convergence claim. Signal permits a future bounded duration probe, not automatic training or default promotion.',
        test_evaluated=False,new_training=False)
    save(root/'curve_summary.json',result);print(json.dumps(result,indent=2),flush=True)
