"""Duration-only extension of the frozen MPC-residual/full-action pilot.

Uses the unchanged pilot trainer/evaluator. Restart from identical seeds because
short checkpoints do not contain replay/environment state for exact resumption.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como import residual_study as pilot

SHORT=PROJECT_ROOT/'results/residual/pilot_v1'


def extended_config(original):
    result=deepcopy(original);result['steps']=36500
    if [k for k in result if result[k]!=original[k]]!=['steps']:
        raise ValueError('Only the interaction budget may change')
    return result


def check_prefix(short,long):
    if len(long)<len(short): raise ValueError('Extended prefix incomplete')
    pd.testing.assert_frame_equal(short.reset_index(drop=True),long.iloc[:len(short)].reset_index(drop=True),check_exact=True)


def prepare(root):
    if root.exists() and any(root.iterdir()): raise ValueError('Choose an empty new output')
    original=pilot.verify(SHORT)
    if not json.loads((SHORT/'run.json').read_text())['complete']: raise ValueError('Completed pilot required')
    root.mkdir(parents=True,exist_ok=True)
    plan=deepcopy(original);plan['config']=extended_config(original['config'])
    plan['scope']='Duration-only matched extension, 100 annual episodes/seed. Restart from identical initial weights/seeds; no validation checkpoint selection.'
    plan['budget']='36500 steps incl unchanged 365-step base-only warmup, 36135 SAC updates per arm/seed. Replay remains 3650. Six jobs; daily MPC cost counted.'
    plan['comparison']='Same 3 seeds, arms, warmup, bound, architecture, optimizer, capacity, profiles, observations/reward/physics as short pilot. Only steps differ.'
    files=[Path(__file__),SHORT/'plan.json',SHORT/'run.json',SHORT/'smoke.json',SHORT/'annual.csv',SHORT/'summary.csv']
    for mode in plan['config']['modes']:
        for seed in plan['config']['seeds']:
            files += [SHORT/f'{mode}_{seed}/training_episodes.csv',SHORT/f'{mode}_{seed}/final.pt']
    plan['source_sha256'].update({str(p.relative_to(PROJECT_ROOT)):digest(p) for p in files})
    pilot.write_json(root/'plan.json',plan)
    smoke=json.loads((SHORT/'smoke.json').read_text())
    smoke['reused_from']=str(SHORT/'smoke.json')
    smoke['reason']='Training/evaluation implementation and smoke configuration unchanged; only full-study duration differs.'
    pilot.write_json(root/'smoke.json',smoke)
    print('Prepared duration-only extension',plan['config'],flush=True)


def aggregate(root):
    pilot.aggregate(root)
    plan=pilot.verify(root)
    for mode in plan['config']['modes']:
        for seed in plan['config']['seeds']:
            short=pd.read_csv(SHORT/f'{mode}_{seed}/training_episodes.csv',float_precision='round_trip')
            long=pd.read_csv(root/f'{mode}_{seed}/training_episodes.csv',float_precision='round_trip')
            check_prefix(short,long)
            run=json.loads((root/f'{mode}_{seed}/run.json').read_text())
            assert run['steps']==36500 and run['updates']==36135 and run['annual_episodes']==100
            assert run['planner_calls']==36500
    short=pd.read_csv(SHORT/'annual.csv');long=pd.read_csv(root/'annual.csv')
    pair=long.merge(short,on=['mode','variant','training_seed','year'],suffixes=('_long','_short'),validate='one_to_one')
    deltas=pair[['mode','variant','training_seed','year']].copy()
    for metric in pilot.METRICS: deltas['delta_'+metric]=pair[metric+'_long']-pair[metric+'_short']
    deltas.to_csv(root/'paired_duration.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    short_means=pd.read_csv(SHORT/'seed_means.csv');long_means=pd.read_csv(root/'seed_means.csv')
    for ax,metric,title in zip(axes.flat,['full_return','flood_days','low_water_days','deficit_million_m3'],
        ['Annual reward ↑','Flood days ↓','Low-water days ↓','Unmet demand, million m³ ↓']):
        for mode,color in [('residual','#27856a'),('full','#267ead')]:
            for variant,style in [('plain','-'),('shared_filter','--')]:
                y=[f.query('mode==@mode and variant==@variant')[metric].mean() for f in [short_means,long_means]]
                ax.plot([3650,36500],y,color=color,ls=style,marker='o',label=f'{mode} · {variant}')
        ax.set(title=title,xlabel='Training steps per seed',xticks=[3650,36500]);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched duration extension · 3 seeds · final checkpoints only\nOnly steps change; warmup, replay capacity, MPC, and network settings stay fixed')
    fig.savefig(root/'residual_pilot.png',dpi=160);fig.savefig(root/'duration_comparison.png',dpi=160);plt.close(fig)
    # Replace the historical caption produced by the reused pilot aggregator.
    run=json.loads((root/'run.json').read_text());run.update(duration_steps=36500,short_steps=3650,
        exact_short_prefix_reproduction=True,changed_config_keys=['steps'],replay_capacity=3650)
    pilot.write_json(root/'run.json',run)
    pilot.verify(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','train','evaluate'],required=True)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/residual/duration_36500_v1')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();root=args.output.resolve()
    if args.stage=='prepare': prepare(root)
    elif args.stage=='train': pilot.launch_jobs(root,'train',args.workers)
    else:
        pilot.launch_jobs(root,'evaluate',args.workers);aggregate(root)


if __name__=='__main__': main()
