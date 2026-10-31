"""Inspect existing capped-PPO training logs without training or test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lake_como.paths import PROJECT_ROOT
from lake_como.precision_study import check_control
from lake_como.stabilize import SEEDS

TAGS = {
    'episode_reward': 'Reward / Total reward (mean)',
    'daily_reward': 'Reward / Instantaneous reward (mean)',
    'value_loss': 'Loss / Value loss',
    'policy_loss': 'Loss / Policy loss',
    'action_std': 'Policy / Standard deviation',
    'learning_rate': 'Learning / Learning rate',
}
WINDOWS = [(100000,125000),(125000,150000),(150000,175000),(175000,200000)]


def window_summary(traces):
    """Descriptive window means only; overlapping logs are not independent trials."""
    rows=[]
    for (seed,metric),frame in traces.groupby(['training_seed','metric']):
        for lo,hi in WINDOWS:
            samples=frame[(frame.step>lo)&(frame.step<=hi)]
            if samples.empty:raise ValueError(f'Missing {metric} window for seed {seed}')
            rows.append(dict(training_seed=seed,metric=metric,start_step=lo,end_step=hi,
                             mean=samples.value.mean(),logged_points=len(samples)))
    windows=pd.DataFrame(rows)
    changes=[]
    for seed in SEEDS:
        row={'training_seed':seed}
        for metric in TAGS:
            frame=traces[(traces.training_seed==seed)&(traces.metric==metric)]
            previous=frame[(frame.step>100000)&(frame.step<=150000)].value.mean()
            late=frame[(frame.step>150000)&(frame.step<=200000)].value.mean()
            row[f'{metric}_previous_50k']=previous
            row[f'{metric}_last_50k']=late
            row[f'{metric}_change']=late-previous
        changes.append(row)
    return windows,pd.DataFrame(changes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/stabilization/curve_inspection_v1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows=[];sources=[]
    for seed in SEEDS:
        directory,metadata=check_control(seed)
        events=list((directory/'logs').rglob('events.out.tfevents.*'))
        if len(events)!=1:raise ValueError('Expected one log per completed control run')
        event=EventAccumulator(str(events[0]));event.Reload()
        for metric,tag in TAGS.items():
            points=event.Scalars(tag)
            if not points or any(not np.isfinite(p.value) for p in points):raise ValueError('Missing/nonfinite training series')
            rows.extend(dict(training_seed=seed,metric=metric,step=p.step,value=p.value) for p in points)
        sources.append(dict(training_seed=seed,checkpoint_sha256=metadata['checkpoint_sha256'],
                            event_file=str(events[0].relative_to(PROJECT_ROOT)),
                            event_sha256=hashlib.sha256(events[0].read_bytes()).hexdigest()))
    traces=pd.DataFrame(rows)
    windows,changes=window_summary(traces)
    traces.to_csv(args.output/'training_traces.csv',index=False)
    windows.to_csv(args.output/'window_summary.csv',index=False)
    changes.to_csv(args.output/'late_changes.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,seed in enumerate(SEEDS):
        frame=traces[(traces.training_seed==seed)&(traces.metric=='episode_reward')].sort_values('step')
        smooth=frame.value.rolling(20,min_periods=1).mean()  # 20 × 500-step log windows = 10k steps
        axes[0].plot(frame.step,smooth,color=colors[i],label=str(seed),lw=1.8)
        f=windows[(windows.training_seed==seed)&(windows.metric=='episode_reward')]
        axes[1].plot((f.start_step+f.end_step)/2,f['mean'],marker='o',color=colors[i],label=str(seed))
    axes[0].set(title='Sampled-policy training return (10k-step smoothing)',xlabel='Training steps',ylabel='Logged mean full episode reward')
    axes[1].set(title='Late training: non-overlapping 25k-step windows',xlabel='Training window',ylabel='Mean of logged full episode rewards')
    axes[0].set_xticks([0,50000,100000,150000,200000],['0','50k','100k','150k','200k'])
    axes[1].set_xticks([112500,137500,162500,187500],['100–125k','125–150k','150–175k','175–200k'],rotation=10)
    axes[1].legend(title='Learner seed',fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Capped mixed-precision PPO, zero entropy · training data only')
    fig.savefig(args.output/'learning_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,title in zip(axes,['value_loss','action_std','learning_rate'],['Normalized critic loss','Gaussian exploration spread','Learning rate']):
        for i,seed in enumerate(SEEDS):
            f=traces[(traces.training_seed==seed)&(traces.metric==metric)].sort_values('step')
            ax.plot(f.step,f.value,color=colors[i],label=str(seed),alpha=.8)
        ax.set(title=title,xlabel='Training steps')
        ax.set_xticks([0,100000,200000],['0','100k','200k']);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Logged normalized value loss');axes[1].set_ylabel('Normalized action standard deviation')
    axes[2].set_ylabel('Learning rate');axes[2].set_yscale('log');axes[2].legend(title='Seed',fontsize=8)
    fig.savefig(args.output/'training_diagnostics.png',dpi=170);plt.close(fig)
    metadata=dict(setting='capped_scheduler, mixed precision, entropy coefficient 0, 200000 steps',
                  source_runs=sources,new_training=False,new_validation_evaluations=False,test_evaluated=False,
                  window_convention='(start, end]; 25k windows; long-window comparison 100–150k vs 150–200k',
                  smoothing='20 consecutive 500-step reward log points, used for visualization only',
                  interpretation=['Reward logs describe sampled training actions and varying historical years, not deterministic validation checkpoints.',
                                  'skrl write_tracking_data clears episode tracking every 500 steps; total-reward scalar averages tracked episode-return means within that logging window.',
                                  'Adjacent logs and smoothed points are dependent; no slope p-values or convergence claim.',
                                  'Value loss is normalized with changing value-preprocessor statistics; decreasing loss is not proof of policy optimality.',
                                  'Only final checkpoints were saved; an intermediate deterministic validation learning curve cannot be reconstructed.'])
    (args.output/'inspection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(changes[['training_seed','episode_reward_previous_50k','episode_reward_last_50k','episode_reward_change']].round(2).to_string(index=False))
    print('Saved inspection:',args.output)


if __name__=='__main__':main()
