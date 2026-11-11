"""One small entropy bonus against the completed capped zero-entropy controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.experiment_plans import record_plan
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.precision_study import check_control
from reservoir_control.systems.lake_como.stabilize import COMMON, SEEDS, VARIANTS

COEFFICIENT = .01
RULE = ('Select entropy bonus 0.01 only if its five-seed mean full validation return is non-worse, '
        'sample SD is non-larger, and weakest seed is non-worse than capped zero-entropy controls; '
        'otherwise retain zero entropy. No test evaluation or best-seed selection.')


def decision_from_means(means):
    control=means.loc['zero_entropy'].full_return
    candidate=means.loc['entropy_bonus'].full_return
    qualifies=bool(candidate.mean()>=control.mean() and candidate.std()<=control.std() and candidate.min()>=control.min())
    return dict(selected='entropy_bonus' if qualifies else 'zero_entropy',entropy_bonus_qualifies=qualifies,
                coefficient=COEFFICIENT,rule=RULE,test_evaluated=False,
                scope='Validation-only evidence on five paired learner seeds')


def aggregate(output):
    import torch
    frames=[];diagnostics=[]
    for seed in SEEDS:
        control,control_meta=check_control(seed)
        candidate=output/'entropy_bonus'/f'seed_{seed}'
        meta=json.loads((candidate/'run.json').read_text())
        expected=deepcopy(COMMON);expected['entropy_loss_scale']=COEFFICIENT
        if not meta['complete'] or meta['job']['common']!=expected:raise ValueError('Incorrect entropy candidate')
        if meta['versions']!=control_meta['versions']:raise ValueError('Library versions differ')
        if meta['test_evaluated']:raise ValueError('Candidate used test data')
        for arm,directory,metadata in [('zero_entropy',control,control_meta),('entropy_bonus',candidate,meta)]:
            if hashlib.sha256((directory/'final.pt').read_bytes()).hexdigest()!=metadata['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            f=pd.read_csv(directory/'validation_annual.csv')
            if f.year.tolist()!=list(settings('validation')['inflow']):raise ValueError('Incorrect validation years')
            frames.append(f.assign(exploration=arm,training_seed=seed))
            checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
            std=torch.exp(checkpoint['policy']['log_std_parameter']).item()
            diagnostics.append(dict(exploration=arm,training_seed=seed,training_seconds=metadata['training_seconds'],
                                    final_action_std_normalized=std,checkpoint_sha256=metadata['checkpoint_sha256']))
    annual=pd.concat(frames,ignore_index=True)
    metrics=['full_return','objective_return','flood_days','low_water_days','deficit_million_m3']
    means=annual.groupby(['exploration','training_seed'])[metrics].mean()
    summary=means.groupby('exploration').agg(['mean','std','min','max'])
    paired=means.loc['entropy_bonus']-means.loc['zero_entropy']
    annual.to_csv(output/'validation_annual.csv',index=False);means.to_csv(output/'seed_means.csv')
    summary.to_csv(output/'aggregate.csv');paired.to_csv(output/'paired_changes.csv')
    pd.DataFrame(diagnostics).to_csv(output/'policy_diagnostics.csv',index=False)
    decision=decision_from_means(means)
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['full_return','objective_return'],['Full five-term validation return','Three-term validation objective']):
        for seed in SEEDS:
            values=[means.loc[(arm,seed),metric] for arm in ['zero_entropy','entropy_bonus']]
            ax.plot([0,1],values,marker='o',alpha=.75,label=str(seed))
        ax.set_xticks([0,1],['Entropy coefficient 0','Entropy coefficient 0.01'])
        ax.set(title=title,ylabel='Mean annual validation return per seed')
        ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(title='Learner seed',fontsize=8)
    fig.suptitle('Entropy ablation · capped scheduler + mixed precision · five paired seeds')
    fig.savefig(output/'entropy_comparison.png',dpi=170);plt.close(fig)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,layout='constrained')
    trace=[]
    for ax,arm in zip(axes,['zero_entropy','entropy_bonus']):
        for seed in SEEDS:
            directory=check_control(seed)[0] if arm=='zero_entropy' else output/'entropy_bonus'/f'seed_{seed}'
            files=list((directory/'logs').rglob('events.out.tfevents.*'))
            event=EventAccumulator(str(files[0]));event.Reload()
            points=event.Scalars('Policy / Standard deviation')
            ax.plot([p.step for p in points],[p.value for p in points],label=str(seed))
            trace.extend(dict(exploration=arm,training_seed=seed,step=p.step,action_std_normalized=p.value) for p in points)
        ax.set(title=arm.replace('_',' '),xlabel='Training steps')
        ax.set_xticks([0,50000,100000,150000,200000],['0','50k','100k','150k','200k'])
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Learned action standard deviation (normalized units)')
    axes[-1].legend(title='Learner seed',fontsize=8)
    fig.suptitle('Does the bonus preserve exploration during training?')
    fig.savefig(output/'exploration_traces.png',dpi=170);plt.close(fig)
    pd.DataFrame(trace).to_csv(output/'exploration_traces.csv',index=False)
    print(means.round(2).to_string(),flush=True);print(decision,flush=True)
    return means,summary,decision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/stabilization/entropy_v1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    controls=[]
    for seed in SEEDS:
        directory,metadata=check_control(seed)
        controls.append(dict(seed=seed,path=str(directory.relative_to(PROJECT_ROOT)),checkpoint_sha256=metadata['checkpoint_sha256']))
    candidate=deepcopy(COMMON);candidate['entropy_loss_scale']=COEFFICIENT
    plan=dict(control_common=COMMON,candidate_common=candidate,scheduler=VARIANTS['capped_scheduler'],
              seeds=SEEDS,controls=controls,rule=RULE,test_evaluated=False,
              train_years=list(settings('train')['inflow']),validation_years=list(settings('validation')['inflow']),
              entropy='Actor loss adds -0.01 × Gaussian policy entropy; environment reward is unchanged',
              evaluation='Both arms: saved frozen normalizer and deterministic mean actions on CPU FP32',
              note='Only one coefficient is tested. No post-hoc search or test-set selection.')
    for path in [args.output/'plan.json']:
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Predeclared entropy plan differs')
        path.write_text(json.dumps(plan,indent=2)+'\n')
    record_plan('entropy', plan)
    jobs=[]
    for seed in SEEDS:
        directory=args.output/'entropy_bonus'/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
        job=dict(variant='capped_scheduler',seed=seed,common=candidate,scheduler=VARIANTS['capped_scheduler'],output=str(directory.relative_to(PROJECT_ROOT)))
        path=directory/'job.json';path.write_text(json.dumps(job,indent=2)+'\n');jobs.append(path)
    def launch(path):
        seed=json.loads(path.read_text())['seed'];print('Start entropy bonus',seed,flush=True)
        with (path.parent/'console.log').open('w') as log:
            subprocess.run([sys.executable,'-m','reservoir_control.systems.lake_como.stabilize','--worker-config',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Complete entropy bonus',seed,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(launch,jobs))
    aggregate(args.output)


if __name__=='__main__':main()
