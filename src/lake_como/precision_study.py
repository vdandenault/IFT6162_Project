"""Disable CPU mixed precision, reusing the completed capped-scheduler controls."""
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
from lake_como.paths import PROJECT_ROOT
from lake_como.experiment_plans import record_plan
from lake_como.environment import settings
from lake_como.stabilize import COMMON, SEEDS, VARIANTS

RULE = ('Select FP32 only if its five-seed mean full validation return is non-worse, '
        'sample SD is non-larger, and weakest seed is non-worse than capped mixed precision; '
        'otherwise retain capped mixed precision. No test evaluation or best-seed selection.')


def check_control(seed):
    directory=PROJECT_ROOT/'results/stabilization/lr_v1/capped_scheduler'/f'seed_{seed}'
    metadata=json.loads((directory/'run.json').read_text())
    job=metadata['job']
    if not metadata.get('complete') or job['seed']!=seed or job['variant']!='capped_scheduler':
        raise ValueError('Missing or incorrect completed control')
    if job['common']!=COMMON or job['scheduler']!=VARIANTS['capped_scheduler']:
        raise ValueError('Control configuration differs')
    if metadata['training_years']!=list(settings('train')['inflow']) or metadata['validation_years']!=list(settings('validation')['inflow']):
        raise ValueError('Control year membership differs')
    if metadata['test_evaluated']:raise ValueError('Control used the test set')
    if hashlib.sha256((directory/'final.pt').read_bytes()).hexdigest()!=metadata['checkpoint_sha256']:
        raise ValueError('Control checkpoint hash mismatch')
    return directory,metadata


def decision_from_means(means):
    control=means.loc['mixed_precision'].full_return
    fp32=means.loc['fp32'].full_return
    qualifies=bool(fp32.mean()>=control.mean() and fp32.std()<=control.std() and fp32.min()>=control.min())
    return dict(selected='fp32' if qualifies else 'mixed_precision',fp32_qualifies=qualifies,
                rule=RULE,test_evaluated=False,scope='Validation-only evidence on five paired learner seeds')


def aggregate(output):
    frames=[];runtimes=[]
    for seed in SEEDS:
        control,metadata=check_control(seed)
        fp32=output/'fp32'/f'seed_{seed}'
        fp_meta=json.loads((fp32/'run.json').read_text())
        expected=deepcopy(COMMON);expected['mixed_precision']=False
        if not fp_meta['complete'] or fp_meta['job']['common']!=expected:raise ValueError('Incorrect FP32 run')
        if fp_meta['versions']!=metadata['versions']:raise ValueError('Library versions differ')
        if fp_meta['test_evaluated']:raise ValueError('FP32 run used test data')
        for arm,directory,meta in [('mixed_precision',control,metadata),('fp32',fp32,fp_meta)]:
            f=pd.read_csv(directory/'validation_annual.csv')
            if f.year.tolist()!=list(settings('validation')['inflow']):raise ValueError('Incorrect validation membership')
            frames.append(f.assign(precision=arm,training_seed=seed))
            runtimes.append(dict(precision=arm,training_seed=seed,training_seconds=meta['training_seconds']))
    annual=pd.concat(frames,ignore_index=True)
    metrics=['full_return','objective_return','flood_days','low_water_days','deficit_million_m3']
    means=annual.groupby(['precision','training_seed'])[metrics].mean()
    summary=means.groupby('precision').agg(['mean','std','min','max'])
    paired=means.loc['fp32']-means.loc['mixed_precision']
    annual.to_csv(output/'validation_annual.csv',index=False);means.to_csv(output/'seed_means.csv')
    summary.to_csv(output/'aggregate.csv');paired.to_csv(output/'paired_changes.csv')
    pd.DataFrame(runtimes).to_csv(output/'runtimes.csv',index=False)
    decision=decision_from_means(means)
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['full_return','objective_return'],['Full five-term validation return','Three-term validation objective']):
        for seed in SEEDS:
            values=[means.loc[(arm,seed),metric] for arm in ['mixed_precision','fp32']]
            ax.plot([0,1],values,marker='o',alpha=.75,label=str(seed))
        ax.set_xticks([0,1],['Capped + mixed precision','Capped + FP32 training'])
        ax.set(title=title,ylabel='Mean annual validation return per seed')
        ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(title='Learner seed',fontsize=8)
    fig.suptitle('Precision ablation · same capped scheduler · 42 train / 10 validation years')
    fig.savefig(output/'precision_comparison.png',dpi=170);plt.close(fig)
    print(means.round(2).to_string(),flush=True);print(decision,flush=True)
    return means,summary,decision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/stabilization/precision_v1')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    controls=[]
    for seed in SEEDS:
        directory,metadata=check_control(seed)
        controls.append(dict(seed=seed,path=str(directory.relative_to(PROJECT_ROOT)),checkpoint_sha256=metadata['checkpoint_sha256']))
    candidate=deepcopy(COMMON);candidate['mixed_precision']=False
    plan=dict(control_common=COMMON,candidate_common=candidate,scheduler=VARIANTS['capped_scheduler'],
              seeds=SEEDS,controls=controls,rule=RULE,test_evaluated=False,
              train_years=list(settings('train')['inflow']),validation_years=list(settings('validation')['inflow']),
              evaluation='Both arms: frozen normalization and deterministic mean actions on CPU FP32',
              control_precision='CPU autocast enabled (PyTorch default bfloat16)',
              candidate_precision='CPU autocast disabled; neural-network parameters/operations float32',
              inference='Mixed precision changes training only; inference is the same for both arms')
    for path in [args.output/'plan.json']:
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Predeclared precision plan differs')
        path.write_text(json.dumps(plan,indent=2)+'\n')
    record_plan('precision', plan)
    jobs=[]
    for seed in SEEDS:
        directory=args.output/'fp32'/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
        job=dict(variant='capped_scheduler',seed=seed,common=candidate,scheduler=VARIANTS['capped_scheduler'],output=str(directory.relative_to(PROJECT_ROOT)))
        path=directory/'job.json';path.write_text(json.dumps(job,indent=2)+'\n');jobs.append(path)
    def launch(path):
        seed=json.loads(path.read_text())['seed'];print('Start FP32',seed,flush=True)
        with (path.parent/'console.log').open('w') as log:
            subprocess.run([sys.executable,'-m','lake_como.stabilize','--worker-config',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Complete FP32',seed,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(launch,jobs))
    aggregate(args.output)


if __name__=='__main__':main()
