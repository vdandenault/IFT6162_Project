"""Predeclared learning-rate ablation: chronological training and validation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from lake_como.experiment_plans import record_plan
from lake_como.environment import make_env, settings
from lake_como.paths import PROJECT_ROOT
from lake_como.reproduce import evaluate, summarize

VARIANTS = {
    'original_scheduler': {'scheduler': 'KLAdaptiveRL', 'min_lr': 1e-6, 'max_lr': 1e-2},
    'capped_scheduler': {'scheduler': 'KLAdaptiveRL', 'min_lr': 1e-6, 'max_lr': 1e-4},
    'fixed_lr': {'scheduler': None},
}
SEEDS = [123,124,125,126,127]
COMMON = dict(steps=200000, rollouts=2048, learning_epochs=10, mini_batches=32,
              discount_factor=.995, gae_lambda=.95, learning_rate=8e-6,
              net_arch=[16,16], initial_log_std=-.5, entropy_loss_scale=0.,
              mixed_precision=True, environment_seed=123, threads=1,
              protocol='chronological', correct_terminal_level=True,
              time_limit_bootstrap=False, checkpoint_selection='final training step')
SELECTION = ('Candidate must have validation mean full return >= control, sample SD <= control, '
             'and worst-seed full return >= control. Among qualifying candidates choose highest mean; '
             'if none qualifies, retain control and report no stabilization demonstrated. '
             'All metrics are seed means over the same 10 validation years.')


def agent_config(variant, output, *, mixed_precision=None, entropy_loss_scale=None):
    from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveRL
    cfg=deepcopy(PPO_DEFAULT_CONFIG)
    cfg.update(rollouts=COMMON['rollouts'],learning_epochs=COMMON['learning_epochs'],
               mini_batches=COMMON['mini_batches'],discount_factor=COMMON['discount_factor'],
               **{'lambda':COMMON['gae_lambda']}, learning_rate=COMMON['learning_rate'],
               grad_norm_clip=.5,ratio_clip=.2,value_clip=.2,clip_predicted_values=False,
               entropy_loss_scale=COMMON['entropy_loss_scale'] if entropy_loss_scale is None else entropy_loss_scale,value_loss_scale=.5,kl_threshold=0,
               mixed_precision=COMMON['mixed_precision'] if mixed_precision is None else mixed_precision,time_limit_bootstrap=COMMON['time_limit_bootstrap'],
               state_preprocessor=RunningStandardScaler,
               state_preprocessor_kwargs={'size':4,'device':'cpu'},
               value_preprocessor=RunningStandardScaler,
               value_preprocessor_kwargs={'size':1,'device':'cpu'})
    option=VARIANTS[variant]
    cfg['learning_rate_scheduler']=KLAdaptiveRL if option['scheduler'] else None
    cfg['learning_rate_scheduler_kwargs']=dict(kl_threshold=.008,min_lr=option['min_lr'],max_lr=option['max_lr']) if option['scheduler'] else {}
    cfg['experiment'].update(directory=str(output/'logs'),experiment_name='ppo',
                             write_interval=500,checkpoint_interval=0)
    return cfg


def run_worker(job_path):
    import torch
    from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction
    from gym4real.algorithms.dam.ppo_skrl import Policy, Value
    from skrl.agents.torch.ppo import PPO
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed
    job=json.loads(job_path.read_text());output=PROJECT_ROOT/job['output']
    output.mkdir(parents=True,exist_ok=True)
    metadata_path=output/'run.json';checkpoint=output/'final.pt'
    if metadata_path.exists():
        saved=json.loads(metadata_path.read_text())
        if saved['job']!=job:raise ValueError('Cannot resume a run with changed settings')
        if checkpoint.exists() and saved.get('complete'):
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=saved['checkpoint_sha256']:
                raise ValueError('Saved checkpoint changed')
            return
    for key, value in COMMON.items():
        if key not in ('mixed_precision', 'entropy_loss_scale') and job['common'].get(key) != value:
            raise ValueError(f'Unexpected training override: {key}')
    torch.set_num_threads(COMMON['threads']);set_seed(job['seed'])
    env=make_env('train',protocol='chronological',correct_terminal_level=True)
    env=wrap_env(RescaleAction(RecordEpisodeStatistics(env),-1.,1.))
    models=dict(policy=Policy(env.observation_space,env.action_space,device='cpu',net_arch=[16,16],initial_log_std=-.5),
                value=Value(env.observation_space,env.action_space,device='cpu',net_arch=[16,16]))
    agent=PPO(models=models,memory=RandomMemory(memory_size=2048,num_envs=env.num_envs,device='cpu'),
              cfg=agent_config(job['variant'],output,mixed_precision=job['common']['mixed_precision'],
                               entropy_loss_scale=job['common']['entropy_loss_scale']),observation_space=env.observation_space,
              action_space=env.action_space,device='cpu')
    metadata=dict(job=job,complete=False,training_years=list(settings('train')['inflow']),
                  validation_years=list(settings('validation')['inflow']),test_evaluated=False,
                  versions={p:version(p) for p in ['torch','skrl','gymnasium','numpy']},
                  mixed_precision=job['common']['mixed_precision'],
                  entropy_loss_scale=job['common']['entropy_loss_scale'],
                  autocast_dtype=str(torch.get_autocast_dtype('cpu')) if job['common']['mixed_precision'] else 'disabled; model parameters float32')
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    started=time.perf_counter()
    trainer=SequentialTrainer(cfg={'timesteps':COMMON['steps'],'headless':True,'disable_progressbar':True},env=env,agents=agent)
    trainer.train();agent.save(str(checkpoint));metadata['training_seconds']=time.perf_counter()-started
    frame,seconds=evaluate('ppo',checkpoint=checkpoint,split='validation',protocol='chronological',correct_terminal_level=True)
    frame.to_csv(output/'validation_trajectories.csv',index=False)
    summarize(frame).assign(training_seed=job['seed'],variant=job['variant']).to_csv(output/'validation_annual.csv',index=False)
    metadata.update(complete=True,evaluation_seconds=seconds,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')


def select_candidate(means):
    """Validation-only mean, sample-SD, and worst-seed guardrails."""
    control=means.loc['original_scheduler'].full_return
    accepted=[]
    for variant in ['capped_scheduler','fixed_lr']:
        candidate=means.loc[variant].full_return
        if candidate.mean()>=control.mean() and candidate.std()<=control.std() and candidate.min()>=control.min():accepted.append(variant)
    selected=max(accepted,key=lambda v:means.loc[v].full_return.mean()) if accepted else 'original_scheduler'
    decision=dict(selection_rule=SELECTION,qualifying_candidates=accepted,selected=selected,
                  test_evaluated=False,claim='Validation selection only; not proof of generalization or statistical significance')
    return decision


def seed_summary(output):
    frames=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            path=output/variant/f'seed_{seed}'/'validation_annual.csv'
            f=pd.read_csv(path)
            if f.year.tolist()!=list(settings('validation')['inflow']):raise ValueError('Incorrect validation years')
            frames.append(f)
    annual=pd.concat(frames,ignore_index=True)
    columns=['full_return','objective_return','flood_days','low_water_days','deficit_million_m3']
    means=annual.groupby(['variant','training_seed'])[columns].mean()
    table=means.groupby('variant').agg(['mean','std','min','max'])
    annual.to_csv(output/'validation_annual.csv',index=False);means.to_csv(output/'seed_means.csv')
    table.to_csv(output/'aggregate.csv')
    decision=select_candidate(means)
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    baseline=[]
    for policy in ['random','mean','max','ead']:
        f,_=evaluate(policy,split='validation',protocol='chronological',correct_terminal_level=True)
        baseline.append(summarize(f))
    pd.concat(baseline).to_csv(output/'validation_rules.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels=['Original scheduler','Capped at 1e-4','Fixed at 8e-6']
    for ax,metric,title in zip(axes,['full_return','objective_return'],['Full five-term validation return','Three-term validation objective']):
        for i,(variant,label) in enumerate(zip(VARIANTS,labels)):
            values=means.loc[variant][metric]
            ax.scatter(np.full(5,i)+np.linspace(-.08,.08,5),values,label=label)
            ax.plot([i-.2,i+.2],[values.mean()]*2,color='#173650',lw=2)
        ax.set_xticks(range(3),labels,rotation=12);ax.set(ylabel='Mean annual return per learner seed',title=title)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Learning-rate ablation · 42 training years / 10 validation years · five seeds')
    fig.savefig(output/'stabilization.png',dpi=170);plt.close(fig)
    print(means.round(2).to_string(),flush=True)
    print('Validation selection:',decision,flush=True)
    return means,table,decision


def learning_rate_diagnostics(output):
    """Plot recorded scheduler rates; verify a fixed rate from optimizer state."""
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True,layout='constrained')
    records=[]
    for ax,variant in zip(axes,VARIANTS):
        for seed in SEEDS:
            directory=output/variant/f'seed_{seed}'
            files=list((directory/'logs').rglob('events.out.tfevents.*'))
            event=EventAccumulator(str(files[0]));event.Reload()
            tag='Learning / Learning rate'
            if tag in event.Tags()['scalars']:
                points=event.Scalars(tag)
                steps=[v.step for v in points];rates=[v.value for v in points]
                kind='TensorBoard measurements'
            else:
                checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
                rate=checkpoint['optimizer']['param_groups'][0]['lr']
                if rate!=COMMON['learning_rate']:raise ValueError('Fixed optimizer learning rate changed')
                steps=[0,COMMON['steps']];rates=[rate,rate]
                kind='Fixed schedule, verified against final optimizer state'
            for step,rate in zip(steps,rates):records.append(dict(variant=variant,training_seed=seed,step=step,learning_rate=rate,source=kind))
            if variant=='capped_scheduler' and max(rates)>1e-4+1e-10:raise ValueError('Learning rate exceeded cap')
            ax.plot(steps,rates,label=str(seed),alpha=.7,lw=1)
        ax.set(title=variant.replace('_',' '),xlabel='Training steps',yscale='log',ylim=(5e-7,2e-2))
        ax.set_xticks([0,50000,100000,150000,200000],['0','50k','100k','150k','200k'])
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Learning rate (log scale)');axes[-1].legend(title='Learner seed',fontsize=8)
    fig.suptitle('Learning-rate behavior: all other algorithm settings fixed')
    fig.savefig(output/'learning_rate_traces.png',dpi=170);plt.close(fig)
    pd.DataFrame(records).to_csv(output/'learning_rate_traces.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config',type=Path)
    parser.add_argument('--workers',type=int,default=3,choices=[1,2,3,4])
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/stabilization/lr_v1')
    args=parser.parse_args()
    if args.worker_config:
        run_worker(args.worker_config);return
    args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(common=COMMON,variants=VARIANTS,seeds=SEEDS,selection_rule=SELECTION,
              train_years=list(settings('train')['inflow']),validation_years=list(settings('validation')['inflow']),
              test_evaluated=False,notes=['Baseline reproduction snapshot is unchanged.',
              'All arms use chronological membership and the corrected terminal level, unlike original-membership reproduction.',
              'Annual reset and time-limit bootstrapping remain fixed across arms; continuous years are not implemented.',
              'Only scheduler changes; mixed precision and entropy remain unchanged.'])
    plan_path=args.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Plan differs; use a new experiment directory')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    record_plan('learning_rate', plan)
    jobs=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            directory=args.output/variant/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
            job=dict(variant=variant,seed=seed,common=COMMON,scheduler=VARIANTS[variant],output=str(directory.relative_to(PROJECT_ROOT)))
            job_path=directory/'job.json';job_path.write_text(json.dumps(job,indent=2)+'\n');jobs.append(job_path)
    def launch(path):
        job=json.loads(path.read_text());print('Start',job['variant'],job['seed'],flush=True)
        with (path.parent/'console.log').open('w') as log:
            subprocess.run([sys.executable,'-m','lake_como.stabilize','--worker-config',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Complete',job['variant'],job['seed'],flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(launch,jobs))
    seed_summary(args.output)
    learning_rate_diagnostics(args.output)


if __name__=='__main__':main()
