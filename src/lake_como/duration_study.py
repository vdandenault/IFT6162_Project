"""Continuous PPO duration comparisons with verified earlier checkpoints; validation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from lake_como.precision_study import check_control
from lake_como.reproduce import evaluate, summarize
from lake_como.stabilize import COMMON, SEEDS, VARIANTS, agent_config

BUDGETS = (200000,400000)
RULE = ('Select 400k only if its five-seed mean full validation return is non-worse, '
        'sample SD is non-larger, and weakest seed is non-worse than its paired 200k prefixes; '
        'otherwise retain 200k. Report added compute. No best intermediate checkpoint or test selection.')


def trees_equal(a,b):
    """Exact comparison includes optimizer/scalers, not just actor weights."""
    import torch
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(trees_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(trees_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def compare_prefix(reference, checkpoint):
    import torch
    expected=torch.load(reference,map_location='cpu',weights_only=True)
    actual=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if expected.keys()!=actual.keys():raise ValueError('Checkpoint component set differs')
    matches={k:bool(trees_equal(expected[k],actual[k])) for k in expected}
    if not all(matches.values()):raise ValueError(f'Checkpoint prefix mismatch: {matches}')
    return matches


def check_duration_control(seed):
    """Verify the completed 400k run before using it as an extension control."""
    directory = PROJECT_ROOT / f'results/stabilization/duration_v1/seed_{seed}'
    metadata = json.loads((directory / 'run.json').read_text())
    if not metadata['complete'] or metadata['test_evaluated'] or not metadata['prefix_check']['exact']:
        raise ValueError('Invalid 400k control')
    if metadata['job']['common'] != COMMON:
        raise ValueError('400k control settings differ')
    for budget in BUDGETS:
        path = directory / f'step_{budget}.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['checkpoint_sha256'][str(budget)]:
            raise ValueError('400k control checkpoint changed')
    return directory, metadata


def check_600k_control(seed):
    directory = PROJECT_ROOT / f'results/stabilization/duration_600k_v1/seed_{seed}'
    metadata = json.loads((directory / 'run.json').read_text())
    if not (metadata['complete'] and metadata['prefix_check']['exact']
            and metadata['prefix_400k_check']['exact']
            and metadata['validation_200k_matches_control']
            and metadata['validation_400k_matches_control']) or metadata['test_evaluated']:
        raise ValueError('Invalid 600k control')
    if metadata['job']['common'] != COMMON:
        raise ValueError('600k control settings differ')
    for budget in (200000, 400000, 600000):
        path = directory / f'step_{budget}.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['checkpoint_sha256'][str(budget)]:
            raise ValueError('600k control checkpoint changed')
    return directory, metadata


def run_worker(job_path):
    import torch
    from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction
    from gym4real.algorithms.dam.ppo_skrl import Policy, Value
    from skrl.agents.torch.ppo import PPO
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed
    job=json.loads(job_path.read_text());output=PROJECT_ROOT/job['output'];output.mkdir(parents=True,exist_ok=True)
    control,control_metadata=check_control(job['seed'])
    budgets=tuple(job['budgets'])
    extension_control=check_duration_control(job['seed'])[0] if max(budgets)>=600000 else None
    control_600k=check_600k_control(job['seed'])[0] if max(budgets)==1000000 else None
    metadata_path=output/'run.json'
    if metadata_path.exists():
        saved=json.loads(metadata_path.read_text())
        if saved['job']!=job:raise ValueError('Cannot resume with changed duration plan')
        if saved.get('complete'):
            for budget in budgets:
                path=output/f'step_{budget}.pt'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=saved['checkpoint_sha256'][str(budget)]:raise ValueError('Checkpoint changed')
            return
    if job['common']!=COMMON:raise ValueError('Non-duration settings changed')
    class CapturePPO(PPO):
        def post_interaction(self,timestep,timesteps):
            super().post_interaction(timestep,timesteps)
            step=timestep+1
            if step in budgets:
                path=output/f'step_{step}.pt';self.save(str(path))
                if step==200000:
                    matches=compare_prefix(control/'final.pt',path)
                    (output/'prefix_check.json').write_text(json.dumps(dict(exact=True,components=matches),indent=2)+'\n')
                    print('200k prefix matches existing control exactly:',job['seed'],flush=True)
                if extension_control is not None and step==400000:
                    matches=compare_prefix(extension_control/'step_400000.pt',path)
                    (output/'prefix_400k_check.json').write_text(json.dumps(dict(exact=True,components=matches),indent=2)+'\n')
                    print('400k prefix matches existing control exactly:',job['seed'],flush=True)
                if control_600k is not None and step==600000:
                    matches=compare_prefix(control_600k/'step_600000.pt',path)
                    (output/'prefix_600k_check.json').write_text(json.dumps(dict(exact=True,components=matches),indent=2)+'\n')
                    print('600k prefix matches existing control exactly:',job['seed'],flush=True)
                print('Captured checkpoint:',step,flush=True)
    torch.set_num_threads(COMMON['threads']);set_seed(job['seed'])
    env=wrap_env(RescaleAction(RecordEpisodeStatistics(make_env('train',protocol='chronological',correct_terminal_level=True)),-1.,1.))
    models=dict(policy=Policy(env.observation_space,env.action_space,device='cpu',net_arch=[16,16],initial_log_std=-.5),
                value=Value(env.observation_space,env.action_space,device='cpu',net_arch=[16,16]))
    agent=CapturePPO(models=models,memory=RandomMemory(memory_size=2048,num_envs=env.num_envs,device='cpu'),
                     cfg=agent_config('capped_scheduler',output),observation_space=env.observation_space,
                     action_space=env.action_space,device='cpu')
    metadata=dict(job=job,complete=False,training_years=list(settings('train')['inflow']),
                  validation_years=list(settings('validation')['inflow']),test_evaluated=False,
                  versions={p:version(p) for p in ['torch','skrl','gymnasium','numpy']},
                  continuous_training=True,old_checkpoint_resumed=False)
    if metadata['versions']!=control_metadata['versions']:raise ValueError('Control library versions differ')
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    started=time.perf_counter()
    trainer=SequentialTrainer(cfg={'timesteps':max(budgets),'headless':True,'disable_progressbar':True},env=env,agents=agent)
    trainer.train();metadata['training_seconds']=time.perf_counter()-started
    # Evaluate only after the continuous run ends so evaluation RNG cannot alter training.
    for budget in budgets:
        frame,_=evaluate('ppo',checkpoint=output/f'step_{budget}.pt',split='validation',protocol='chronological',correct_terminal_level=True)
        frame.to_csv(output/f'validation_trajectories_{budget}.csv',index=False)
        annual=summarize(frame)
        annual.assign(training_seed=job['seed'],budget=budget).to_csv(output/f'validation_annual_{budget}.csv',index=False)
        reference = None
        if budget==200000:
            reference=control/'validation_annual.csv'
        elif budget==400000 and extension_control is not None:
            reference=extension_control/'validation_annual_400000.csv'
        elif budget==600000 and control_600k is not None:
            reference=control_600k/'validation_annual_600000.csv'
        if reference is not None:
            expected=pd.read_csv(reference)
            columns=['year','objective_return','full_return','flood_days','low_water_days','deficit_million_m3','terminal_storage_m3']
            pd.testing.assert_frame_equal(annual[columns].reset_index(drop=True),expected[columns].reset_index(drop=True),check_exact=False,atol=1e-7,rtol=1e-10)
    metadata.update(complete=True,checkpoint_sha256={str(b):hashlib.sha256((output/f'step_{b}.pt').read_bytes()).hexdigest() for b in budgets},
                    prefix_check=json.loads((output/'prefix_check.json').read_text()),validation_200k_matches_control=True)
    if extension_control is not None:
        metadata.update(prefix_400k_check=json.loads((output/'prefix_400k_check.json').read_text()),validation_400k_matches_control=True)
    if control_600k is not None:
        metadata.update(prefix_600k_check=json.loads((output/'prefix_600k_check.json').read_text()),validation_600k_matches_control=True)
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')


def decision_from_means(means, baseline=200000, candidate=400000):
    a=means.loc[baseline].full_return;b=means.loc[candidate].full_return
    qualifies=bool(b.mean()>=a.mean() and b.std()<=a.std() and b.min()>=a.min())
    return dict(selected_steps=candidate if qualifies else baseline,longer_qualifies=qualifies,
                rule=RULE if candidate==400000 else extension_rule() if candidate==600000 else million_rule(),test_evaluated=False,scope=f'Five paired seeds, same validation years, exact verified prefixes through {baseline//1000}k')


def aggregate(output):
    budgets=tuple(json.loads((output/'seed_123/job.json').read_text())['budgets'])
    baseline,candidate=(600000,1000000) if max(budgets)==1000000 else budgets[-2:]
    frames=[];runtime=[]
    for seed in SEEDS:
        directory=output/f'seed_{seed}';metadata=json.loads((directory/'run.json').read_text())
        if not metadata['complete'] or not metadata['prefix_check']['exact'] or not metadata['validation_200k_matches_control']:
            raise ValueError('Incomplete or unmatched continuous training run')
        if candidate>=600000 and not (metadata['prefix_400k_check']['exact'] and metadata['validation_400k_matches_control']):
            raise ValueError('400k prefix not verified')
        if candidate==1000000 and not (metadata['prefix_600k_check']['exact'] and metadata['validation_600k_matches_control']):
            raise ValueError('600k prefix not verified')
        if metadata['test_evaluated']:raise ValueError('Test set was evaluated')
        runtime.append(dict(training_seed=seed,**{f'continuous_{candidate//1000}k_seconds':metadata['training_seconds']},
                            original_200k_seconds=check_control(seed)[1]['training_seconds']))
        for budget in budgets:
            path=directory/f'step_{budget}.pt'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['checkpoint_sha256'][str(budget)]:raise ValueError('Checkpoint hash changed')
            frame=pd.read_csv(directory/f'validation_annual_{budget}.csv')
            if frame.year.tolist()!=list(settings('validation')['inflow']):raise ValueError('Incorrect validation years')
            frames.append(frame)
    annual=pd.concat(frames,ignore_index=True)
    metrics=['full_return','objective_return','flood_days','low_water_days','deficit_million_m3']
    means=annual.groupby(['budget','training_seed'])[metrics].mean()
    summary=means.groupby('budget').agg(['mean','std','min','max'])
    annual.to_csv(output/'validation_annual.csv',index=False);means.to_csv(output/'seed_means.csv');summary.to_csv(output/'aggregate.csv')
    (means.loc[candidate]-means.loc[baseline]).to_csv(output/'paired_changes.csv')
    pd.DataFrame(runtime).to_csv(output/'runtimes.csv',index=False)
    decision=decision_from_means(means,baseline,candidate)
    if candidate==1000000:
        decision=decide_million(means)
        for endpoint in (800000,1000000):
            (means.loc[endpoint]-means.loc[600000]).to_csv(output/f'paired_changes_{endpoint}.csv')
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['full_return','objective_return'],['Full five-term validation return','Three-term validation objective']):
        for seed in SEEDS:
            ax.plot(budgets,[means.loc[(b,seed),metric] for b in budgets],marker='o',label=str(seed),alpha=.8)
        ax.set_xticks(budgets,['1M' if b==1000000 else f'{b//1000}k' for b in budgets]);ax.set(title=title,xlabel='Training steps',ylabel='Mean annual validation return per seed')
        ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(title='Learner seed',fontsize=8)
    fig.suptitle('Continuous training duration · capped mixed precision · entropy 0')
    fig.savefig(output/'duration_comparison.png',dpi=170);plt.close(fig)
    print(means.round(2).to_string(),flush=True);print(decision,flush=True)
    return means,summary,decision


def extension_rule():
    return ('600k qualifies versus 400k only if five-seed mean full validation return, '
            'sample SD, and weakest seed are all non-worse. This comparison does not '
            'retroactively change the earlier 200k-versus-400k decision. No test selection.')


def million_rule():
    return ('Compare fixed 800k and 1M endpoints independently against 600k. Each qualifies '
            'only if five-seed mean full validation return, sample SD, and weakest seed '
            'are all non-worse. Among qualifying endpoints choose higher mean return, '
            'breaking ties toward fewer steps; otherwise retain 600k. Stop at 1M. No test selection.')


def decide_million(means):
    comparisons = {str(endpoint): decision_from_means(means,600000,endpoint)
                   for endpoint in (800000,1000000)}
    eligible = [600000] + [int(endpoint) for endpoint, result in comparisons.items()
                          if result['longer_qualifies']]
    selected = max(eligible,key=lambda endpoint: (means.loc[endpoint].full_return.mean(),-endpoint))
    return dict(selected_steps=selected,comparisons=comparisons,rule=million_rule(),
                test_evaluated=False,stop_steps=1000000,
                scope='Five paired seeds, shared validation years, exact prefixes through 600k')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config',type=Path)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-steps',type=int,choices=[400000,600000,1000000],default=400000)
    args=parser.parse_args()
    if args.worker_config:run_worker(args.worker_config);return
    budgets=tuple(range(200000,args.max_steps+1,200000))
    suffix={400000:'duration_v1',600000:'duration_600k_v1',1000000:'duration_1m_v1'}[args.max_steps]
    args.output=args.output or PROJECT_ROOT/'results/stabilization'/suffix
    args.output.mkdir(parents=True,exist_ok=True)
    controls=[]
    for seed in SEEDS:
        directory,metadata=check_control(seed)
        controls.append(dict(seed=seed,path=str(directory.relative_to(PROJECT_ROOT)),checkpoint_sha256=metadata['checkpoint_sha256']))
    plan=dict(common=COMMON,scheduler=VARIANTS['capped_scheduler'],seeds=SEEDS,budgets=list(budgets),
              controls=controls,rule=RULE,test_evaluated=False,continuous_training=True,
              train_years=list(settings('train')['inflow']),validation_years=list(settings('validation')['inflow']),
              selection='Compare fixed 200k and 400k endpoints, no best intermediate checkpoint',
              prefix_verification='Exact equality of all saved checkpoint components including optimizer/scalers; validation scores also checked',
              evaluation_timing='After 400k training ends; no evaluation-induced RNG changes mid-run')
    if args.max_steps==600000:
        plan.update(rule=extension_rule(),selection='Fixed 400k-versus-600k comparison; 200k retained for context',
                    evaluation_timing='After 600k training ends; no evaluation-induced RNG changes mid-run',
                    controls_400k=[dict(seed=seed,checkpoint_sha256=check_duration_control(seed)[1]['checkpoint_sha256']) for seed in SEEDS])
    if args.max_steps==1000000:
        plan.update(rule=million_rule(),selection='Fixed 800k and 1M versus 600k; earlier budgets for context',
                    evaluation_timing='After 1M training ends; no evaluation-induced RNG changes mid-run',
                    stop_steps=1000000,
                    controls_400k=[dict(seed=seed,checkpoint_sha256=check_duration_control(seed)[1]['checkpoint_sha256']) for seed in SEEDS],
                    controls_600k=[dict(seed=seed,checkpoint_sha256=check_600k_control(seed)[1]['checkpoint_sha256']) for seed in SEEDS])
    for path in [args.output/'plan.json']:
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Predeclared duration plan differs')
        path.write_text(json.dumps(plan,indent=2)+'\n')
    record_plan({400000:'duration_400k',600000:'duration_600k',1000000:'duration_1m'}[args.max_steps], plan)
    jobs=[]
    for seed in SEEDS:
        directory=args.output/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
        path=directory/'job.json';path.write_text(json.dumps(dict(seed=seed,common=COMMON,budgets=list(budgets),output=str(directory.relative_to(PROJECT_ROOT))),indent=2)+'\n');jobs.append(path)
    def launch(path):
        seed=json.loads(path.read_text())['seed'];print('Start continuous',args.max_steps,seed,flush=True)
        with (path.parent/'console.log').open('w') as log:
            subprocess.run([sys.executable,'-m','lake_como.duration_study','--worker-config',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Complete duration pair',seed,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(launch,jobs))
    aggregate(args.output)


if __name__=='__main__':main()
