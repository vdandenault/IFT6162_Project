"""Staged first MPC-residual pilot and matched proposal-aware full-action control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.environment import make_env,settings
from reservoir_control.systems.lake_como.inflow_observation import PastInflowObservation
from reservoir_control.controllers.sac.agent import SACConfig,ReplayBuffer
from reservoir_control.systems.lake_como.sac_study import verify_plan
from reservoir_control.systems.lake_como.residual_mpc import make_base,MPCProposalEnv,ProposalSACAgent,combine_request
from reservoir_control.systems.lake_como.mpc import MPCConfig,SeasonalForecast,PlanningModel
from reservoir_control.systems.lake_como.safety_filter import FloodSafetyFilter,fit_error_limits
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como.reproduce import evaluate
from reservoir_control.systems.lake_como.compare import annual_metrics,verify_matched_inputs
from reservoir_control.systems.lake_como.mpc_objective_closed_loop import level_severity
from reservoir_control.systems.lake_como.splits import split_years

BASE=PROJECT_ROOT/'results/sac/baseline_200k_v1'
METRICS=['full_return','flood_days','low_water_days','deficit_million_m3','flood_excess_m_days','low_excess_m_days']
CONFIG=dict(seeds=[123,124,125],modes=['residual','full'],steps=3650,warmup_steps=365,
            batch_size=128,replay_capacity=3650,residual_bound_m3_s=50.,
            smoke_steps=730,smoke_seed=901,environment_seed=123)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if digest(PROJECT_ROOT/name)!=expected: raise ValueError(f'Changed residual-study source: {name}')
    verify_plan(BASE)
    return plan


def prepare(root):
    if root.exists() and any(root.iterdir()): raise ValueError('Choose a new output directory')
    root.mkdir(parents=True,exist_ok=True)
    original=verify_plan(BASE)
    norm=json.loads((BASE/'normalization.json').read_text())
    high=float(settings('train')['action']['high'])
    files=[Path(__file__),PROJECT_ROOT/'src/reservoir_control/systems/lake_como/residual_mpc.py',BASE/'plan.json',BASE/'normalization.json',
           PROJECT_ROOT/'src/reservoir_control/systems/lake_como/safety_filter.py',PROJECT_ROOT/'src/reservoir_control/systems/lake_como/mpc_retention.py',
           PROJECT_ROOT/'src/reservoir_control/systems/lake_como/mpc_causal.py',PROJECT_ROOT/'configs/experiments/control_protocol_v1.json']
    plan=dict(config=CONFIG,agent=asdict(SACConfig(observation_dim=6)),high=high,fallback=original['fallback'],
        observation_mean=norm['mean']+[high/2],observation_scale=norm['scale']+[high/2],
        training_years=split_years('train'),validation_years=split_years('validation'),test_evaluated=False,
        source_sha256={str(p.relative_to(PROJECT_ROOT)):digest(p) for p in files},
        scope='First small 3-seed/10-annual-episode pilot, not convergence, paper reproduction, or budget-matched to old SAC/PPO.',
        paper='https://arxiv.org/abs/2602.10539',code='https://github.com/Guozheng-Ma/DAWN',
        adaptation='Frozen seven-day persistence/retention MPC, added causal base-proposal feature, +/-50 m3/s residual, base-only warmup, LayerNorm critics, no progressive exploration or explicit critic-only warmup.',
        matched_control='Full-action actor receives identical six features, architecture/LN, warmup, steps, seeds and reward; learns full request rather than bounded additive correction.',
        replay='Normalized residual for residual arm; normalized nominal request for full arm. Critics receive combined nominal request. Warmup executes exact base request.',
        entropy='Density of normalized learnable action before nominal saturation, not delivered release; physical clipping is not included in log probability.',
        selection='Final checkpoint only. All three seeds, ten validation years, plain/shared-filter tracks. No setting or seed selection from validation.',
        bounds='Residual proposal bound before common filter; filter may make a larger physical correction. No hard flood guarantee.',
        budget='3650 steps incl 365 base-only warmup, 3285 SAC updates per arm/seed; six jobs. MPC calls counted in training and evaluation. Old 5-feature SAC200k and PPO1M are contextual references only.')
    write_json(root/'plan.json',plan)
    print('Prepared residual pilot',CONFIG,flush=True)


def new_agent(plan,mode,seed):
    return ProposalSACAgent(SACConfig(**plan['agent']),observation_mean=plan['observation_mean'],
        observation_scale=plan['observation_scale'],mode=mode,bound=plan['config']['residual_bound_m3_s'],high=plan['high'],seed=seed)


def smoke_zero_control(plan):
    """Exact daily zero-correction reproduction on one training year, before learning."""
    params=settings('train');seasonal=SeasonalForecast.fit(params['inflow'],params['demand'])
    reference=make_base(params,seasonal)
    env=MPCProposalEnv(PastInflowObservation(make_env('train',1946),plan['fallback']),lambda:make_base(params,seasonal))
    try:
        obs,info=env.reset()
        for day in range(365):
            expected=reference(obs[:4],info)
            np.testing.assert_array_equal([env.base_request],expected)
            obs,_,terminated,truncated,info=env.step(np.zeros(1,np.float32))
            np.testing.assert_array_equal(info['action'],expected)
            assert not terminated and truncated==(day==364)
        assert env.planner_calls==365
        return dict(passed=True,year=1946,days=365,exact_base_action_reproduction=True,terminal_does_not_plan=True,
                    planner_calls=env.planner_calls,planner_seconds=env.planner_seconds)
    finally: env.close()


def train_one(root,mode,seed,smoke=False):
    plan=verify(root);cfg=plan['config'];torch.set_num_threads(1)
    if mode not in cfg['modes'] or (not smoke and seed not in cfg['seeds']): raise ValueError('Unknown arm/seed')
    if not smoke:
        gate=json.loads((root/'smoke.json').read_text());assert gate['complete']
    out=root/(f'smoke_{mode}' if smoke else f'{mode}_{seed}')
    if out.exists(): raise ValueError('Preserve existing job output')
    out.mkdir();steps=cfg['smoke_steps'] if smoke else cfg['steps']
    params=settings('train');seasonal=SeasonalForecast.fit(params['inflow'],params['demand'])
    env=MPCProposalEnv(PastInflowObservation(make_env('train'),plan['fallback']),lambda:make_base(params,seasonal),mode=mode,bound=cfg['residual_bound_m3_s'])
    agent=new_agent(plan,mode,seed);buffer=ReplayBuffer(cfg['replay_capacity'],6,1,seed)
    obs,info=env.reset(seed=cfg['environment_seed'])
    initial_actor={k:v.clone() for k,v in agent.actor.state_dict().items()}
    episodes=[];trace=[];warmup=[];return_sum=0.;length=0;max_balance=0.;max_delta=0.;year_counts={}
    last_storage=float(info['storage']);started=time.perf_counter();metrics=None
    max_gradient=0.
    try:
        for step in range(1,steps+1):
            base=float(obs[-1])
            if step<=cfg['warmup_steps']:
                action=np.array([0. if mode=='residual' else 2*base/plan['high']-1],np.float32)
                physical=np.array([0. if mode=='residual' else base],np.float32)
            else:
                action=agent.act(obs);physical=agent.physical_action(torch.from_numpy(action)).numpy()
            next_obs,reward,terminated,truncated,info=env.step(physical)
            sim=env.unwrapped;year=int(sim.curr_year_data)
            assert year in split_years('train')
            year_counts[year]=year_counts.get(year,0)+1
            assert np.isfinite(np.r_[obs,next_obs,physical,reward]).all()
            np.testing.assert_allclose(sum(info['weighted_reward'].values()),reward,atol=1e-7,rtol=1e-10)
            np.testing.assert_allclose(next_obs[0],info['level'],atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(next_obs[4],sim.inflow[sim.current_step-1],atol=1e-5,rtol=1e-6)
            error=abs(info['storage']-last_storage-86400*(sim.inflow[sim.current_step-1]-info['release']))
            max_balance=max(max_balance,error);assert error<1e-4
            max_delta=max(max_delta,abs(info['effective_correction_m3_s']))
            if mode=='residual': assert abs(info['effective_correction_m3_s'])<=cfg['residual_bound_m3_s']+1e-4
            buffer.add(obs,action,reward,next_obs,terminated,truncated)
            if step<=cfg['warmup_steps']:
                assert info['nominal_request_m3_s']==base
                warmup.append((year,length+1,base,float(reward)))
            if step>cfg['warmup_steps']:
                metrics=agent.update(buffer.sample(cfg['batch_size']))
                assert all(np.isfinite(v) for v in metrics.values())
                max_gradient=max(max_gradient,metrics['critic_gradient_norm'],metrics['actor_gradient_norm'])
            length+=1;return_sum+=reward;last_storage=float(info['storage']);obs=next_obs
            assert not terminated and truncated==(length==365)
            if truncated:
                assert obs[-1]==0.
                episodes.append(dict(step=step,year=year,length=length,full_return=return_sum))
                if step<steps:
                    obs,info=env.reset();last_storage=float(info['storage'])
                length=0;return_sum=0.
            if step%365==0 or step==steps:
                trace.append(dict(step=step,updates=agent.updates,elapsed_seconds=time.perf_counter()-started,
                                  **(metrics or {})))
                pd.DataFrame(trace).to_csv(out/'learning_trace.csv',index=False)
                pd.DataFrame(episodes).to_csv(out/'training_episodes.csv',index=False)
                print(mode,seed,step,'/',steps,'seconds',round(time.perf_counter()-started,1),flush=True)
    finally: env.close()
    assert agent.updates==steps-cfg['warmup_steps']
    assert any(not torch.equal(v,agent.actor.state_dict()[k]) for k,v in initial_actor.items())
    agent.save(out/'final.pt');restored=ProposalSACAgent.load(out/'final.pt')
    np.testing.assert_array_equal(agent.act(obs,deterministic=True),restored.act(obs,deterministic=True))
    write_json(out/'run.json',dict(complete=True,mode=mode,seed=seed,steps=steps,updates=agent.updates,
        annual_episodes=len(episodes),max_balance_error_m3=max_balance,max_effective_correction_m3_s=max_delta,
        max_gradient_norm=max_gradient,year_transition_counts=year_counts,warmup=warmup,
        planner_calls=env.planner_calls,planner_seconds=env.planner_seconds,
        training_seconds=time.perf_counter()-started,checkpoint_sha256=digest(out/'final.pt'),
        plan_sha256=digest(root/'plan.json'),test_evaluated=False))


class ProposalController:
    def __init__(self,agent,base,fallback,safety_filter=None):
        self.agent,self.base,self.fallback,self.filter=agent,base,fallback,safety_filter
        self.previous_storage=None;self.previous_day=None;self.records=[]
    def __call__(self,observation,info):
        obs=np.asarray(observation,np.float32)
        day=int(np.rint((np.arctan2(obs[1],obs[2])%(2*np.pi))*365/(2*np.pi)))%365+1
        if day==1: self.previous_storage=None
        elif self.previous_day!=day-1: raise ValueError('Consecutive observations required')
        storage=float(info['storage'])
        past=self.fallback if self.previous_storage is None else max(0.,(storage-self.previous_storage)/86400+float(info['release']))
        five=np.r_[obs,past].astype(np.float32)
        base=float(self.base(obs,info)[0]);six=np.r_[five,base].astype(np.float32)
        a=self.agent.act(six,deterministic=True)
        physical=float(self.agent.physical_action(torch.from_numpy(a))[0])
        proposed=(combine_request(base,physical,self.agent.high) if self.agent.mode=='residual' else np.float32(physical))
        final=self.filter(np.array([proposed],np.float32),five,info) if self.filter else np.array([proposed],np.float32)
        self.records.append(dict(day=day,base_request_m3_s=base,learned_action_m3_s=physical,
            prefilter_request_m3_s=float(proposed),prefilter_correction_m3_s=float(proposed)-base,
            final_request_m3_s=float(final[0]),final_correction_m3_s=float(final[0])-base,
            solver_success=self.base.diagnostics[-1]['solver_success'],
            retained_iteration_limit=self.base.diagnostics[-1]['retained_iteration_limit']))
        self.previous_storage,self.previous_day=storage,day
        return final


def evaluate_one(root,mode,seed):
    plan=verify(root);torch.set_num_threads(1);out=root/f'{mode}_{seed}'
    run=json.loads((out/'run.json').read_text())
    assert run['complete'] and run['checkpoint_sha256']==digest(out/'final.pt')
    assert run['plan_sha256']==digest(root/'plan.json') and run['steps']==plan['config']['steps']
    if (out/'evaluation.json').exists(): raise ValueError('Preserve completed evaluation')
    agent=ProposalSACAgent.load(out/'final.pt');before={k:v.clone() for k,v in agent.state_dict().items()}
    rng=agent.generator.get_state().clone()
    params=settings('train');seasonal=SeasonalForecast.fit(params['inflow'],params['demand'])
    model=PlanningModel(params,MPCConfig.load());limits=fit_error_limits(params['inflow'],seasonal.inflow)
    annuals=[];decisions=[]
    for variant in ['plain','shared_filter']:
        filt=FloodSafetyFilter(model,plan['fallback'],limits) if variant=='shared_filter' else None
        controller=ProposalController(agent,make_base(params,seasonal),plan['fallback'],filt)
        daily,_=evaluate('mpc_residual' if mode=='residual' else 'proposal_full',split='validation',protocol='chronological',
            correct_terminal_level=True,sequential_years=True,measure_decisions=True,controller_factory=lambda env:controller)
        daily['mode'],daily['training_seed'],daily['variant']=mode,seed,variant
        daily['controller']=f'{mode}_{seed}_{variant}'
        annual=annual_metrics(daily)
        severity=level_severity(daily.assign(objective=variant)).reset_index().drop(columns=['objective','max_level_m','min_level_m'])
        annual=annual.merge(severity,on='year',validate='one_to_one')
        annual['mode'],annual['training_seed'],annual['variant']=mode,seed,variant
        annuals.append(annual)
        dec=pd.DataFrame(controller.records);dec['mode'],dec['training_seed'],dec['variant']=mode,seed,variant
        dec['year']=daily.year.to_numpy();decisions.append(dec)
        daily.to_csv(out/f'trajectories_{variant}.csv',index=False)
        if filt: pd.DataFrame(filt.diagnostics).to_csv(out/'filter_diagnostics.csv',index=False)
        print('Evaluated',mode,seed,variant,flush=True)
    pd.concat(annuals,ignore_index=True).to_csv(out/'annual.csv',index=False)
    pd.concat(decisions,ignore_index=True).to_csv(out/'decisions.csv',index=False)
    assert all(torch.equal(v,agent.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,agent.generator.get_state())
    write_json(out/'evaluation.json',dict(complete=True,actor_and_statistics_unchanged=True,annual_episodes=20,test_evaluated=False))


def aggregate(root):
    plan=verify(root);frames=[];annuals=[];decisions=[];runs=[]
    for mode in plan['config']['modes']:
        for seed in plan['config']['seeds']:
            out=root/f'{mode}_{seed}';run=json.loads((out/'run.json').read_text());runs.append(run)
            assert json.loads((out/'evaluation.json').read_text())['complete']
            annuals.append(pd.read_csv(out/'annual.csv'))
            decisions.append(pd.read_csv(out/'decisions.csv'))
            frames += [pd.read_csv(out/f'trajectories_{v}.csv',float_precision='round_trip') for v in ['plain','shared_filter']]
    daily=pd.concat(frames,ignore_index=True);annual=pd.concat(annuals,ignore_index=True)
    reference=pd.read_csv(PROJECT_ROOT/'results/safety/common_filter_v1/trajectories.csv',float_precision='round_trip')
    verify_matched_inputs(pd.concat([daily,reference],ignore_index=True),split_years('validation'))
    assert all(run['warmup']==runs[0]['warmup'] for run in runs)
    assert all(run['year_transition_counts']==runs[0]['year_transition_counts'] for run in runs)
    seed_means=annual.groupby(['mode','variant','training_seed'])[METRICS].mean().reset_index()
    summary=seed_means.groupby(['mode','variant'])[METRICS].mean().reset_index()
    variability=seed_means.groupby(['mode','variant']).full_return.agg(seed_sd='std',weakest_seed='min',best_seed='max').reset_index()
    all_decisions=pd.concat(decisions,ignore_index=True)
    for name,table in [('trajectories',daily),('annual',annual),('seed_means',seed_means),('summary',summary),('variability',variability),('decisions',all_decisions)]:
        table.to_csv(root/f'{name}.csv',index=False)
    base_annual=pd.read_csv(PROJECT_ROOT/'results/safety/common_filter_v1/annual.csv')
    matched=[]
    for variant in ['plain','shared_filter']:
        left=annual.query('mode=="residual" and variant==@variant')
        right=annual.query('mode=="full" and variant==@variant')
        pair=left.merge(right,on=['training_seed','year'],suffixes=('_residual','_full'),validate='one_to_one')
        row=pair[['training_seed','year']].copy();row['variant']=variant
        for metric in METRICS: row[f'delta_{metric}']=pair[f'{metric}_residual']-pair[f'{metric}_full']
        matched.append(row)
    pd.concat(matched,ignore_index=True).to_csv(root/'paired_residual_minus_full.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,metric,title in zip(axes,['full_return','low_water_days','deficit_million_m3'],['Annual reward ↑','Low-water days ↓','Unmet demand, million m³ ↓']):
        for track,offset in [('plain',0),('shared_filter',4)]:
            ref_variant='unfiltered' if track=='plain' else 'filtered'
            values=[base_annual.query('family=="mpc" and variant==@ref_variant')[metric].mean(),
                    summary.query('mode=="full" and variant==@track')[metric].iloc[0],
                    summary.query('mode=="residual" and variant==@track')[metric].iloc[0]]
            ax.bar(np.arange(3)+offset,values,color=['#b78219','#267ead','#27856a'])
        ax.set_xticks([0,1,2,4,5,6],['MPC','Full','Residual']*2,rotation=25);ax.set_title(title)
        ax.text(1,-.25,'Plain',ha='center',transform=ax.get_xaxis_transform());ax.text(5,-.25,'Shared filter',ha='center',transform=ax.get_xaxis_transform())
    fig.suptitle('First MPC residual pilot · 3 seeds × 3,650 steps per learned arm\nMatched full-action control also sees the MPC proposal; validation only')
    fig.savefig(root/'residual_pilot.png',dpi=160);plt.close(fig)
    verify(root)
    write_json(root/'run.json',dict(complete=True,test_evaluated=False,training_jobs=6,training_interactions=6*plan['config']['steps'],
        annual_validation_episodes=120,warmup_and_year_exposure_matched=True,sources_unchanged=True,
        training_planner_calls=sum(r['planner_calls'] for r in runs),training_planner_seconds=sum(r['planner_seconds'] for r in runs),
        original_defaults_unchanged=True))
    print(summary.to_string(index=False),flush=True);print(variability.to_string(index=False),flush=True)


def launch_jobs(root,stage,workers):
    plan=verify(root)
    def launch(job):
        mode,seed=job
        with (root/f'{stage}_{mode}_{seed}.log').open('x') as log:
            subprocess.run([sys.executable,'-m','reservoir_control.systems.lake_como.residual_study','--stage',stage+'_worker',
                '--output',str(root),'--mode',mode,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[(mode,seed) for mode in plan['config']['modes'] for seed in plan['config']['seeds']]
    with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(launch,jobs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','smoke','train','evaluate','train_worker','evaluate_worker'],required=True)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/residual/pilot_v1')
    parser.add_argument('--mode',choices=['residual','full']);parser.add_argument('--seed',type=int)
    parser.add_argument('--workers',type=int,default=6);args=parser.parse_args();root=args.output.resolve()
    if args.stage=='prepare': prepare(root)
    elif args.stage=='smoke':
        plan=verify(root);zero=smoke_zero_control(plan)
        for mode in plan['config']['modes']: train_one(root,mode,plan['config']['smoke_seed'],smoke=True)
        write_json(root/'smoke.json',dict(complete=True,zero_control=zero,both_modes_passed=True))
    elif args.stage=='train_worker': train_one(root,args.mode,args.seed)
    elif args.stage=='evaluate_worker': evaluate_one(root,args.mode,args.seed)
    elif args.stage=='train': launch_jobs(root,'train',args.workers)
    else:
        launch_jobs(root,'evaluate',args.workers);aggregate(root)


if __name__=='__main__': main()
