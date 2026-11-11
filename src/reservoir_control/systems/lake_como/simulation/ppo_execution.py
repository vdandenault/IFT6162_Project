"""Frozen PPO execution alignment: deterministic mean versus stochastic policy.

All learner/action seeds retained. Compare action-seed means within policy/year;
never duplicate deterministic trajectories as independent action replications.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.ppo_study import SEEDS,METRICS,save,verify
from reservoir_control.systems.lake_como.simulation.ppo_curve_audit import SOURCES,run as curves
from reservoir_control.systems.lake_como.simulation.ppo_agent import PPOAgent
from reservoir_control.systems.lake_como.simulation.hydraulic_ppo import HydraulicPPOAgent
from reservoir_control.systems.lake_como.simulation.ppo_protocol import ObservationHistory
from reservoir_control.systems.lake_como.simulation.ppo_diagnosis import make_env
from reservoir_control.systems.lake_como.simulation.carryover_check import take_step,FIELDS
from reservoir_control.systems.lake_como.simulation.precision_validation import annual_records
from reservoir_control.systems.lake_como.simulation.target_year_check import validation_pairs
from reservoir_control.systems.lake_como.simulation.outlet import OutletDomainError

ROOT=PROJECT_ROOT/'results/simulator/ppo_execution_v1'
ACTION_SEEDS=(2101,2102,2103,2104,2105)
MODELS=('original','mapped')
MODES=('deterministic','stochastic')


def specs():
    rows=[]
    for model in MODELS:
        for seed in SEEDS:
            for years in validation_pairs():
                for mode in MODES:
                    for action_seed in ((None,) if mode=='deterministic' else ACTION_SEEDS):
                        name=f'{model}_p{seed}_{years[0]}_{mode}_a{action_seed}'
                        rows.append(dict(name=name,model=model,seed=seed,years=list(years),mode=mode,action_seed=action_seed))
    return rows


def noise_seed(action_seed,block_start):return int(action_seed+10000*block_start)


def frozen_digest(agent):
    h=hashlib.sha256()
    for key,value in sorted(agent.model.state_dict().items()):h.update(key.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    reference=json.loads((SOURCES['mapped']/'plan.json').read_text());verify(reference)
    paths=[Path(__file__),Path(__file__).with_name('ppo_curve_audit.py')]
    for source in SOURCES.values():
        paths += [source/'plan.json',source/'summary.json']
        for seed in SEEDS:paths += [p for p in (source/f'seed{seed}').iterdir() if p.is_file()]
    hashes=dict(reference['source_sha256']);hashes.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(root/'plan.json',dict(source_sha256=hashes,cases=specs(),learner_seeds=list(SEEDS),action_seeds=list(ACTION_SEEDS),
        quartiles=reference['quartiles'],
        execution='Same final frozen checkpoints. Deterministic is transform(tanh(Gaussian mean)); stochastic samples the learned Gaussian before the same transforms. Deterministic is not generally the mean physical action. No policy/critic/normalizer updates.',
        randomness='Torch reseeded after loading/resetting: action_seed + 10000*block_start. Common standard-Normal draws across policy seeds and original/mapped actors. Five action seeds are conditional noise repetitions, not new weather years.',
        environment='Same five two-year validation blocks, .35m reset, initial EMA/prior, causal history, daily commands and protected hourly physical model. State/history carry through annual boundary. No costs/physics/task changes.',
        controls='All 50 deterministic episodes must replay the saved original trajectories bitwise; source/checkpoint/state-digest/counter integrity verified.',
        aggregation='First average stochastic repetitions within learner seed and historical year, then pair with its single deterministic outcome. No deterministic pseudo-replication, action-seed selection, p-values or best-mode/default promotion.',
        outcomes='Annual cost, benchmark target shortage, flood/low-water days and severity, episode-ending stock; full-block physical and gamma .995 discounted costs. Report action-seed sensitivity and completion/domain failures.',
        curves='Completed training episodes only, 730-day accounting; exclude unfinished 710-step prefixes. Within-weather-window slopes over early/late 100k and last50k; log optimization and Gaussian spread. No fixed-policy learning curve without saved intermediate checkpoints.',
        duration_rule='Future duration probe supported only if >=3/5 mapped seeds have late fitted cost decline >=2% per100k and >=3/5 still decline in last50k. No new training as part of this study; failure does not prove convergence.',
        failures='No full-block cost for incomplete trajectories. Required model/mode means and paired comparisons suppressed if any member incomplete. Preserve failure/prefix records; no outside-domain penalty, clamping or hidden omission.',
        test_evaluated=False,new_training=False,default_changed=False))


def run_case(args):
    root,spec=args;root=Path(root)
    if spec not in specs():raise ValueError('Declared validation/model/seed/mode case required')
    folder=root/spec['name'];folder.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    source=SOURCES[spec['model']];kind=PPOAgent if spec['model']=='original' else HydraulicPPOAgent
    agent=kind.load(source/f"seed{spec['seed']}/final.pt");before=frozen_digest(agent);counters=(agent.steps,agent.updates)
    years=tuple(spec['years']);env,obs,info=make_env('validation',years);memory=ObservationHistory()
    if spec['mode']=='stochastic':torch.manual_seed(noise_seed(spec['action_seed'],years[0]))
    rows=[];failure=None;started=time.monotonic()
    def controller(raw,metadata):return agent.act(memory.encode(raw,metadata),deterministic=spec['mode']=='deterministic')[0]
    try:
        for day in range(730):
            obs,info,row,term,trunc=take_step(env,controller,obs,info)
            if term!=(day==729) or trunc:raise ValueError('Finite task boundary mismatch')
            rows.append(row)
    except (OutletDomainError,ValueError,FloatingPointError) as error:failure=dict(type=type(error).__name__,message=str(error))
    frame=pd.DataFrame(rows,columns=FIELDS);frame.to_csv(folder/'daily.csv',index=False)
    complete=len(frame)==730 and failure is None
    control=None
    if spec['mode']=='deterministic':
        reference=pd.read_csv(source/f"seed{spec['seed']}/{years[0]}_{years[1]}.csv",float_precision='round_trip')
        np.testing.assert_array_equal(frame[list(FIELDS)].to_numpy(float),reference[list(FIELDS)].to_numpy(float));control=True
    if frozen_digest(agent)!=before or (agent.steps,agent.updates)!=counters:raise AssertionError('Frozen checkpoint changed during execution')
    record=dict(**spec,complete=complete,observed_days=len(frame),failure=failure,model_digest=before,model_unchanged=True,
        deterministic_reference_bitwise_match=control,elapsed_seconds=time.monotonic()-started,
        two_year_cost=float(-frame.reward.sum()) if complete else None,
        discounted_cost=float(np.dot(agent.config.gamma**np.arange(730),-frame.reward.to_numpy())) if complete else None,
        end_storage_million_m3=env.state.storage/1e6 if complete else None)
    save(folder/'run.json',record);env.close()
    return f"{spec['name']}: {len(frame)}/730, failure={failure}"


def pair_annual(annual):
    keys=['model','seed','block_start','year'];columns=list(METRICS)+['end_storage_million_m3']
    deterministic=annual[annual['mode'].eq('deterministic')].set_index(keys)
    stochastic=annual[annual['mode'].eq('stochastic')].groupby(keys)[columns].mean()
    if not deterministic.index.is_unique:raise ValueError('Single deterministic outcome per policy/year required')
    stochastic=stochastic.reindex(deterministic.index);paired=deterministic[[]].copy()
    for metric in columns:
        paired['deterministic_'+metric]=deterministic[metric];paired['stochastic_'+metric]=stochastic[metric]
        paired['delta_'+metric]=stochastic[metric]-deterministic[metric]
    return paired.reset_index()


def aggregate(root,plan):
    area=settings('train')['lake_params']['surface'];annual=[];blocks=[]
    for spec in specs():
        folder=root/spec['name'];record=json.loads((folder/'run.json').read_text());frame=pd.read_csv(folder/'daily.csv',float_precision='round_trip')
        rows=annual_records(frame,dict(**spec,arm=spec['model']),plan['quartiles'],area)
        for row in rows:row.update(model=spec['model'],mode=spec['mode'],action_seed=spec['action_seed'])
        annual.extend(rows);blocks.append(record)
    annual=pd.DataFrame(annual);annual.to_csv(root/'annual.csv',index=False);blocks=pd.DataFrame(blocks);blocks.to_csv(root/'blocks.csv',index=False)
    summaries=[];completeness={}
    for (model,mode),group in annual.groupby(['model','mode']):
        expected=50 if mode=='deterministic' else 250
        complete=len(group)==expected and bool(group.complete.all())
        completeness[(model,mode)]=complete
        row=dict(model=model,mode=mode,complete=complete,annual_records=len(group))
        # Every policy/year has the same five action repetitions. Collapse them
        # first so the deterministic endpoint is counted exactly once.
        averaged=group.groupby(['seed','block_start','year'])[list(METRICS)].mean()
        for metric in METRICS:row[metric]=float(averaged[metric].mean()) if complete else None
        block=blocks[blocks.model.eq(model)&blocks['mode'].eq(mode)]
        row['discounted_two_year_cost']=float(block.discounted_cost.mean()) if complete else None
        row['episode_end_storage_million_m3']=float(block.end_storage_million_m3.mean()) if complete else None
        summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(root/'mode_summary.csv',index=False)
    paired=[]
    for model in MODELS:
        if all(completeness[(model,mode)] for mode in MODES):paired.append(pair_annual(annual[annual.model.eq(model)]))
    paired=pd.concat(paired,ignore_index=True) if paired else pd.DataFrame()
    if len(paired):
        paired.to_csv(root/'paired_annual.csv',index=False)
        paired.groupby(['model','seed'])[[c for c in paired if c.startswith('delta_')]].mean().to_csv(root/'paired_learner_means.csv')
    stochastic=annual[annual['mode'].eq('stochastic')]
    if bool(stochastic.complete.all()):stochastic.groupby(['model','action_seed'])[list(METRICS)].mean().to_csv(root/'action_seed_means.csv')
    result=dict(all_required_complete=bool(all(completeness.values())),required_episodes=300,complete_episodes=int(blocks.complete.sum()),
        deterministic_reference_replays=int(blocks.deterministic_reference_bitwise_match.fillna(False).sum()),
        means=summary.astype(object).where(pd.notna(summary),None).to_dict('records'),
        failures=[{k:row[k] for k in ['name','model','seed','action_seed','observed_days','failure']} for row in blocks.to_dict('records') if row['failure'] is not None],
        test_evaluated=False,new_training=False,default_changed=False)
    save(root/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['prepare','curves','execute','aggregate','all'],default='all');args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare(args.root)
    plan=json.loads((args.root/'plan.json').read_text());verify(plan)
    if args.stage in ['curves','all']:curves(args.root)
    if args.stage in ['execute','all']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for index,result in enumerate(pool.map(run_case,[(str(args.root),s) for s in specs()])):
                if (index+1)%10==0:print(f'{index+1}/300: {result}',flush=True)
    if args.stage in ['aggregate','all']:aggregate(args.root,plan)
    verify(plan)

if __name__=='__main__':main()
