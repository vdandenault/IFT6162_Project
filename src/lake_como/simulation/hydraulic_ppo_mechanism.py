"""Predeclared fixed-training-state mechanism audit of a request transform.

Compare original actor/raw requests, same actor with the warp, and trained warped
actor. Identical standard-Normal draws; same factual state and recorded weather.
This separates the immediate map effect from subsequent learning descriptively.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .ppo_study import ROOT as BASELINE,SEEDS,save,verify
from .ppo_identification import ROOT as IDENTIFICATION,FIT_PAIRS,SATURATION_ANCHORS,DRAWS,restore,daily_mean
from .hydraulic_ppo_study import ROOT
from .hydraulic_ppo import HydraulicPPOAgent
from .ppo_agent import PPOAgent
from .outlet import OutletDomainError


def prepare(root):
    plan=json.loads((root/'plan.json').read_text());verify(plan)
    if (root/'summary.json').exists():raise ValueError('Prepare mechanism plan before final outcomes exist')
    paths=[Path(__file__),root/'plan.json',IDENTIFICATION/'plan.json']
    save(root/'mechanism_plan.json',dict(source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        seeds=list(SEEDS),pairs=[list(p) for p in FIT_PAIRS],anchors=list(SATURATION_ANCHORS),draws=DRAWS,
        arms=['baseline','map_only','trained_map'],
        protocol='Same 180 frozen FIT states from original actor rep0. Same standard-Normal draws per state in all arms. Baseline raw requests, same baseline actor with deterministic warp, trained warped actor. Score full-day physical release at the same recorded weather; policy never receives that forcing.',
        failures='No full required-scope means unless all five training seeds and all state evaluations complete. No policy/checkpoint selection.',
        test_evaluated=False,validation_evaluated=False))


def run(root):
    verify(json.loads((root/'plan.json').read_text()));verify(json.loads((root/'mechanism_plan.json').read_text()))
    statuses=[json.loads((root/f'seed{s}/training.json').read_text()) for s in SEEDS]
    if not all(r['complete'] for r in statuses):
        save(root/'mechanism_summary.json',dict(complete=False,reason='Incomplete training seed; endpoint comparison suppressed',test_evaluated=False));return
    torch.set_num_threads(1);rows=[]
    for seed in SEEDS:
        baseline=PPOAgent.load(BASELINE/f'seed{seed}/final.pt')
        mapped=HydraulicPPOAgent(baseline.config,action_low=baseline.model.action_low.numpy(),action_high=baseline.model.action_high.numpy(),seed=seed)
        mapped.model.load_state_dict(baseline.model.state_dict());trained=HydraulicPPOAgent.load(root/f'seed{seed}/final.pt')
        for years in FIT_PAIRS:
            snapshots=json.loads((IDENTIFICATION/f'seed{seed}/fit_{years[0]}.json').read_text())['snapshots']
            for ordinal in SATURATION_ANCHORS:
                env,memory=restore(years,snapshots[str(ordinal)]);x=torch.tensor(memory.last_encoded).reshape(1,-1).expand(DRAWS,-1)
                low=daily_mean(env,0);high=daily_mean(env,float(env.action_space.high[0]))
                for arm,agent in [('baseline',baseline),('map_only',mapped),('trained_map',trained)]:
                    torch.manual_seed(700000+seed*10000+years[0]*10+ordinal)
                    with torch.no_grad():z=agent.model.distribution(x).sample();raw=agent.model.physical(z).reshape(-1).numpy()
                    requests=raw if arm=='baseline' else np.array([agent.transform(memory.last_encoded,[q])[0] for q in raw])
                    try:
                        effective=np.array([daily_mean(env,q) for q in requests]);complete=True;failure=None
                    except OutletDomainError as error:effective=np.array([]);complete=False;failure=str(error)
                    rows.append(dict(seed=seed,block_start=years[0],ordinal=ordinal,arm=arm,complete=complete,failure=failure,
                        daily_span=high-low,upper_mass=float(np.mean(abs(effective-high)<=1e-5)) if complete else None,
                        effective_std=float(effective.std()) if complete else None,requested_std=float(requests.std()),
                        mean_request=float(requests.mean()),mean_release=float(effective.mean()) if complete else None))
                env.close()
        print('Mechanism seed',seed,'complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(root/'mechanism.csv',index=False)
    complete=len(frame)==540 and frame.complete.all();valid=frame[frame.daily_span.gt(1e-5)]
    result=dict(complete=bool(complete),state_arm_rows=len(frame),
        means={str(arm):{key:float(group[key].mean()) if complete else None for key in ['upper_mass','effective_std','requested_std','mean_request','mean_release']} for arm,group in valid.groupby('arm')},
        interpretation='Own-state adaptation is excluded: these are the same original baseline fitting states. Changed physical initialization/distribution and state-dependent map are part of the intervention; this is not proof of a unique learning mechanism.',
        test_evaluated=False,validation_evaluated=False)
    save(root/'mechanism_summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['prepare','run'],required=True);args=parser.parse_args()
    prepare(args.root) if args.stage=='prepare' else run(args.root)
