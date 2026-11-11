"""Explain short/long residual behavior from frozen saved validation data only.
No new transitions, training, counterfactual outcomes, or test-year evaluation.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.residual_mpc import ProposalSACAgent
from reservoir_control.systems.lake_como.residual_study import verify
from reservoir_control.systems.lake_como.sac_diagnostics import water_allocations
from reservoir_control.systems.lake_como.mpc_hindsight import digest

ROOTS=dict(short=PROJECT_ROOT/'results/residual/pilot_v1',long=PROJECT_ROOT/'results/residual/duration_36500_v1')
KEYS=['training_seed','year','day']


def recorded_observations(frame,fallback,ema_coeff):
    """Reconstruct exactly the causal six-feature input from pre-decision telemetry."""
    result=[]
    for (_,year),group in frame.groupby(['training_seed','year'],sort=False):
        group=group.sort_values('day');assert group.day.tolist()==list(range(1,366))
        average=0.;previous=None
        for row in group.itertuples(index=False):
            angle=2*math.pi*(row.day-1)/365
            past=fallback if previous is None else max(0.,(row.storage_before_m3-previous.storage_before_m3)/86400+previous.release_m3_s)
            result.append([row.level_before_m,math.sin(angle),math.cos(angle),average,past,row.base_request_m3_s])
            average=ema_coeff*average+(1-ema_coeff)*row.demand_m3_s;previous=row
    # Caller sorts by seed/year/day before reconstruction.
    return np.asarray(result,np.float32)


@torch.no_grad()
def critic_snapshot(agent,raw,recorded_delta):
    obs=agent.normalize(torch.from_numpy(raw))
    action,_=agent.actor(obs,deterministic=True)
    physical=agent.physical_action(action).numpy().ravel()
    q1=agent.q1(obs,action).numpy().ravel();q2=agent.q2(obs,action).numpy().ravel()
    zero=torch.zeros_like(action);qzero=torch.minimum(agent.q1(obs,zero),agent.q2(obs,zero)).numpy().ravel()
    grid=np.linspace(-1,1,21,dtype=np.float32)
    surfaces=[]
    for value in grid:
        a=torch.full_like(action,float(value))
        surfaces.append(torch.minimum(agent.q1(obs,a),agent.q2(obs,a)).numpy().ravel())
    surface=np.stack(surfaces,axis=1)
    return pd.DataFrame(dict(actor_delta_m3_s=physical,min_q=np.minimum(q1,q2),twin_gap=np.abs(q1-q2),
        q_actor_minus_zero=np.minimum(q1,q2)-qzero,grid_argmax_delta=grid[surface.argmax(axis=1)]*agent.bound,
        grid_q_range=surface.max(axis=1)-surface.min(axis=1),recorded_delta_m3_s=recorded_delta))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/residual/diagnosis_v3');root=p.parse_args().output
    if root.exists() and any(root.iterdir()):raise ValueError('Preserve prior output; choose empty directory')
    root.mkdir(parents=True);torch.set_num_threads(1);params=settings('train');plans={k:verify(v) for k,v in ROOTS.items()}
    sources=[Path(__file__)]
    for source in ROOTS.values():
        sources += [source/'plan.json',source/'trajectories.csv',source/'decisions.csv']
        sources += [source/f'residual_{seed}/{name}' for seed in [123,124,125] for name in ['final.pt','learning_trace.csv']]
    hashes={str(f.relative_to(PROJECT_ROOT)):digest(f) for f in sources}
    (root/'plan.json').write_text(json.dumps(dict(scope='Post-hoc read-only diagnosis on previously evaluated validation years. All seeds, both final checkpoints, no tuning or new simulator calls. Common-state critics are hypotheses, not calibrated returns.',source_sha256=hashes),indent=2)+'\n')
    frames=[];traces=[]
    for stage,source in ROOTS.items():
        frame=pd.read_csv(source/'trajectories.csv',float_precision='round_trip').query('mode=="residual" and variant=="plain"').copy()
        dec=pd.read_csv(source/'decisions.csv',float_precision='round_trip').query('mode=="residual" and variant=="plain"')
        frame=frame.merge(dec.drop(columns=['mode','variant']),on=KEYS,validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
        frame['stage']=stage;frames.append(water_allocations(frame,params['lake_params']['min_env_flow']))
        for seed in [123,124,125]:
            t=pd.read_csv(source/f'residual_{seed}/learning_trace.csv');t['training_seed']=seed;t['stage']=stage;traces.append(t)
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(root/'daily.csv',index=False)
    pair=frames[1].merge(frames[0],on=KEYS,suffixes=('_long','_short'),validate='one_to_one')
    for col in ['inflow_m3_s','demand_m3_s']:np.testing.assert_array_equal(pair[col+'_long'],pair[col+'_short'])
    allocation=['useful_m3','surplus_after_mef_m3','environmental_allocation_m3','released_m3','deficit_m3','low_water','flood']
    penalties=[c for c in frame if c.startswith('weighted_')]
    annual=frame.groupby(['stage','training_seed','year'])[allocation+penalties].sum().reset_index();annual.to_csv(root/'annual_accounting.csv',index=False)
    means=annual.groupby('stage')[allocation+penalties].mean();delta=means.loc['long']-means.loc['short']
    terminal=frame.query('day==365').groupby('stage').storage_m3.mean()
    np.testing.assert_allclose(delta.deficit_m3,delta.surplus_after_mef_m3+delta.environmental_allocation_m3+terminal['long']-terminal['short'],atol=1e-6)
    monthly=frame.groupby(['stage','month']).agg(request=('request_m3_s','mean'),release=('release_m3_s','mean'),
        correction=('prefilter_correction_m3_s','mean'),level=('level_m','mean'),demand=('demand_m3_s','mean'),
        low_water_days=('low_water',lambda x:x.sum()/30),unmet_million_m3=('deficit_m3',lambda x:x.sum()/30/1e6),
        surplus_million_m3=('surplus_after_mef_m3',lambda x:x.sum()/30/1e6)).reset_index();monthly.to_csv(root/'monthly.csv',index=False)
    pd.concat(traces).to_csv(root/'learning_traces.csv',index=False)
    scores=[]
    for state_stage,states in zip(ROOTS,frames):
        raw=recorded_observations(states,plans[state_stage]['fallback'],params['exponential_average_coeff'])
        for seed in [123,124,125]:
            mask=states.training_seed.to_numpy()==seed
            for model_stage,source in ROOTS.items():
                agent=ProposalSACAgent.load(source/f'residual_{seed}/final.pt');before={k:v.clone() for k,v in agent.state_dict().items()};rng=agent.generator.get_state().clone()
                scored=critic_snapshot(agent,raw[mask],states.loc[mask,'learned_action_m3_s'].to_numpy())
                if state_stage==model_stage:
                    # Saved evaluation used single-state forwards; batched GEMM can differ slightly.
                    replayed=np.array([float(agent.physical_action(torch.from_numpy(agent.act(r,deterministic=True)))[0]) for r in raw[mask]])
                    np.testing.assert_allclose(replayed,scored.recorded_delta_m3_s,rtol=0,atol=1e-11)
                for k,v in before.items():assert torch.equal(v,agent.state_dict()[k])
                assert torch.equal(rng,agent.generator.get_state())
                scored['state_stage']=state_stage;scored['model_stage']=model_stage;scored['training_seed']=seed
                scored['year']=states.loc[mask,'year'].to_numpy();scored['day']=states.loc[mask,'day'].to_numpy();scores.append(scored)
    scored=pd.concat(scores);scored.to_csv(root/'common_state_critics.csv',index=False)
    scored.groupby(['state_stage','model_stage'])[['actor_delta_m3_s','min_q','twin_gap','q_actor_minus_zero','grid_argmax_delta','grid_q_range']].mean().reset_index().to_csv(root/'critic_summary.csv',index=False)
    summary=dict(complete=True,new_simulator_transitions=0,new_updates=0,test_evaluated=False,
        extra_unmet_million_m3=delta.deficit_m3/1e6,extra_surplus_million_m3=delta.surplus_after_mef_m3/1e6,
        extra_environmental_allocation_million_m3=delta.environmental_allocation_m3/1e6,
        extra_terminal_storage_million_m3=(terminal['long']-terminal['short'])/1e6,
        extra_penalties={k:float(-delta[k]) for k in penalties},recorded_actor_requests_reproduced=True,
        interpretation='Exact water/penalty accounting and frozen common-state Q surfaces, not a causal explanation of training instability or evidence that Q differences equal real returns.')
    for f,h in hashes.items():assert digest(PROJECT_ROOT/f)==h,f
    (root/'run.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print(monthly.to_string(index=False))

if __name__=='__main__':main()
