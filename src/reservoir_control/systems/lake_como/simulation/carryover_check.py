"""Four-way 1995 storage/controller intervention, plus MPC memory controls.

Preserves the 1994–1995 episode clock, available exogenous history, forecasts,
physical rules and costs. Boundary storage is the only physical intervention.
No test years or newly fitted controller parameters are used.
"""
from reservoir_control.evaluation.provenance import verify_sources
import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT,DAM_CONFIG,DAM_DATA
from reservoir_control.systems.lake_como.simulation.low_water import make_protected_outlet_env,LowWaterRule
from reservoir_control.systems.lake_como.simulation.model import WaterState,PhysicalWeights
from reservoir_control.systems.lake_como.simulation.outlet import OutletSpec,OutletDomainError
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank,ProtectedMPCConfig
from reservoir_control.systems.lake_como.simulation.mpc_diagnosis import DiagnosticMPC

DEFAULT=PROJECT_ROOT/'results/simulator/carryover_check_v1'
REFERENCE=PROJECT_ROOT/'results/simulator/target_year_check_v1'
PRIMARY=(('mpc_low','mpc','low'),('feedback_low','feedback','low'),
         ('mpc_high','mpc','high'),('feedback_high','feedback','high'))
FIELDS=('elapsed_hour','year','model_day','level_before_m','level_m','minimum_level_m','maximum_level_m',
    'request_m3_s','release_m3_s','inflow_m3_s','target_m3_s','past_target_ema','past_inflow_m3_s',
    'reward','unmet_target_m3','delivered_target_m3','remaining_river_m3','environmental_shortage_m3',
    'flood_excess_m_hours','low_water_excess_m_hours','balance_error_m3')


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@dataclass(frozen=True)
class BoundaryHistory:
    hour: int
    target_ema: float
    last_inflow: float
    common_command: float
    previous_hour: int
    previous_plan: tuple


def make_controller(training,bank):
    physical={k:training[k] for k in ['lake_params','action','flood_level','starving_level']}
    return DiagnosticMPC(physical,bank,forecast='conditional',target_mode='ema_residual',
                        search='restricted',ema_coefficient=training['exponential_average_coeff'])


def restore_branch(bank,boundary,storage):
    if boundary.hour!=365*24 or boundary.previous_hour!=364*24 or len(boundary.previous_plan)!=7:
        raise ValueError('Original 1995 boundary clock and previous seven-day plan required')
    if not np.isfinite([storage,boundary.target_ema,boundary.last_inflow,boundary.common_command]).all():
        raise ValueError('Finite boundary state and history required')
    env=make_protected_outlet_env('validation',(1994,1995));env.fallback=bank.initial_last_inflow
    env.reset(seed=901)
    env.model.lake.check_domain(env.model.lake.storage_to_level(storage))
    if min(boundary.target_ema,boundary.last_inflow)<0 or not env.model.low<=boundary.common_command<=env.model.high:
        raise ValueError('Nonnegative history and bounded common command required')
    plan=np.asarray(boundary.previous_plan,float)
    if not np.isfinite(plan).all() or (plan<env.model.low).any() or (plan>env.model.high).any():
        raise ValueError('Finite bounded carried plan required')
    env.hour=boundary.hour;env.state=WaterState(storage,boundary.common_command,0.)
    env.demand_ema=boundary.target_ema;env.last_inflow=boundary.last_inflow
    return env,env._observation(),env._info()


def restore_memory(controller,boundary,*,cold=False):
    controller.reset()
    if not cold:
        controller.previous_hour=boundary.previous_hour
        controller.previous_plan=np.asarray(boundary.previous_plan,np.float32).copy()


def decompose(costs):
    names=('mpc_low','mpc_high','feedback_low','feedback_high')
    if any(name not in costs or not np.isfinite(costs[name]) for name in names):
        raise ValueError('Four complete finite costs required for decomposition')
    ml,mh,fl,fh=(costs[name] for name in names)
    gap=ml-fh;storage_mpc=ml-mh;storage_feedback=fl-fh
    controller_low=ml-fl;controller_high=mh-fh
    assert abs(gap-(storage_mpc+controller_high))<1e-8
    assert abs(gap-(controller_low+storage_feedback))<1e-8
    return dict(observed_gap=gap,storage_benefit_mpc=storage_mpc,storage_benefit_feedback=storage_feedback,
        controller_gap_low=controller_low,controller_gap_high=controller_high,
        interaction=storage_mpc-storage_feedback,
        symmetric_storage_component=.5*(storage_mpc+storage_feedback),
        symmetric_controller_component=.5*(controller_low+controller_high),
        interpretation='Two exact order-dependent decompositions; symmetric components are an averaging convention, not unique percentages caused.')


def take_step(env,controller,obs,info):
    hour=env.hour;ema=float(obs[3]);last=float(obs[4])
    action=(np.array([np.clip(float(obs[3])+20+150*(float(obs[0])-.35),0,env.model.high)],np.float32)
            if controller is None else controller(obs,info))
    obs,reward,term,trunc,info=env.step(action);rows=info['hourly']
    record=dict(elapsed_hour=hour,year=rows[0]['year'],model_day=rows[0]['model_day'],
        level_before_m=rows[0]['level_before_m'],level_m=rows[-1]['level_m'],
        minimum_level_m=min(r['level_m'] for r in rows),maximum_level_m=max(r['level_m'] for r in rows),
        request_m3_s=float(action[0]),release_m3_s=float(np.mean([r['release_m3_s'] for r in rows])),
        inflow_m3_s=float(np.mean([r['inflow_m3_s'] for r in rows])),target_m3_s=rows[0]['demand_m3_s'],
        past_target_ema=ema,past_inflow_m3_s=last,reward=reward,
        unmet_target_m3=sum(r['unmet_demand_m3'] for r in rows),
        delivered_target_m3=sum(r['diversion_m3'] for r in rows),
        remaining_river_m3=sum(r['river_out_m3'] for r in rows),
        environmental_shortage_m3=sum(r['environmental_shortage_m3'] for r in rows),
        flood_excess_m_hours=sum(r['flood_excess_m_hours'] for r in rows),
        low_water_excess_m_hours=sum(r['low_water_excess_m_hours'] for r in rows),
        balance_error_m3=max(abs(r['lake_balance_error_m3']) for r in rows))
    return obs,info,record,term,trunc


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    references={};paths=(list(Path(__file__).parent.glob('*.py')) + (list((PROJECT_ROOT/'src/reservoir_control/controllers').rglob('*.py')) + [PROJECT_ROOT/'src/reservoir_control/systems/lake_como/sac.py', PROJECT_ROOT/'src/reservoir_control/systems/lake_como/fqi.py']))
    paths += [PROJECT_ROOT/'src/reservoir_control/systems/lake_como/environment.py',PROJECT_ROOT/'src/reservoir_control/systems/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/reservoir_control/systems/lake_como/splits.py',PROJECT_ROOT/'configs/splits.json',PROJECT_ROOT/'pyproject.toml',PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',DAM_CONFIG/'world_train.yaml',DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv',
        REFERENCE/'plan.json',REFERENCE/'annual.csv']
    paths += [DAM_DATA/s/f'water_{s}_{p}.csv' for s in ['inflow','demand'] for p in ['train','test']]
    for controller,arm in [('mpc','ema_restricted'),('feedback','causal_feedback')]:
        source=REFERENCE/('1994_1995__'+arm);record=json.loads((source/'run.json').read_text())
        references[controller]=dict(run=str(source/'run.json'),data=record['data_directory'])
        paths += [source/'run.json',Path(record['data_directory'])/'daily.csv',Path(record['data_directory'])/'decisions.jsonl']
    save(root/'plan.json',dict(primary=[dict(name=n,controller=c,storage=s,memory='carried' if c=='mpc' else 'not_applicable') for n,c,s in PRIMARY],
        supplementary=[dict(name='mpc_'+s+'_cold',controller='mpc',storage=s,memory='cold') for s in ['low','high']],
        references=references,fit_years=list(range(1946,1988)),mpc=asdict(ProtectedMPCConfig()),
        weights=asdict(PhysicalWeights()),outlet=asdict(OutletSpec()),rule=asdict(LowWaterRule()),
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        intervention='1995 initial relative storage only; all exogenous history, time/remaining horizon, common initial command, forecasts and fitted parameters held fixed.',
        memory='Primary MPC arms share the same carried 1994 MPC plan. Two cold-start MPC controls isolate numerical warm-start context.',
        prefix='Reconstruct each original 1994 trajectory and verify against saved data. Keep original (1994,1995) episode with hour=8760, not a new one-year reset.',
        attribution='Two order-specific decompositions and a symmetric averaging convention. No unique causal percentage or proof of a missing terminal value.',
        accounting='High/low initial volume difference reconciles extra release and end storage. Model diversion and remaining river flow are distinct; remaining flow is not called wasted.',
        limits='Boundary swap is a conditional model experiment. The extra water is not free: original 1994 costs and factual two-year totals are reported. One dry validation case, one forecast seed.',
        test_evaluated=False,model_or_costs_changed=False,operationally_calibrated=False))


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    verify_sources(plan['source_sha256'])
    return plan


def reconstruct(root,plan,training,bank):
    if (root/'boundary.json').exists():return
    snapshots={};memory=None
    for name in ['mpc','feedback']:
        env=make_protected_outlet_env('validation',(1994,1995));env.fallback=bank.initial_last_inflow
        obs,info=env.reset(seed=901);controller=make_controller(training,bank) if name=='mpc' else None
        records=[]
        for _ in range(365):
            obs,info,row,term,trunc=take_step(env,controller,obs,info)
            assert not(term or trunc);records.append(row)
        frame=pd.DataFrame(records);source=pd.read_csv(Path(plan['references'][name]['data'])/'daily.csv').iloc[:365]
        for field in ['level_before_m','level_m','request_m3_s','release_m3_s','reward']:
            np.testing.assert_allclose(frame[field],source[field],rtol=0,atol=1e-9)
        frame.to_csv(root/('prefix_1994_'+name+'.csv'),index=False)
        snapshots[name]=dict(state=asdict(env.state),target_ema=env.demand_ema,last_inflow=env.last_inflow,hour=env.hour)
        if controller is not None:
            memory=dict(previous_hour=controller.previous_hour,previous_plan=controller.previous_plan.tolist())
        env.close()
    a,b=snapshots['mpc'],snapshots['feedback']
    for field in ['target_ema','last_inflow','hour']:assert a[field]==b[field],field
    assert a['state']['storage']<b['state']['storage']
    history=BoundaryHistory(a['hour'],a['target_ema'],a['last_inflow'],a['state']['command'],
                           memory['previous_hour'],tuple(memory['previous_plan']))
    save(root/'boundary.json',dict(common_history=asdict(history),low_storage_m3=a['state']['storage'],
        high_storage_m3=b['state']['storage'],original_snapshots=snapshots,prefix_reference_match=True))


def run_branch(root,spec,training,bank,boundary):
    folder=root/spec['name'];folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():return
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    raw=boundary['common_history'];history=BoundaryHistory(**raw)
    storage=boundary[spec['storage']+'_storage_m3']
    env,obs,info=restore_branch(bank,history,storage)
    initial=obs.copy();controller=make_controller(training,bank) if spec['controller']=='mpc' else None
    if controller is not None:restore_memory(controller,history,cold=spec['memory']=='cold')
    records=[];failure=None
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(365):
                obs,info,row,term,trunc=take_step(env,controller,obs,info);records.append(row)
                if controller is not None:log.write(json.dumps(controller.diagnostics[-1],allow_nan=False)+'\n')
                if term or trunc:break
        except OutletDomainError as error:failure=dict(error=str(error),elapsed_hour=env.hour)
        finally:
            final_storage=env.state.storage;env.close()
    frame=pd.DataFrame(records,columns=FIELDS);frame.to_csv(data/'daily.csv',index=False)
    complete=len(frame)==365 and failure is None
    metrics=dict(annual_cost=float(-frame.reward.sum()) if complete else None,
        observed_prefix_cost=float(-frame.reward.sum()),
        total_release_m3=float((frame.release_m3_s*86400).sum()),
        total_inflow_m3=float((frame.inflow_m3_s*86400).sum()),
        target_diversion_m3=float(frame.delivered_target_m3.sum()),remaining_river_m3=float(frame.remaining_river_m3.sum()),
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6),
        environmental_shortage_million_m3=float(frame.environmental_shortage_m3.sum()/1e6),
        flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
        flood_days=int((frame.flood_excess_m_hours>0).sum()),low_water_days=int((frame.low_water_excess_m_hours>0).sum()),
        minimum_level_m=float(frame.minimum_level_m.min()) if len(frame) else None,
        maximum_level_m=float(frame.maximum_level_m.max()) if len(frame) else None,
        maximum_balance_error_m3=float(frame.balance_error_m3.abs().max()) if len(frame) else None)
    # Endpoint accounting is asserted for complete branches only; a rejected
    # partial day may have advanced valid hours that do not form a scored day.
    if complete:assert abs(final_storage-storage-metrics['total_inflow_m3']+metrics['total_release_m3'])<1e-4
    save(folder/'run.json',dict(**spec,complete=complete,observed_days=len(frame),failure=failure,
        data_directory=str(data),initial_observation=initial.tolist(),initial_storage_m3=storage,
        final_storage_m3=final_storage,**metrics))
    print('FINISHED',spec['name'],metrics['annual_cost'],'complete',complete,flush=True)


def aggregate(root,plan,boundary):
    records={}
    for spec in plan['primary']+plan['supplementary']:
        records[spec['name']]=json.loads((root/spec['name']/'run.json').read_text())
    pd.DataFrame([{k:v for k,v in record.items() if k not in ['failure','initial_observation']} for record in records.values()]).to_csv(root/'summary.csv',index=False)
    primary=all(records[name]['complete'] for name,_,_ in PRIMARY)
    if primary:
        decomposition=decompose({name:records[name]['annual_cost'] for name,_,_ in PRIMARY})
        for name,source in [('mpc_low','mpc'),('feedback_high','feedback')]:
            actual=pd.read_csv(Path(records[name]['data_directory'])/'daily.csv')
            reference=pd.read_csv(Path(plan['references'][source]['data'])/'daily.csv').iloc[365:]
            for field in ['level_before_m','level_m','request_m3_s','release_m3_s','reward']:
                np.testing.assert_allclose(actual[field],reference[field],rtol=0,atol=1e-9)
        decomposition['own_storage_reference_match']=True
        # Context: the higher boundary inventory required a different 1994 path.
        costs94={name:float(-pd.read_csv(root/('prefix_1994_'+name+'.csv')).reward.sum()) for name in ['mpc','feedback']}
        decomposition['factual_1994_costs']=costs94
        decomposition['factual_two_year_gap']=(costs94['mpc']+records['mpc_low']['annual_cost']
            -costs94['feedback']-records['feedback_high']['annual_cost'])
        save(root/'decomposition.json',dict(available=True,**decomposition))
    else:save(root/'decomposition.json',dict(available=False,reason='At least one incomplete primary branch; no full-year attribution'))
    budgets=[]
    for name in ['mpc','feedback']:
        low=records[name+'_low'];high=records[name+'_high']
        if not low['complete'] or not high['complete']:continue
        initial=high['initial_storage_m3']-low['initial_storage_m3']
        released=high['total_release_m3']-low['total_release_m3']
        remaining=high['final_storage_m3']-low['final_storage_m3']
        delivered=high['target_diversion_m3']-low['target_diversion_m3']
        river=high['remaining_river_m3']-low['remaining_river_m3']
        assert abs(initial-released-remaining)<1e-4
        assert abs(released-delivered-river)<1e-4
        budgets.append(dict(controller=name,initial_extra_million_m3=initial/1e6,
            extra_release_million_m3=released/1e6,extra_delivery_million_m3=delivered/1e6,
            extra_remaining_river_million_m3=river/1e6,extra_end_storage_million_m3=remaining/1e6,
            volume_residual_m3=initial-released-remaining))
    pd.DataFrame(budgets).to_csv(root/'inventory_accounting.csv',index=False)
    memory=[]
    for storage in ['low','high']:
        warm=records['mpc_'+storage];cold=records['mpc_'+storage+'_cold']
        memory.append(dict(storage=storage,complete=warm['complete'] and cold['complete'],
            cold_minus_carried_cost=cold['annual_cost']-warm['annual_cost'] if warm['complete'] and cold['complete'] else None))
    save(root/'memory_controls.json',memory)
    save(root/'run.json',dict(complete=True,all_primary_complete=primary,
        all_branches_complete=all(r['complete'] for r in records.values()),test_evaluated=False,
        source_hashes_verified=True,model_or_costs_changed=False,operationally_calibrated=False,
        interpretation='Conditional four-way intervention, not a unique percentage attribution or a free-stock policy improvement.'))
    print((root/'decomposition.json').read_text(),flush=True)
    print(pd.DataFrame(budgets).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','prepare','prefix','run','aggregate'],default='all')
    args=parser.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    plan=verify(root);training=settings('train');bank=CausalForecastBank(training['inflow'],training['demand'],before_year=1988)
    if args.stage in ('all','prefix'):reconstruct(root,plan,training,bank)
    if args.stage in ('all','run','aggregate'):
        boundary=json.loads((root/'boundary.json').read_text())
        if args.stage in ('all','run'):
            for spec in plan['primary']+plan['supplementary']:run_branch(root,spec,training,bank,boundary)
        if args.stage in ('all','aggregate'):aggregate(root,plan,boundary)
    verify(root)


if __name__=='__main__':main()
