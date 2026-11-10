"""Trace 1995 marginal reserve releases and isolate target-forecast information.

Perfect targets are explicitly privileged diagnostic inputs, never an operational
policy feature. Inflow forecasts, physics, search and realized costs are fixed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_CONFIG,DAM_DATA
from .carryover_check import BoundaryHistory,restore_branch,restore_memory,take_step,FIELDS
from .mpc_diagnosis import DiagnosticMPC
from .protected_mpc import CausalForecastBank,ProtectedMPCConfig,evaluate_schedules
from .model import PhysicalWeights,WaterState,physical_cost
from .outlet import OutletSpec,OutletDomainError
from .low_water import LowWaterRule

DEFAULT=PROJECT_ROOT/'results/simulator/reserve_use_check_v2'
REFERENCE=PROJECT_ROOT/'results/simulator/carryover_check_v1'
CASES=tuple((f'{mode}_{storage}',mode,storage) for mode in ['causal','perfect_target'] for storage in ['low','high'])


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class PerfectTargetRecord:
    """Restricted 1995 target labels only; accepts no inflow records."""
    def __init__(self,values,*,year=1995):
        if year!=1995:raise ValueError('This privileged diagnostic is restricted to 1995')
        values=np.asarray(values,float)
        if values.shape!=(365,) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('A finite nonnegative 1995 model-year target record is required')
        self.values=values.copy();self.values.setflags(write=False)

    def horizon(self,elapsed_days,length):
        if (not isinstance(elapsed_days,int) or not 365<=elapsed_days<730 or
                not isinstance(length,int) or not 1<=length<=7 or elapsed_days+length>730):
            raise ValueError('Only the 1995 portion of the original two-year clock is allowed')
        return self.values[elapsed_days-365:elapsed_days-365+length].copy()


class _PerfectTargetBank:
    def __init__(self,inner,record):self.inner=inner;self.record=record

    @property
    def context(self):return self.inner.context

    @context.setter
    def context(self,value):self.inner.context=value

    def forecast(self,*args,**kwargs):
        paths,causal,metadata=self.inner.forecast(*args,**kwargs)
        if self.context is None:raise RuntimeError('Original model clock required')
        targets=self.record.horizon(self.context[2],kwargs['horizon'])
        return paths,targets,dict(metadata,oracle_targets=True,
            target_information='Perfect future benchmark targets; non-deployable diagnostic',
            causal_target_forecast=causal.tolist())


class PerfectTargetMPC(DiagnosticMPC):
    def __init__(self,params,bank,record,**kwargs):
        super().__init__(params,bank,target_mode='ema_residual',search='restricted',**kwargs)
        self.bank=_PerfectTargetBank(self.bank,record)

    def __call__(self,observation,info):
        action=super().__call__(observation,info)
        self.diagnostics[-1].update(target_mode='perfect_target_diagnostic',oracle_targets=True)
        return action


def model_for(training):
    return {k:training[k] for k in ['lake_params','action','flood_level','starving_level']}


def make_controller(training,bank,mode,targets):
    kwargs=dict(forecast='conditional',ema_coefficient=training['exponential_average_coeff'])
    if mode=='perfect_target':return PerfectTargetMPC(model_for(training),bank,targets,**kwargs)
    if mode=='causal':return DiagnosticMPC(model_for(training),bank,target_mode='ema_residual',search='restricted',**kwargs)
    raise ValueError('Known diagnostic mode required')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    paths=list(Path(__file__).parent.glob('*.py'))+[REFERENCE/'plan.json',REFERENCE/'boundary.json',
        REFERENCE/'summary.csv',REFERENCE/'inventory_accounting.csv',PROJECT_ROOT/'src/lake_como/environment.py',
        PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',PROJECT_ROOT/'src/lake_como/splits.py',
        PROJECT_ROOT/'configs/splits.json',PROJECT_ROOT/'pyproject.toml',PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',DAM_CONFIG/'world_train.yaml',DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv']
    paths += [DAM_DATA/s/f'water_{s}_{p}.csv' for s in ['inflow','demand'] for p in ['train','test']]
    for storage in ['low','high']:
        folder=REFERENCE/('mpc_'+storage);record=json.loads((folder/'run.json').read_text())
        paths += [folder/'run.json',Path(record['data_directory'])/'daily.csv',Path(record['data_directory'])/'decisions.jsonl']
    save(root/'plan.json',dict(cases=[dict(name=n,mode=m,storage=s) for n,m,s in CASES],
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        fit_years=list(range(1946,1988)),mpc=asdict(ProtectedMPCConfig()),weights=asdict(PhysicalWeights()),
        outlet=asdict(OutletSpec()),rule=asdict(LowWaterRule()),
        intervention='Planning target forecast only: original causal history correction versus perfect future 1995 targets. Environment targets/rewards unchanged.',
        held_fixed='Same conditional inflow paths/seed, physical state per storage, original clock/history/MPC memory, search rule/grid, costs and constraints.',
        caveat='Target-dependent service proposals change with the supplied target forecast as in the existing search. This is not a fixed action-set ablation.',
        trace='Signed high-minus-low daily release/delivery/remaining-river deltas and cumulative inventory reconciliation; counterfactual accounting, not physical parcel tracking.',
        selected_days='First seven model days, ten largest positive causal high-minus-low remaining-river increments, and first 50/90% cumulative net-river crossings. Selected diagnostic days, not a representative sample.',
        audit='Freeze decisions, then inspect first-day predictions versus realized execution and already-included service/zero-first alternatives. Cost tie tolerance 1e-8; no policy changes.',
        interpretation='One selected validation case and one forecast seed. Perfect targets are privileged information, not a deployable controller or an optimal performance bound.',
        test_evaluated=False,model_or_costs_changed=False,operationally_calibrated=False))


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed source '+name)
    return plan


def run_case(root,spec,training,bank,boundary,targets):
    folder=root/spec['name'];folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():return
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    history=BoundaryHistory(**boundary['common_history']);storage=boundary[spec['storage']+'_storage_m3']
    env,obs,info=restore_branch(bank,history,storage)
    controller=make_controller(training,bank,spec['mode'],targets);restore_memory(controller,history)
    records=[];failure=None
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(365):
                before=obs.copy()
                obs,info,row,term,trunc=take_step(env,controller,obs,info);records.append(row)
                diagnostic=dict(controller.diagnostics[-1],selected_plan=controller.previous_plan.tolist(),
                                planning_observation=before.tolist())
                plan=np.asarray(diagnostic['selected_plan'])
                days=(row['elapsed_hour']//24+np.arange(len(plan)))%365+1
                state=WaterState(controller.model.lake.level_to_storage(float(before[0])),float(before[5]),float(before[6]))
                result=evaluate_schedules(controller.model,state,plan[None,:],diagnostic['forecast_inflows'],
                    diagnostic['forecast_target'],days,controller.weights)
                diagnostic.update(predicted_daily_end_levels=result['daily_levels'][0].tolist(),
                    predicted_daily_costs=result['daily_costs'][0].tolist(),scenario_valid=result['valid'][0].tolist())
                log.write(json.dumps(diagnostic,allow_nan=False)+'\n');log.flush()
                if len(records)%100==0:print(spec['name'],len(records),'days',flush=True)
                if term or trunc:break
        except OutletDomainError as error:failure=dict(error=str(error),elapsed_hour=env.hour)
        finally:final_storage=env.state.storage;env.close()
    frame=pd.DataFrame(records,columns=FIELDS);frame.to_csv(data/'daily.csv',index=False)
    complete=len(frame)==365 and failure is None
    if spec['mode']=='causal' and complete:
        ref=json.loads((REFERENCE/('mpc_'+spec['storage'])/'run.json').read_text())
        original=pd.read_csv(Path(ref['data_directory'])/'daily.csv',float_precision='round_trip')
        np.testing.assert_allclose(frame.to_numpy(float),original[list(FIELDS)].to_numpy(float),rtol=0,atol=1e-9)
    save(folder/'run.json',dict(**spec,complete=complete,observed_days=len(frame),failure=failure,data_directory=str(data),
        initial_storage_m3=storage,final_storage_m3=final_storage,
        annual_cost=float(-frame.reward.sum()) if complete else None,
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6),
        remaining_river_million_m3=float(frame.remaining_river_m3.sum()/1e6),
        flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),
        low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
        low_water_days=int((frame.low_water_excess_m_hours>0).sum()),
        maximum_balance_error_m3=float(frame.balance_error_m3.abs().max()) if len(frame) else None))
    print('FINISHED',spec['name'],'cost',-frame.reward.sum(),'failure',failure,flush=True)


def inventory_trace(low,high,area):
    if len(low)!=len(high) or not np.array_equal(low.elapsed_hour,high.elapsed_hour):
        raise ValueError('Equal completed-day clocks required for marginal accounting')
    result=pd.DataFrame(dict(model_day=low.model_day,elapsed_hour=low.elapsed_hour,
        extra_release_m3=(high.release_m3_s-low.release_m3_s)*86400,
        extra_delivery_m3=high.delivered_target_m3-low.delivered_target_m3,
        extra_remaining_river_m3=high.remaining_river_m3-low.remaining_river_m3,
        extra_storage_before_m3=area*(high.level_before_m-low.level_before_m),
        extra_storage_after_m3=area*(high.level_m-low.level_m)))
    assert (result.extra_release_m3-result.extra_delivery_m3-result.extra_remaining_river_m3).abs().max()<1e-5
    assert (result.extra_storage_after_m3-result.extra_storage_before_m3+result.extra_release_m3).abs().max()<1e-5
    for name in ['release','delivery','remaining_river']:
        result['cumulative_extra_'+name+'_m3']=result['extra_'+name+'_m3'].cumsum()
    return result


def decision_audit(training,decision,actual_row,mode,storage):
    model=make_controller(training,CausalForecastBank(training['inflow'],training['demand'],before_year=1988),
                          'causal',None).model
    obs=np.asarray(decision['planning_observation']);plan=np.asarray(decision['selected_plan'],np.float32)
    flows=np.asarray(decision['forecast_inflows']);targets=np.asarray(decision['forecast_target'])
    days=(decision['elapsed_hours']//24+np.arange(len(plan)))%365+1
    state=WaterState(model.lake.level_to_storage(float(obs[0])),float(obs[5]),float(obs[6]))
    service=np.clip(targets+model.lake.min_env_flow[days-1],model.low,model.high).astype(np.float32)
    zero_first=service.copy();zero_first[0]=np.float32(model.low)
    alternatives=np.array([plan,service,zero_first],np.float32)
    result=evaluate_schedules(model,state,alternatives,flows,targets,days)
    costs=result['scenario_costs'].mean(axis=1);selected_cost=costs[0]
    assert not np.isfinite(selected_cost) or np.all(costs[1:]>=selected_cost-1e-8)
    expected=[]
    for alternative,requests in enumerate(alternatives):
        first=[]
        for path in flows:
            final,rows=model.rollout(state,np.repeat(requests[0],24),np.repeat(path[0],24),
                np.repeat(targets[0],24),np.repeat(days[0],24))
            first.append(dict(release_m3_s=float(np.mean([r['release_m3_s'] for r in rows])),
                remaining_river_m3=float(sum(r['river_out_m3'] for r in rows)),
                delivery_m3=float(sum(r['diversion_m3'] for r in rows))))
        expected.append({key:float(np.mean([f[key] for f in first])) for key in first[0]})
    item=dict(mode=mode,storage=storage,model_day=int(actual_row.model_day),
        initial_level_m=float(obs[0]),request_m3_s=float(plan[0]),actual_target_m3_s=float(actual_row.target_m3_s),
        predicted_target_m3_s=float(targets[0]),actual_release_m3_s=float(actual_row.release_m3_s),
        actual_remaining_river_m3=float(actual_row.remaining_river_m3),
        predicted_maximum_level_m=decision['predicted_maximum_level_m'],
        predicted_minimum_level_m=decision['predicted_minimum_level_m'],
        actual_end_level_m=float(actual_row.level_m),
        predicted_first_end_level_mean_m=float(np.mean(result['daily_levels'][0,:,0])),
        initial_forced_full_open=bool(obs[0]>model.lake.max_level),
        expected_first_release_m3_s=expected[0]['release_m3_s'],
        expected_first_remaining_river_m3=expected[0]['remaining_river_m3'],
        expected_cost=float(selected_cost) if np.isfinite(selected_cost) else None)
    for index,name in [(1,'service'),(2,'zero_first')]:
        tie=np.isfinite(costs[index]) and np.isfinite(selected_cost) and abs(costs[index]-selected_cost)<=1e-8
        item.update({name+'_expected_cost':float(costs[index]) if np.isfinite(costs[index]) else None,
            name+'_expected_first_release_m3_s':expected[index]['release_m3_s'],
            name+'_same_cost_lower_first_release':bool(tie and expected[index]['release_m3_s']<expected[0]['release_m3_s']-1e-6),
            name+'_predicted_end_level_mean_m':float(np.mean(result['daily_levels'][index,:,-1]))})
    return item


def analyze(root,plan,training):
    records={};frames={};decisions={}
    for spec in plan['cases']:
        name=spec['name'];record=json.loads((root/name/'run.json').read_text());records[name]=record
        data=Path(record['data_directory']);frames[name]=pd.read_csv(data/'daily.csv')
        decisions[name]={int(d['elapsed_hours']):d for d in
            (json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines())}
    pd.DataFrame([{k:v for k,v in r.items() if k!='failure'} for r in records.values()]).to_csv(root/'summary.csv',index=False)
    all_complete=all(record['complete'] for record in records.values())
    if not all_complete:
        save(root/'run.json',dict(complete=True,all_branches_complete=False,test_evaluated=False,
            interpretation='A branch is incomplete; full-year marginal/target attribution suppressed.'))
        return
    area=training['lake_params']['surface'];timing={};budgets=[]
    for mode in ['causal','perfect_target']:
        delta=inventory_trace(frames[mode+'_low'],frames[mode+'_high'],area)
        delta.to_csv(root/(mode+'_inventory_trace.csv'),index=False)
        final=float(delta.cumulative_extra_remaining_river_m3.iloc[-1]);crossings={}
        for fraction in [.5,.9,.99]:
            reached=delta.loc[delta.cumulative_extra_remaining_river_m3>=fraction*final,'model_day']
            crossings[str(fraction)]=int(reached.iloc[0]) if len(reached) and final>0 else None
        timing[mode]=dict(net_extra_remaining_river_million_m3=final/1e6,
            positive_daily_increments_million_m3=float(delta.extra_remaining_river_m3.clip(lower=0).sum()/1e6),
            negative_daily_increments_million_m3=float(delta.extra_remaining_river_m3.clip(upper=0).sum()/1e6),
            first_cumulative_fraction_crossing_days=crossings)
        budgets.append(dict(mode=mode,extra_initial_million_m3=float(delta.extra_storage_before_m3.iloc[0]/1e6),
            extra_delivery_million_m3=float(delta.cumulative_extra_delivery_m3.iloc[-1]/1e6),
            extra_remaining_river_million_m3=final/1e6,extra_final_storage_million_m3=float(delta.extra_storage_after_m3.iloc[-1]/1e6)))
    save(root/'release_timing.json',timing);pd.DataFrame(budgets).to_csv(root/'marginal_inventory.csv',index=False)
    causal=pd.read_csv(root/'causal_inventory_trace.csv')
    days=set(range(1,8))|set(causal.nlargest(10,'extra_remaining_river_m3').model_day.astype(int))
    days |= {value for key,value in timing['causal']['first_cumulative_fraction_crossing_days'].items() if key in ['0.5','0.9'] and value is not None}
    audits=[]
    for spec in plan['cases']:
        name=spec['name']
        for day in sorted(days):
            row=frames[name].loc[frames[name].model_day.eq(day)].iloc[0]
            audits.append(decision_audit(training,decisions[name][int(row.elapsed_hour)],row,spec['mode'],spec['storage']))
    pd.DataFrame(audits).to_csv(root/'selected_decision_audits.csv',index=False)
    save(root/'selected_days.json',dict(days=sorted(days),selection=plan['selected_days']))
    save(root/'run.json',dict(complete=True,all_branches_complete=True,test_evaluated=False,
        source_hashes_verified=True,model_or_costs_changed=False,operationally_calibrated=False,
        interpretation='Perfect-target information diagnostic, not a deployable controller, optimal bound, or proof that terminal value alone is missing.'))
    print(pd.DataFrame(records.values()).to_string(index=False),flush=True)
    print(json.dumps(timing,indent=2),flush=True)
    print(pd.DataFrame(budgets).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','prepare','run','analyze'],default='all')
    args=parser.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    plan=verify(root);training=settings('train')
    if args.stage in ('all','run'):
        bank=CausalForecastBank(training['inflow'],training['demand'],before_year=1988)
        boundary=json.loads((REFERENCE/'boundary.json').read_text())
        targets=PerfectTargetRecord(settings('validation')['demand'][1995][:365])
        for spec in plan['cases']:run_case(root,spec,training,bank,boundary,targets)
        verify(root)
    if args.stage in ('all','analyze'):analyze(root,plan,training)
    verify(root)


if __name__=='__main__':main()
