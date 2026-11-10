"""Frozen 2x2 target/search diagnosis; reference costs/physics/inflow fixed."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_CONFIG,DAM_DATA
from lake_como.simulation.low_water import make_protected_outlet_env,ProtectedOutletModel
from lake_como.simulation.model import WaterState
from lake_como.simulation.outlet import OutletDomainError
from lake_como.simulation.protected_mpc import CausalForecastBank,evaluate_schedules
from lake_como.simulation.mpc_diagnosis import DiagnosticMPC,TargetForecaster,SearchExpansion

REF=PROJECT_ROOT/'results/simulator/protected_mpc_v1'
DEFAULT=PROJECT_ROOT/'results/simulator/mpc_target_search_v1'
VARIANTS=dict(seasonal_restricted=('seasonal','restricted'),ema_restricted=('ema_residual','restricted'),
              seasonal_expanded=('seasonal','expanded'),ema_expanded=('ema_residual','expanded'))


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed source '+name)
    return plan


def metrics(frame):
    return dict(completed_days=len(frame),return_=float(frame.reward.sum()),
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6),
        environmental_shortage_million_m3=float(frame.environmental_shortage_m3.sum()/1e6),
        flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),
        low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
        maximum_level_m=float(frame.maximum_level_m.max()),minimum_level_m=float(frame.minimum_level_m.min()),
        maximum_balance_error_m3=float(frame.balance_error_m3.max()))


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    reference=json.loads((REF/'plan.json').read_text())
    paths=list((PROJECT_ROOT/'src/lake_como/simulation').glob('*.py'))+[Path(__file__),REF/'plan.json']
    paths += [PROJECT_ROOT/'src/lake_como/environment.py',PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/lake_como/splits.py',PROJECT_ROOT/'configs/splits.json',
        DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',DAM_CONFIG/'world_train.yaml',DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv']
    paths += [DAM_DATA/k/f'water_{k}_{p}.csv' for k in ['inflow','demand'] for p in ['train','test']]
    for case in reference['cases']:
        for name,source in [('seasonal_restricted','conditional_7d'),('causal_feedback','causal_feedback')]:
            folder=REF/(case['name']+'__'+source);run=json.loads((folder/'run.json').read_text())
            paths += [folder/'run.json',folder/run['attempt']/'daily.csv',folder/run['attempt']/'decisions.jsonl']
            destination=root/(case['name']+'__'+name);destination.mkdir()
            save(destination/'run.json',dict(run,arm=name,data_directory=str(folder/run['attempt']),
                 source_reference=str(folder/'run.json')))
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(root/'plan.json',dict(cases=reference['cases'],variants=VARIANTS,source_sha256=hashes,
        held_fixed=dict(inflow_forecast='conditional historical, five paths, seven-day source blocks, seed 901 + hour',
            horizon_days=7,weights=reference['weights'],physics=reference['outlet'],rule=reference['rule'],
            initial_conditions=reference['initialization'],feasibility=reference['feasibility']),
        target='Seasonal versus prior-only per-lead OLS residual correction using existing EMA; account for reset bias. No new policy observation.',
        search=dict(restricted=reference['search'],expanded=asdict(SearchExpansion()),
            rule='Two coordinate sweeps, retain incumbent, no global optimum claim; higher compute budget is explicit.'),
        reference='Frozen seasonal/restricted conditional-7d and feedback artifacts are reused, not reranked or retuned.',
        audits='Fresh controllers on reference MPC states every 30 days in full windows / every 7 days in shared-state branch. Ex-post realized seven-day plan scores after plans are frozen.',
        validation='Same two-year validation screen as prior work; already inspected, not independent confirmation. No validation fitting/selection.',
        test_evaluated=False,model_or_costs_changed=False,operationally_calibrated=False))
    training=settings('train')
    for case in reference['cases']:
        bank=CausalForecastBank(training['inflow'],training['demand'],before_year=case['before_year'])
        target=TargetForecaster(bank,training['exponential_average_coeff'])
        save(root/(case['name']+'_target_fit.json'),dict(years=list(bank.years),ema_coefficient=target.coefficient,
            slopes=target.slopes.tolist(),window_rows_per_lead=target.counts.tolist(),
            note='Counts refer to overlapping prior two-year windows, not independent sample sizes. No evaluation observations fitted.'))


def controller_for(training,bank,arm):
    target,search=VARIANTS[arm]
    physical={k:training[k] for k in ['lake_params','action','flood_level','starving_level']}
    return DiagnosticMPC(physical,bank,forecast='conditional',target_mode=target,search=search,
                         ema_coefficient=training['exponential_average_coeff'])


def run_arm(root,case,arm,training):
    folder=root/(case['name']+'__'+arm);folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():return
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    bank=CausalForecastBank(training['inflow'],training['demand'],before_year=case['before_year'])
    env=make_protected_outlet_env(case['split'],tuple(case['years']));env.fallback=bank.initial_last_inflow
    obs,info=env.reset(seed=901)
    if case['start_hour']:
        anchor=case['anchor'];env.hour=anchor['hour'];env.state=WaterState(**anchor['state']);env.demand_ema=anchor['demand_ema']
        previous=env.hour//24-1;year=case['years'][previous//365]
        env.last_inflow=float(env.forcing[year][previous%365].mean());obs=env._observation();info=env._info()
    controller=controller_for(training,bank,arm);rows=[];failure=None;started=time.monotonic()
    with (data/'decisions.jsonl').open('w') as log:
        try:
            while env.hour<case['end_hour']:
                hour=env.hour;past_ema=float(obs[3]);action=controller(obs,info)
                log.write(json.dumps(controller.diagnostics[-1],allow_nan=False)+'\n');log.flush()
                obs,reward,term,trunc,info=env.step(action);hourly=info['hourly']
                rows.append(dict(elapsed_hour=hour,year=hourly[0]['year'],model_day=hourly[0]['model_day'],
                    level_before_m=hourly[0]['level_before_m'],level_m=hourly[-1]['level_m'],past_target_ema=past_ema,
                    minimum_level_m=min(r['level_m'] for r in hourly),maximum_level_m=max(r['level_m'] for r in hourly),
                    request_m3_s=float(action[0]),release_m3_s=float(np.mean([r['release_m3_s'] for r in hourly])),
                    inflow_m3_s=float(np.mean([r['inflow_m3_s'] for r in hourly])),target_m3_s=hourly[0]['demand_m3_s'],
                    reward=reward,unmet_target_m3=sum(r['unmet_demand_m3'] for r in hourly),
                    environmental_shortage_m3=sum(r['environmental_shortage_m3'] for r in hourly),
                    flood_excess_m_hours=sum(r['flood_excess_m_hours'] for r in hourly),
                    low_water_excess_m_hours=sum(r['low_water_excess_m_hours'] for r in hourly),
                    balance_error_m3=max(abs(r['lake_balance_error_m3']) for r in hourly)))
                if len(rows)%50==0:print(case['name'],arm,len(rows),'days',flush=True)
                if term or trunc:break
        except OutletDomainError as error:
            failure=dict(error=str(error),attempted_decision_hour=hour,elapsed_hour=env.hour)
        finally:env.close()
    frame=pd.DataFrame(rows)
    if frame.empty:raise RuntimeError('No complete interval; inspect preserved attempt')
    frame.to_csv(data/'daily.csv',index=False)
    gains=[d['search_expected_gain'] for d in controller.diagnostics if d['search_expected_gain'] is not None]
    save(folder/'run.json',dict(case=case['name'],arm=arm,data_directory=str(data),failure=failure,
        complete=failure is None and len(frame)==(case['end_hour']-case['start_hour'])//24,
        expected_days=(case['end_hour']-case['start_hour'])//24,fit_years=list(bank.years),
        domain_recoveries=sum(d['domain_recovery'] for d in controller.diagnostics),
        mean_expected_search_gain=float(np.mean(gains)) if gains else None,
        elapsed_seconds=time.monotonic()-started,**metrics(frame)))
    print('FINISHED',case['name'],arm,len(rows),'days','failure',failure,'seconds',round(time.monotonic()-started,2),flush=True)


def audit_states(root,plan,training):
    if (root/'same_state_audits.csv').exists():return
    results=[]
    for case in plan['cases']:
        record=json.loads((root/(case['name']+'__seasonal_restricted')/'run.json').read_text())
        reference=pd.read_csv(Path(record['data_directory'])/'daily.csv')
        bank=CausalForecastBank(training['inflow'],training['demand'],before_year=case['before_year'])
        actual=settings(case['split']);coefficient=training['exponential_average_coeff']
        ema=case.get('anchor',{}).get('demand_ema',0.);command=case.get('anchor',{}).get('state',{}).get('command',0.)
        if case['start_hour']:
            previous=case['start_hour']//24-1;last=actual['inflow'][case['years'][previous//365]][previous%365]
        else:last=bank.initial_last_inflow
        stride=7 if case['start_hour'] else 30
        for index,row in reference.iterrows():
            hour=int(row.elapsed_hour);remaining=len(case['years'])*365*24-hour
            if index%stride==0:
                phase=2*np.pi*(hour%(365*24))/(365*24)
                obs=np.array([row.level_before_m,np.sin(phase),np.cos(phase),ema,last,command,0,remaining/(len(case['years'])*365*24)],np.float32)
                info=dict(elapsed_hours=hour,remaining_hours=remaining,decision_hours=24)
                frozen=[];plans=[]
                for arm in VARIANTS:
                    controller=controller_for(training,bank,arm);action=controller(obs,info);d=controller.diagnostics[-1]
                    plans.append(controller.previous_plan.copy());frozen.append(dict(case=case['name'],elapsed_hour=hour,arm=arm,
                        request_m3_s=float(action[0]),expected_cost=d['expected_cost'],restricted_seed_cost=d['restricted_seed_cost']))
                # Only now access realized future values, for scoring frozen plans.
                horizon=len(plans[0]);ordinal=hour//24+np.arange(horizon)
                flows=np.array([[actual['inflow'][case['years'][i//365]][i%365] for i in ordinal]])
                targets=np.array([actual['demand'][case['years'][i//365]][i%365] for i in ordinal])
                model=ProtectedOutletModel(training);state=WaterState(model.lake.level_to_storage(float(obs[0])),float(obs[5]))
                scored=evaluate_schedules(model,state,plans,flows,targets,ordinal%365+1)
                for i,item in enumerate(frozen):
                    cost=float(scored['scenario_costs'][i,0]);item.update(realized_open_loop_cost=cost if np.isfinite(cost) else None,
                        realized_plan_in_domain=bool(scored['valid'][i,0]))
                    results.append(item)
            ema=coefficient*ema+(1-coefficient)*row.target_m3_s;last=row.inflow_m3_s;command=row.request_m3_s
    pd.DataFrame(results).to_csv(root/'same_state_audits.csv',index=False)


def aggregate(root,plan):
    summaries=[];paired=[];forecast_errors=[];search=[]
    for case in plan['cases']:
        frames={};records={}
        for arm in [*VARIANTS,'causal_feedback']:
            record=json.loads((root/(case['name']+'__'+arm)/'run.json').read_text());records[arm]=record
            summaries.append({k:v for k,v in record.items() if k not in ('failure','fit_years')})
            data=Path(record['data_directory']);frames[arm]=pd.read_csv(data/'daily.csv')
            if arm=='causal_feedback':continue
            actual=settings(case['split'])
            for line in (data/'decisions.jsonl').read_text().splitlines():
                d=json.loads(line);start=d['elapsed_hours']//24
                for lead,prediction in enumerate(d['forecast_target']):
                    index=start+lead;value=float(actual['demand'][case['years'][index//365]][index%365])
                    forecast_errors.append(dict(case=case['name'],arm=arm,lead_days=lead+1,
                        error_m3_s=prediction-value,absolute_error_m3_s=abs(prediction-value)))
                if d.get('search_mode')=='expanded':
                    search.append(dict(case=case['name'],arm=arm,elapsed_hour=d['elapsed_hours'],
                        restricted_seed_cost=d['restricted_seed_cost'],expanded_cost=d['expected_cost'],
                        expected_gain=d['search_expected_gain'],evaluations=d['expanded_candidate_evaluations']))
        for before,after,factor in [('seasonal_restricted','ema_restricted','target'),
            ('seasonal_restricted','seasonal_expanded','search'),('seasonal_expanded','ema_expanded','target_expanded'),
            ('ema_restricted','ema_expanded','search_ema'),('causal_feedback','ema_expanded','both_vs_feedback')]:
            n=min(len(frames[before]),len(frames[after]));a=metrics(frames[before].iloc[:n]);b=metrics(frames[after].iloc[:n])
            paired.append(dict(case=case['name'],factor=factor,before=before,after=after,matched_days=n,
                full_window=records[before]['complete'] and records[after]['complete'],
                **{f'delta_{k}':b[k]-a[k] for k in a if k!='completed_days'}))
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(root/'paired_factors.csv',index=False)
    detail=pd.DataFrame(forecast_errors)
    detail.groupby(['case','arm','lead_days']).agg(count=('error_m3_s','count'),bias_m3_s=('error_m3_s','mean'),
        mae_m3_s=('absolute_error_m3_s','mean')).reset_index().to_csv(root/'target_forecast_skill.csv',index=False)
    pd.DataFrame(search).to_csv(root/'search_gains.csv',index=False)
    save(root/'run.json',dict(complete=True,test_evaluated=False,model_or_costs_changed=False,source_hashes_verified=True,
        operationally_calibrated=False,interpretation='Fixed diagnostic including recorded failures; no final-test claims or validation-selected tuning.'))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','prepare','run','audit','aggregate'],default='all')
    args=parser.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    plan=verify(root);training=settings('train')
    if args.stage in ('all','run'):
        for case in plan['cases']:
            for arm in VARIANTS:run_arm(root,case,arm,training)
        verify(root)
    if args.stage in ('all','audit'):audit_states(root,plan,training)
    if args.stage in ('all','aggregate'):aggregate(root,plan)
    verify(root)


if __name__=='__main__':main()
