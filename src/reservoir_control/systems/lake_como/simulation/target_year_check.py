"""Broader validation of the fixed target-history forecast and original search.

Same two-year reset protocol, conditional inflow paths, costs and controller
parameters. Previously inspected 1988–1989 is separated from additional years.
Final test years are forbidden. A failed year cannot receive a full-year score.
"""
from reservoir_control.evaluation.provenance import verify_sources
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT,DAM_CONFIG,DAM_DATA
from reservoir_control.systems.lake_como.splits import split_years
from reservoir_control.systems.lake_como.simulation.low_water import LowWaterRule,make_protected_outlet_env
from reservoir_control.systems.lake_como.simulation.model import PhysicalWeights
from reservoir_control.systems.lake_como.simulation.outlet import OutletSpec,OutletDomainError
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank,ProtectedMPCConfig
from reservoir_control.systems.lake_como.simulation.mpc_diagnosis import DiagnosticMPC,TargetForecaster

DEFAULT=PROJECT_ROOT/'results/simulator/target_year_check_v1'
REFERENCE=PROJECT_ROOT/'results/simulator/mpc_target_search_v1'
ARMS=('seasonal_restricted','ema_restricted','causal_feedback')
DAILY_FIELDS=('elapsed_hour','year','model_day','level_before_m','level_m','minimum_level_m',
    'maximum_level_m','request_m3_s','release_m3_s','inflow_m3_s','target_m3_s','reward',
    'unmet_target_m3','environmental_shortage_m3','flood_excess_m_hours',
    'low_water_excess_m_hours','balance_error_m3')
SCOPES={'reference_1988_1989':tuple(range(1988,1990)),
        'additional_1990_1997':tuple(range(1990,1998)),
        'all_validation':tuple(range(1988,1998))}


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validation_pairs(years=None):
    allowed=tuple(split_years('validation'))
    selected=allowed if years is None else tuple(years)
    if selected!=allowed or len(selected)%2 or any(b!=a+1 for a,b in zip(selected,selected[1:])):
        raise ValueError('This frozen check requires all validation years, paired consecutively; no test years')
    return [selected[i:i+2] for i in range(0,len(selected),2)]


def annual_metrics(frame,years,arm,quartiles):
    """Report missing/partial years explicitly; never score them as full years."""
    rows=[]
    for position,year in enumerate(years,1):
        group=frame.loc[frame.year.eq(year)]
        days=group.model_day.to_numpy(int)
        if len(np.unique(days))!=len(days) or len(days)>365:
            raise ValueError('Duplicate or excess model days')
        complete=np.array_equal(days,np.arange(1,366))
        observed_cost=float(-group.reward.sum()) if len(group) else None
        row=dict(year=int(year),arm=arm,block_start=int(years[0]),position_in_pair=position,
            complete=complete,observed_days=len(group),annual_cost=observed_cost if complete else None,
            observed_prefix_cost=observed_cost)
        row.update({name:None for name in ['unmet_target_million_m3','environmental_shortage_million_m3',
            'flood_excess_m_hours','low_water_excess_m_hours','flood_days','low_water_days',
            'minimum_level_m','maximum_level_m','initial_level_m','mean_inflow_m3_s',
            'regime','maximum_balance_error_m3']})
        if complete:
            mean_flow=float(group.inflow_m3_s.mean())
            row.update(unmet_target_million_m3=float(group.unmet_target_m3.sum()/1e6),
                environmental_shortage_million_m3=float(group.environmental_shortage_m3.sum()/1e6),
                flood_excess_m_hours=float(group.flood_excess_m_hours.sum()),
                low_water_excess_m_hours=float(group.low_water_excess_m_hours.sum()),
                flood_days=int((group.flood_excess_m_hours>0).sum()),
                low_water_days=int((group.low_water_excess_m_hours>0).sum()),
                minimum_level_m=float(group.minimum_level_m.min()),maximum_level_m=float(group.maximum_level_m.max()),
                initial_level_m=float(group.level_before_m.iloc[0]),mean_inflow_m3_s=mean_flow,
                regime='dry' if mean_flow<quartiles[0] else ('wet' if mean_flow>quartiles[1] else 'middle'),
                maximum_balance_error_m3=float(group.balance_error_m3.max()))
        rows.append(row)
    return rows


def paired_scope(annual,before,after,years):
    """A missing pair suppresses the primary mean, rather than disappearing."""
    selected=annual[annual.year.isin(years)]
    a=selected[selected.arm.eq(before)].set_index('year').reindex(years)
    b=selected[selected.arm.eq(after)].set_index('year').reindex(years)
    keep=a.complete.fillna(False).astype(bool)&b.complete.fillna(False).astype(bool)
    delta=(b.annual_cost-a.annual_cost)[keep]
    all_complete=bool(keep.all())
    result=dict(before=before,after=after,expected_years=len(years),complete_pairs=int(keep.sum()),
        all_pairs_complete=all_complete,wins=int((delta < -1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),
        losses=int((delta>1e-8).sum()),mean_annual_cost_delta=float(delta.mean()) if all_complete else None,
        median_annual_cost_delta=float(delta.median()) if all_complete else None,
        best_annual_cost_delta=float(delta.min()) if all_complete else None,
        worst_annual_cost_delta=float(delta.max()) if all_complete else None)
    blocks=[]
    for i in range(0,len(years),2):
        block=years[i:i+2]
        if len(block)==2 and keep.loc[list(block)].all():blocks.append(float(delta.loc[list(block)].sum()))
    result.update(complete_blocks=len(blocks),block_wins=sum(value < -1e-8 for value in blocks),
                  block_losses=sum(value>1e-8 for value in blocks))
    return result


def prepare(root):
    pairs=validation_pairs();root.mkdir(parents=True,exist_ok=False)
    cases=[dict(name=f'{a}_{b}',years=[a,b],before_year=1988) for a,b in pairs]
    paths=(list(Path(__file__).parent.glob('*.py')) + (list((PROJECT_ROOT/'src/reservoir_control/controllers').rglob('*.py')) + [PROJECT_ROOT/'src/reservoir_control/systems/lake_como/sac.py', PROJECT_ROOT/'src/reservoir_control/systems/lake_como/fqi.py']))+[PROJECT_ROOT/'src/reservoir_control/systems/lake_como/environment.py',
        PROJECT_ROOT/'src/reservoir_control/systems/lake_como/hourly_telemetry.py',PROJECT_ROOT/'src/reservoir_control/systems/lake_como/splits.py',
        PROJECT_ROOT/'configs/splits.json',PROJECT_ROOT/'pyproject.toml',PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',
        DAM_CONFIG/'world_train.yaml',DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv',REFERENCE/'plan.json']
    paths += [DAM_DATA/s/f'water_{s}_{part}.csv' for s in ['inflow','demand'] for part in ['train','test']]
    for arm in ARMS:
        source=REFERENCE/('validation_1988_1989__'+arm);record=json.loads((source/'run.json').read_text())
        folder=root/('1988_1989__'+arm);folder.mkdir()
        paths += [source/'run.json',Path(record['data_directory'])/'daily.csv',Path(record['data_directory'])/'decisions.jsonl']
        save(folder/'run.json',dict(record,case='1988_1989',source_reference=str(source/'run.json')))
    training=settings('train');bank=CausalForecastBank(training['inflow'],training['demand'],before_year=1988)
    fitted=TargetForecaster(bank,training['exponential_average_coeff'])
    thresholds=np.quantile([np.mean(training['inflow'][y][:365]) for y in sorted(training['inflow'])],[.25,.75])
    save(root/'plan.json',dict(cases=cases,arms=list(ARMS),scopes={k:list(v) for k,v in SCOPES.items()},
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        training_years=list(bank.years),mpc=asdict(ProtectedMPCConfig()),weights=asdict(PhysicalWeights()),
        outlet=asdict(OutletSpec()),low_water_rule=asdict(LowWaterRule()),
        target_fit=dict(slopes=fitted.slopes.tolist(),window_rows_per_lead=fitted.counts.tolist(),
                        ema_coefficient=fitted.coefficient),
        training_inflow_quartiles_m3_s=thresholds.tolist(),
        fit='Freeze all inflow/target fits at 1946–1987, including for later validation windows. No validation refitting.',
        initialization='Reset level .35 m, target EMA zero and prior last-model-day inflow climatology at each two-year block; carry state/history through its year boundary.',
        protocol='365 samples per model year, unchanged. The ten validation years are five non-overlapping two-year episodes, not a ten-year continuous run.',
        scope='Additional 1990–1997 is the primary expansion; inspected 1988–1989 reported separately. These are validation data, not independent final-test confirmation.',
        failures='No full-year score for incomplete years. Primary scope means/deltas suppressed if any required year/pair is missing. Matched prefixes reported separately.',
        statistics='Paired annual and block win counts, mean/median/range, first/second-year breakdown. No p-value or independent-year confidence claim.',
        regimes='Prior training annual-inflow quartiles; ex-post context only, not a policy feature or fitting criterion.',
        forecast_seed='Unchanged single seed 901 + elapsed hour; no seed or hyperparameter selection.',
        numpy_version=np.__version__,pandas_version=pd.__version__,
        test_evaluated=False,model_or_costs_changed=False,operationally_calibrated=False))


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    if [tuple(case['years']) for case in plan['cases']]!=validation_pairs():raise ValueError('Changed validation cases')
    if any(case['before_year']!=1988 for case in plan['cases']):raise ValueError('Validation refitting forbidden')
    verify_sources(plan['source_sha256'])
    return plan


def run_arm(root,case,arm,training,plan):
    folder=root/(case['name']+'__'+arm);folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():return
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    bank=CausalForecastBank(training['inflow'],training['demand'],before_year=1988)
    env=make_protected_outlet_env('validation',tuple(case['years']));env.fallback=bank.initial_last_inflow
    obs,info=env.reset(seed=901)
    controller=None
    if arm!='causal_feedback':
        physical={k:training[k] for k in ['lake_params','action','flood_level','starving_level']}
        controller=DiagnosticMPC(physical,bank,forecast='conditional',
            target_mode='ema_residual' if arm=='ema_restricted' else 'seasonal',search='restricted',
            ema_coefficient=training['exponential_average_coeff'])
        np.testing.assert_array_equal(controller.target_forecaster.slopes,plan['target_fit']['slopes'])
    rows=[];failure=None;started=time.monotonic()
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for index in range(730):
                hour=env.hour
                if controller is None:
                    action=np.array([np.clip(float(obs[3])+20+150*(float(obs[0])-.35),0,env.model.high)],np.float32)
                else:
                    action=controller(obs,info)
                    log.write(json.dumps(controller.diagnostics[-1],allow_nan=False)+'\n');log.flush()
                obs,reward,term,trunc,info=env.step(action);hourly=info['hourly']
                rows.append(dict(elapsed_hour=hour,year=hourly[0]['year'],model_day=hourly[0]['model_day'],
                    level_before_m=hourly[0]['level_before_m'],level_m=hourly[-1]['level_m'],
                    minimum_level_m=min(r['level_m'] for r in hourly),maximum_level_m=max(r['level_m'] for r in hourly),
                    request_m3_s=float(action[0]),release_m3_s=float(np.mean([r['release_m3_s'] for r in hourly])),
                    inflow_m3_s=float(np.mean([r['inflow_m3_s'] for r in hourly])),target_m3_s=hourly[0]['demand_m3_s'],
                    reward=reward,unmet_target_m3=sum(r['unmet_demand_m3'] for r in hourly),
                    environmental_shortage_m3=sum(r['environmental_shortage_m3'] for r in hourly),
                    flood_excess_m_hours=sum(r['flood_excess_m_hours'] for r in hourly),
                    low_water_excess_m_hours=sum(r['low_water_excess_m_hours'] for r in hourly),
                    balance_error_m3=max(abs(r['lake_balance_error_m3']) for r in hourly)))
                if len(rows)%100==0:print(case['name'],arm,len(rows),'days',flush=True)
                if term or trunc:break
        except OutletDomainError as error:failure=dict(error=str(error),attempted_decision_hour=hour,elapsed_hour=env.hour)
        finally:env.close()
    pd.DataFrame(rows,columns=DAILY_FIELDS).to_csv(data/'daily.csv',index=False)
    save(folder/'run.json',dict(case=case['name'],arm=arm,data_directory=str(data),
        complete=len(rows)==730 and failure is None,expected_days=730,observed_days=len(rows),failure=failure,
        fit_years=list(bank.years),elapsed_seconds=time.monotonic()-started,
        domain_recoveries=0 if controller is None else sum(d['domain_recovery'] for d in controller.diagnostics)))
    print('FINISHED',case['name'],arm,len(rows),'days','failure',failure,flush=True)


def aggregate(root,plan):
    annual=[];prefix=[];failures=[];forecast=[]
    for case in plan['cases']:
        frames={}
        for arm in ARMS:
            record=json.loads((root/(case['name']+'__'+arm)/'run.json').read_text())
            data=Path(record['data_directory']);frame=pd.read_csv(data/'daily.csv');frames[arm]=frame
            annual.extend(annual_metrics(frame,case['years'],arm,plan['training_inflow_quartiles_m3_s']))
            if record.get('failure'):failures.append(dict(case=case['name'],arm=arm,**record['failure']))
            if arm=='causal_feedback':continue
            actual=settings('validation')['demand']
            for line in (data/'decisions.jsonl').read_text().splitlines():
                decision=json.loads(line);start=decision['elapsed_hours']//24
                for lead,prediction in enumerate(decision['forecast_target']):
                    ordinal=start+lead;year=case['years'][ordinal//365]
                    truth=float(actual[year][ordinal%365])
                    forecast.append(dict(arm=arm,year=case['years'][start//365],lead_days=lead+1,
                        error_m3_s=prediction-truth,absolute_error_m3_s=abs(prediction-truth)))
        for year in case['years']:
            for before in ['seasonal_restricted','causal_feedback']:
                a=frames[before].loc[frames[before].year.eq(year)];b=frames['ema_restricted'].loc[frames['ema_restricted'].year.eq(year)]
                n=min(len(a),len(b))
                prefix.append(dict(year=year,before=before,after='ema_restricted',matched_days=n,
                    full_year=n==365,cost_delta=float(-b.reward.iloc[:n].sum()+a.reward.iloc[:n].sum()) if n else None))
    frame=pd.DataFrame(annual);frame.to_csv(root/'annual.csv',index=False)
    pd.DataFrame(prefix).to_csv(root/'matched_prefixes.csv',index=False)
    save(root/'failures.json',failures)
    summaries=[];paired=[];positions=[]
    for scope,years in SCOPES.items():
        for arm in ARMS:
            selected=frame[frame.year.isin(years)&frame.arm.eq(arm)]
            complete=len(selected)==len(years) and bool(selected.complete.all())
            metrics=['annual_cost','unmet_target_million_m3','environmental_shortage_million_m3',
                     'flood_excess_m_hours','low_water_excess_m_hours','flood_days','low_water_days']
            summaries.append(dict(scope=scope,arm=arm,expected_years=len(years),complete_years=int(selected.complete.sum()),
                all_required_complete=complete,**{f'mean_{name}':float(selected[name].mean()) if complete else None for name in metrics}))
            for position in [1,2]:
                subset=selected[selected.position_in_pair.eq(position)]
                positions.append(dict(scope=scope,arm=arm,position_in_pair=position,expected_years=len(years)//2,
                    complete_years=int(subset.complete.sum()),mean_annual_cost=float(subset.annual_cost.mean()) if subset.complete.all() else None))
        for before in ['seasonal_restricted','causal_feedback']:
            paired.append(dict(scope=scope,**paired_scope(frame,before,'ema_restricted',years)))
    pd.DataFrame(summaries).to_csv(root/'scope_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(root/'paired_summary.csv',index=False)
    pd.DataFrame(positions).to_csv(root/'episode_position.csv',index=False)
    errors=pd.DataFrame(forecast)
    errors.groupby(['arm','year','lead_days']).agg(count=('error_m3_s','count'),bias_m3_s=('error_m3_s','mean'),
        mae_m3_s=('absolute_error_m3_s','mean')).reset_index().to_csv(root/'target_forecast_skill.csv',index=False)
    save(root/'run.json',dict(complete=True,all_arms_complete=bool(frame.complete.all()),domain_failures=len(failures),
        test_evaluated=False,model_or_costs_changed=False,source_hashes_verified=True,
        operationally_calibrated=False,interpretation='Fixed validation expansion; report inspected reference separately, no automatic best-controller promotion.'))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','prepare','run','aggregate'],default='all')
    args=parser.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    plan=verify(root)
    if args.stage in ('all','run'):
        training=settings('train')
        for case in plan['cases']:
            for arm in ARMS:run_arm(root,case,arm,training,plan)
        verify(root)
    if args.stage in ('all','aggregate'):aggregate(root,plan)
    verify(root)


if __name__=='__main__':main()
