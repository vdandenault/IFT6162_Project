"""Chronological training-only forecast selection, then frozen event diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, DAM_CONFIG, DAM_DATA
from .history_forecast import HistoryForecastBank, HistoryMPC, METHODS
from .protected_mpc import ProtectedMPCConfig
from .precision_validation import SEEDS
from .flood_diagnosis import DEFAULT as FLOOD_REFERENCE, source, get_snapshot, restore, restore_plan
from .carryover_check import take_step, FIELDS
from .feasibility_audit import audit_forecast
from .reserve_use_check import save, verify
from .outlet import OutletDomainError
from .model import WaterState

DEFAULT = PROJECT_ROOT/'results/simulator/forecast_study_v2'
PHASES = dict(tuning=tuple(range(1976,1982)), confirmation=tuple(range(1982,1988)))


def crps(samples, truth):
    samples = np.asarray(samples, float)
    return float(np.mean(abs(samples-truth))-.5*np.mean(abs(samples[:,None]-samples[None,:])))


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = list(Path(__file__).parent.glob('*.py')) + [PROJECT_ROOT/'src/lake_como/environment.py',
        PROJECT_ROOT/'src/lake_como/hourly_telemetry.py', PROJECT_ROOT/'src/lake_como/splits.py',
        PROJECT_ROOT/'configs/splits.json', PROJECT_ROOT/'pyproject.toml', PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py', DAM_CONFIG/'lake.yaml', DAM_CONFIG/'world_train.yaml', DAM_CONFIG/'world_test.yaml',
        DAM_DATA/'MEF.csv', FLOOD_REFERENCE/'plan.json', FLOOD_REFERENCE/'run.json', FLOOD_REFERENCE/'branches.csv']
    paths += [DAM_DATA/s/f'water_{s}_{p}.csv' for s in ['inflow','demand'] for p in ['train','test']]
    paths += list((FLOOD_REFERENCE/'snapshots').glob('*.json'))
    paths += list((FLOOD_REFERENCE/'forecast_audits').glob('*.json'))
    reference = pd.read_csv(FLOOD_REFERENCE/'branches.csv')
    for r in reference[reference.controller.eq('combined') & reference.memory.eq('original')].itertuples():
        paths += [FLOOD_REFERENCE/r.name/'run.json', Path(r.data_directory)/'daily.csv', Path(r.data_directory)/'decisions.jsonl']
    save(root/'plan.json', dict(source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        methods=list(METHODS), seeds=list(SEEDS), fit_years=list(range(1946,1976)),
        phases={key:list(years) for key,years in PHASES.items()},
        score='Seven-day-volume ensemble CRPS (million m3), first-day mean MAE, ensemble envelope coverage and wet upper-envelope coverage. Five equally weighted scenarios for every method.',
        wet_definition='Seven-day volume above the 90th percentile of all complete fitting-period seven-day windows; outcome labels only, not policy inputs.',
        origins='Every model day index 7..358 inclusive within each holdout year; prior seven flows observed, future seven scored. Input observations cast to float32 like the environment. Seed + 24*day index; no rolling refit.',
        selection='Tuning only: candidate must strictly lower volume CRPS, not lower wet upper-envelope coverage, and first-day MAE <=110% of baseline. Lowest CRPS among eligible methods, declared method order breaks ties; otherwise retain baseline.',
        confirmation='Freeze selection before computing confirmation. Same 1946–1975 fitting bank. Report whether CRPS and wet coverage beat/match baseline and MAE gate holds; no reselection or confirmation refit.',
        event_protocol='After selection/confirmation, refit selected forecast on all 1946–1987. Compare old and selected inflow forecasts with combined precision/tie MPC from both saved stocks, both 14/7-day anchors, every seed. Common history and original warm plan; original record clock/horizon retained.',
        current_states='Repeat forecast capacity/search audits at the 33 prior failed states. Realized futures remain offline labels only.',
        interpretation='Holdouts select a forecast, not a controller winner. Reused validation events are diagnostic; no final test use, cost/physics/default changes or safety guarantee.',
        test_evaluated=False, model_or_costs_changed=False, default_changed=False))


def summarize(frame):
    rows=[]
    for method, group in frame.groupby('method', sort=False):
        wet=group[group.wet]
        rows.append(dict(method=method, origins=len(group), wet_origins=len(wet),
            volume_crps=float(group.volume_crps.mean()), first_mae=float(group.first_mae.mean()),
            volume_mae=float(group.volume_mae.mean()), envelope_coverage=float(group.covered.mean()),
            wet_upper_coverage=float(wet.upper_covered.mean()), mean_envelope_width=float(group.width.mean())))
    return pd.DataFrame(rows)


def select(summary):
    indexed=summary.set_index('method');base=indexed.loc['baseline']
    eligible=[method for method in METHODS[1:] if
        indexed.loc[method,'volume_crps'] < base.volume_crps and
        indexed.loc[method,'wet_upper_coverage'] >= base.wet_upper_coverage and
        indexed.loc[method,'first_mae'] <= 1.1*base.first_mae]
    selected=min(eligible,key=lambda m:(indexed.loc[m,'volume_crps'],METHODS.index(m))) if eligible else 'baseline'
    return dict(selected_method=selected, eligible_methods=eligible, selected_on='1976–1981 only',
                baseline_retained=selected=='baseline', validation_events_used=False)


def score_phase(root, phase, training):
    path=root/(phase+'.csv')
    if path.exists():
        return pd.read_csv(path)
    fit=HistoryForecastBank(training['inflow'],training['demand'],before_year=1976,method='baseline')
    wet_threshold=float(np.quantile(fit.paths.sum(axis=1)*86400/1e6,.9))
    rows=[]
    for method in METHODS:
        bank=HistoryForecastBank(training['inflow'],training['demand'],before_year=1976,method=method)
        for year in PHASES[phase]:
            q=np.asarray(training['inflow'][year][:365],float)
            for n in range(7,359):
                past=np.asarray(q[n-7:n],np.float32).astype(float)
                bank.set_history(past)
                truth=q[n:n+7];volume=float(truth.sum()*86400/1e6)
                for seed in SEEDS:
                    paths,_,meta=bank.forecast('conditional',n+1,float(past[-1]),seed=seed+n*24)
                    volumes=paths.sum(axis=1)*86400/1e6
                    rows.append(dict(method=method,year=year,model_day=n+1,seed=seed,
                        volume_crps=crps(volumes,volume),first_mae=abs(float(paths[:,0].mean())-truth[0]),
                        volume_mae=abs(float(volumes.mean())-volume),
                        covered=bool(volumes.min()<=volume<=volumes.max()),upper_covered=bool(volume<=volumes.max()),
                        wet=bool(volume>wet_threshold),width=float(volumes.max()-volumes.min())))
            print(phase,method,year,'done',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
    summarize(frame).to_csv(root/(phase+'_summary.csv'),index=False)
    save(root/(phase+'_record.json'),dict(complete=True,fit_years=list(fit.years),wet_threshold_million_m3=wet_threshold,
        scored_years=list(PHASES[phase]),scored_day_seed_forecasts=len(frame)))
    return frame


def forecasts(root):
    training=settings('train')
    tuning=score_phase(root,'tuning',training)
    selected=select(summarize(tuning))
    if (root/'selection.json').exists():
        assert json.loads((root/'selection.json').read_text())==selected
    else:
        save(root/'selection.json',selected)
    # This selection checksum is written before computing confirmation outcomes.
    checksum=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()
    guard=root/'confirmation_plan.json'
    if guard.exists():
        assert json.loads(guard.read_text())['selection_sha256']==checksum
    else:
        save(guard,dict(selection_sha256=checksum,reselection_allowed=False,fit_before_year=1976))
    confirmation=score_phase(root,'confirmation',training)
    summary=summarize(confirmation).set_index('method');base=summary.loc['baseline'];chosen=summary.loc[selected['selected_method']]
    passed=bool(chosen.volume_crps<=base.volume_crps and chosen.wet_upper_coverage>=base.wet_upper_coverage and chosen.first_mae<=1.1*base.first_mae)
    save(root/'confirmation.json',dict(complete=True,selection_sha256=checksum,selected_method=selected['selected_method'],
        gates_passed=passed,new_forecast_selected=not selected['baseline_retained'],reselection=False,
        interpretation='Confirmation is reported without selecting another method; event reruns remain labeled diagnostics.'))
    print('SELECTION',selected,'confirmation gates',passed,flush=True)


def event_plan(root):
    verify(root)
    selection=json.loads((root/'selection.json').read_text())
    confirm=json.loads((root/'confirmation.json').read_text())
    checksum=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()
    if confirm['selection_sha256']!=checksum:
        raise ValueError('Selection changed after confirmation')
    path=root/'event_plan.json'
    value=dict(selected_method=selection['selected_method'],selection_sha256=checksum,
        confirmation_sha256=hashlib.sha256((root/'confirmation.json').read_bytes()).hexdigest(),
        confirmation_gates_passed=confirm['gates_passed'],fit_years=list(range(1946,1988)),
        controller='Fixed combined precision/tie MPC; inflow forecast only changes',test_evaluated=False)
    if path.exists():
        assert json.loads(path.read_text())==value
    else:
        save(path,value)
    return value


def controller(training, method, seed):
    bank=HistoryForecastBank(training['inflow'],training['demand'],before_year=1988,method=method)
    physical={k:training[k] for k in ['lake_params','action','flood_level','starving_level']}
    c=HistoryMPC(physical,bank,forecast='conditional',config=ProtectedMPCConfig(seed=seed),
                 ema_coefficient=training['exponential_average_coeff'])
    return c,bank


def event_specs():
    plan=json.loads((FLOOD_REFERENCE/'plan.json').read_text())
    return [dict(year=e['year'],seed=seed,lead=a['lead'],start_day=a['start_day'],end_day=e['end_day'],stock=stock,forecast=forecast,
        name=f"{e['year']}_seed{seed}_lead{a['lead']}__stock_{stock}__{forecast}")
        for e in plan['events'] for seed in SEEDS for a in e['anchors'] for stock in ['original','combined'] for forecast in ['baseline','selected']]


def run_event(root,spec,training,selected_method):
    folder=root/'events'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():return
    n=365+spec['start_day']-1;hour=n*24
    common=get_snapshot(FLOOD_REFERENCE,spec['year'],spec['seed'],'original',hour)
    stock=get_snapshot(FLOOD_REFERENCE,spec['year'],spec['seed'],spec['stock'],hour)
    method='baseline' if spec['forecast']=='baseline' else selected_method
    c,bank=controller(training,method,spec['seed'])
    env,obs,info=restore(spec['year'],bank,common,stock)
    restore_plan(c,common)
    _,reference,_=source(spec['year'],spec['seed'],'original')
    past=reference.inflow_m3_s.iloc[n-7:n].to_numpy(np.float32).astype(float)
    c.prime_history(past,hour)
    expected=spec['end_day']-spec['start_day']+1
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    rows=[];failure=None
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(expected):
                obs,info,row,term,trunc=take_step(env,c,obs,info);rows.append(row)
                log.write(json.dumps(c.diagnostics[-1],allow_nan=False)+'\n')
                if term or trunc:raise AssertionError('Do not shorten the original episode')
        except OutletDomainError as error:
            failure=dict(error=str(error),elapsed_hour=env.hour)
        finally:env.close()
    frame=pd.DataFrame(rows,columns=FIELDS);frame.to_csv(data/'daily.csv',index=False)
    complete=len(frame)==expected and failure is None
    if complete and spec['forecast']=='baseline':
        name=f"{spec['year']}_seed{spec['seed']}_lead{spec['lead']}__stock_{spec['stock']}__combined__memory_original"
        old=json.loads((FLOOD_REFERENCE/name/'run.json').read_text())
        original=pd.read_csv(Path(old['data_directory'])/'daily.csv',float_precision='round_trip')
        np.testing.assert_array_equal(frame.to_numpy(float),original.to_numpy(float))
    save(folder/'run.json',dict(**spec,method=method,complete=complete,observed_days=len(frame),expected_days=expected,
        failure=failure,data_directory=str(data),
        window_cost=float(-frame.reward.sum()) if complete else None,
        flood_severity=float(frame.flood_excess_m_hours.sum()) if complete else None,
        low_water_severity=float(frame.low_water_excess_m_hours.sum()) if complete else None,
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6) if complete else None,
        peak_level_m=float(max(frame.level_before_m.max(),frame.maximum_level_m.max())) if complete else None,
        domain_recoveries=sum(d['domain_recovery'] for d in c.diagnostics)))


def events(root):
    plan=event_plan(root);training=settings('train')
    for index,spec in enumerate(event_specs(),1):
        run_event(root,spec,training,plan['selected_method'])
        if index%10==0:print('Event branches',index,'/ 80',flush=True)


def current_state_audits(root):
    selected=event_plan(root)['selected_method'];training=settings('train')
    out=root/'current_state_audits';out.mkdir(exist_ok=True)
    rows=[]
    for path in sorted((FLOOD_REFERENCE/'forecast_audits').glob('*.json')):
        old=json.loads(path.read_text());year=old['year'];seed=old['seed'];hour=old['elapsed_hour'];n=hour//24
        c,bank=controller(training,selected,seed)
        snapshot=get_snapshot(FLOOD_REFERENCE,year,seed,old['arm'],hour)
        env,obs,info=restore(year,bank,snapshot,snapshot);restore_plan(c,snapshot)
        _,frame,_=source(year,seed,'original')
        c.prime_history(frame.inflow_m3_s.iloc[n-7:n].to_numpy(np.float32).astype(float),hour)
        c(obs,info);d=c.diagnostics[-1]
        state=WaterState(c.model.lake.level_to_storage(float(obs[0])),float(obs[5]),0.)
        days=(hour//24+np.arange(d['horizon_days']))%365+1
        result=audit_forecast(c.model,state,d,days)
        save(out/path.name,dict(year=year,seed=seed,source_arm=old['arm'],model_day=old['model_day'],
            method=selected,forecast_inflows=d['forecast_inflows'],**result))
        rows.append(dict(year=year,seed=seed,source_arm=old['arm'],model_day=old['model_day'],
            certified_infeasible=result['certified_infeasible_forecast'],
            feasible_audit_candidates=result['feasible_candidates'],controller_recovery=d['domain_recovery']))
        env.close()
    pd.DataFrame(rows).to_csv(root/'current_state_summary.csv',index=False)


def aggregate(root):
    records=[json.loads((root/'events'/s['name']/'run.json').read_text()) for s in event_specs()]
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='failure'} for r in records]);frame.to_csv(root/'events.csv',index=False)
    pairs=[]
    for keys,g in frame.groupby(['year','seed','lead','stock']):
        a=g[g.forecast.eq('baseline')].iloc[0];b=g[g.forecast.eq('selected')].iloc[0]
        complete=bool(a.complete and b.complete)
        paired=dict(zip(['year','seed','lead','stock'],map(lambda x:int(x) if isinstance(x,(np.integer,int)) else x,keys)))
        paired.update(complete=complete)
        for metric in ['window_cost','flood_severity','low_water_severity','unmet_target_million_m3','peak_level_m','domain_recoveries']:
            paired['baseline_'+metric]=a[metric] if a.complete else None
            paired['selected_'+metric]=b[metric] if b.complete else None
            paired['delta_'+metric]=b[metric]-a[metric] if complete else None
        pairs.append(paired)
        data_a=Path(a.data_directory);data_b=Path(b.data_directory)
        fa=pd.read_csv(data_a/'daily.csv',float_precision='round_trip');fb=pd.read_csv(data_b/'daily.csv',float_precision='round_trip')
        n=min(len(fa),len(fb));exog=['elapsed_hour','year','model_day','inflow_m3_s','target_m3_s','past_target_ema','past_inflow_m3_s']
        np.testing.assert_array_equal(fa[exog].iloc[:n],fb[exog].iloc[:n])
        da=[json.loads(line) for line in (data_a/'decisions.jsonl').read_text().splitlines()]
        db=[json.loads(line) for line in (data_b/'decisions.jsonl').read_text().splitlines()]
        for first,second in zip(da[:n],db[:n],strict=True):
            np.testing.assert_array_equal(first['forecast_target'],second['forecast_target'])
            assert first['horizon_days']==second['horizon_days']==7
        for f in [fa,fb]:
            if len(f):
                np.testing.assert_allclose(-f.reward,f.unmet_target_m3/1e6+f.flood_excess_m_hours+f.low_water_excess_m_hours,atol=1e-12,rtol=0)
                np.testing.assert_allclose(145900000*(f.level_m-f.level_before_m),(f.inflow_m3_s-f.release_m3_s)*86400,atol=1e-6,rtol=0)
    pd.DataFrame(pairs).to_csv(root/'paired_events.csv',index=False)
    audits=pd.read_csv(root/'current_state_summary.csv')
    save(root/'run.json',dict(complete=True,all_events_complete=bool(frame.complete.all()),
        expected_events=80,completed_events=int(frame.complete.sum()),
        selected_method=json.loads((root/'selection.json').read_text())['selected_method'],
        confirmation_gates_passed=json.loads((root/'confirmation.json').read_text())['gates_passed'],
        audited_previous_failed_states=len(audits),certified_selected_forecast_states=int(audits.certified_infeasible.sum()),
        paired_exogenous_inputs_and_target_forecasts_verified=True,cost_and_balance_verified=True,
        test_evaluated=False,default_changed=False,model_or_costs_changed=False))
    print(pd.DataFrame(pairs).groupby(['year','lead','stock']).mean(numeric_only=True).to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['all','prepare','forecasts','events','audit','aggregate'],default='all')
    args=p.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    verify(root)
    if args.stage in ['all','forecasts']:forecasts(root)
    if args.stage in ['all','events']:events(root)
    if args.stage in ['all','audit']:current_state_audits(root)
    if args.stage in ['all','aggregate']:aggregate(root)
    verify(root)


if __name__=='__main__':main()
