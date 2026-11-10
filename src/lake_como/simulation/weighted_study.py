"""Training-fold selection of explicit-probability tail scenarios, then fixed events."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_CONFIG,DAM_DATA
from .weighted_forecast import WeightedForecastBank,WeightedMPC,CONFIGS,weighted_crps
from .history_forecast import HistoryForecastBank
from .protected_mpc import ProtectedMPCConfig
from .precision_validation import SEEDS
from .forecast_study import DEFAULT as REFERENCE,event_specs
from .flood_diagnosis import DEFAULT as FLOOD_REFERENCE,source,get_snapshot,restore,restore_plan
from .carryover_check import take_step,FIELDS
from .outlet import OutletDomainError
from .model import WaterState
from .feasibility_audit import full_open_lower_bound
from .reserve_use_check import save,verify

DEFAULT=PROJECT_ROOT/'results/simulator/weighted_study_v1'
FOLDS=(dict(name='early_development',before_year=1966,years=list(range(1966,1976))),
       dict(name='later_development',before_year=1976,years=list(range(1976,1982))))
LATE=dict(name='late_training_check',before_year=1976,years=list(range(1982,1988)))
ARMS=('legacy','weighted','uniform_ablation')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    paths=list(Path(__file__).parent.glob('*.py'))+[REFERENCE/'plan.json',REFERENCE/'run.json',
        REFERENCE/'event_plan.json',REFERENCE/'events.csv',PROJECT_ROOT/'src/lake_como/environment.py',
        PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',PROJECT_ROOT/'src/lake_como/splits.py',
        PROJECT_ROOT/'configs/splits.json',PROJECT_ROOT/'pyproject.toml',PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',DAM_CONFIG/'world_train.yaml',DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv',
        FLOOD_REFERENCE/'plan.json',PROJECT_ROOT/'results/simulator/tail_diagnosis_v1/plan.json']
    paths += [DAM_DATA/s/f'water_{s}_{part}.csv' for s in ['inflow','demand'] for part in ['train','test']]
    paths += list((FLOOD_REFERENCE/'snapshots').glob('*.json'))
    for r in pd.read_csv(REFERENCE/'events.csv').query('forecast=="selected"').itertuples():
        paths += [REFERENCE/'events'/r.name/'run.json',Path(r.data_directory)/'daily.csv',Path(r.data_directory)/'decisions.jsonl']
    save(root/'plan.json',dict(source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        configs=CONFIGS,folds=list(FOLDS),late_check=LATE,seeds=list(SEEDS),
        distribution='Finite historical seasonal/history mixture. Alpha grid 0/.05/.1/.2; zero is an ablation. Same bounded path scaling and history features as frozen selected forecast. No future observed inflow as policy input.',
        probabilities='Five CDF bands: 0/.25/.5/.75/(1-tail)/1 with tail .01 or .05. Draw once within each band; attach the band mass, not 20%, as scenario weight. Historical atoms may span bands.',
        score='Weighted empirical seven-day-volume CRPS, weighted first-day mean MAE, wet upper-envelope coverage. All models use five paths. Wet labels exceed each fit-prefix 90th-percentile seven-day volume.',
        origins='Same complete-history/full-horizon origins, model days 8..359, all five seeds. Fits are strictly earlier than each fold. Overlapping days/seeds are dependent.',
        selection='Pool both development folds. Strictly lower volume CRPS than legacy, wet upper coverage >= legacy, first-day MAE <=110% of legacy. Lowest CRPS among eligible, declared config order for ties; otherwise legacy. No forced positive mixture mass.',
        late_check_protocol='Freeze selection checksum before the 1982–1987 check. These years were inspected earlier and are not independent confirmation. No reselection.',
        events='Refit only 1946–1987, then 120 matched event branches: legacy forecast/controller, selected forecast with declared probabilities, and same selected paths uniformly weighted as an explicitly distorted-probability ablation. Original stocks/anchors/seeds/history/warm plan/clock.',
        objective='Physical reward weights unchanged. Only declared forecast probabilities weight expected horizon cost. Every positive-mass path must stay inside the same model domain; no infeasible path removed, renormalized or scored after extrapolation.',
        failures='Incomplete windows retain status and no full-window score; required-group averages suppressed if any run is missing.',
        interpretation='A selected probability model is not a calibrated storm distribution or general controller promotion. No final test, physical changes or default changes.',
        test_evaluated=False,model_or_reward_weights_changed=False,default_changed=False))


def summarize(frame):
    rows=[]
    for variant,g in frame.groupby('variant',sort=False):
        wet=g[g.wet]
        rows.append(dict(variant=variant,day_seed_forecasts=len(g),wet_day_seed_forecasts=len(wet),
            volume_crps=float(g.volume_crps.mean()),first_mae=float(g.first_mae.mean()),
            wet_upper_coverage=float(wet.upper_covered.mean()),upper_coverage=float(g.upper_covered.mean()),
            mean_scenario_volume=float(g.mean_volume.mean())))
    return pd.DataFrame(rows)


def choose(summary):
    index=summary.set_index('variant');base=index.loc['legacy']
    eligible=[name for name in CONFIGS if name!='legacy' and
        index.loc[name,'volume_crps']<base.volume_crps and
        index.loc[name,'wet_upper_coverage']>=base.wet_upper_coverage and
        index.loc[name,'first_mae']<=1.1*base.first_mae]
    chosen=min(eligible,key=lambda name:(index.loc[name,'volume_crps'],list(CONFIGS).index(name))) if eligible else 'legacy'
    return dict(selected_variant=chosen,eligible_variants=eligible,config=CONFIGS[chosen],
        selected_on=[f['name'] for f in FOLDS],late_outcomes_used=False,validation_events_used=False)


def score_fold(root,fold,training):
    path=root/(fold['name']+'.csv')
    if path.exists():return pd.read_csv(path)
    banks={name:WeightedForecastBank(training['inflow'],training['demand'],before_year=fold['before_year'],variant=name) for name in CONFIGS}
    fit=next(iter(banks.values()));threshold=float(np.quantile(fit.paths.sum(axis=1)*86400/1e6,.9))
    rows=[]
    for year in fold['years']:
        q=np.asarray(training['inflow'][year][:365],float)
        for n in range(7,359):
            past=np.asarray(q[n-7:n],np.float32).astype(float);truth=float(q[n:n+7].sum()*86400/1e6)
            for variant,bank in banks.items():
                bank.set_history(past)
                for seed in SEEDS:
                    paths,_,meta=bank.forecast('conditional',n+1,float(past[-1]),seed=seed+n*24)
                    p=np.asarray(meta['scenario_probabilities']);volumes=paths.sum(axis=1)*86400/1e6
                    rows.append(dict(fold=fold['name'],variant=variant,year=year,model_day=n+1,seed=seed,
                        volume_crps=weighted_crps(volumes,truth,p),
                        first_mae=abs(float(paths[:,0]@p)-q[n]),wet=truth>threshold,
                        upper_covered=bool(truth<=volumes.max()),mean_volume=float(volumes@p)))
        print(fold['name'],year,'nine variants done',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
    summarize(frame).to_csv(root/(fold['name']+'_summary.csv'),index=False)
    save(root/(fold['name']+'_record.json'),dict(complete=True,fit_years=list(fit.years),scored_years=fold['years'],
        wet_threshold_million_m3=threshold,day_seed_variant_forecasts=len(frame)))
    return frame


def forecasts(root):
    training=settings('train')
    folds=[score_fold(root,fold,training) for fold in FOLDS]
    development=pd.concat(folds,ignore_index=True);summary=summarize(development)
    summary.to_csv(root/'development_summary.csv',index=False);selected=choose(summary)
    if (root/'selection.json').exists():assert json.loads((root/'selection.json').read_text())==selected
    else:save(root/'selection.json',selected)
    checksum=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()
    path=root/'late_check_plan.json'
    guard=dict(selection_sha256=checksum,reselection_allowed=False)
    if path.exists():assert json.loads(path.read_text())==guard
    else:save(path,guard)
    late=summarize(score_fold(root,LATE,training)).set_index('variant')
    a=late.loc['legacy'];b=late.loc[selected['selected_variant']]
    passed=bool(b.volume_crps<=a.volume_crps and b.wet_upper_coverage>=a.wet_upper_coverage and b.first_mae<=1.1*a.first_mae)
    save(root/'late_check.json',dict(complete=True,selection_sha256=checksum,gates_passed=passed,
        reused_training_years=True,reselection=False))
    print(summary.to_string(index=False),flush=True);print('FROZEN SELECTION',selected,'late gates',passed,flush=True)


def locked_selection(root):
    verify(root);selected=json.loads((root/'selection.json').read_text())
    checksum=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()
    assert json.loads((root/'late_check_plan.json').read_text())['selection_sha256']==checksum
    late=json.loads((root/'late_check.json').read_text());assert late['selection_sha256']==checksum
    record=dict(selected_variant=selected['selected_variant'],selection_sha256=checksum,
        late_check_sha256=hashlib.sha256((root/'late_check.json').read_bytes()).hexdigest(),
        late_gates_passed=late['gates_passed'],refit_years=list(range(1946,1988)),test_evaluated=False)
    if (root/'event_plan.json').exists():assert json.loads((root/'event_plan.json').read_text())==record
    else:save(root/'event_plan.json',record)
    return selected['selected_variant']


def make_controller(training,variant,seed,mode='weighted'):
    bank=WeightedForecastBank(training['inflow'],training['demand'],before_year=1988,variant=variant)
    params={k:training[k] for k in ['lake_params','action','flood_level','starving_level']}
    return WeightedMPC(params,bank,probability_mode=mode,forecast='conditional',config=ProtectedMPCConfig(seed=seed),
        ema_coefficient=training['exponential_average_coeff']),bank


def specs():
    return [dict(year=s['year'],seed=s['seed'],lead=s['lead'],start_day=s['start_day'],end_day=s['end_day'],stock=s['stock'],arm=arm,
                 name=f"{s['year']}_seed{s['seed']}_lead{s['lead']}__stock_{s['stock']}__{arm}")
            for s in event_specs() if s['forecast']=='baseline' for arm in ARMS]


def run_case(root,spec,selected,training):
    folder=root/'events'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():return
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    variant='legacy' if spec['arm']=='legacy' else selected
    mode='uniform_ablation' if spec['arm']=='uniform_ablation' else 'weighted'
    c,bank=make_controller(training,variant,spec['seed'],mode)
    n=365+spec['start_day']-1;hour=n*24
    common=get_snapshot(FLOOD_REFERENCE,spec['year'],spec['seed'],'original',hour)
    stock=get_snapshot(FLOOD_REFERENCE,spec['year'],spec['seed'],spec['stock'],hour)
    env,obs,info=restore(spec['year'],bank,common,stock);restore_plan(c,common)
    _,original,_=source(spec['year'],spec['seed'],'original')
    c.prime_history(original.inflow_m3_s.iloc[n-7:n].to_numpy(np.float32).astype(float),hour)
    rows=[];failure=None;expected=spec['end_day']-spec['start_day']+1
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(expected):
                obs,info,row,term,trunc=take_step(env,c,obs,info);rows.append(row)
                log.write(json.dumps(c.diagnostics[-1],allow_nan=False)+'\n')
                if term or trunc:raise AssertionError('Event end must not truncate the original record')
        except OutletDomainError as error:failure=dict(error=str(error),elapsed_hour=env.hour)
        finally:env.close()
    frame=pd.DataFrame(rows,columns=FIELDS);frame.to_csv(data/'daily.csv',index=False)
    complete=len(frame)==expected and failure is None
    if complete and spec['arm']=='legacy':
        name=f"{spec['year']}_seed{spec['seed']}_lead{spec['lead']}__stock_{spec['stock']}__selected"
        record=json.loads((REFERENCE/'events'/name/'run.json').read_text())
        old=pd.read_csv(Path(record['data_directory'])/'daily.csv',float_precision='round_trip')
        np.testing.assert_array_equal(frame.to_numpy(float),old.to_numpy(float))
    save(folder/'run.json',dict(**spec,variant=variant,complete=complete,expected_days=expected,observed_days=len(frame),
        data_directory=str(data),failure=failure,
        window_cost=float(-frame.reward.sum()) if complete else None,
        flood_severity=float(frame.flood_excess_m_hours.sum()) if complete else None,
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6) if complete else None,
        low_water_severity=float(frame.low_water_excess_m_hours.sum()) if complete else None,
        peak_level_m=float(max(frame.level_before_m.max(),frame.maximum_level_m.max())) if complete else None,
        domain_recoveries=sum(d['domain_recovery'] for d in c.diagnostics)))


def events(root):
    selected=locked_selection(root);training=settings('train')
    for index,spec in enumerate(specs(),1):
        run_case(root,spec,selected,training)
        if index%10==0:print('Weighted event branches',index,'/ 120',flush=True)


def aggregate(root):
    records=[json.loads((root/'events'/s['name']/'run.json').read_text()) for s in specs()]
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='failure'} for r in records]);frame.to_csv(root/'events.csv',index=False)
    paired=[];coverage=[];fallbacks=[];model=make_controller(settings('train'),'legacy',901)[0].model
    for keys,group in frame.groupby(['year','seed','lead','stock']):
        frames={};decisions={}
        for r in group.itertuples():
            data=Path(r.data_directory);f=pd.read_csv(data/'daily.csv',float_precision='round_trip')
            ds=[json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines()]
            frames[r.arm]=f;decisions[r.arm]=ds
            _,truth,_=source(r.year,r.seed,'original');onset=268 if r.year==1993 else 178
            for d,row in zip(ds,f.itertuples(),strict=True):
                probs=np.asarray(d['applied_probabilities']);assert (probs>0).all() and abs(probs.sum()-1)<1e-12
                if d['domain_recovery']:
                    state=WaterState(model.lake.level_to_storage(float(np.float32(row.level_before_m))))
                    certs=[full_open_lower_bound(model,state,path)['certified_upper_exit'] for path in d['forecast_inflows']]
                    assert any(certs)
                    fallbacks.append(dict(year=r.year,seed=r.seed,lead=r.lead,stock=r.stock,arm=r.arm,elapsed_hour=d['elapsed_hours'],certified_paths=sum(certs)))
                if r.lead==14 and r.stock=='original' and onset-7<=row.model_day<onset:
                    n=d['elapsed_hours']//24;actual=float(truth.inflow_m3_s.iloc[n:n+7].sum()*86400/1e6)
                    volumes=np.asarray(d['forecast_inflows']).sum(axis=1)*86400/1e6
                    coverage.append(dict(year=r.year,seed=r.seed,model_day=int(row.model_day),arm=r.arm,
                        truth_million_m3=actual,largest_scenario_million_m3=float(volumes.max()),upper_covered=bool(actual<=volumes.max())))
            if len(f):
                np.testing.assert_allclose(-f.reward,f.unmet_target_m3/1e6+f.flood_excess_m_hours+f.low_water_excess_m_hours,atol=1e-12,rtol=0)
                np.testing.assert_allclose(145900000*(f.level_m-f.level_before_m),(f.inflow_m3_s-f.release_m3_s)*86400,atol=1e-6,rtol=0)
        n=min(len(f) for f in frames.values());exog=['elapsed_hour','year','model_day','inflow_m3_s','target_m3_s','past_target_ema','past_inflow_m3_s']
        for arm in ARMS[1:]:
            np.testing.assert_array_equal(frames['legacy'][exog].iloc[:n],frames[arm][exog].iloc[:n])
            for a,b in zip(decisions['legacy'][:n],decisions[arm][:n],strict=True):
                np.testing.assert_array_equal(a['forecast_target'],b['forecast_target'])
                assert a['horizon_days']==b['horizon_days']==7
        for a,b in zip(decisions['weighted'][:n],decisions['uniform_ablation'][:n],strict=True):
            np.testing.assert_array_equal(a['forecast_inflows'],b['forecast_inflows'])
        index=group.set_index('arm');base=index.loc['legacy']
        for arm in ARMS[1:]:
            candidate=index.loc[arm];complete=bool(base.complete and candidate.complete)
            row=dict(year=int(keys[0]),seed=int(keys[1]),lead=int(keys[2]),stock=keys[3],arm=arm,complete=complete)
            for metric in ['window_cost','flood_severity','unmet_target_million_m3','low_water_severity','domain_recoveries','peak_level_m']:
                row['legacy_'+metric]=base[metric] if base.complete else None
                row['candidate_'+metric]=candidate[metric] if candidate.complete else None
                row['delta_'+metric]=candidate[metric]-base[metric] if complete else None
            paired.append(row)
    pd.DataFrame(paired).to_csv(root/'paired_events.csv',index=False)
    pd.DataFrame(coverage).to_csv(root/'preflood_coverage.csv',index=False)
    pd.DataFrame(fallbacks).to_csv(root/'fallback_certificates.csv',index=False)
    save(root/'run.json',dict(complete=True,all_events_complete=bool(frame.complete.all()),
        expected_events=120,completed_events=int(frame.complete.sum()),selected_variant=locked_selection(root),
        declared_probabilities_verified=True,uniform_ablation_paths_identical=True,
        paired_target_forecasts_and_exogenous_history_verified=True,all_fallbacks_capacity_certified=True,
        test_evaluated=False,model_or_reward_weights_changed=False,feasibility_rule_changed=False,default_changed=False))
    summaries=[]
    for (year,lead,arm),g in frame.groupby(['year','lead','arm']):
        complete=len(g)==10 and bool(g.complete.all())
        row=dict(year=int(year),lead=int(lead),arm=arm,expected_runs=10,complete_runs=int(g.complete.sum()),all_complete=complete)
        for metric in ['window_cost','flood_severity','unmet_target_million_m3','domain_recoveries']:
            row['mean_'+metric]=float(g[metric].mean()) if complete else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(root/'event_summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['all','prepare','forecasts','events','aggregate'],default='all')
    args=p.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    verify(root)
    if args.stage in ['all','forecasts']:forecasts(root)
    if args.stage in ['all','events']:events(root)
    if args.stage in ['all','aggregate']:aggregate(root)
    verify(root)


if __name__=='__main__':main()
