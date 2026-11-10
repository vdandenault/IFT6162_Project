"""Read-only support/sampling and matched-state fallback diagnosis.

No forecast selection, fitting changes, controller runs, costs or physics changes.
Future inflows are offline labels only. Previously used training holdouts are
reused for diagnosis, never called independent confirmation here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, DAM_CONFIG, DAM_DATA
from .history_forecast import HistoryForecastBank, features
from .forecast_study import DEFAULT as REFERENCE, PHASES, controller
from .flood_diagnosis import source
from .precision_validation import SEEDS
from .feasibility_audit import full_open_lower_bound
from .model import WaterState
from .reserve_use_check import save, verify

DEFAULT=PROJECT_ROOT/'results/simulator/tail_diagnosis_v1'
CATEGORIES=('outside_seasonal_support','excluded_by_history_pool','missed_by_five_sampling','covered_by_five')


def pool_details(bank, day, past):
    """Exactly reconstruct the frozen scaled/stratified generator before sampling."""
    if bank.method!='history_scaled_stratified':
        raise ValueError('This audit is specific to the frozen selected method')
    past=np.asarray(past,float)
    if past.shape!=(7,) or not 1<=day<=365:
        raise ValueError('Seven past observations and aligned calendar required')
    indices=bank.pools[day]
    distance=np.mean(((bank.past[indices]-features(past))/bank.scale)**2,axis=1)
    order=np.argsort(distance,kind='stable')[:128]
    ratios=(past[-1]+1)/(bank.flat[bank.starts[indices]-1]+1)
    factors=np.clip(ratios,.5,2.)
    raw=bank.paths[indices]
    scaled=raw*factors[:,None]
    return dict(seasonal=scaled,conditioned=scaled[order].copy(),raw_conditioned=raw[order].copy(),
        uncapped_conditioned=raw[order]*ratios[order,None],starts=bank.starts[indices[order]].copy(),
        factors=factors[order],ratios=ratios[order])


def draw_indices(paths,count,seed):
    if count<1 or count>len(paths):raise ValueError('Nonempty volume strata required')
    order=np.argsort(np.asarray(paths).sum(axis=1),kind='stable')
    rng=np.random.default_rng(seed)
    return np.array([rng.choice(group) for group in np.array_split(order,count)])


def nested_draws(paths,seed):
    five=draw_indices(paths,5,seed)
    fifteen=np.r_[five,draw_indices(paths,10,seed+100000)]
    twentyfive=np.r_[fifteen,draw_indices(paths,10,seed+200000)]
    return {5:five,15:fifteen,25:twentyfive}


def support_record(details,truth,seed):
    volume=lambda paths:np.asarray(paths).sum(axis=1)*86400/1e6
    conditioned=volume(details['conditioned']);seasonal=volume(details['seasonal'])
    draws=nested_draws(details['conditioned'],seed)
    maxima={count:float(conditioned[ids].max()) for count,ids in draws.items()}
    if truth<=maxima[5]:category='covered_by_five'
    elif truth>seasonal.max():category='outside_seasonal_support'
    elif truth>conditioned.max():category='excluded_by_history_pool'
    else:category='missed_by_five_sampling'
    # Probability of hitting truth under one independent draw per original
    # volume stratum, conditional on this frozen finite candidate population.
    order=np.argsort(conditioned,kind='stable')
    miss_probability=float(np.prod([np.mean(conditioned[group]<truth) for group in np.array_split(order,5)]))
    record=dict(category=category,truth_million_m3=float(truth),
        seasonal_max_million_m3=float(seasonal.max()),pool_max_million_m3=float(conditioned.max()),
        raw_pool_max_million_m3=float(volume(details['raw_conditioned']).max()),
        uncapped_pool_max_million_m3=float(volume(details['uncapped_conditioned']).max()),
        conditional_five_upper_coverage_probability=1-miss_probability,
        upper_ratio_clips=int((details['ratios']>2.).sum()),
        lower_ratio_clips=int((details['ratios']<.5).sum()))
    for count,ids in draws.items():
        record[f'covered_{count}']=bool(truth<=maxima[count])
        record[f'max_{count}_million_m3']=maxima[count]
        record[f'unique_{count}']=len(np.unique(ids))
    return record


def fallback_contrasts(cells):
    """Two exact conditional orders, not unique causal percentages."""
    a,b,c,d=(np.asarray(cells[key],int) for key in ['old_state_old_forecast','old_state_new_forecast',
                                                'new_state_old_forecast','new_state_new_forecast'])
    if not(a.shape==b.shape==c.shape==d.shape):raise ValueError('Matched time grids required')
    return dict(observed_extra_fallbacks=int((d-a).sum()),forecast_effect_at_old_states=int((b-a).sum()),
        state_effect_under_new_forecast=int((d-b).sum()),state_effect_under_old_forecast=int((c-a).sum()),
        forecast_effect_at_new_states=int((d-c).sum()),
        forecast_new_warnings_at_old_states=int(((a==0)&(b==1)).sum()),
        forecast_removed_warnings_at_old_states=int(((a==1)&(b==0)).sum()))


def forecast_infeasible(model,level,paths):
    """Exact capacity certificate, with a cheap zero-release non-exit bound."""
    paths=np.asarray(paths,float)
    if not np.isfinite(paths).all() or (paths<0).any():raise ValueError('Nonnegative finite paths required')
    state=WaterState(model.lake.level_to_storage(float(np.float32(level))))
    model.lake.check_domain(model.lake.storage_to_level(state.storage))
    for path in paths:
        # With nonnegative inflow and nonnegative release, zero release is an
        # upper level bound. If it stays below the guard, full opening must too.
        if float(np.float32(level))+path.sum()*86400/model.lake.surface<=model.lake.spec.upper_evaluation_m:
            continue
        if full_open_lower_bound(model,state,path)['certified_upper_exit']:
            return True
    return False


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    selection=json.loads((REFERENCE/'selection.json').read_text())
    if selection['selected_method']!='history_scaled_stratified':raise ValueError('Frozen selected method changed')
    paths=list(Path(__file__).parent.glob('*.py'))+[REFERENCE/'plan.json',REFERENCE/'run.json',
        REFERENCE/'selection.json',REFERENCE/'confirmation_plan.json',REFERENCE/'event_plan.json',
        REFERENCE/'events.csv',REFERENCE/'tuning.csv',REFERENCE/'confirmation.csv',
        PROJECT_ROOT/'src/lake_como/environment.py',PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/lake_como/splits.py',PROJECT_ROOT/'configs/splits.json',
        PROJECT_ROOT/'pyproject.toml',PROJECT_ROOT/'uv.lock',DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',
        DAM_CONFIG/'world_train.yaml',DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv']
    paths += [DAM_DATA/s/f'water_{s}_{p}.csv' for s in ['inflow','demand'] for p in ['train','test']]
    for r in pd.read_csv(REFERENCE/'events.csv').itertuples():
        paths += [REFERENCE/'events'/r.name/'run.json',Path(r.data_directory)/'daily.csv',Path(r.data_directory)/'decisions.jsonl']
    save(root/'plan.json',dict(source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        support='Partition upper misses: outside all seasonally eligible scaled paths; removed by nearest-128 history conditioning; in pool but missed by original five sampled paths. Future truth is an offline label only.',
        scaling='Report raw and uncapped conditional-pool support only as diagnostics. No unbounded rescaling is applied to a controller.',
        density='Nested 5/15/25 scenario sets retain the original five and add independent stratified batches of ten, with seed offsets 100000/200000. Coverage-only sensitivity, not a policy or weighted-score comparison.',
        holdouts='Reuse 1976–1981 and 1982–1987 with frozen 1946–1975 fit. These are already inspected development holdouts, not new confirmation or selection.',
        events='Both fixed events, all five seeds, last seven pre-flood days from the 14-day/original-stock branch. Fit remains 1946–1987; labels are realized seven-day volumes.',
        fallback='For every paired time in the 80 prior event replays, cross both saved states with both forecasts. Reconstruct forecasts from actual past only; certify domain infeasibility with the fixed full-outlet hourly bound. Verify diagonal decisions against logged fallbacks.',
        attribution='Two conditional orders for forecast/state contributions; counts include overlapping anchors and stocks and are not independent samples or unique causal percentages.',
        limits='No new forecast fitting, selection, controller runs, weights, physics, defaults or final tests. The outlet-domain guard is distinct from the flood threshold.',
        test_evaluated=False,model_or_costs_changed=False,default_changed=False))


def holdouts(root):
    training=settings('train')
    bank=HistoryForecastBank(training['inflow'],training['demand'],before_year=1976,method='history_scaled_stratified')
    for phase,years in PHASES.items():
        path=root/(phase+'_support.csv')
        if path.exists():continue
        prior=pd.read_csv(REFERENCE/(phase+'.csv'))
        prior=prior[prior.method.eq('history_scaled_stratified')].set_index(['year','model_day','seed'])
        wet_threshold=json.loads((REFERENCE/(phase+'_record.json')).read_text())['wet_threshold_million_m3']
        rows=[]
        for year in years:
            q=np.array(training['inflow'][year][:365],float)
            for n in range(7,359):
                past=np.asarray(q[n-7:n],np.float32).astype(float)
                details=pool_details(bank,n+1,past);truth=float(q[n:n+7].sum()*86400/1e6)
                for seed in SEEDS:
                    sampled_seed=seed+n*24
                    row=support_record(details,truth,sampled_seed)
                    old=prior.loc[(year,n+1,seed)]
                    assert bool(old.upper_covered)==row['covered_5']
                    # Reconstructed five paths reproduce the frozen generator.
                    bank.set_history(past)
                    frozen,_,_=bank.forecast('conditional',n+1,float(past[-1]),seed=sampled_seed)
                    ids=nested_draws(details['conditioned'],sampled_seed)[5]
                    np.testing.assert_array_equal(frozen,details['conditioned'][ids])
                    rows.append(dict(phase=phase,year=year,model_day=n+1,seed=seed,wet=truth>wet_threshold,**row))
            print(phase,year,'support done',flush=True)
        pd.DataFrame(rows).to_csv(path,index=False)


def events(root):
    path=root/'event_support.csv'
    if path.exists():return
    training=settings('train')
    bank=HistoryForecastBank(training['inflow'],training['demand'],before_year=1988,method='history_scaled_stratified')
    rows=[]
    for r in pd.read_csv(REFERENCE/'events.csv').query('lead==14 and stock=="original" and forecast=="selected"').itertuples():
        _,actual,_=source(r.year,r.seed,'original');onset=268 if r.year==1993 else 178
        ds=[json.loads(line) for line in (Path(r.data_directory)/'decisions.jsonl').read_text().splitlines()]
        for d in ds:
            n=d['elapsed_hours']//24;day=n%365+1
            if not onset-7<=day<onset:continue
            past=actual.inflow_m3_s.iloc[n-7:n].to_numpy(np.float32).astype(float)
            details=pool_details(bank,day,past);truth=float(actual.inflow_m3_s.iloc[n:n+7].sum()*86400/1e6)
            seed=r.seed+d['elapsed_hours'];ids=nested_draws(details['conditioned'],seed)[5]
            np.testing.assert_array_equal(details['conditioned'][ids],d['forecast_inflows'])
            rows.append(dict(year=r.year,model_day=day,seed=r.seed,**support_record(details,truth,seed)))
    pd.DataFrame(rows).to_csv(path,index=False)


def fallbacks(root):
    path=root/'fallback_cells.csv'
    if path.exists():return
    training=settings('train');model=controller(training,'baseline',901)[0].model
    bank=HistoryForecastBank(training['inflow'],training['demand'],before_year=1988,method='history_scaled_stratified')
    baseline=HistoryForecastBank(training['inflow'],training['demand'],before_year=1988,method='baseline')
    frame=pd.read_csv(REFERENCE/'events.csv');rows=[]
    for keys,group in frame.groupby(['year','seed','lead','stock']):
        a=group[group.forecast.eq('baseline')].iloc[0];b=group[group.forecast.eq('selected')].iloc[0]
        fa=pd.read_csv(Path(a.data_directory)/'daily.csv',float_precision='round_trip')
        fb=pd.read_csv(Path(b.data_directory)/'daily.csv',float_precision='round_trip')
        da=[json.loads(line) for line in (Path(a.data_directory)/'decisions.jsonl').read_text().splitlines()]
        db=[json.loads(line) for line in (Path(b.data_directory)/'decisions.jsonl').read_text().splitlines()]
        assert len(fa)==len(fb)==len(da)==len(db) and a.complete and b.complete
        _,actual,_=source(int(a.year),int(a.seed),'original')
        for old,new,ra,rb in zip(da,db,fa.itertuples(),fb.itertuples(),strict=True):
            hour=old['elapsed_hours'];n=hour//24;day=n%365+1
            past=actual.inflow_m3_s.iloc[n-7:n].to_numpy(np.float32).astype(float)
            for fitted in [baseline,bank]:fitted.set_history(past)
            oldpaths=baseline.forecast('conditional',day,float(past[-1]),seed=int(a.seed)+hour)[0]
            newpaths=bank.forecast('conditional',day,float(past[-1]),seed=int(a.seed)+hour)[0]
            np.testing.assert_array_equal(oldpaths,old['forecast_inflows']);np.testing.assert_array_equal(newpaths,new['forecast_inflows'])
            cells=dict(old_state_old_forecast=forecast_infeasible(model,ra.level_before_m,oldpaths),
                old_state_new_forecast=forecast_infeasible(model,ra.level_before_m,newpaths),
                new_state_old_forecast=forecast_infeasible(model,rb.level_before_m,oldpaths),
                new_state_new_forecast=forecast_infeasible(model,rb.level_before_m,newpaths))
            assert cells['old_state_old_forecast']==old['domain_recovery']
            assert cells['new_state_new_forecast']==new['domain_recovery']
            rows.append(dict(year=int(a.year),seed=int(a.seed),lead=int(a.lead),stock=a.stock,model_day=day,
                old_level_m=float(ra.level_before_m),new_level_m=float(rb.level_before_m),
                old_first_inflow_mean=float(oldpaths[:,0].mean()),new_first_inflow_mean=float(newpaths[:,0].mean()),**cells))
        print('Crossed fallback states',keys,flush=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def aggregate(root):
    summaries=[];categories=[]
    for scope in ['tuning','confirmation','event_1993','event_1997']:
        if scope in PHASES:
            frame=pd.read_csv(root/(scope+'_support.csv'));subsets={'all':frame,'wet':frame[frame.wet]}
        else:
            frame=pd.read_csv(root/'event_support.csv');frame=frame[frame.year.eq(int(scope[-4:]))];subsets={'selected_preflood_week':frame}
        for subset,f in subsets.items():
            assert set(f.category).issubset(CATEGORIES)
            assert (f.covered_5.astype(int)<=f.covered_15.astype(int)).all()
            assert (f.covered_15.astype(int)<=f.covered_25.astype(int)).all()
            assert (f.category.eq('covered_by_five')==f.covered_5).all()
            summaries.append(dict(scope=scope,subset=subset,origins=len(f),upper_misses_5=int((~f.covered_5).sum()),
                coverage_5=float(f.covered_5.mean()),coverage_15=float(f.covered_15.mean()),coverage_25=float(f.covered_25.mean()),
                pool_coverage=float((f.truth_million_m3<=f.pool_max_million_m3).mean()),
                seasonal_coverage=float((f.truth_million_m3<=f.seasonal_max_million_m3).mean()),
                uncapped_pool_coverage=float((f.truth_million_m3<=f.uncapped_pool_max_million_m3).mean()),
                raw_pool_coverage=float((f.truth_million_m3<=f.raw_pool_max_million_m3).mean()),
                expected_five_coverage=float(f.conditional_five_upper_coverage_probability.mean())))
            for category in CATEGORIES:
                categories.append(dict(scope=scope,subset=subset,category=category,count=int(f.category.eq(category).sum())))
    pd.DataFrame(summaries).to_csv(root/'coverage_summary.csv',index=False)
    pd.DataFrame(categories).to_csv(root/'miss_categories.csv',index=False)
    cells=pd.read_csv(root/'fallback_cells.csv')
    contrasts=[]
    fields=['old_state_old_forecast','old_state_new_forecast','new_state_old_forecast','new_state_new_forecast']
    for label,g in [('all',cells),('1993',cells[cells.year.eq(1993)]),('1997',cells[cells.year.eq(1997)])]:
        contrasts.append(dict(scope=label,matched_times=len(g),**fallback_contrasts({k:g[k].to_numpy() for k in fields}),
            **{k:int(g[k].sum()) for k in fields}))
    pd.DataFrame(contrasts).to_csv(root/'fallback_contrasts.csv',index=False)
    save(root/'run.json',dict(complete=True,matched_event_forecast_origins=70,
        reused_holdout_day_seed_origins=21120,matched_fallback_times=len(cells),
        nested_coverage_monotone_verified=True,original_five_paths_and_fallbacks_reproduced=True,
        no_new_controller_runs=True,test_evaluated=False,model_or_costs_changed=False,default_changed=False))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(pd.DataFrame(categories).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['all','prepare','holdouts','events','fallbacks','aggregate'],default='all')
    args=p.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    verify(root)
    if args.stage in ['all','holdouts']:holdouts(root)
    if args.stage in ['all','events']:events(root)
    if args.stage in ['all','fallbacks']:fallbacks(root)
    if args.stage in ['all','aggregate']:aggregate(root)
    verify(root)


if __name__=='__main__':main()
