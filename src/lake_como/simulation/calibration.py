"""Source-backed calibration evidence, with explicit identifiability limits.

Fits an effective historical balance coefficient, NOT surveyed lake geometry or
unobserved gate/routing parameters. Uses only 1946–1987 with a nested year split;
1998–2010 test outcomes are never inspected or fitted.
"""
import argparse
from io import StringIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_DATA,DAM_CONFIG
from lake_como.mpc_hindsight import digest

SOURCE_ROOT=PROJECT_ROOT/'results/simulator/calibration_research/sources'
SOURCES={
    'pangaea_919886_inflow.tsv':'https://doi.pangaea.de/10.1594/PANGAEA.919886?format=textfile',
    'pangaea_919889_outflow.tsv':'https://doi.pangaea.de/10.1594/PANGAEA.919889?format=textfile',
    'operator_manual_2013.pdf':'https://www.addaconsorzio.it/wp-content/uploads/2021/07/dispensa-consorzio-adda.pdf',
    'ranzi_2021_historical_methods.pdf':'https://iris.unibs.it/retrieve/13f3dac1-dd91-411a-9473-b16b4e69f0dc/2021_Ranzi-et-al%40L_Acqua_Adda_HR%2BFrontespizio_compressed.pdf',
}
CITATION='Ranzi, R.; Michailidi, E. M.; Tomirotti, M.; Crespi, A.; Brunetti, M.; Maugeri, M. (2020). Daily inflows at Como lake, PANGAEA.919886; Daily outflows from Como lake, PANGAEA.919889. CC BY 4.0.'


def read_pangaea(path,years):
    """Retain only declared years before checking or scoring numerical observations."""
    years=tuple(years)
    if not years or any(y<1946 or y>1987 for y in years):
        raise ValueError('This calibration accepts chronological training years only')
    text=Path(path).read_text()
    if '*/\n' not in text or 'Flow rate, daily mean' not in text:raise ValueError('Missing PANGAEA metadata/units')
    frame=pd.read_csv(StringIO(text.split('*/\n',1)[1]),sep='\t')
    if frame.columns.tolist()!=['Date/Time','Q [m**3/s]']:raise ValueError('Unexpected source schema')
    frame['date']=pd.to_datetime(frame['Date/Time'],format='%Y-%m-%d')
    frame=frame[frame.date.dt.year.isin(years)].copy().sort_values('date')
    if frame.date.duplicated().any():raise ValueError('Duplicate dates in selected calibration years')
    for year in years:
        expected=pd.date_range(f'{year}-01-01',f'{year}-12-31')
        actual=pd.DatetimeIndex(frame.loc[frame.date.dt.year.eq(year),'date'])
        if not actual.equals(expected):raise ValueError(f'Incomplete dated source for {year}')
    values=frame['Q [m**3/s]'].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid daily flow')
    return frame.set_index('date')['Q [m**3/s]']


def fit_balance_area(delta_level,net_flow):
    delta=np.asarray(delta_level,float);flow=np.asarray(net_flow,float)
    if delta.shape!=flow.shape or delta.ndim!=1 or not np.isfinite([delta,flow]).all():raise ValueError('Aligned finite one-dimensional arrays required')
    keep=abs(delta)>=.005-1e-9
    if not keep.any():raise ValueError('No informative level changes')
    ratios=86400*flow[keep]/delta[keep]
    result=float(np.median(ratios))
    if not np.isfinite(result) or result<=0:raise ValueError('Invalid effective balance coefficient')
    return result


def balance_scores(delta,net,area):
    delta=np.asarray(delta,float);net=np.asarray(net,float)
    flow_residual=net-area*delta/86400
    height_error=86400*net/area-delta
    return dict(transitions=len(delta),flow_mae_m3_s=float(abs(flow_residual).mean()),
        flow_rmse_m3_s=float(np.sqrt(np.mean(flow_residual**2))),
        one_step_level_mae_mm=float(abs(height_error).mean()*1000),
        one_step_level_rmse_mm=float(np.sqrt(np.mean(height_error**2))*1000))


def require_operational_calibration(profile):
    """Prevent a source audit or circular balance fit being promoted as full calibration."""
    required=['confirmed_flow_interval','independent_outlet_measurements','gate_ramp_identified',
              'river_routing_identified','downstream_reference_verified','irrigation_demand_verified']
    missing=[k for k in required if profile.get('evidence',{}).get(k) is not True]
    if missing:raise ValueError('Not an operationally calibrated profile: '+', '.join(missing))
    return True


def collect_rows(params,levels,source_in,source_out):
    records=[]
    for year in sorted(params['inflow']):
        dates=pd.date_range(f'{year}-01-01',f'{year}-12-31')
        h=levels.loc[year].dropna().to_numpy(float)
        inflow=np.asarray(params['inflow'][year],float);target=np.asarray(params['demand'][year],float)
        if not(len(dates)==len(h)==len(inflow)==len(target)):raise ValueError('Raw leap samples must be preserved in this audit')
        for k,date in enumerate(dates):
            records.append(dict(date=str(date.date()),year=year,ordinal=k,level_m=float(h[k]),
                benchmark_inflow_m3_s=float(inflow[k]),benchmark_target_m3_s=float(target[k]),
                published_inflow_m3_s=float(source_in.loc[date]),published_outflow_m3_s=float(source_out.loc[date]),
                next_level_m=None if k+1==len(h) else float(h[k+1])))
    return pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/simulator/calibration_evidence_v2')
    p.add_argument('--sources',type=Path,default=SOURCE_ROOT);p.add_argument('--fetch',action='store_true')
    args=p.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):p.error('Choose an empty output; prior evidence is preserved')
    if args.fetch:
        args.sources.mkdir(parents=True,exist_ok=True)
        for name,url in SOURCES.items():
            dest=args.sources/name
            if not dest.exists():
                with urlopen(url,timeout=90) as response:dest.write_bytes(response.read())
    if any(not(args.sources/name).exists() for name in SOURCES):p.error('Sources missing; run with --fetch or supply --sources')
    root.mkdir(parents=True,exist_ok=True);params=settings('train');years=sorted(params['inflow'])
    source_in=read_pangaea(args.sources/'pangaea_919886_inflow.tsv',years)
    source_out=read_pangaea(args.sources/'pangaea_919889_outflow.tsv',years)
    levels=pd.concat([pd.read_csv(DAM_DATA/'level'/f'water_level_{part}.csv',index_col='year') for part in ['train','test']])
    frame=collect_rows(params,levels,source_in,source_out);frame.to_csv(root/'training_source_crosswalk.csv',index=False)
    comparisons=[]
    for signal in ['benchmark_inflow_m3_s','benchmark_target_m3_s']:
        for source in ['published_inflow_m3_s','published_outflow_m3_s']:
            for shift in [-1,0,1]:
                left=[];right=[]
                for _,group in frame.groupby('year'):
                    target=group[source].shift(-shift);mask=target.notna()
                    left.extend(group.loc[mask,signal]);right.extend(target[mask])
                a=np.asarray(left);b=np.asarray(right)
                comparisons.append(dict(benchmark=signal,source=source,source_date_shift=shift,points=len(a),
                    correlation=float(np.corrcoef(a,b)[0,1]),mae_m3_s=float(abs(a-b).mean()),exact_fraction=float(np.mean(a==b))))
    pd.DataFrame(comparisons).to_csv(root/'dated_flow_comparisons.csv',index=False)
    pairs=frame[frame.next_level_m.notna()].copy();pairs['delta_level_m']=pairs.next_level_m-pairs.level_m
    pairs['net_proxy_m3_s']=pairs.benchmark_inflow_m3_s-pairs.benchmark_target_m3_s
    fit=pairs[pairs.year.le(1975)];check=pairs[pairs.year.ge(1976)]
    effective=fit_balance_area(fit.delta_level_m,fit.net_proxy_m3_s)
    yearly=[]
    for year,group in fit.groupby('year'):
        yearly.append(dict(year=int(year),effective_area_m2=fit_balance_area(group.delta_level_m,group.net_proxy_m3_s)))
    pd.DataFrame(yearly).to_csv(root/'fitting_year_coefficients.csv',index=False)
    scores=[]
    for partition,group in [('fit_1946_1975',fit),('check_1976_1987',check)]:
        for name,area in [('pinned',145900000.),('published_geometry',145033000.),('effective_balance_fit',effective)]:
            scores.append(dict(partition=partition,model=name,area_m2=area,**balance_scores(group.delta_level_m,group.net_proxy_m3_s,area)))
    pd.DataFrame(scores).to_csv(root/'balance_scores.csv',index=False)
    # External published outlet reconstruction is assessed without fitting it.
    external=[]
    for shift in [-1,0,1]:
        net=[];delta=[]
        for _,group in frame.groupby('year'):
            q=group.published_inflow_m3_s.shift(-shift)-group.published_outflow_m3_s.shift(-shift)
            valid=q.notna()&group.next_level_m.notna()
            net.extend(q[valid]);delta.extend((group.next_level_m-group.level_m)[valid])
        external.append(dict(source_date_shift=shift,**balance_scores(delta,net,145033000.)))
    pd.DataFrame(external).to_csv(root/'external_reconstruction_checks.csv',index=False)
    anchors=[]
    for date,documented,tolerance in [('1984-02-13',-.71,.00501),('1987-07-21',2.64,1e-12)]:
        found=float(frame.loc[frame.date.eq(date),'level_m'].iloc[0]);anchors.append(dict(date=date,
            csv_level_m=found,documented_malgrate_level_m=documented,tolerance_m=tolerance,matches=bool(abs(found-documented)<=tolerance)))
    if not all(r['matches'] for r in anchors):raise ValueError('Documented level/date anchors do not match')
    outlet_anchors=[]
    for date,documented in [('1953-04-04',18.),('1987-07-20',917.7)]:
        found=float(frame.loc[frame.date.eq(date),'benchmark_target_m3_s'].iloc[0])
        outlet_anchors.append(dict(date=date,benchmark_target_m3_s=found,documented_outlet_flow_m3_s=documented,
            matches=bool(abs(found-documented)<1e-8)))
    if not all(r['matches'] for r in outlet_anchors):raise ValueError('Dated outlet-flow anchors do not match')
    domain=pairs.assign(below_operating_floor=(pairs.level_m<-.5)|(pairs.next_level_m<-.5),
        positive_proxy_release=pairs.benchmark_target_m3_s>0)
    domain.to_csv(root/'balance_transitions.csv',index=False)
    domain_summary=domain.groupby('below_operating_floor').agg(transitions=('year','count'),
        positive_proxy_fraction=('positive_proxy_release','mean'),minimum_observed_level_m=('level_m','min')).reset_index()
    domain_summary.to_csv(root/'operating_domain_summary.csv',index=False)
    replays=[]
    for year,group in pairs.groupby('year'):
        for name,area in [('pinned',145900000.),('effective_balance_fit',effective)]:
            predicted=group.level_m.iloc[0]+np.cumsum(group.net_proxy_m3_s.to_numpy()*86400/area)
            residual=predicted-group.next_level_m.to_numpy()
            replays.append(dict(year=int(year),partition='fit' if year<=1975 else 'check',model=name,
                level_rmse_mm=float(np.sqrt(np.mean(residual**2))*1000),final_level_error_mm=float(residual[-1]*1000)))
    pd.DataFrame(replays).to_csv(root/'prescribed_proxy_release_replay.csv',index=False)
    flows=frame.benchmark_target_m3_s.to_numpy();lags=[]
    for _,group in frame.groupby('year'):lags.extend(abs(np.diff(group.published_outflow_m3_s)))
    lags=np.asarray(lags)
    profile=dict(version='calibration-evidence-v1',fully_calibrated=False,
        purpose='Source-confirmed rating-curve datum and dataset balance reconstruction; not an operational calibration',
        evidence=dict(calendar_date_anchor='Strong training-only dated crossmatch and documented date/level anchors; no exact source identity claim',
            rating_curve_gauge='Malgrate, historical official zero 197.37 m above sea level',
            level_series_gauge='Supported by training-date extremes, not independently authenticated for every row',
            observation_clock='Malgrate readings normally 08:00 since 1946 in Ranzi et al. 2021; exact per-row clock/timezone not established',
            confirmed_flow_interval=False,independent_outlet_measurements=False,
            gate_ramp_identified=False,river_routing_identified=False,downstream_reference_verified=False,irrigation_demand_verified=False),
        source_supported_rating=dict(coefficient=33.37,zero_flow_level_m=-2.5,exponent=2.015,
            interpretation='Published fully-open outlet curve in Malgrate level; not an observed gate-position model'),
        geometry=dict(pinned_area_m2=145900000.,published_area_at_199m_m2=145033000.,
            fitted_effective_balance_area_m2=effective,
            warning='The fit nearly reconstructs a data-generation identity. It is not an independent surveyed-area calibration.'),
        target_semantics='The benchmark demand target strongly tracks published historical outlet flow; treat irrigation interpretation as unverified.',
        documentary_anchors=anchors,dated_outlet_anchors=outlet_anchors,
        structural_domain_finding='The -0.5 m operating reference is not a physical no-flow elevation. Published curve has zero at -2.5 m; observed experimental low levels extend below -0.5 m. Pinned zero-release behavior there cannot be calibrated by surface-area tuning.',
        outlet_change_statistics=dict(p50_daily_change_m3_s=float(np.quantile(lags,.5)),p99_daily_change_m3_s=float(np.quantile(lags,.99)),
            maximum_daily_change_m3_s=float(lags.max()),warning='Daily changes do not identify hourly actuator limits'),
        operational_parameters=dict(ramp_m3_s_per_hour=None,river_residence_hours=None,downstream_reference_m3_s=None,
            status='Not identified; do not promote illustrative 5/6/300 settings'),
        fit_years=list(range(1946,1976)),checking_years=list(range(1976,1988)),test_evaluated=False,
        provenance_warning='PANGAEA inflows/outflows are reconstructed series; balance consistency is not independent measurement validation',
        citation=CITATION)
    paths=[Path(__file__),*list(args.sources/name for name in SOURCES),DAM_CONFIG/'lake.yaml',PROJECT_ROOT/'configs/splits.json']
    paths += [DAM_DATA/s/f'water_{s}_{part}.csv' for s in ['inflow','demand','level'] for part in ['train','test']]
    manifest={str(path.resolve()):digest(path) for path in paths}
    profile['source_sha256']=manifest;profile['source_urls']=SOURCES
    save=lambda path,value:path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save(root/'profile.json',profile)
    try:require_operational_calibration(profile)
    except ValueError as error:save(root/'readiness.json',dict(operationally_calibrated=False,reason=str(error),data_reconstruction_fit_complete=True))
    else:raise AssertionError('Unidentified operational parameters incorrectly accepted')
    for name,sha in manifest.items():assert digest(Path(name))==sha,name
    save(root/'run.json',dict(complete=True,test_evaluated=False,fit_years=profile['fit_years'],checking_years=profile['checking_years'],
        effective_balance_area_m2=effective,source_hashes_verified=True,operationally_calibrated=False))
    print(json.dumps({k:profile[k] for k in ['geometry','target_semantics','documentary_anchors','dated_outlet_anchors','operational_parameters']},indent=2))
    print(pd.DataFrame(scores).to_string(index=False))

if __name__=='__main__':main()
