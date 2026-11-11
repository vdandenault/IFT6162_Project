"""Conditional historical consistency checks, plus a strict measured-data gateway."""
import json
import numpy as np
import pandas as pd
from gym4real.envs.dam.lake import Lake
from reservoir_control.systems.lake_como.paths import DAM_DATA
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.splits import split_years


def observed_level_consistency():
    """Retain every timing assumption; never select an offset or fit a curve.

    Release inferred by balance is a proxy, not an observed operator decision.
    Extreme-request trajectories are model envelopes, not impossibility proofs.
    """
    params=settings('train');lake=Lake(params['lake_params'])
    levels=pd.concat([pd.read_csv(DAM_DATA/'level'/f'water_level_{p}.csv',index_col='year') for p in ['train','test']])
    rows=[]
    for year in split_years('train'):
        h=levels.loc[year].to_numpy(float)[:365];flow=np.asarray(params['inflow'][year][:365])
        for offset in [-1,0,1]:
            for t in range(364):
                index=t+offset
                if not 0<=index<365:continue
                storage=lake.level_to_storage(h[t]);observed=lake.level_to_storage(h[t+1])
                ends=[lake.integration(24,storage,q,float(flow[index]),t+1)[0] for q in [0.,float(params['action']['high'])]]
                lo,hi=min(ends),max(ends)
                rows.append(dict(year=year,model_day=t+1,inflow_offset=offset,
                    inferred_release_m3_s=float(flow[index]-(observed-storage)/86400),
                    observed_level_m=float(h[t]),next_level_m=float(h[t+1]),
                    outside_extreme_request_envelope=bool(observed<lo-1e-6 or observed>hi+1e-6),
                    envelope_distance_m=max(lo-observed,observed-hi,0)/lake.surface))
    frame=pd.DataFrame(rows)
    summary=frame.groupby('inflow_offset').agg(transitions=('year','count'),
        outside_fraction=('outside_extreme_request_envelope','mean'),
        mean_envelope_distance_m=('envelope_distance_m','mean'),
        negative_proxy_fraction=('inferred_release_m3_s',lambda x:float((x<0).mean()))).reset_index()
    return frame,summary


def rating_curve_audit():
    params=settings('train');lake=Lake(params['lake_params']);rows=[]
    for level in np.linspace(-.5,2.,101):
        storage=lake.level_to_storage(float(level))
        for day in range(1,366):
            lower=float(lake.min_release(storage,day));upper=float(lake.max_release(storage,day))
            rows.append(dict(level_m=float(level),model_day=day,minimum_release=lower,maximum_release=upper,
                environmental_minimum_above_hydraulic_capacity=bool(lower>upper+1e-10),
                finite_nonnegative=bool(np.isfinite([lower,upper]).all() and min(lower,upper)>=0)))
    return pd.DataFrame(rows)


def measured_balance(frame,metadata,surface=145900000.):
    """Validate a supplied aligned measured series; no remote data is fabricated.

    Rows: timestamp, level_m, inflow_m3_s, outflow_m3_s. Flows are means from
    that row until the next timestamp. A consistent gauge datum and alignment
    must be explicitly declared. This checks balance, not gate model calibration.
    """
    required=['source_url','timezone','level_datum']
    if any(not metadata.get(k) for k in required) or metadata.get('alignment_confirmed') is not True:
        raise ValueError('Source, timezone, datum and confirmed alignment are required')
    if metadata.get('flow_convention')!='interval_mean_until_next_timestamp':
        raise ValueError('Explicit interval-mean flow convention required')
    if metadata.get('level_convention')!='at_timestamp' or metadata.get('units')!={'level':'m','inflow':'m3/s','outflow':'m3/s'}:
        raise ValueError('Explicit level convention and SI units required')
    if len(frame)<2 or not np.isfinite(surface) or surface<=0:raise ValueError('At least two records and positive surface required')
    timestamps=pd.DatetimeIndex(pd.to_datetime(frame.timestamp))
    timestamps=timestamps.tz_localize(metadata['timezone'],ambiguous='raise',nonexistent='raise') if timestamps.tz is None else timestamps.tz_convert(metadata['timezone'])
    dt=np.diff(timestamps.asi8)/1e9
    values=frame[['level_m','inflow_m3_s','outflow_m3_s']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,1:]<0).any() or (dt<=0).any():raise ValueError('Finite, ordered, nonnegative-flow measurements required')
    residual=surface*np.diff(values[:,0])-dt*(values[:-1,1]-values[:-1,2])
    return dict(intervals=len(dt),balance_rmse_m3=float(np.sqrt(np.mean(residual**2))),
        balance_mae_m3=float(np.mean(abs(residual))),source=metadata['source_url'],
        limitation='Constant pinned surface; evaporation/other inputs omitted; alignment is supplied rather than independently verified')
