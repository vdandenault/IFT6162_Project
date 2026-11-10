"""Fixed-model flood replay; stops at the first out-of-envelope trial.

Refinement keeps daily requests and hourly reserve-ceiling updates fixed. Only
evaluation of the state-dependent hydraulic release is refined inside the hour.
No rating curve is evaluated beyond its declared numerical envelope.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from lake_como.hourly_telemetry import linear_exposure
from .low_water import ProtectedOutletModel
from .model import WaterState


@dataclass(frozen=True)
class ReplayAnchor:
    hour: int
    state: WaterState
    demand_ema: float


def daily_request(policy, model, state, demand_ema):
    if policy == 'constant_100':
        value = 100.
    elif policy == 'causal_feedback':
        # Match the existing float32 observations/action interface exactly.
        level = float(np.float32(model.lake.storage_to_level(state.storage)))
        average = float(np.float32(demand_ema))
        value = np.clip(average+20+150*(level-.35),model.low,model.high)
    elif policy == 'maximum_nominal':
        value = model.high
    else:
        raise ValueError('Unknown fixed diagnostic policy')
    return float(np.float32(value))


def flood_replay(params, years, *, policy='constant_100', anchor=None,
                 end_hour=None, substeps_per_hour=1):
    years=tuple(years)
    if not years or any(y not in params['inflow'] for y in years):
        raise ValueError('Only supplied training years allowed')
    if any(b!=a+1 for a,b in zip(years,years[1:])):
        raise ValueError('Consecutive model years required')
    if not isinstance(substeps_per_hour,int) or substeps_per_hour<1:
        raise ValueError('Positive integer refinement required')
    model=ProtectedOutletModel(params)
    anchor=anchor or ReplayAnchor(0,WaterState(model.lake.level_to_storage(.35)),0.)
    end_hour=len(years)*365*24 if end_hour is None else end_hour
    if anchor.hour%24 or not anchor.hour<end_hour<=len(years)*365*24:
        raise ValueError('Daily-boundary anchor and valid replay interval required')
    if anchor.state.river_storage or not math.isfinite(anchor.demand_ema):
        raise ValueError('Bypass routing and finite demand history required')
    state=anchor.state;ema=anchor.demand_ema;rows=[];snapshots={};failure=None
    dt=3600/substeps_per_hour
    for hour in range(anchor.hour,end_hour):
        year=years[hour//(365*24)];day=hour//24%365+1
        if hour%24==0:
            snapshots[hour]=ReplayAnchor(hour,state,ema)
            request=daily_request(policy,model,state,ema)
        inflow=float(params['inflow'][year][day-1]);target=float(params['demand'][year][day-1])
        if not math.isfinite(inflow) or inflow<0 or not math.isfinite(target) or target<0:
            raise ValueError('Finite nonnegative forcings required')
        hourly_ceiling=model.lake.rule.release_cap(model.lake.storage_to_level(state.storage),model.lake.surface)
        for part in range(substeps_per_hour):
            before=model.lake.storage_to_level(state.storage)
            model.lake.check_domain(before)
            capacity=float(model.lake.max_release(state.storage,day))
            minimum=float(model.lake.min_release(state.storage,day))
            hydraulic=float(model.lake.unprotected_release(request,state.storage,day))
            release=min(hydraulic,hourly_ceiling)
            following=state.storage+dt*(inflow-release)
            level=model.lake.storage_to_level(following)
            low=model.lake.spec.lower_evaluation_m;high=model.lake.spec.upper_evaluation_m
            accepted=low-1e-12<=level<=high+1e-12
            flood,fa=linear_exposure(before,level,model.flood)
            low_hours,la=linear_exposure(before,level,model.low_water,above=False)
            mef=float(model.lake.min_env_flow[day-1]);diversion=min(target,max(release-mef,0.))
            rows.append(dict(hour=hour,substep=part,elapsed_end_hour=hour+(part+1)/substeps_per_hour,
                year=year,model_day=day,step_seconds=dt,level_before_m=before,level_m=level,
                request_m3_s=request,release_m3_s=release,capacity_m3_s=capacity,
                operating_minimum_m3_s=minimum,low_water_cap_m3_s=hourly_ceiling,
                inflow_m3_s=inflow,target_m3_s=target,accepted=accepted,
                forced_full_open=bool(before>model.lake.max_level),
                low_water_intervened=bool(hydraulic>release+1e-10),
                flood_hours=flood*dt/3600,low_water_hours=low_hours*dt/3600,
                flood_excess_m_hours=fa*dt/3600,low_water_excess_m_hours=la*dt/3600,
                unmet_target_m3=(target-diversion)*dt,
                environmental_shortage_m3=max(mef-(release-diversion),0.)*dt,
                balance_error_m3=following-state.storage-dt*(inflow-release)))
            if not accepted:
                failure=dict(hour=hour,substep=part,elapsed_end_hour=hour+(part+1)/substeps_per_hour,
                    year=year,model_day=day,level_before_m=before,trial_level_m=level,
                    boundary='upper' if level>high else 'lower',
                    capacity_m3_s=capacity,release_m3_s=release,request_m3_s=request,
                    inflow_m3_s=inflow,forced_full_open=before>model.lake.max_level)
                break
            state=WaterState(following,request,0.)
        if failure is not None:break
        if (hour+1)%24==0:
            coefficient=params['exponential_average_coeff']
            ema=coefficient*ema+(1-coefficient)*target
    return pd.DataFrame(rows),snapshots,failure


def capacity_volume_bound(params, years, anchor, end_hour):
    """Necessary-feasibility bound, not an optimized release schedule.

    While h is in the declared envelope, monotone full-open capacity <= Q(upper).
    Even allowing that maximum at all levels, cumulative input must fit available
    headroom plus this optimistic discharge volume. Positive excess certifies an
    upper-envelope exit for any trajectory staying in the envelope over the prefix.
    """
    model=ProtectedOutletModel(params)
    start=anchor.hour
    if start%24 or not start<end_hour<=len(years)*365*24:
        raise ValueError('Valid daily anchor and end required')
    level=model.lake.storage_to_level(anchor.state.storage)
    model.lake.check_domain(level)
    upper=model.lake.spec.upper_evaluation_m
    maximum=model.lake.max_release(model.lake.level_to_storage(upper),1)
    headroom=model.lake.surface*(upper-level)
    excess=-headroom;largest=excess;first=None
    for hour in range(start,end_hour):
        year=years[hour//(365*24)];day=hour//24%365
        excess+=3600*(float(params['inflow'][year][day])-maximum)
        largest=max(largest,excess)
        if excess>1e-6 and first is None:first=hour+1
    return dict(start_hour=start,end_hour=end_hour,initial_level_m=level,
        envelope_capacity_upper_m3_s=float(maximum),headroom_m3=float(headroom),
        maximum_volume_excess_m3=float(largest),certifies_upper_exit=first is not None,
        first_certified_end_hour=first,
        scope='Fixed forcing, constant area, zero evaporation, no extra outlet; any trajectory staying in declared envelope.')
