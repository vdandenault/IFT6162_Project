"""Transparent hourly balance, command dynamics and conservative river routing.

Optional parameter values are experiment assumptions, not calibrated Olginate data.
The active-storage datum remains the pinned model's datum, not total lake volume.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from gym4real.envs.dam.lake import Lake
from reservoir_control.systems.lake_como.hourly_telemetry import linear_exposure


@dataclass(frozen=True)
class Operations:
    ramp_m3_s_per_hour: float | None = None
    river_residence_hours: float = 0.
    downstream_reference_m3_s: float | None = None

    def __post_init__(self):
        if self.river_residence_hours < 0 or not np.isfinite(self.river_residence_hours):
            raise ValueError('Finite nonnegative residence time required')
        for v in (self.ramp_m3_s_per_hour, self.downstream_reference_m3_s):
            if v is not None and (not np.isfinite(v) or v <= 0):
                raise ValueError('Optional ramp/reference must be finite and positive')


@dataclass(frozen=True)
class PhysicalWeights:
    shortage_per_million_m3: float = 1.
    flood_per_m_hour: float = 1.
    low_water_per_m_hour: float = 1.
    downstream_per_million_m3: float = 1.
    terminal_reserve_per_million_m3: float = 0.
    reserve_target_level_m: float = .35

    def __post_init__(self):
        values=asdict(self)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Finite objective values required')
        if any(v<0 for k,v in values.items() if k!='reserve_target_level_m'):
            raise ValueError('Objective weights must be nonnegative')


@dataclass(frozen=True)
class WaterState:
    storage: float
    command: float = 0.
    river_storage: float = 0.


class OperationalModel:
    def __init__(self, params, operations=Operations()):
        self.lake=Lake(params['lake_params']);self.operations=operations
        if self.lake.evaporation:raise ValueError('Supply evaporation explicitly, not via the legacy flag')
        self.high=float(params['action']['high']);self.low=float(params['action']['low'])
        self.flood=float(params['flood_level']);self.low_water=float(params['starving_level'])

    def advance(self,state,request,inflow,demand,day,evaporation_mm_day=0.):
        if not np.isfinite([state.storage,state.command,state.river_storage,request,inflow,demand,evaporation_mm_day]).all():
            raise ValueError('Finite state and inputs required')
        if not self.low<=request<=self.high or min(inflow,demand,evaporation_mm_day,state.river_storage)<0 or not 1<=day<=365:
            raise ValueError('Invalid physical input')
        ramp=self.operations.ramp_m3_s_per_hour
        command=request if ramp is None else state.command+float(np.clip(request-state.command,-ramp,ramp))
        release=float(self.lake.actual_release(command,state.storage,day))
        evap=evaporation_mm_day*self.lake.surface/1000/86400
        storage=state.storage+3600*(inflow-release-evap)
        tau=self.operations.river_residence_hours*3600
        if tau:
            fraction=-math.expm1(-3600/tau)
            routed=state.river_storage*(1-fraction)+release*tau*fraction
            volume=state.river_storage+release*3600-routed
        else:
            if state.river_storage:raise ValueError('Bypass routing requires zero river storage')
            routed=0.;volume=release*3600
        river_in=max(0.,volume/3600)
        mef=float(self.lake.min_env_flow[day-1])
        diversion=min(demand,max(river_in-mef,0.))
        river_out=river_in-diversion  # MEF remains in the river; it is not consumed.
        before=self.lake.storage_to_level(state.storage);level=self.lake.storage_to_level(storage)
        flood,fa=linear_exposure(before,level,self.flood)
        low,la=linear_exposure(before,level,self.low_water,above=False)
        threshold=self.operations.downstream_reference_m3_s
        row=dict(storage_before_m3=state.storage,storage_m3=storage,level_before_m=before,level_m=level,
            request_m3_s=request,command_m3_s=command,release_m3_s=release,inflow_m3_s=inflow,
            evaporation_m3=evap*3600,demand_m3_s=demand,environmental_flow_m3_s=mef,
            river_storage_before_m3=state.river_storage,river_storage_m3=routed,
            routed_flow_m3_s=river_in,diversion_m3=diversion*3600,river_out_m3=river_out*3600,
            unmet_demand_m3=(demand-diversion)*3600,
            environmental_shortage_m3=max(mef-river_out,0.)*3600,
            downstream_excess_m3=0. if threshold is None else max(river_out-threshold,0.)*3600,
            flood_hours=flood,low_water_hours=low,flood_excess_m_hours=fa,low_water_excess_m_hours=la,
            lake_balance_error_m3=storage-state.storage-3600*(inflow-release-evap),
            river_balance_error_m3=state.river_storage+release*3600-routed-(diversion+river_out)*3600,
            below_active_storage_datum=bool(storage<0))
        if not np.isfinite(list(v for v in row.values())).all():raise RuntimeError('Nonfinite model trajectory')
        return WaterState(storage,command,routed),row

    def rollout(self,state,requests,inflows,demands,days):
        """Explicit forecast inputs only; this model holds no live future profiles."""
        if not(len(requests)==len(inflows)==len(demands)==len(days)):raise ValueError('Aligned forecast arrays required')
        rows=[]
        for request,inflow,demand,day in zip(requests,inflows,demands,days):
            state,row=self.advance(state,float(request),float(inflow),float(demand),int(day));rows.append(row)
        return state,rows


def physical_cost(rows,weights):
    return sum(weights.shortage_per_million_m3*r['unmet_demand_m3']/1e6
        +weights.flood_per_m_hour*r['flood_excess_m_hours']
        +weights.low_water_per_m_hour*r['low_water_excess_m_hours']
        +weights.downstream_per_million_m3*r['downstream_excess_m3']/1e6 for r in rows)
