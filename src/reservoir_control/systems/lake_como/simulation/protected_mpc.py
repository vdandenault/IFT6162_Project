"""Bounded causal MPC using the protected outlet model and unchanged costs.

The NumPy planning kernel is checked against ProtectedOutletModel, not a new
surrogate penalty. Invalid model-domain predictions are infeasible schedules;
if none is feasible, maximum release is an explicit, uncertified fallback.
"""
from dataclasses import dataclass

import numpy as np

from reservoir_control.systems.lake_como.splits import split_years
from reservoir_control.systems.lake_como.simulation.low_water import ProtectedOutletModel
from reservoir_control.systems.lake_como.simulation.model import PhysicalWeights, WaterState, physical_cost


class CausalForecastBank:
    def __init__(self, inflows, targets, *, before_year):
        years=tuple(y for y in split_years('train') if y<before_year)
        if not years or any(y not in inflows or y not in targets for y in years):
            raise ValueError('Complete prior training years required')
        self.years=years
        self.flows=np.array([inflows[y][:365] for y in years],float)
        self.targets=np.array([targets[y][:365] for y in years],float)
        if self.flows.shape!=(len(years),365) or self.targets.shape!=self.flows.shape:
            raise ValueError('Complete model-year profiles required')
        if not all(np.isfinite(a).all() and (a>=0).all() for a in (self.flows,self.targets)):
            raise ValueError('Finite nonnegative training records required')
        self.seasonal=self.flows.mean(axis=0)
        self.target_mean=self.targets.mean(axis=0)
        self.flat=self.flows.ravel()
        for a in (self.flows,self.targets,self.seasonal,self.target_mean,self.flat):a.setflags(write=False)

    @property
    def initial_last_inflow(self):
        return float(self.seasonal[-1])  # prior climatology, not a claimed measurement

    def forecast(self, kind, day, last_inflow, *, horizon=7, count=5, seed=901):
        if kind not in ('persistence','seasonal','conditional') or not 1<=day<=365:
            raise ValueError('Known forecast and valid model day required')
        if not 1<=horizon<=7 or count<1 or not np.isfinite(last_inflow) or last_inflow<0:
            raise ValueError('Valid causal forecast dimensions required')
        days=(day-1+np.arange(horizon))%365
        metadata=dict(training_years=list(self.years),source_start_indices=[],kind=kind)
        if kind=='persistence':paths=np.full((1,horizon),last_inflow)
        elif kind=='seasonal':paths=self.seasonal[days][None,:].copy()
        else:
            # Always choose seven-day blocks, even for one-day MPC. This keeps
            # the first-day sample identical in the horizon ablation.
            starts=np.arange(1,len(self.flat)-7+1)
            distance=np.abs(starts%365-(day-1));distance=np.minimum(distance,365-distance)
            starts=starts[distance<=15]
            order=np.argsort(abs(self.flat[starts-1]-last_inflow),kind='stable')
            pool=starts[order[:min(128,len(order))]]
            chosen=np.random.default_rng(seed).choice(pool,count,replace=True)
            paths=self.flat[chosen[:,None]+np.arange(horizon)]
            metadata.update(source_start_indices=chosen.tolist(),conditioning_pool_size=len(pool),
                            sampling='Uniform prior historical blocks, not calibrated probabilities')
        return paths.copy(),self.target_mean[days].copy(),metadata


def _exposure_area(start, end, threshold, *, above=True):
    sign=1 if above else -1
    a=sign*(start-threshold);b=sign*(end-threshold)
    positive=np.maximum(a,b)
    crossing=np.divide(.5*positive**2,abs(b-a),out=np.zeros_like(a),where=abs(b-a)>0)
    return np.where((a<=0)&(b<=0),0.,np.where((a>=0)&(b>=0),.5*(a+b),crossing))


def evaluate_schedules(model,state,schedules,inflows,targets,days,weights=PhysicalWeights()):
    """Vectorized equivalent of protected hourly rollout; no domain extrapolation.

    Shapes: schedules [candidate, day], inflows [scenario, day], targets [day].
    The stored state for a failed path remains its last valid state; its cost is
    infinity. No invalid state is clamped and continued as a physical trajectory.
    """
    if type(model) is not ProtectedOutletModel or state.river_storage:
        raise ValueError('The protected, zero-evaporation/bypass model is required')
    if weights.terminal_reserve_per_million_m3:
        raise ValueError('This study holds terminal cost at zero')
    schedules=np.asarray(schedules,float);inflows=np.asarray(inflows,float)
    targets=np.asarray(targets,float);days=np.asarray(days)
    if schedules.ndim!=2 or inflows.ndim!=2 or schedules.shape[1]!=inflows.shape[1]:
        raise ValueError('Aligned candidate/scenario day matrices required')
    horizon=schedules.shape[1]
    if targets.shape!=(horizon,) or days.shape!=(horizon,) or horizon<1:
        raise ValueError('Aligned daily target/calendar required')
    if (not all(np.isfinite(a).all() for a in (schedules,inflows,targets,days))
            or (schedules<model.low).any() or (schedules>model.high).any()
            or (inflows<0).any() or (targets<0).any() or (days<1).any() or (days>365).any()
            or (days!=days.astype(int)).any()):
        raise ValueError('Finite bounded inputs required')
    lake=model.lake;lake.check_domain(lake.storage_to_level(state.storage))
    shape=(len(schedules),len(inflows))
    storage=np.full(shape,state.storage);alive=np.ones(shape,bool);cost=np.zeros(shape)
    daily_levels=np.empty((*shape,horizon));daily_cost=np.empty((*shape,horizon))
    maximum=np.full(shape,lake.storage_to_level(state.storage))
    minimum=maximum.copy()
    for index,day in enumerate(days.astype(int)):
        request=schedules[:,index,None];inflow=inflows[None,:,index]
        mef=float(lake.min_env_flow[day-1]);start_cost=cost.copy()
        for _ in range(24):
            before=storage/lake.surface+lake.spec.storage_reference_m
            capacity=lake.C_r*(before-lake.alpha)**lake.beta
            operating_min=np.where(before<=lake.min_level,0.,np.where(before<=lake.max_level,mef,capacity))
            ceiling=np.maximum(0.,lake.surface*(before-lake.rule.floor_m)/(lake.rule.reserve_hours*3600))
            release=np.minimum(np.minimum(capacity,np.maximum(operating_min,request)),ceiling)
            following=storage+3600*(inflow-release)
            level=following/lake.surface+lake.spec.storage_reference_m
            valid=(level>=lake.spec.lower_evaluation_m-1e-12)&(level<=lake.spec.upper_evaluation_m+1e-12)
            active=alive&valid
            diversion=np.minimum(targets[index],np.maximum(release-mef,0.))
            stage=(weights.shortage_per_million_m3*(targets[index]-diversion)*3600/1e6
                +weights.flood_per_m_hour*_exposure_area(before,level,model.flood)
                +weights.low_water_per_m_hour*_exposure_area(before,level,model.low_water,above=False))
            cost+=np.where(active,stage,0.)
            maximum=np.where(active,np.maximum(maximum,level),maximum)
            minimum=np.where(active,np.minimum(minimum,level),minimum)
            storage=np.where(active,following,storage)
            alive=active
        daily_levels[:,:,index]=storage/lake.surface+lake.spec.storage_reference_m
        daily_cost[:,:,index]=cost-start_cost
    return dict(scenario_costs=np.where(alive,cost,np.inf),valid=alive,
                daily_levels=daily_levels,daily_costs=daily_cost,
                maximum_levels=maximum,minimum_levels=minimum)


@dataclass(frozen=True)
class ProtectedMPCConfig:
    horizon_days: int = 7
    grid_points: int = 9
    scenarios: int = 5
    seed: int = 901

    def __post_init__(self):
        if self.horizon_days not in (1,7) or self.grid_points<2 or self.scenarios<1:
            raise ValueError('One/seven-day horizon and positive search dimensions required')


class ProtectedMPC:
    """Finite schedule MPC; executes only the first daily bounded request."""
    def __init__(self,params,bank,*,forecast='persistence',config=ProtectedMPCConfig(),weights=PhysicalWeights()):
        self.model=ProtectedOutletModel(params);self.bank=bank;self.forecast=forecast
        self.config=config;self.weights=weights;self.diagnostics=[]
        self.reset()

    def reset(self):
        self.previous_hour=None;self.previous_plan=None

    def __call__(self,observation,info):
        obs=np.asarray(observation,float);hour=int(info['elapsed_hours'])
        if obs.shape!=(8,) or not np.isfinite(obs).all() or hour%24 or info['decision_hours']!=24:
            raise ValueError('Daily eight-feature operational observation required')
        horizon=min(self.config.horizon_days,int(info['remaining_hours'])//24)
        if horizon<1:raise ValueError('Do not plan from a completed record')
        day=hour//24%365+1;days=(day-1+np.arange(horizon))%365+1
        flows,targets,metadata=self.bank.forecast(self.forecast,day,float(obs[4]),horizon=horizon,
            count=self.config.scenarios,seed=self.config.seed+hour)
        state=WaterState(self.model.lake.level_to_storage(obs[0]),float(obs[5]),float(obs[6]))
        grid=np.linspace(self.model.low,self.model.high,self.config.grid_points).astype(np.float32)
        service=np.clip(targets+self.model.lake.min_env_flow[days-1],self.model.low,self.model.high).astype(np.float32)
        feedback=np.float32(np.clip(obs[3]+20+150*(obs[0]-.35),self.model.low,self.model.high))
        candidates=[np.full(horizon,v,np.float32) for v in grid]+[service.copy(),np.full(horizon,feedback,np.float32)]
        for first in [*grid,feedback]:
            sequence=service.copy();sequence[0]=first;candidates.append(sequence)
        warm=self.previous_hour==hour-24 and self.previous_plan is not None
        if warm:
            shifted=self.previous_plan[1:]
            if len(shifted):candidates.append(np.pad(shifted,(0,max(0,horizon-len(shifted))),mode='edge')[:horizon])
        # Stable deduplication retains the declared candidate order for ties.
        schedules=np.array(list({tuple(row):row for row in candidates}.values()),np.float32)
        result=evaluate_schedules(self.model,state,schedules,flows,targets,days,self.weights)
        costs=result['scenario_costs'].mean(axis=1)
        valid=np.isfinite(costs)
        recovery=not valid.any()
        if recovery:
            plan=np.full(horizon,np.float32(self.model.high));index=None;parity_error=None
        else:
            index=int(np.argmin(costs));plan=schedules[index].copy()
            # Check the selected first-day predictions against the actual scalar
            # protected model on every decision and scenario, including cost.
            errors=[]
            for scenario,path in enumerate(flows):
                final,rows=self.model.rollout(state,np.repeat(plan[0],24),np.repeat(path[0],24),
                    np.repeat(targets[0],24),np.repeat(days[0],24))
                error=abs(self.model.lake.storage_to_level(final.storage)-result['daily_levels'][index,scenario,0])
                cost_error=abs(physical_cost(rows,self.weights)-result['daily_costs'][index,scenario,0])
                if error>1e-8 or cost_error>1e-6:raise AssertionError('Planning kernel differs from protected simulator')
                errors.append(error)
            parity_error=max(errors)
        self.previous_hour=hour;self.previous_plan=plan.copy()
        self.diagnostics.append(dict(elapsed_hours=hour,horizon_days=horizon,forecast_kind=self.forecast,
            forecast_inflows=flows.tolist(),forecast_target=targets.tolist(),forecast_metadata=metadata,
            request_m3_s=float(plan[0]),candidate_count=len(schedules),feasible_candidates=int(valid.sum()),
            domain_recovery=recovery,expected_cost=None if recovery else float(costs[index]),
            predicted_maximum_level_m=None if recovery else float(result['maximum_levels'][index].max()),
            predicted_minimum_level_m=None if recovery else float(result['minimum_levels'][index].min()),
            shifted_previous_plan=warm,first_day_parity_error_m=parity_error))
        return np.array([plan[0]],np.float32)
