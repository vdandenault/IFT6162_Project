"""Orthogonal target-forecast and schedule-search diagnostics at fixed costs.

Uses only the existing target EMA observation. Forecast coefficients are fitted
on prior training records, with zero-initialization bias accounted for explicitly.
Expanded search retains the restricted solution as its incumbent.
"""
from dataclasses import dataclass

import numpy as np

from .model import WaterState,physical_cost
from .protected_mpc import ProtectedMPC,evaluate_schedules


class TargetForecaster:
    def __init__(self,bank,ema_coefficient=.8):
        if not np.isfinite(ema_coefficient) or not 0<=ema_coefficient<1:
            raise ValueError('EMA coefficient in [0,1) required')
        self.bank=bank;self.coefficient=float(ema_coefficient)
        self.expected=np.zeros(731)
        for n in range(730):
            self.expected[n+1]=self.coefficient*self.expected[n]+(1-self.coefficient)*bank.target_mean[n%365]
        xx=np.zeros(7);xy=np.zeros(7);counts=np.zeros(7,int)
        # Each prior two-year window reproduces the evaluation's reset semantics.
        # Features at n use observations through n-1, never the target at n.
        for i in range(len(bank.years)-1):
            if bank.years[i+1]!=bank.years[i]+1:continue
            targets=bank.targets[i:i+2].ravel();ema=np.zeros(731)
            for n,value in enumerate(targets):
                ema[n+1]=self.coefficient*ema[n]+(1-self.coefficient)*value
            indices=np.arange(1,730)
            mass=1-self.coefficient**indices
            residual=(ema[indices]-self.expected[indices])/mass
            for lead in range(7):
                keep=indices+lead<730;n=indices[keep];x=residual[keep]
                y=targets[n+lead]-bank.target_mean[(n+lead)%365]
                xx[lead]+=np.dot(x,x);xy[lead]+=np.dot(x,y);counts[lead]+=len(n)
        if not counts.any():raise ValueError('At least one prior consecutive two-year window required')
        self.slopes=np.divide(xy,xx,out=np.zeros(7),where=xx>0)
        self.counts=counts
        for a in (self.expected,self.slopes,self.counts):a.setflags(write=False)

    def predict(self,day,observed_ema,elapsed_days,*,horizon=7):
        if (not isinstance(elapsed_days,int) or not 0<=elapsed_days<=730 or
                day!=elapsed_days%365+1 or not 1<=horizon<=7 or
                not np.isfinite(observed_ema) or observed_ema<0):
            raise ValueError('Aligned daily clock, finite observed EMA and valid horizon required')
        seasonal=self.bank.target_mean[(day-1+np.arange(horizon))%365]
        # At reset there is no target history; use the same seasonal prior.
        residual=0. if elapsed_days==0 else (
            observed_ema-self.expected[elapsed_days])/(1-self.coefficient**elapsed_days)
        return np.maximum(0.,seasonal+self.slopes[:horizon]*residual),float(residual)


class _TargetBank:
    def __init__(self,bank,forecaster,mode):
        self.bank=bank;self.forecaster=forecaster;self.mode=mode;self.context=None

    def forecast(self,*args,**kwargs):
        flows,seasonal,metadata=self.bank.forecast(*args,**kwargs)
        if self.context is None:raise RuntimeError('Target observation context required')
        day,ema,elapsed=self.context
        if self.mode=='ema_residual':
            targets,residual=self.forecaster.predict(day,ema,elapsed,horizon=kwargs['horizon'])
        else:targets=seasonal;residual=0.
        metadata=dict(metadata,target_forecast_mode=self.mode,target_history_residual=residual)
        return flows,targets,metadata


@dataclass(frozen=True)
class SearchExpansion:
    grid_points: int = 17
    sweeps: int = 2

    def __post_init__(self):
        if not isinstance(self.grid_points,int) or not isinstance(self.sweeps,int) or self.grid_points<2 or self.sweeps<1:
            raise ValueError('Positive bounded search dimensions required')


class DiagnosticMPC(ProtectedMPC):
    def __init__(self,params,bank,*,target_mode='seasonal',search='restricted',
                 ema_coefficient=.8,expansion=SearchExpansion(),**kwargs):
        if target_mode not in ('seasonal','ema_residual') or search not in ('restricted','expanded'):
            raise ValueError('Known target forecast and search mode required')
        self.target_forecaster=TargetForecaster(bank,ema_coefficient)
        adapted=_TargetBank(bank,self.target_forecaster,target_mode)
        super().__init__(params,adapted,**kwargs)
        self.target_mode=target_mode;self.search=search;self.expansion=expansion

    def __call__(self,observation,info):
        hour=int(info['elapsed_hours']);obs=np.asarray(observation,float)
        if obs.shape!=(8,) or not np.isfinite(obs).all() or hour%24:
            raise ValueError('Daily eight-feature observation required')
        self.bank.context=(hour//24%365+1,float(obs[3]),hour//24)
        action=super().__call__(observation,info)
        diagnostic=self.diagnostics[-1]
        diagnostic.update(target_mode=self.target_mode,search_mode=self.search,
            restricted_seed_cost=diagnostic['expected_cost'],expanded_candidate_evaluations=0,
            restricted_candidate_count=diagnostic['candidate_count'],
            restricted_feasible_candidates=diagnostic['feasible_candidates'],
            search_expected_gain=0. if diagnostic['expected_cost'] is not None else None)
        if self.search=='restricted':return action
        plan=self.previous_plan.copy();horizon=len(plan)
        flows=np.asarray(diagnostic['forecast_inflows']);targets=np.asarray(diagnostic['forecast_target'])
        days=(hour//24+np.arange(horizon))%365+1
        state=WaterState(self.model.lake.level_to_storage(obs[0]),float(obs[5]),float(obs[6]))
        grid=np.linspace(self.model.low,self.model.high,self.expansion.grid_points).astype(np.float32)
        service=np.clip(targets+self.model.lake.min_env_flow[days-1],self.model.low,self.model.high).astype(np.float32)
        best=float('inf') if diagnostic['expected_cost'] is None else diagnostic['expected_cost']
        evaluations=0
        for _ in range(self.expansion.sweeps):
            for coordinate in range(horizon):
                # Incumbent first preserves ties. Grid contains the old nine
                # points plus finer values; service/current values remain exact.
                values=list(dict.fromkeys([plan[coordinate],*grid,service[coordinate]]))
                proposals=np.repeat(plan[None,:],len(values),axis=0)
                proposals[:,coordinate]=values
                scores=evaluate_schedules(self.model,state,proposals,flows,targets,days,self.weights)
                costs=scores['scenario_costs'].mean(axis=1);index=int(np.argmin(costs))
                evaluations+=len(proposals)
                if costs[index]<best:
                    best=float(costs[index]);plan=proposals[index].copy()
        final=evaluate_schedules(self.model,state,plan[None,:],flows,targets,days,self.weights)
        actual=float(final['scenario_costs'].mean())
        if np.isfinite(best) and actual>best+1e-8:raise AssertionError('Search lost its retained incumbent')
        if diagnostic['restricted_seed_cost'] is not None and actual>diagnostic['restricted_seed_cost']+1e-8:
            raise AssertionError('Expanded search increased expected cost')
        recovery=not np.isfinite(actual)
        parity=[]
        if not recovery:
            for scenario,path in enumerate(flows):
                final_state,rows=self.model.rollout(state,np.repeat(plan[0],24),np.repeat(path[0],24),
                    np.repeat(targets[0],24),np.repeat(days[0],24))
                error=abs(self.model.lake.storage_to_level(final_state.storage)-final['daily_levels'][0,scenario,0])
                if error>1e-8 or abs(physical_cost(rows,self.weights)-final['daily_costs'][0,scenario,0])>1e-6:
                    raise AssertionError('Expanded plan differs from scalar physics/cost')
                parity.append(error)
        else:plan=np.full(horizon,np.float32(self.model.high))
        self.previous_plan=plan.copy()
        diagnostic.update(request_m3_s=float(plan[0]),expected_cost=None if recovery else actual,
            domain_recovery=recovery,expanded_candidate_evaluations=evaluations,
            search_expected_gain=None if diagnostic['restricted_seed_cost'] is None or recovery else diagnostic['restricted_seed_cost']-actual,
            predicted_maximum_level_m=None if recovery else float(final['maximum_levels'].max()),
            predicted_minimum_level_m=None if recovery else float(final['minimum_levels'].min()),
            first_day_parity_error_m=max(parity) if parity else None)
        return np.array([plan[0]],np.float32)
