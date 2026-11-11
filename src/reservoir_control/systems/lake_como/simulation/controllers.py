"""Explicit-reset daily MPC/residual adapters and operational scenario search.

These are new continuous/operational experiments, never replacements of frozen
annual controller code. Scenario search is a finite schedule/grid method, not a
reproduction of the published stochastic-MPC solver or a global optimum.
"""
import math
import numpy as np
from scipy.optimize import minimize
from reservoir_control.systems.lake_como.mpc import MPCConfig,PlanningModel,SeasonalForecast
from reservoir_control.systems.lake_como.simulation.model import WaterState,physical_cost


def observed_state(model,observation):
    obs=np.asarray(observation,float)
    if obs.shape!=(8,) or not np.isfinite(obs).all():raise ValueError('Expected eight operational observation features')
    return WaterState(model.lake.level_to_storage(obs[0]),float(obs[5]),float(obs[6]))


class ContinuousMPC:
    """Persistence MPC with horizon/warm start crossing model-year boundaries."""
    def __init__(self,training_params,config=None):
        self.config=config or MPCConfig.load();self.seasonal=SeasonalForecast.fit(training_params['inflow'],training_params['demand'])
        self.model=PlanningModel(training_params,self.config);self.resets=0;self.diagnostics=[]
        self.previous_hour=None;self.previous_plan=None

    def reset(self):
        self.previous_hour=None;self.previous_plan=None;self.resets+=1

    def __call__(self,observation,info):
        obs=np.asarray(observation,float);hour=int(info['elapsed_hours'])
        if obs.shape!=(8,) or info['decision_hours']!=24 or hour%24:raise ValueError('Daily operational observation required')
        if info['next_day'] is None:raise ValueError('Do not plan from a completed episode')
        horizon=self.config.horizon_days
        if info['terminal_mode']=='finite_horizon':horizon=min(horizon,int(info['remaining_hours'])//24)
        day=hour//24%365;days=(day+np.arange(horizon))%365+1
        inflows=np.full(horizon,float(obs[4]));demands=self.seasonal.demand[days-1]
        storage=self.model.lake.level_to_storage(obs[0]);span=self.model.high-self.model.low
        demand=np.clip((demands+self.model.lake.min_env_flow[days-1]-self.model.low)/span,0,1)
        candidates=[demand,np.zeros(horizon),np.ones(horizon)]
        if self.previous_hour==hour-24 and self.previous_plan is not None:
            shifted=self.previous_plan[1:]
            if len(shifted):candidates.append(np.pad(shifted,(0,max(0,horizon-len(shifted))),mode='edge')[:horizon])
        objective=lambda x:self.model.cost(x,storage,inflows,demands,days)
        values=[objective(x) for x in candidates];initial=candidates[int(np.argmin(values))]
        result=minimize(objective,initial,method='SLSQP',bounds=[(0,1)]*horizon,
            options=dict(maxiter=self.config.max_iterations,ftol=self.config.ftol,eps=self.config.finite_difference_step))
        bounded=np.isfinite(result.x).all() and (result.x>=-1e-8).all() and (result.x<=1+1e-8).all()
        candidate=np.clip(result.x,0,1) if bounded else initial
        retain=bounded and (result.success or result.status==9) and objective(candidate)<=min(values)+1e-9
        plan=candidate if retain else initial
        self.previous_hour=hour;self.previous_plan=plan.copy()
        self.diagnostics.append(dict(elapsed_hours=hour,horizon=horizon,days=days.tolist(),
            used_shifted_plan=len(candidates)==4,solver_success=bool(result.success),solver_status=int(result.status),
            retained_iteration_limit=bool(retain and not result.success)))
        return np.array([self.model.low+plan[0]*span],np.float32)


class ContinuousResidual:
    """Transfer diagnostic for a frozen 6-feature proposal-aware checkpoint.

    Ignores the new command/river/time-to-go features. Daily only; no performance
    or information-matched retraining claim. Reset is explicit, never inferred
    from January 1. Call reset once per actual episode.
    """
    def __init__(self,agent,base):self.agent=agent;self.base=base
    def reset(self):self.base.reset()
    def __call__(self,observation,info):
        import torch
        proposal=float(self.base(observation,info)[0]);six=np.r_[observation[:5],proposal].astype(np.float32)
        learned=float(self.agent.physical_action(torch.from_numpy(self.agent.act(six,deterministic=True)))[0])
        request=np.clip(proposal+learned,0,self.agent.high) if self.agent.mode=='residual' else learned
        return np.array([request],np.float32)


def forecast_hours(bank,observation,info,*,horizon_hours,count,seed):
    hour=int(info['elapsed_hours']);day=hour//24%365+1;offset=hour%24
    needed=math.ceil((offset+horizon_hours)/24)
    paths,meta=bank.paths(day,float(observation[4]),horizon=needed,count=count,seed=seed+hour)
    inflows=np.repeat(paths,24,axis=1)[:,offset:offset+horizon_hours]
    days=(hour+np.arange(horizon_hours))//24%365+1
    demands=bank.demand_climatology[days-1]
    return inflows,demands,days,meta


class ScenarioPlanner:
    def __init__(self,model,bank,weights,*,horizon_days=7,scenarios=5,grid_points=9,seed=901):
        if not 1<=horizon_days<=28 or scenarios<1 or grid_points<2:raise ValueError('Invalid bounded planner dimensions')
        self.model,self.bank,self.weights=model,bank,weights
        self.horizon_days,self.scenarios,self.grid_points,self.seed=horizon_days,scenarios,grid_points,seed
        self.diagnostics=[]
    def reset(self):pass
    def __call__(self,observation,info):
        hours=self.horizon_days*24
        if info['terminal_mode']=='finite_horizon':hours=min(hours,int(info['remaining_hours']))
        flows,demands,days,meta=forecast_hours(self.bank,observation,info,horizon_hours=hours,count=self.scenarios,seed=self.seed)
        # Candidate commands are constant over each decision interval.
        service=np.clip(demands+self.model.lake.min_env_flow[days-1],self.model.low,self.model.high)
        interval=info['decision_hours']
        service=np.repeat([service[i:i+interval].mean() for i in range(0,hours,interval)],interval)[:hours]
        schedules=[np.full(hours,v,dtype=np.float32) for v in np.linspace(self.model.low,self.model.high,self.grid_points).astype(np.float32)]+[service.astype(np.float32)]
        state=observed_state(self.model,observation);costs=[]
        for requests in schedules:
            scenario_costs=[]
            for path in flows:
                final,rows=self.model.rollout(state,requests,path,demands,days)
                cost=physical_cost(rows,self.weights)
                if info['terminal_mode']=='finite_horizon' and hours==info['remaining_hours']:
                    target=self.model.lake.level_to_storage(self.weights.reserve_target_level_m)
                    cost+=self.weights.terminal_reserve_per_million_m3*max(target-final.storage,0)/1e6
                scenario_costs.append(cost)
            costs.append(np.mean(scenario_costs))
        index=int(np.argmin(costs))
        self.diagnostics.append(dict(elapsed_hours=info['elapsed_hours'],expected_physical_cost=float(costs[index]),
            candidate_schedules=len(schedules),scenario_count=self.scenarios,forecast_metadata=meta))
        return np.array([schedules[index][0]],np.float32)


class ScenarioConstraintFilter:
    """Finite-scenario/grid feasibility and explicit least-violation recovery.

    No action is certified for unseen inflows; absence of a grid witness is not
    a proof that continuous actions cannot satisfy the constraints.
    """
    def __init__(self,model,bank,weights,*,scenarios=5,grid_points=17,seed=901):
        if scenarios<1 or grid_points<2:raise ValueError('Invalid filter size')
        self.model,self.bank,self.weights=model,bank,weights;self.scenarios,self.grid_points,self.seed=scenarios,grid_points,seed
        self.diagnostics=[]
    def __call__(self,proposal,observation,info):
        raw=float(np.asarray(proposal).item())
        if not np.isfinite(raw):raise ValueError('Finite proposal required')
        hours=info['decision_hours'];flows,demands,days,_=forecast_hours(self.bank,observation,info,horizon_hours=hours,count=self.scenarios,seed=self.seed)
        state=observed_state(self.model,observation);raw=float(np.float32(np.clip(raw,self.model.low,self.model.high)))
        grid=np.unique(np.r_[np.linspace(self.model.low,self.model.high,self.grid_points).astype(np.float32).astype(float),raw]);results=[]
        for request in grid:
            peak_violation=0.;end_violation=0.;costs=[]
            for flow in flows:
                final,rows=self.model.rollout(state,np.full(hours,request),flow,demands,days)
                levels=[rows[0]['level_before_m']]+[r['level_m'] for r in rows]
                flood=max(max(levels)-self.model.flood,0)/.05
                low=max(self.model.low_water-min(levels),0)/.05
                ref=self.model.operations.downstream_reference_m3_s
                river=0 if ref is None else max(max(r['river_out_m3']/3600 for r in rows)-ref,0)/ref
                peak_violation=max(peak_violation,flood,low,river)
                level=self.model.lake.storage_to_level(final.storage)
                end_violation=max(end_violation,max(level-self.model.flood,self.model.low_water-level,0)/.05)
                costs.append(physical_cost(rows,self.weights))
            results.append((peak_violation,end_violation,float(np.mean(costs)),abs(request-raw),request))
        feasible=[r for r in results if r[0]<=1e-8]
        chosen=min(feasible,key=lambda r:(r[3],r[2],r[4])) if feasible else min(results)
        self.diagnostics.append(dict(elapsed_hours=info['elapsed_hours'],finite_grid_feasible=bool(feasible),
            recovery=not bool(feasible),worst_normalized_violation=chosen[0],proposal=raw,request=chosen[4],
            guarantee='Finite sampled scenarios only; no unconditional safety claim'))
        return np.array([chosen[4]],np.float32)
