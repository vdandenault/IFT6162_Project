"""Explicit-probability seasonal/history mixtures and weighted, unchanged-cost MPC.

No path is dropped from the all-path domain rule because its probability is small.
"""
import numpy as np

from reservoir_control.systems.lake_como.simulation.history_forecast import HistoryForecastBank, HistoryMPC
from reservoir_control.systems.lake_como.simulation.tail_diagnosis import pool_details
from reservoir_control.systems.lake_como.simulation.precision_mpc import service_requests,select_schedule
from reservoir_control.systems.lake_como.simulation.protected_mpc import evaluate_schedules
from reservoir_control.systems.lake_como.simulation.model import WaterState,physical_cost

CONFIGS={'legacy':dict(alpha=None,tail=None)}
for _alpha in [0.,.05,.1,.2]:
    for _tail in [.01,.05]:
        CONFIGS[f'mix{int(_alpha*100):02d}_tail{int(_tail*100):02d}']=dict(alpha=_alpha,tail=_tail)


def validate_probabilities(probabilities,count):
    p=np.asarray(probabilities,float)
    if p.shape!=(count,) or not np.isfinite(p).all() or (p<=0).any() or not np.isclose(p.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError('One positive finite probability per path, summing to one, required')
    return p


def weighted_crps(values,truth,probabilities):
    x=np.asarray(values,float);p=validate_probabilities(probabilities,len(x))
    return float(np.sum(p*abs(x-truth))-.5*np.sum(p[:,None]*p[None,:]*abs(x[:,None]-x[None,:])))


def mixture_distribution(bank,day,past,alpha):
    if not 0<=alpha<=1:raise ValueError('Mixture mass in [0,1] required')
    details=pool_details(bank,day,past)
    indices=bank.pools[day];starts=bank.starts[indices]
    probabilities=np.full(len(indices),alpha/len(indices))
    probabilities[np.isin(starts,details['starts'])]+=(1-alpha)/len(details['starts'])
    positive=probabilities>0
    paths=details['seasonal'][positive].copy();starts=starts[positive]
    probabilities=probabilities[positive];probabilities/=probabilities.sum()
    order=np.argsort(paths.sum(axis=1),kind='stable')
    return paths[order],probabilities[order],starts[order]


def probability_strata(paths,probabilities,starts,tail,seed):
    """One draw per CDF band, with each band's actual mass as scenario weight.

    Atoms crossing a boundary are split by their CDF interval; duplicated source
    paths may therefore appear in two bands. This preserves each band's mass.
    """
    if tail not in [.01,.05]:raise ValueError('Declared upper-tail band required')
    paths=np.asarray(paths,float);starts=np.asarray(starts,int)
    if paths.ndim!=2 or paths.shape[1]!=7 or starts.shape!=(len(paths),) or not np.isfinite(paths).all() or (paths<0).any():
        raise ValueError('Aligned finite nonnegative seven-day paths and source indices required')
    if (np.diff(paths.sum(axis=1))<0).any():raise ValueError('Volume-sorted distribution required')
    p=validate_probabilities(probabilities,len(paths));cdf=np.cumsum(p);cdf[-1]=1.
    boundaries=np.array([0.,.25,.5,.75,1-tail,1.])
    masses=np.diff(boundaries)
    u=boundaries[:-1]+masses*np.random.default_rng(seed).random(5)
    u=np.minimum(u,np.nextafter(1.,0.))
    indices=np.searchsorted(cdf,u,side='right')
    return paths[indices].copy(),masses,starts[indices].copy()


class WeightedForecastBank(HistoryForecastBank):
    def __init__(self,inflows,targets,*,before_year,variant):
        if variant not in CONFIGS:raise ValueError('Declared weighted forecast variant required')
        super().__init__(inflows,targets,before_year=before_year,method='history_scaled_stratified')
        self.variant=variant
        self._context_key=None
        self._distribution=None

    def forecast(self,kind,day,last_inflow,*,horizon=7,count=5,seed=901):
        if count!=5:raise ValueError('Five declared scenarios required')
        if self.variant=='legacy' or len(self.history)<7 or kind!='conditional':
            paths,targets,meta=super().forecast(kind,day,last_inflow,horizon=horizon,count=count,seed=seed)
            return paths,targets,dict(meta,forecast_variant=self.variant,scenario_probabilities=[.2]*5)
        if not 1<=day<=365 or not 1<=horizon<=7 or count!=5 or self.history[-1]!=last_inflow:
            raise ValueError('Aligned observed history, calendar and five scenarios required')
        config=CONFIGS[self.variant]
        key=(self.variant,day,tuple(self.history))
        if key!=self._context_key:
            self._distribution=mixture_distribution(self,day,self.history,config['alpha'])
            self._context_key=key
        paths,p,starts=self._distribution
        selected,masses,chosen=probability_strata(paths,p,starts,config['tail'],seed)
        calendar=(day-1+np.arange(horizon))%365
        return selected[:,:horizon],self.target_mean[calendar].copy(),dict(
            training_years=list(self.years),source_start_indices=chosen.tolist(),
            observed_history=self.history.tolist(),forecast_variant=self.variant,
            seasonal_mixture_mass=config['alpha'],upper_tail_mass=config['tail'],
            scenario_probabilities=masses.tolist(),cold_history=False,
            probability_model='Prior finite mixture; one conditional draw per declared CDF band. Not calibrated storm probabilities.')


def expected_costs(scenario_costs,probabilities):
    costs=np.asarray(scenario_costs,float);p=validate_probabilities(probabilities,costs.shape[1])
    if np.array_equal(p,np.full(len(p),1/len(p))):
        return costs.mean(axis=1)  # exact parity for legacy and uniform controls
    # Domain-invalid paths stay infeasible even with 1% mass. No truncation,
    # renormalization over surviving paths, or replacement by a finite penalty.
    return np.where(np.isfinite(costs).all(axis=1),costs@p,np.inf)


def schedules_for(controller,obs,hour,targets,previous_hour,previous_plan):
    horizon=len(targets);days=(hour//24+np.arange(horizon))%365+1
    grid=np.linspace(controller.model.low,controller.model.high,controller.config.grid_points).astype(np.float32)
    service=service_requests(controller.model,targets,days,precision_safe=True)
    feedback=np.float32(np.clip(obs[3]+20+150*(obs[0]-.35),controller.model.low,controller.model.high))
    candidates=[np.full(horizon,v,np.float32) for v in grid]+[service.copy(),np.full(horizon,feedback,np.float32)]
    for first in [*grid,feedback]:
        plan=service.copy();plan[0]=first;candidates.append(plan)
    if previous_hour==hour-24 and previous_plan is not None and len(previous_plan[1:]):
        shifted=previous_plan[1:]
        candidates.append(np.pad(shifted,(0,max(0,horizon-len(shifted))),mode='edge')[:horizon])
    return np.array(list({tuple(row):row for row in candidates}.values()),np.float32),days


class WeightedMPC(HistoryMPC):
    def __init__(self,params,bank,*,probability_mode='weighted',**kwargs):
        if probability_mode not in ['weighted','uniform_ablation']:raise ValueError('Declared probability mode required')
        self.probability_mode=probability_mode
        super().__init__(params,bank,**kwargs)

    def __call__(self,observation,info):
        previous_hour=self.previous_hour
        previous_plan=None if self.previous_plan is None else self.previous_plan.copy()
        # Parent retains causal history, generates the unchanged candidate family
        # and validates the scalar kernel. Its returned action is not executed:
        # rescore the very same candidates before returning an action to the env.
        action=super().__call__(observation,info)
        d=self.diagnostics[-1];declared=d['forecast_metadata']['scenario_probabilities']
        p=np.asarray(declared,float) if self.probability_mode=='weighted' else np.full(len(declared),.2)
        validate_probabilities(p,len(d['forecast_inflows']))
        d.update(probability_mode=self.probability_mode,applied_probabilities=p.tolist(),
                 unweighted_request_m3_s=d['request_m3_s'])
        if np.array_equal(p,np.full(len(p),1/len(p))):return action
        obs=np.asarray(observation,float);hour=int(info['elapsed_hours'])
        plans,days=schedules_for(self,obs,hour,np.asarray(d['forecast_target']),previous_hour,previous_plan)
        assert len(plans)==d['candidate_count']
        state=WaterState(self.model.lake.level_to_storage(obs[0]),float(obs[5]),float(obs[6]))
        result=evaluate_schedules(self.model,state,plans,d['forecast_inflows'],d['forecast_target'],days,self.weights)
        costs=expected_costs(result['scenario_costs'],p)
        chosen,ties=select_schedule(costs,plans,conservative=True)
        recovery=chosen is None
        if recovery:plan=np.full(len(days),np.float32(self.model.high));parity=None
        else:
            plan=plans[chosen].copy();errors=[]
            for scenario,path in enumerate(d['forecast_inflows']):
                final,rows=self.model.rollout(state,np.repeat(plan[0],24),np.repeat(path[0],24),
                                              np.repeat(d['forecast_target'][0],24),np.repeat(days[0],24))
                error=abs(self.model.lake.storage_to_level(final.storage)-result['daily_levels'][chosen,scenario,0])
                if error>1e-8 or abs(physical_cost(rows,self.weights)-result['daily_costs'][chosen,scenario,0])>1e-6:
                    raise AssertionError('Weighted selection differs from scalar physical model')
                errors.append(error)
            parity=max(errors)
        first_min,_=select_schedule(costs,plans)
        self.previous_plan=plan.copy()
        d.update(request_m3_s=float(plan[0]),selected_plan=plan.tolist(),domain_recovery=recovery,
            feasible_candidates=int(np.isfinite(costs).sum()),exact_minimum_tie_count=ties,
            tie_changed_selection=bool(not recovery and chosen!=first_min),
            expected_cost=None if recovery else float(costs[chosen]),
            predicted_maximum_level_m=None if recovery else float(result['maximum_levels'][chosen].max()),
            predicted_minimum_level_m=None if recovery else float(result['minimum_levels'][chosen].min()),
            first_day_parity_error_m=parity)
        return np.array([plan[0]],np.float32)
