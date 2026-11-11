"""Training-only seasonal historical-block forecasts and labeled stress inputs."""
import numpy as np
from reservoir_control.systems.lake_como.splits import split_years


class HistoricalScenarios:
    def __init__(self,inflows,demands):
        years=split_years('train')
        if set(inflows)!=set(years) or set(demands)!=set(years):raise ValueError('Exactly the chronological training years required')
        self.years=tuple(years)
        self.flows=np.array([inflows[y][:365] for y in years],float)
        self.demands=np.array([demands[y][:365] for y in years],float)
        if self.flows.shape!=(42,365) or self.demands.shape!=(42,365) or not all(np.isfinite(a).all() and (a>=0).all() for a in [self.flows,self.demands]):
            raise ValueError('Complete finite nonnegative training profiles required')
        self.climatology=self.flows.mean(axis=0);self.demand_climatology=self.demands.mean(axis=0)
        self.flat=self.flows.ravel()
        for a in [self.flows,self.demands,self.climatology,self.demand_climatology,self.flat]:a.setflags(write=False)

    def paths(self,day,last_inflow,*,horizon=7,count=8,seed=0,season_window=15):
        if not 1<=day<=365 or not 1<=horizon<=30 or count<1 or not np.isfinite(last_inflow) or last_inflow<0:raise ValueError('Invalid causal forecast request')
        starts=np.arange(1,len(self.flat)-horizon+1)
        distance=np.abs(starts%365-(day-1));distance=np.minimum(distance,365-distance)
        starts=starts[distance<=season_window]
        # Condition on only the observed previous flow, then retain the entire
        # contiguous historical future block rather than sampling days independently.
        order=np.argsort(abs(self.flat[starts-1]-last_inflow),kind='stable')
        pool=starts[order[:min(len(order),128)]]
        chosen=np.random.default_rng(seed).choice(pool,count,replace=True)
        paths=self.flat[chosen[:,None]+np.arange(horizon)]
        return paths.copy(),dict(training_years=list(self.years),source_start_indices=chosen.tolist(),
            conditioning_pool_size=len(pool),
            scenario_type='Empirical conditional seasonal historical blocks; no probability/safety guarantee')

    def point(self,day,last_inflow,horizon=7):
        return np.full(horizon,last_inflow),self.demand_climatology[(day-1+np.arange(horizon))%365].copy()


def within_day_shape(daily,kind='flat'):
    daily=np.asarray(daily,float)
    if daily.ndim!=1 or not np.isfinite(daily).all() or (daily<0).any():raise ValueError('Finite daily flows required')
    if kind=='flat':weights=np.ones(24)
    elif kind=='front_loaded':weights=np.r_[np.full(6,4.),np.zeros(18)]
    elif kind=='late_loaded':weights=np.r_[np.zeros(18),np.full(6,4.)]
    else:raise ValueError('Unknown synthetic intraday shape')
    return daily[:,None]*weights[None,:]


def stress_profiles(params,years,kind):
    """Independent sensitivity scenarios, not fitted climate probabilities."""
    from copy import deepcopy
    result=deepcopy(params)
    if kind not in ['historical','drought_half','wet_150pct','front_loaded','late_loaded']:
        raise ValueError('Unknown stress scenario')
    factor={'drought_half':.5,'wet_150pct':1.5}.get(kind,1.)
    hourly={}
    for year in years:
        daily=np.asarray(result['inflow'][year][:365],float)*factor
        result['inflow'][year]=daily.tolist()
        hourly[year]=within_day_shape(daily,kind if kind in ['front_loaded','late_loaded'] else 'flat')
    return result,hourly
