"""Daily/subdaily experimental Gym; physical mechanisms and rewards explicit."""
from copy import deepcopy
import math
import gymnasium as gym
import numpy as np
from lake_como.environment import settings
from lake_como.splits import split_years
from .model import OperationalModel,Operations,PhysicalWeights,WaterState,physical_cost


class OperationalEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,params,years,*,decision_hours=24,terminal_mode='finite_horizon',
                 operations=Operations(),objective='physical',weights=PhysicalWeights(),
                 initial_level_range=(.35,.35),hourly_inflow=None,evaporation_mm_day=None):
        super().__init__();self.params=deepcopy(params);self.years=tuple(years)
        if not self.years or any(b!=a+1 for a,b in zip(self.years,self.years[1:])):raise ValueError('Consecutive years required')
        if decision_hours not in [1,3,6,12,24]:raise ValueError('Decision interval must divide a 24-hour day')
        if terminal_mode not in ['finite_horizon','continuing_record']:raise ValueError('Unknown terminal mode')
        if objective not in ['physical','benchmark']:raise ValueError('Unknown objective')
        if terminal_mode=='continuing_record' and weights.terminal_reserve_per_million_m3:
            raise ValueError('A terminal reserve penalty belongs to a finite task, not an artificial record cutoff')
        if objective=='benchmark' and (decision_hours!=24 or operations.river_residence_hours or weights.terminal_reserve_per_million_m3):
            raise ValueError('Benchmark reward requires daily decisions, bypass routing and no new terminal cost')
        bounds=np.asarray(initial_level_range,float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:raise ValueError('Ordered finite initial levels required')
        if bounds[0]<params['lake_params']['min_level'] or bounds[1]>params['lake_params']['max_level']:
            raise ValueError('Initial levels must lie in the model operating range')
        self.decision_hours=decision_hours;self.terminal_mode=terminal_mode;self.objective=objective
        self.weights=weights;self.initial_level_range=tuple(bounds);self.model=OperationalModel(params,operations)
        self.total_hours=len(self.years)*365*24;self.forcing={};self.demands={};self.evaps={}
        hourly_inflow=hourly_inflow or {};evaporation_mm_day=evaporation_mm_day or {}
        for year in self.years:
            inflow=np.asarray(params['inflow'][year][:365],float);demand=np.asarray(params['demand'][year][:365],float)
            hourly=np.asarray(hourly_inflow.get(year,np.repeat(inflow[:,None],24,axis=1)),float).copy()
            evap=np.asarray(evaporation_mm_day.get(year,np.zeros(365)),float).copy()
            if inflow.shape!=(365,) or demand.shape!=(365,) or hourly.shape!=(365,24) or evap.shape!=(365,):raise ValueError('Invalid forcing shapes')
            if not all(np.isfinite(a).all() and (a>=0).all() for a in [inflow,demand,hourly,evap]):raise ValueError('Finite nonnegative forcing required')
            np.testing.assert_allclose(hourly.mean(axis=1),inflow,rtol=1e-10,atol=1e-8,err_msg='Hourly shapes must preserve daily inflow volume')
            self.forcing[year]=hourly;self.demands[year]=demand.copy();self.evaps[year]=evap
        self.fallback=float(np.mean([v[0] for v in settings('train')['inflow'].values()]))
        self.action_space=gym.spaces.Box(self.model.low,self.model.high,(1,),np.float32)
        self.observation_space=gym.spaces.Box(np.array([-np.inf,-1,-1,0,0,0,0,0],np.float32),
            np.array([np.inf,1,1,np.inf,np.inf,self.model.high,np.inf,1],np.float32),dtype=np.float32)
        self.hour=None

    def _observation(self):
        phase=2*np.pi*(self.hour%(365*24))/(365*24)
        remaining=(self.total_hours-self.hour)/self.total_hours if self.terminal_mode=='finite_horizon' else 1.
        return np.array([self.model.lake.storage_to_level(self.state.storage),np.sin(phase),np.cos(phase),
            self.demand_ema,self.last_inflow,self.state.command,self.state.river_storage,remaining],np.float32)

    def _info(self):
        finished=self.hour==self.total_hours
        return dict(storage=self.state.storage,command=self.state.command,river_storage=self.state.river_storage,
            elapsed_hours=self.hour,decision_hours=self.decision_hours,remaining_hours=self.total_hours-self.hour,
            next_year=None if finished else self.years[self.hour//(365*24)],
            next_day=None if finished else self.hour//24%365+1,next_hour=None if finished else self.hour%24,
            terminal_mode=self.terminal_mode)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options:raise ValueError('Configure the initial distribution in the protocol')
        level=float(self.np_random.uniform(*self.initial_level_range))
        self.state=WaterState(self.model.lake.level_to_storage(level));self.hour=0
        self.demand_ema=0.;self.last_inflow=self.fallback
        return self._observation(),self._info()

    def step(self,action):
        if self.hour is None or self.hour>=self.total_hours:raise RuntimeError('Reset before stepping a completed record')
        action=np.asarray(action,np.float32)
        if not self.action_space.contains(action):raise ValueError('Nominal bounded action required')
        request=float(action[0]);rows=[];start_hour=self.hour
        for _ in range(self.decision_hours):
            year=self.years[self.hour//(365*24)];day=self.hour//24%365+1;hour=self.hour%24
            self.state,row=self.model.advance(self.state,request,float(self.forcing[year][day-1,hour]),
                float(self.demands[year][day-1]),day,float(self.evaps[year][day-1]))
            row.update(year=year,model_day=day,hour=hour+1);rows.append(row);self.hour+=1
            if self.hour%24==0:
                coeff=self.params['exponential_average_coeff']
                self.demand_ema=coeff*self.demand_ema+(1-coeff)*row['demand_m3_s']
        self.last_inflow=float(np.mean([r['inflow_m3_s'] for r in rows]))
        if self.objective=='benchmark':
            release=float(np.mean([r['release_m3_s'] for r in rows]));day=rows[-1]['model_day'];demand=rows[-1]['demand_m3_s']
            level=rows[-1]['level_m'];c=self.params['reward_coeff'];mef=rows[-1]['environmental_flow_m3_s']
            seasonal=.5*(3-math.cos((day-1)*2*math.pi/365)) if self.params['smooth_daily_deficit_coeff'] else (2 if 120<day<=243 else 1)
            terms=[c['overflow_coeff']*int(level>self.model.flood),c['starving_coeff']*int(level<self.model.low_water),
                c['daily_deficit_coeff']*max(demand-max(release-mef,0),0)*seasonal,
                c['wasted_water_coeff']*max(release-demand,0),c['clip_action_coeff']*(request-release)**2]
            cost=sum(terms)
        else:cost=physical_cost(rows,self.weights)
        end=self.hour==self.total_hours
        reserve_cost=0.
        if end and self.objective=='physical':
            target=self.model.lake.level_to_storage(self.weights.reserve_target_level_m)
            reserve_cost=self.weights.terminal_reserve_per_million_m3*max(target-self.state.storage,0)/1e6
        info=self._info();info.update(hourly=rows,transition_start_hour=start_hour,year_boundary=self.hour%(365*24)==0,
            physical_cost=physical_cost(rows,self.weights),terminal_reserve_cost=reserve_cost,
            reward_definition=self.objective,relative_storage_outside_model_domain=any(r['below_active_storage_datum'] for r in rows))
        return self._observation(),float(-cost-reserve_cost),bool(end and self.terminal_mode=='finite_horizon'),bool(end and self.terminal_mode=='continuing_record'),info


def make_operational_env(split='train',years=(1946,1947),**kwargs):
    if any(y not in split_years(split) for y in years):raise ValueError('Sequence must stay within one selected split')
    return OperationalEnv(settings(split),years,**kwargs)
