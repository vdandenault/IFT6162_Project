"""Causal state-aware request transform retaining the original nominal range.

The middle 80% of policy fractions spans the current hydraulic interval. Each
10% tail retains commands outside that interval, which can matter as lake level
changes during the day. The transformed request is held fixed for all 24 hours.
PPO importance ratios are for stored pre-transform actions; the actuator map is
fixed given the observed state and independent of policy parameters.
"""
import json
import math
from pathlib import Path
import numpy as np
from lake_como.environment import settings
from .low_water import ProtectedOutletModel
from .ppo_agent import PPOAgent

CONTRACT=dict(version=1,kind='hydraulic_full_range_warp',lower_fraction=.1,upper_fraction=.9,
    collapsed_interval='identity_nominal',decision='one_constant_daily_request',
    probability='pre_transform_tanh_action_ratios',entropy_coefficient=0.)


class HydraulicTransform:
    def __init__(self):
        training=settings('train')
        self.model=ProtectedOutletModel({key:training[key] for key in ['lake_params','action','flood_level','starving_level']})
        self.low=float(np.float32(self.model.low));self.high=float(np.float32(self.model.high))

    def interval(self,observation):
        obs=np.asarray(observation,np.float32)
        if obs.shape!=(15,) or not np.isfinite(obs).all():raise ValueError('Finite fifteen-feature observation required')
        if abs(float(obs[1])**2+float(obs[2])**2-1)>1e-5:raise ValueError('Valid seasonal sine/cosine required')
        phase=math.atan2(float(obs[1]),float(obs[2]))%(2*math.pi)
        day=int(round(phase*365/(2*math.pi)))%365+1
        storage=self.model.lake.level_to_storage(float(obs[0]))
        lo=float(np.clip(self.model.lake.actual_release(self.low,storage,day),self.low,self.high))
        hi=float(np.clip(self.model.lake.actual_release(self.high,storage,day),self.low,self.high))
        if hi<lo:raise ValueError('Hydraulic endpoints reversed')
        return lo,hi

    def __call__(self,observation,nominal_request):
        request=np.asarray(nominal_request,np.float32)
        if request.shape!=(1,) or not np.isfinite(request).all() or not self.low<=request[0]<=self.high:
            raise ValueError('Bounded scalar nominal request required')
        lo,hi=self.interval(observation)
        if hi-lo<=1e-6:return request.copy()
        fraction=(float(request[0])-self.low)/(self.high-self.low)
        mapped=np.interp(fraction,[0.,.1,.9,1.],[self.low,lo,hi,self.high])
        return np.array([np.clip(mapped,self.low,self.high)],np.float32)


class HydraulicPPOAgent(PPOAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.config.entropy_coefficient!=0:raise ValueError('Warped entropy requires a separately derived objective')
        self.transform=HydraulicTransform()

    def act(self,observation,*,deterministic=False):
        nominal,latent,log_probability,value=super().act(observation,deterministic=deterministic)
        return self.transform(observation,nominal),latent,log_probability,value

    def save(self,path):
        super().save(path)
        Path(path).with_suffix('.action.json').write_text(json.dumps(CONTRACT,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        contract=json.loads(Path(path).with_suffix('.action.json').read_text())
        if contract!=CONTRACT:raise ValueError('Checkpoint action-transform contract mismatch')
        return super().load(path)
