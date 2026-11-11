"""Bounded learned MPC corrections, with a matched proposal-aware full-action control.

DAWN-inspired base-data warmup and critic LayerNorm; not a paper reproduction.
Existing frozen SAC/MPC implementations are not modified.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import SACConfig
from reservoir_control.systems.lake_como.mpc import MPCConfig,SeasonalForecast
from reservoir_control.systems.lake_como.mpc_causal import CausalMPC,fit_horizon_weights
from reservoir_control.systems.lake_como.mpc_retention import RetainingMPC


def make_base(params,seasonal=None):
    seasonal=seasonal or SeasonalForecast.fit(params['inflow'],params['demand'])
    config=MPCConfig.load()
    policy=CausalMPC(params,seasonal,config,'persistence',
                     fit_horizon_weights(params['inflow'],seasonal.inflow,7))
    policy.controller=RetainingMPC(params,policy.forecast,config,retain_iteration_limit=True)
    return policy


def combine_request(base,correction,high=491.61):
    if not np.isfinite([base,correction,high]).all() or high<=0:
        raise ValueError('Finite base/correction and positive high required')
    return np.float32(np.clip(float(base)+float(correction),0.,high))


class MPCProposalEnv(gym.Wrapper):
    """Append causal MPC proposal (m³/s). Residual action is a delta in m³/s.

    The terminal sixth feature is zero and is never bootstrapped. Planner calls
    happen only before actual decisions, not on terminal observations.
    """
    def __init__(self,env,base_factory,*,mode='residual',bound=50.):
        super().__init__(env)
        if env.observation_space.shape!=(5,) or mode not in ('residual','full') or bound<=0:
            raise ValueError('Five-feature base observation, known mode, positive bound required')
        self.base_factory,self.mode,self.bound=base_factory,mode,float(bound)
        self.high=float(env.action_space.high[0]);self.mpc=None;self.base_request=None
        self.planner_calls=0;self.planner_seconds=0.;self.last_solver=None
        self.observation_space=gym.spaces.Box(
            np.r_[env.observation_space.low,0.].astype(np.float32),
            np.r_[env.observation_space.high,self.high].astype(np.float32),dtype=np.float32)
        self.action_space=(gym.spaces.Box(-bound,bound,(1,),np.float32) if mode=='residual'
                           else gym.spaces.Box(0.,self.high,(1,),np.float32))

    def _observation(self,obs,info):
        import time
        start=time.perf_counter()
        self.base_request=float(self.mpc(obs[:4],info)[0])
        self.planner_seconds+=time.perf_counter()-start;self.planner_calls+=1
        self.last_solver=dict(self.mpc.diagnostics[-1])
        return np.r_[obs,self.base_request].astype(np.float32)

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.mpc=self.base_factory()
        return self._observation(obs,info),info

    def step(self,action):
        if self.base_request is None:
            raise RuntimeError('Reset before stepping')
        a=np.asarray(action,dtype=np.float32)
        if not self.action_space.contains(a):
            raise ValueError('Action outside declared physical action space')
        base=self.base_request
        request=(combine_request(base,float(a[0]),self.high) if self.mode=='residual'
                 else np.float32(a[0]))
        obs,reward,terminated,truncated,info=self.env.step(np.array([request],np.float32))
        info=dict(info,base_request_m3_s=base,learned_action_m3_s=float(a[0]),
                  effective_correction_m3_s=float(request)-base,nominal_request_m3_s=float(request),
                  nominal_saturated=bool(self.mode=='residual' and abs(float(request)-(base+float(a[0])))>1e-4),
                  base_solver=dict(self.last_solver))
        if terminated or truncated:
            following=np.r_[obs,0.].astype(np.float32);self.base_request=None
        else:
            following=self._observation(obs,info)
        return following,reward,terminated,truncated,info


class CombinedCritic(nn.Module):
    """Critic sees combined nominal request, not the unscaled residual coordinate."""
    def __init__(self,config,mode,bound,high):
        super().__init__();self.mode,self.bound,self.high=mode,bound,high
        layers=[];width=config.observation_dim+config.action_dim
        for hidden in config.hidden_sizes:
            layers.extend([nn.Linear(width,hidden),nn.LayerNorm(hidden),nn.ReLU()]);width=hidden
        self.net=nn.Sequential(*layers,nn.Linear(width,1))

    def combined_action(self,observations,actions):
        # Sixth observation is normalized with mean=high/2, scale=high/2.
        return ((observations[:,-1:]+2*self.bound/self.high*actions).clamp(-1,1)
                if self.mode=='residual' else actions)

    def forward(self,observations,actions):
        return self.net(torch.cat([observations,self.combined_action(observations,actions)],dim=-1))


class ProposalSACAgent(SACAgent):
    """Inherited visible SAC updates, LN critics, canonical replay action semantics.

    Replay action is normalized delta for residual mode, normalized full request
    for full mode. Entropy is in those coordinates, before physical saturation.
    """
    def __init__(self,config,*,observation_mean,observation_scale,mode='residual',bound=50.,high=491.61,seed=0):
        if config.observation_dim!=6 or mode not in ('residual','full') or bound<=0:
            raise ValueError('Expected six observations, known mode and positive bound')
        if not (np.isclose(observation_mean[-1],high/2) and np.isclose(observation_scale[-1],high/2)):
            raise ValueError('Proposal normalization must use mean=scale=high/2')
        self.mode,self.bound,self.high=mode,float(bound),float(high)
        low,upper=(-bound,bound) if mode=='residual' else (0.,high)
        super().__init__(config,observation_mean=observation_mean,observation_scale=observation_scale,
                         action_low=[low],action_high=[upper],seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+10000)
            self.q1=CombinedCritic(config,mode,bound,high);self.q2=CombinedCritic(config,mode,bound,high)
        self.target_q1,self.target_q2=deepcopy(self.q1),deepcopy(self.q2)
        self.target_q1.requires_grad_(False);self.target_q2.requires_grad_(False)
        self.critic_optimizer=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=config.learning_rate)
        # Same actor initialization in both modes; residual mean starts at zero.
        with torch.no_grad():
            self.actor.net[-1].weight.zero_();self.actor.net[-1].bias.zero_()
            self.actor.net[-1].bias[config.action_dim:]=-2.

    def save(self,path):
        torch.save(dict(format_version='proposal_sac_v1',mode=self.mode,bound=self.bound,high=self.high,
            config=asdict(self.config),model=self.state_dict(),updates=self.updates,generator=self.generator.get_state(),
            optimizers=[o.state_dict() for o in (self.actor_optimizer,self.critic_optimizer,self.alpha_optimizer)]),path)

    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['format_version']!='proposal_sac_v1':
            raise ValueError('Expected proposal-aware checkpoint')
        model=state['model']
        agent=cls(SACConfig(**state['config']),observation_mean=model['observation_mean'].numpy(),
                  observation_scale=model['observation_scale'].numpy(),mode=state['mode'],bound=state['bound'],high=state['high'])
        agent.load_state_dict(model)
        for opt,saved in zip((agent.actor_optimizer,agent.critic_optimizer,agent.alpha_optimizer),state['optimizers'],strict=True):
            opt.load_state_dict(saved)
        agent.generator.set_state(state['generator']);agent.updates=state['updates']
        return agent
