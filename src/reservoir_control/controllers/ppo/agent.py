from dataclasses import dataclass,asdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOConfig:
    observation_dim:int=15
    hidden:int=64
    rollout:int=2048
    epochs:int=10
    batch_size:int=64
    learning_rate:float=3e-4
    gamma:float=.995
    gae_lambda:float=.95
    clip_ratio:float=.2
    value_coefficient:float=.5
    entropy_coefficient:float=0.
    max_gradient_norm:float=.5
    target_kl:float=.03
    reward_scale:float=.01
    initial_log_std:float=-.5

    def __post_init__(self):
        if min(self.observation_dim,self.hidden,self.rollout,self.epochs,self.batch_size)<1:
            raise ValueError('Declared observation size and positive training dimensions required')
        if not 0<self.gamma<=1 or not 0<=self.gae_lambda<=1 or self.reward_scale<=0:
            raise ValueError('Valid discount, GAE and fixed reward units required')


def gae(rewards,values,next_values,terminated,truncated,gamma,lam):
    arrays=[np.asarray(a) for a in [rewards,values,next_values,terminated,truncated]]
    if not arrays or any(a.shape!=arrays[0].shape for a in arrays) or arrays[0].ndim!=1:
        raise ValueError('Aligned scalar transition arrays required')
    r,v,nv,term,trunc=arrays
    if not all(np.isfinite(a).all() for a in [r,v,nv]):raise ValueError('Finite GAE inputs required')
    advantage=np.zeros(len(r),np.float64);following=0.
    for i in range(len(r)-1,-1,-1):
        delta=float(r[i])+gamma*(not bool(term[i]))*float(nv[i])-float(v[i])
        # Bootstrap truncations from their real stored observation, but never
        # carry GAE across a reset. True finite-task ends do not bootstrap.
        following=delta+gamma*lam*(not(bool(term[i]) or bool(trunc[i])))*following
        advantage[i]=following
    return advantage.astype(np.float32),(advantage+v).astype(np.float32)


def log_probability(distribution,latent):
    jacobian=2*(math.log(2)-latent-F.softplus(-2*latent))
    return (distribution.log_prob(latent)-jacobian).sum(dim=-1)


class Networks(nn.Module):
    def __init__(self, config, action_low, action_high, observation_scales):
        super().__init__()
        scales = np.ones(config.observation_dim) if observation_scales is None else np.asarray(observation_scales)
        if scales.shape != (config.observation_dim,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('One finite positive scale per observation is required')
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.register_buffer('action_low',torch.tensor(action_low,dtype=torch.float32))
        self.register_buffer('action_high',torch.tensor(action_high,dtype=torch.float32))
        def mlp():
            net=nn.Sequential(nn.Linear(config.observation_dim,config.hidden),nn.Tanh(),
                nn.Linear(config.hidden,config.hidden),nn.Tanh(),nn.Linear(config.hidden,1))
            for layer in net:
                if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
            return net
        self.actor=mlp();self.critic=mlp()
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.log_std=nn.Parameter(torch.full((1,),config.initial_log_std))

    def distribution(self,observations):
        mean=self.actor(observations/self.scales)
        return torch.distributions.Normal(mean,self.log_std.clamp(-5,1).exp().expand_as(mean))

    def value(self,observations):return self.critic(observations/self.scales).squeeze(-1)

    def physical(self,latent):
        return self.action_low+(torch.tanh(latent)+1)*.5*(self.action_high-self.action_low)


class PPOAgent:
    def __init__(self,config=PPOConfig(),*,action_low,action_high,seed=123,observation_scales=None):
        torch.manual_seed(seed);self.config=config;self.seed=seed
        self.model=Networks(config,action_low,action_high,observation_scales)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config.learning_rate,eps=1e-5)
        self.shuffle=np.random.default_rng(seed+10000);self.updates=0;self.steps=0

    @torch.no_grad()
    def act(self,observation,*,deterministic=False):
        x=torch.as_tensor(observation,dtype=torch.float32).reshape(1,-1)
        dist=self.model.distribution(x);z=dist.mean if deterministic else dist.sample()
        logp=log_probability(dist,z);value=self.model.value(x)
        action=self.model.physical(z).reshape(-1)
        if not torch.isfinite(action).all():raise ValueError('Nonfinite PPO action')
        return action.numpy().copy(),z.reshape(-1).numpy().copy(),float(logp.item()),float(value.item())

    @torch.no_grad()
    def value(self,observation):
        return float(self.model.value(torch.as_tensor(observation,dtype=torch.float32).reshape(1,-1)).item())

    def update(self,rollout):
        c=self.config
        advantage,returns=gae(rollout['reward'],rollout['value'],rollout['next_value'],
            rollout['terminated'],rollout['truncated'],c.gamma,c.gae_lambda)
        advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        x=torch.tensor(np.asarray(rollout['observation']),dtype=torch.float32)
        z=torch.tensor(np.asarray(rollout['latent']),dtype=torch.float32)
        oldlog=torch.tensor(rollout['log_probability'],dtype=torch.float32)
        adv=torch.tensor(advantage);target=torch.tensor(returns);logs=[];stopped=False
        for epoch in range(c.epochs):
            indices=self.shuffle.permutation(len(x))
            for start in range(0,len(x),c.batch_size):
                batch=indices[start:start+c.batch_size]
                dist=self.model.distribution(x[batch]);logp=log_probability(dist,z[batch]);difference=logp-oldlog[batch]
                ratio=difference.exp();kl=((ratio-1)-difference).mean()
                if not torch.isfinite(ratio).all():raise ValueError('Nonfinite PPO probability ratio')
                if kl.item()>c.target_kl:stopped=True;break
                policy=-torch.minimum(ratio*adv[batch],ratio.clamp(1-c.clip_ratio,1+c.clip_ratio)*adv[batch]).mean()
                value=F.mse_loss(self.model.value(x[batch]),target[batch])
                # Entropy coefficient is frozen at zero. Base Gaussian entropy
                # is logged only; it is not called exact squashed entropy.
                entropy=dist.entropy().sum(dim=-1).mean()
                loss=policy+c.value_coefficient*value-c.entropy_coefficient*entropy
                if not torch.isfinite(loss):raise ValueError('Nonfinite PPO loss')
                self.optimizer.zero_grad();loss.backward()
                actor_params=[*self.model.actor.parameters(),self.model.log_std]
                actor_norm=float(torch.sqrt(sum((p.grad.square().sum() for p in actor_params if p.grad is not None))).item())
                critic_norm=float(torch.sqrt(sum((p.grad.square().sum() for p in self.model.critic.parameters() if p.grad is not None))).item())
                norm=nn.utils.clip_grad_norm_(self.model.parameters(),c.max_gradient_norm)
                if not np.isfinite([actor_norm,critic_norm,float(norm)]).all():raise ValueError('Nonfinite PPO gradient')
                self.optimizer.step();self.updates+=1
                if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('Nonfinite PPO parameters')
                logs.append(dict(policy_loss=float(policy.item()),value_loss=float(value.item()),
                    approx_kl=float(kl.item()),clip_fraction=float(((ratio-1).abs()>c.clip_ratio).float().mean().item()),
                    actor_gradient_norm=actor_norm,critic_gradient_norm=critic_norm,gradient_norm=float(norm),
                    gaussian_entropy=float(entropy.item())))
            if stopped:break
        result={key:float(np.mean([row[key] for row in logs])) for key in logs[0]} if logs else {}
        result.update(minibatch_updates=len(logs),kl_early_stop=stopped,return_min=float(returns.min()),return_max=float(returns.max()),
                      log_std=float(self.model.log_std.item()))
        return result

    def save(self,path):
        torch.save(dict(config=asdict(self.config),seed=self.seed,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),updates=self.updates,steps=self.steps),path)

    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=state['model'];agent=cls(PPOConfig(**state['config']),action_low=model['action_low'].numpy(),
            action_high=model['action_high'].numpy(),seed=state['seed'],
            observation_scales=model['scales'].numpy())
        agent.model.load_state_dict(model,strict=True);agent.model.eval();agent.updates=state['updates'];agent.steps=state['steps']
        return agent
