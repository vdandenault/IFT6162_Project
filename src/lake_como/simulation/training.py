"""Explicit operational SAC smoke loop with separate reset and bootstrap masks."""
from dataclasses import replace
import numpy as np
import torch
from lake_como.sac import SACAgent,SACConfig,ReplayBuffer
from .protocol import replay_terminal


def train_operational(env,*,steps,seed=901,warmup=64,batch_size=32):
    if steps<1 or warmup<0 or batch_size<1:raise ValueError('Positive training dimensions required')
    torch.set_num_threads(1)
    config=replace(SACConfig(observation_dim=8),gamma=env.protocol.gamma_per_decision,reward_scale=1.)
    # Declared physical unit scales, not statistics fitted on validation/test data.
    agent=SACAgent(config,observation_mean=np.zeros(8),observation_scale=[1,1,1,100,100,250,1e7,1],
        action_low=env.action_space.low,action_high=env.action_space.high,seed=seed)
    replay=ReplayBuffer(max(1024,steps),8,1,seed=seed);rng=np.random.default_rng(seed)
    observation,reset_info=env.reset(seed=seed);rows=[];episodes=[];episode_return=0.;length=0
    for step in range(steps):
        normalized=rng.uniform(-1,1,1).astype(np.float32) if step<warmup else agent.act(observation)
        request=agent.physical_action(torch.from_numpy(normalized)).numpy()
        following,reward,term,trunc,info=env.step(request)
        mask=replay_terminal(term,trunc)
        replay.add(observation,normalized,reward,following,mask,False)
        diagnostics=None
        if step>=warmup and len(replay)>=batch_size:diagnostics=agent.update(replay.sample(batch_size))
        episode_return+=reward;length+=1
        rows.append(dict(step=step+1,reward=reward,terminated=term,truncated=trunc,replay_end=mask,
            elapsed_hours=info['elapsed_hours'],year_boundary=info['year_boundary'],
            alpha=float(agent.alpha.item()),critic_loss=None if diagnostics is None else diagnostics['critic_loss']))
        observation=following
        if term or trunc:
            episodes.append(dict(length=length,return_=episode_return));episode_return=0.;length=0
            if step+1<steps:observation,reset_info=env.reset()
    return agent,rows,dict(agent_transitions=env.agent_transitions,burn_in_transitions=env.burn_in_transitions,
        physical_hours=(env.agent_transitions+env.burn_in_transitions)*env.protocol.decision_hours,
        updates=agent.updates,episodes=episodes,unfinished_length=length,
        bootstrap_rule='Mask terminated only; reset on terminated OR truncated',
        normalization='Fixed training-unit scales; no fitted evaluation statistics')
