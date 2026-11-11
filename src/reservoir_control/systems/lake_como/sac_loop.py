"""Explicit single-environment SAC loop; importing this module starts no training."""
import numpy as np
import torch


def train_steps(env, agent, replay, *, steps, warmup_steps, batch_size,
                updates_per_step=1, seed=0, environment_seed=None, on_step=None):
    """A fresh training segment on an ordinary non-autoresetting Gymnasium env.

    Caller must supply a TRAIN-only env (e.g. CorrectTerminalLevel followed by
    PastInflowObservation) and fixed training-only observation statistics. No
    evaluation, split selection, running normalization or filter is hidden here.
    Returns episode original returns and final update diagnostics, not a study.
    """
    if min(steps, batch_size, updates_per_step) < 1 or warmup_steps < 0:
        raise ValueError("Invalid training-loop budget")
    if replay.capacity < batch_size:
        raise ValueError("Replay capacity must accommodate a minibatch")
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    if (not np.array_equal(low, agent.action_low.numpy())
            or not np.array_equal(high, agent.action_high.numpy())
            or env.observation_space.shape != (agent.config.observation_dim,)):
        raise ValueError("Environment spaces must match agent bounds/observation dimensions")
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed if environment_seed is None else environment_seed)
    episode_return, episode_length, episodes, diagnostics = 0.0, 0, [], None
    for step in range(steps):
        action = (rng.uniform(-1, 1, size=agent.config.action_dim).astype(np.float32)
                  if step < warmup_steps else agent.act(observation))
        request = agent.physical_action(torch.from_numpy(action)).numpy()
        following, reward, terminated, truncated, info = env.step(request)
        replay.add(observation, action, reward, following, terminated, truncated)
        episode_return += float(reward)
        episode_length += 1
        previous = observation
        observation = following
        updated = False
        if step >= warmup_steps and len(replay) >= batch_size:
            for _ in range(updates_per_step):
                diagnostics = agent.update(replay.sample(batch_size))
                updated = True
        if on_step is not None:
            on_step(dict(step=step + 1, observation=previous, next_observation=following,
                         normalized_action=action, request=request, reward=float(reward),
                         terminated=terminated, truncated=truncated, info=info,
                         episode_length=episode_length, episode_return=episode_return,
                         diagnostics=diagnostics if updated else None))
        if terminated or truncated:
            episodes.append({"return": episode_return, "length": episode_length})
            observation, _ = env.reset()
            episode_return, episode_length = 0.0, 0
    return dict(steps=steps, updates=agent.updates, episodes=episodes,
                last_update=diagnostics, unfinished_return=episode_return,
                unfinished_length=episode_length)
