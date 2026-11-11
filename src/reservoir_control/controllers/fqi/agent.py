"""Discrete-action fitted Q iteration with an Extra-Trees value regressor."""
import numpy as np


def state_actions(states, actions):
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 1:
        raise ValueError("A matrix of observation features is required")
    return np.column_stack(
        [np.repeat(states, len(actions), axis=0), np.tile(actions, len(states))]
    ).astype(np.float32)


def greedy_values(model, states, actions, batch_size=2048):
    values = []
    for start in range(0, len(states), batch_size):
        features = state_actions(states[start : start + batch_size], actions)
        predicted = model.predict(features).reshape(-1, len(actions))
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite Q prediction")
        values.append(predicted.max(axis=1))
    return np.concatenate(values)


def bellman_targets(rewards, done, next_values, gamma):
    rewards = np.asarray(rewards)
    done = np.asarray(done, dtype=bool)
    next_values = np.asarray(next_values)
    if not (rewards.shape == done.shape == next_values.shape):
        raise ValueError("Transition array mismatch")
    if not np.isfinite(rewards).all() or not np.isfinite(next_values).all():
        raise ValueError("Nonfinite Bellman input")
    return rewards + gamma * np.where(done, 0.0, next_values)


