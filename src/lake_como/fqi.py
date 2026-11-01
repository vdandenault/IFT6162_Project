"""Discrete-action fitted Q iteration with an Extra-Trees value regressor."""
import numpy as np


def state_actions(states, actions):
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 5:
        raise ValueError("Five raw observation features required")
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


class FQIController:
    """Build the same past-inflow observation from completed-day telemetry."""

    def __init__(
        self, model, normalized_actions, physical_high, fallback, safety_filter=None
    ):
        self.model = model
        self.normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        self.physical_high = float(physical_high)
        self.fallback = float(fallback)
        self.filter = safety_filter
        self.previous_storage = None
        self.previous_day = None

    def __call__(self, observation, info):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (4,) or not np.isfinite(obs).all():
            raise ValueError("Finite four-feature base observation required")
        day = (
            int(np.rint((np.arctan2(obs[1], obs[2]) % (2 * np.pi)) * 365 / (2 * np.pi)))
            % 365
            + 1
        )
        if day == 1:
            self.previous_storage = None
        elif self.previous_day != day - 1:
            raise ValueError("Consecutive days required")
        storage = float(info["storage"])
        past = (
            self.fallback
            if self.previous_storage is None
            else max(
                0.0, (storage - self.previous_storage) / 86400 + float(info["release"])
            )
        )
        state = np.r_[obs, past].astype(np.float32)
        q = self.model.predict(state_actions(state[None, :], self.normalized_actions))
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite Q prediction")
        # Ascending release grid breaks ties toward lower requested release.
        action = np.array(
            [self.normalized_actions[int(q.argmax())] * self.physical_high],
            dtype=np.float32,
        )
        self.previous_storage = storage
        self.previous_day = day
        return self.filter(action, state, info) if self.filter is not None else action
