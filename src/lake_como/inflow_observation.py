"""Append a causal past-inflow estimate without changing DamEnv dynamics/reward."""
import gymnasium as gym
import numpy as np


class PastInflowObservation(gym.Wrapper):
    """Append completed-day inflow reconstructed from storage/release telemetry.

    Reset uses the training-only first-day seasonal forecast, matching causal MPC.
    The estimate returned after step t is available for decision t+1, including
    terminal observations. No current/future inflow profile is accessed.
    """

    def __init__(self, env, first_day_fallback):
        super().__init__(env)
        if env.observation_space.shape != (4,):
            raise ValueError("Expected four-feature base observation")
        if env.unwrapped.lake.evaporation:
            raise ValueError("Past-inflow reconstruction requires disabled evaporation")
        if not np.isfinite(first_day_fallback) or first_day_fallback < 0:
            raise ValueError("Finite nonnegative training fallback required")
        self.first_day_fallback = float(first_day_fallback)
        self.previous_storage = None
        self.observation_space = gym.spaces.Box(
            low=np.r_[env.observation_space.low, 0.0].astype(np.float32),
            high=np.r_[env.observation_space.high, np.inf].astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_storage = float(info["storage"])
        return np.r_[observation, self.first_day_fallback].astype(np.float32), info

    def step(self, action):
        if self.previous_storage is None:
            raise RuntimeError("Reset before stepping")
        observation, reward, terminated, truncated, info = self.env.step(action)
        storage = float(info["storage"])
        estimate = (storage - self.previous_storage) / 86400 + float(info["release"])
        if estimate < -1e-8 or not np.isfinite(estimate):
            raise ValueError("Invalid reconstructed inflow")
        self.previous_storage = storage
        return (
            np.r_[observation, max(0.0, estimate)].astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )
