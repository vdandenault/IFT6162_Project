"""Gymnasium application interface, separate from optimization and physics."""
import gymnasium as gym
from reservoir_control.core import EnvironmentSession


class GymnasiumAdapter(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, session: EnvironmentSession, *, action_space, observation_space):
        super().__init__()
        self.session = session
        self.action_space = action_space
        self.observation_space = observation_space
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self.session.reset(seed=seed, options=options)
        self._needs_reset = False
        return result.observation, dict(result.info)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError('Reset before stepping a new or completed episode')
        result = self.session.step(action)
        self._needs_reset = bool(result.terminated or result.truncated)
        return (result.observation, float(result.reward), bool(result.terminated),
                bool(result.truncated), dict(result.info))

    def close(self):
        self._needs_reset = True
        self.session.close()
