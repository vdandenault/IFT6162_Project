import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .coupled import CoupledSession, f_limit_target


class OntarioOttawaEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, model, frame, initial_states, settings, years):
        self.model, self.frame, self.initial_states = model, frame, initial_states
        self.settings, self.years = settings, tuple(years)
        if not self.years or any(y not in initial_states for y in self.years):
            raise ValueError('Available annual initial states required')
        self.action_space = spaces.Box(-2000., 2000., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)
        self.simulation = None

    def observation(self):
        sim = self.simulation
        if sim.session.done:
            decision, time = self._last_information
            time = type(time)(time.index+1, time.year+1, 1, time.duration_seconds)
        else:
            decision, time = sim.information()
        state = sim.session.state
        angle = 2*np.pi*(time.period-1)/48
        return np.asarray([
            state.level_m-74.8, np.sin(angle), np.cos(angle),
            state.previous_release_m3_s/10000, decision.forecast.nts_m3_s[0]/10000,
            decision.forecast.annual_nts_m3_s/10000, decision.forecast.slon_m3_s/10000,
            sim.storage_m3/self.settings['ottawa_capacity_m3'],
            decision.longsault_roughness, decision.pointe_claire_roughness,
            decision.ice/2, (f_limit_target(state.level_m)-22.1),
            (48-sim.elapsed)/48,
            # Expose every stored input available to the MPC rule-curve rollout.
            *(x-74.8 for x in state.mean_level_history_m),
            state.september_level_m-74.8 if np.isfinite(state.september_level_m) else 0.,
            state.september_release_m3_s/10000 if np.isfinite(state.september_release_m3_s) else 0.,
            float(np.isfinite(state.september_level_m)),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        year = int((options or {}).get('year', self.np_random.choice(self.years)))
        if year not in self.years: raise ValueError('Reset year is outside this environment split')
        self.simulation = CoupledSession(self.model, self.frame, self.initial_states[year], self.settings)
        return self.observation(), {'year': year}

    def schedule(self, action):
        action = np.asarray(action, float)
        if action.shape != (1,) or not np.isfinite(action).all() or not -2000 <= action[0] <= 2000:
            raise ValueError('One finite residual request in [-2000, 2000] required')
        decision, time = self.simulation.information()
        neutral = self.simulation.adapter.neutral(self.simulation.session.state, decision, time)
        # PPO controls the first request; the advisory tail follows the rule curve.
        neutral[0] = np.clip(neutral[0]+float(action[0]), 0., 11500.)
        return neutral

    def step(self, action):
        self._last_information = self.simulation.information()
        row = self.simulation.step(self.schedule(action))
        terminated = self.simulation.elapsed == 48
        # The annual task ends, with explicit terminal storage cost; no reset state
        # is substituted for the final real observation.
        observation = self.observation()
        if not np.isfinite(observation).all(): raise RuntimeError('Nonfinite Ontario observation')
        return observation, row['reward'], terminated, False, row
