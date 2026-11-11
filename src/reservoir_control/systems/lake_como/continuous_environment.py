"""Continuous model-year episodes around the pinned DamEnv equations.

This opt-in environment changes year-boundary state continuity, not the lake
physics, reward, observations, or original make_env factory. Model years remain
365 steps pending a separate source-calendar audit.
"""
from copy import deepcopy
from numbers import Integral

import gymnasium as gym
import numpy as np
from gym4real.envs.dam.env import DamEnv

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.inflow_observation import PastInflowObservation
from reservoir_control.systems.lake_como.splits import split_years


class ContinuousDamEnv(DamEnv):
    """One finite episode over an explicitly ordered, consecutive year sequence.

    Each year uses the original 365 daily decisions. Storage, level, last release,
    and demand EMA carry into the next year. The last observation contains actual
    final level; a year boundary is not an episode end. Only exhaustion of the
    supplied sequence truncates the episode. Explicit reset starts the same sequence.

    No new inflow, gate, evaporation or demand model is introduced. This adapter
    deliberately isolates continuity from those future changes.
    """

    def __init__(self, params, years):
        self.years = tuple(years)
        if (not self.years
                or any(not isinstance(y, Integral) or isinstance(y, bool) for y in self.years)
                or any(b != a + 1 for a, b in zip(self.years, self.years[1:]))):
            raise ValueError('Choose a nonempty sequence of consecutive integer years')
        params = deepcopy(params)
        if params['lake_params']['evaporation']:
            raise ValueError('This continuity experiment requires the pinned no-evaporation physics')
        if params['period'] != 365 or params['sim_horizon'] != 365 or params['doy'] != 1:
            raise ValueError('This version requires the pinned 365-day January-start setup')
        for signal in ('inflow', 'demand'):
            selected = {}
            for year in self.years:
                if year not in params[signal]:
                    raise ValueError(f'{signal} does not contain requested year {year}')
                values = np.asarray(params[signal][year], dtype=float)
                if (values.ndim != 1 or len(values) < 365
                        or not np.isfinite(values[:365]).all() or (values[:365] < 0).any()):
                    raise ValueError(f'{signal}/{year} requires 365 finite nonnegative samples')
                selected[year] = values.tolist()
            params[signal] = selected
        params['random_init'] = False
        self.sequence_position = 0
        self.elapsed_steps = None
        self.finished = False
        super().__init__(params)

    def _get_observation(self):
        observation = super()._get_observation()
        if 'level' in self.obs_keys:
            # Parent indexes modulo 365 and otherwise returns the initial level
            # at a terminal/year-boundary observation.
            observation[self.obs_keys.index('level')] = self.level[-1]
        return observation

    def reset(self, *, seed=None, options=None):
        if options and set(options) - {'rewind_profiles'}:
            raise ValueError('Only rewind_profiles is supported; reset always rewinds this sequence')
        gym.Env.reset(self, seed=seed)
        self.sequence_position = 0
        self.elapsed_steps = 0
        self.finished = False
        observation, info = super().reset(seed=seed, options={'rewind_profiles': True})
        info.update(next_year=int(self.years[0]), next_day=1, elapsed_steps=0,
                    episode_steps=365 * len(self.years), calendar='365-day model years')
        return observation, info

    def _advance_year(self):
        # Rebase the parent's per-year history without changing physical state or
        # EMA. Parent reward/physics keep their original local-day indexing.
        self.sequence_position += 1
        self.curr_year_data = self.years[self.sequence_position]
        self.demand = self.demand_data[self.curr_year_data]
        self.inflow = self.inflow_data[self.curr_year_data]
        self.storage = [self.storage[-1]]
        self.level = [self.level[-1]]
        self.release = [self.release[-1]]
        self.actions = []
        self.doy = []
        self.current_step = 0

    def step(self, action):
        if self.elapsed_steps is None or self.finished:
            raise RuntimeError('Reset before stepping an unstarted or completed sequence')
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Provide a finite release request inside nominal action bounds')
        year = int(self.curr_year_data)
        day = self.current_step + 1
        before = float(self.storage[-1])
        observation, reward, terminated, annual_end, info = super().step(action)
        self.elapsed_steps += 1
        self.finished = bool(annual_end and self.sequence_position == len(self.years) - 1)
        if annual_end and not self.finished:
            self._advance_year()
            observation = self._get_observation()
        info.update(transition_year=year, transition_day=day, year_boundary=bool(annual_end),
                    elapsed_steps=self.elapsed_steps, storage_before=before,
                    next_year=None if self.finished else int(self.curr_year_data),
                    next_day=None if self.finished else self.current_step + 1,
                    balance_error_m3=float(info['storage'] - before
                        - 86400 * (self.inflow_data[year][day - 1] - info['release'])))
        # Retain termination semantics from the underlying physical task. The
        # finite supplied record, rather than each calendar year, is the time limit.
        return observation, reward, terminated, self.finished, info


def make_continuous_env(split='train', years=None, *, past_inflow=True):
    """Build an opt-in sequence confined to one chronological split.

    Defaults to two training years. The fifth observation retains the exact
    existing completed-day telemetry semantics; it is not a new feature layout.
    Loading this factory does not train or evaluate a controller.
    """
    membership = split_years(split)
    years = tuple(membership[:2] if years is None else years)
    if any(year not in membership for year in years):
        raise ValueError('A continuous episode must stay within its selected split')
    env = ContinuousDamEnv(settings(split), years)
    if past_inflow:
        training = settings('train')
        fallback = float(np.mean([values[0] for values in training['inflow'].values()]))
        env = PastInflowObservation(env, fallback)
    return env
