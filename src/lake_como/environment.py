"""Project configuration and observation fixes around the unchanged simulator."""
import calendar
import os

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import gymnasium as gym
import pandas as pd
import yaml
import gym4real  # registers environments
from lake_como.paths import UPSTREAM_ROOT, DAM_CONFIG
from lake_como.splits import split_years


def settings(split='train', *, include_leap_years=True, protocol='chronological'):
    """Load chronological profiles by default, or upstream membership for reproduction.

    Leap years retain upstream 365-step episodes; calendar normalization is deferred.
    """
    if protocol not in ('chronological', 'upstream'):
        raise ValueError('protocol must be chronological or upstream')
    valid_splits = ('train', 'validation', 'test') if protocol == 'chronological' else ('train', 'test')
    if split not in valid_splits:
        raise ValueError(f'{protocol} split must be one of {valid_splits}')
    # Validation/test use the same physics/reward, with deterministic year cycling.
    world_split = 'train' if split == 'train' else 'test'
    world = yaml.safe_load((DAM_CONFIG / f'world_{world_split}.yaml').read_text())
    lake = yaml.safe_load((DAM_CONFIG / 'lake.yaml').read_text())
    lake['min_env_flow'] = pd.read_csv(UPSTREAM_ROOT / lake['min_env_flow'])['MEF'].to_numpy()
    lake['evaporation_rates'] = None
    params = {k: world[k] for k in (
        'period', 'integration', 'sim_horizon', 'doy', 'flood_level',
        'starving_level', 'observations', 'action', 'seed', 'random_init',
        'smooth_daily_deficit_coeff', 'exponential_average_coeff')}
    params.update(lake_params=lake, reward_coeff=world['reward'])
    for signal in ('demand', 'inflow'):
        sources = ('train', 'test') if protocol == 'chronological' else (split,)
        frames = [pd.read_csv(UPSTREAM_ROOT / f'gym4real/data/dam/{signal}/water_{signal}_{part}.csv',
                              index_col='year') for part in sources]
        df = pd.concat(frames)
        if not df.index.is_unique:
            raise ValueError(f'Duplicate years in {signal}')
        if protocol == 'chronological':
            years = split_years(split, include_leap_years=include_leap_years)
        else:
            years = [int(y) for y in df.index if include_leap_years or not calendar.isleap(int(y))]
        params[signal] = {y: df.loc[y].dropna().tolist() for y in years}
    return params


class CorrectTerminalLevel(gym.Wrapper):
    """Return actual final level, not the upstream modulo-year lookup of level[0].

    This does not change dynamics, rewards, truncation, or reset initialization.
    """
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and 'level' in self.unwrapped.obs_keys:
            observation = observation.copy()
            observation[self.unwrapped.obs_keys.index('level')] = info['level']
        return observation, reward, terminated, truncated, info


def make_env(split='train', year=None, *, include_leap_years=True,
             protocol='chronological', correct_terminal_level=True):
    params = settings(split, include_leap_years=include_leap_years, protocol=protocol)
    if year is not None:
        if year not in params['inflow']:
            raise ValueError(f'{year} is unavailable in {protocol}/{split}; '
                             'check year membership and include_leap_years.')
        for signal in ('demand', 'inflow'):
            params[signal] = {year: params[signal][year]}
        params['random_init'] = False
    env = gym.make('gym4real/dam-v0', settings=params)
    return CorrectTerminalLevel(env) if correct_terminal_level else env
