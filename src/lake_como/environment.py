"""Load upstream configs without changing the process working directory."""
import os

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import gymnasium as gym
import pandas as pd
import yaml
import gym4real  # registers environments
from lake_como.paths import UPSTREAM_ROOT, DAM_CONFIG


def settings(split='train'):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    directory = DAM_CONFIG
    world = yaml.safe_load((directory / f'world_{split}.yaml').read_text())
    lake = yaml.safe_load((directory / 'lake.yaml').read_text())
    lake['min_env_flow'] = pd.read_csv(UPSTREAM_ROOT / lake['min_env_flow'])['MEF'].to_numpy()
    lake['evaporation_rates'] = None
    params = {k: world[k] for k in (
        'period', 'integration', 'sim_horizon', 'doy', 'flood_level',
        'starving_level', 'observations', 'action', 'seed', 'random_init',
        'smooth_daily_deficit_coeff', 'exponential_average_coeff')}
    params.update(lake_params=lake, reward_coeff=world['reward'])
    for signal in ('demand', 'inflow'):
        df = pd.read_csv(UPSTREAM_ROOT / world[signal], index_col='year')
        params[signal] = {int(y): row.dropna().tolist() for y, row in df.iterrows()}
    return params


def make_env(split='train', year=None):
    params = settings(split)
    if year is not None:
        for signal in ('demand', 'inflow'):
            params[signal] = {year: params[signal][year]}
        params['random_init'] = False
    return gym.make('gym4real/dam-v0', settings=params)
