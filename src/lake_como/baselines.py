import argparse
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from lake_como.environment import make_env, settings
from lake_como.paths import RESULTS


def rollout(policy, year, split='train', seed=123, *, include_leap_years=True, protocol='chronological'):
    env = make_env(split, year, include_leap_years=include_leap_years, protocol=protocol)
    sim = env.unwrapped
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    rows = []
    start = time.perf_counter()
    for day in range(365):
        before = info['storage']
        observed_level = float(obs[0])
        if policy == 'random':
            action = env.action_space.sample()
        elif policy == 'ead':
            action = np.array([info['exponential_average_demand']], dtype=np.float32)
        else:
            rate = {'constant': 100.0, 'mean': float(env.action_space.high.item()) / 2,
                    'max': float(env.action_space.high.item())}[policy]
            action = np.array([rate], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        inflow = sim.inflow[day]
        mef = sim.lake.min_env_flow[day]
        deficit = max(info['demand'] - max(info['release'] - mef, 0), 0)
        rows.append(dict(policy=policy, year=year, day=day + 1,
                         observed_level_before_m=observed_level,
                         level_after_m=info['level'], storage_before_m3=before,
                         storage_after_m3=info['storage'], requested_m3_s=float(action.item()),
                         release_m3_s=info['release'], inflow_m3_s=inflow,
                         demand_m3_s=info['demand'], deficit_m3=deficit * 86400,
                         surplus_m3=max(info['release'] - info['demand'], 0) * 86400,
                         flood=int(info['level'] > sim.flood_level),
                         low_water=int(info['level'] < sim.starving_level), reward=reward,
                         mass_balance_error_m3=info['storage'] - before - 86400 * (inflow - info['release']),
                         **info['pure_reward']))
        if terminated or truncated:
            if day != 364 or terminated or not truncated:
                raise RuntimeError('Unexpected episode boundary')
            break
    elapsed = time.perf_counter() - start
    env.close()
    frame = pd.DataFrame(rows)
    if len(frame) != 365 or not np.isfinite(frame.select_dtypes('number')).all().all():
        raise RuntimeError('Invalid trajectory')
    summary = dict(policy=policy, year=year, seed=seed, steps=len(frame), seconds=elapsed,
                   return_=float(frame.reward.sum()), deficit_m3=float(frame.deficit_m3.sum()),
                   surplus_m3=float(frame.surplus_m3.sum()), flood_days=int(frame.flood.sum()),
                   low_water_days=int(frame.low_water.sum()), max_level_m=float(frame.level_after_m.max()),
                   terminal_storage_m3=float(frame.storage_after_m3.iloc[-1]),
                   adjusted_release_days=int((abs(frame.requested_m3_s - frame.release_m3_s) > 1e-5).sum()),
                   max_mass_balance_error_m3=float(frame.mass_balance_error_m3.abs().max()))
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--years', type=int, nargs='+', help='Default: first eligible year of the selected split')
    parser.add_argument('--policies', nargs='+', choices=['constant', 'random', 'ead', 'mean', 'max'], default=['constant', 'random', 'ead'])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--benchmark', type=int, default=0, help='Extra annual constant-policy rollouts for timing')
    parser.add_argument('--protocol', choices=['chronological', 'upstream'], default='chronological')
    parser.add_argument('--include-leap-years', action=argparse.BooleanOptionalAction, default=True,
                        help='Include leap years (default); --no-include-leap-years excludes them. Episodes remain 365 steps.')
    args = parser.parse_args()
    try:
        available = settings(args.split, include_leap_years=args.include_leap_years,
                             protocol=args.protocol)['inflow']
    except ValueError as error:
        parser.error(str(error))
    years = args.years or [next(iter(available))]
    for year in years:
        if year not in available:
            parser.error(f'{year} is unavailable in {args.protocol}/{args.split}; check split membership and leap-year option')
    frames, summaries = [], []
    for year in years:
        for policy in args.policies:
            frame, summary = rollout(policy, year, args.split, args.seed, include_leap_years=args.include_leap_years, protocol=args.protocol)
            frames.append(frame); summaries.append(summary)
    out = RESULTS
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out / 'trajectories.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    metadata = {'arguments': vars(args), 'years': years, 'eligible_years': list(available),
                'calendar_policy': 'upstream 365 steps; leap years included' if args.include_leap_years else 'non-leap years only',
                'terminal_observation': 'corrected final level; annual reset still restores initial state',
                'platform': platform.platform(),
                'python': platform.python_version(),
                'versions': {p: version(p) for p in ('numpy', 'gymnasium', 'pandas', 'PyYAML', 'pygame')},
                'settings': 'Pinned upstream world/lake YAMLs; see reports/data_audit.json for source hashes',
                'metrics': 'Daily endpoint flood/low-water only; deficit is unweighted physical volume.',
                'timing_scope': 'reset/construction excluded; includes policy, stepping, and row logging'}
    if args.benchmark:
        times = [rollout('constant', years[0], args.split, args.seed, include_leap_years=args.include_leap_years, protocol=args.protocol)[1]['seconds'] for _ in range(args.benchmark)]
        metadata['benchmark'] = {'episodes': len(times), 'median_seconds': float(np.median(times)),
                                 'total_seconds': sum(times)}
    (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    if args.benchmark:
        print('Timing:', metadata['benchmark'])


if __name__ == '__main__':
    main()
