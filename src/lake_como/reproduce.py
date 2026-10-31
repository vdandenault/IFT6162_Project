"""Evaluate the original DamEnv test split and supplied PPO checkpoint."""
import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lake_como.environment import make_env, settings
from lake_como.paths import PROJECT_ROOT, UPSTREAM_ROOT

POLICIES = ('random', 'mean', 'max', 'ead', 'ppo')
OBJECTIVE = ('overflow_reward', 'starving_reward', 'daily_deficit_reward')
CHECKPOINT = UPSTREAM_ROOT / 'examples/dam/trained_agents/agent_200000.pt'


class CheckpointPolicy:
    """Use the authors' architecture and saved observation scaler, frozen on CPU."""
    def __init__(self, observation_space, action_space, checkpoint):
        import torch
        from gym4real.algorithms.dam.ppo_skrl import Policy
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        self.torch = torch
        torch.manual_seed(123)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.policy = Policy(observation_space, action_space, device='cpu',
                             net_arch=(16, 16), initial_log_std=-0.5)
        self.policy.load_state_dict(state['policy'], strict=True)
        self.policy.eval()
        self.scaler = RunningStandardScaler(size=observation_space, device='cpu')
        self.scaler.load_state_dict(state['state_preprocessor'], strict=True)
        self.scaler.eval()

    def __call__(self, observation):
        with self.torch.no_grad():
            state = self.torch.as_tensor(observation, dtype=self.torch.float32).reshape(1, -1)
            # Matches agent.act(...)[-1]['mean_actions'] in the upstream notebook.
            mean, _, _ = self.policy.compute({'states': self.scaler(state, train=False)}, role='policy')
            return mean.reshape(-1).cpu().numpy()


def evaluate(policy_name, seed=123, checkpoint=CHECKPOINT):
    # Original observations and membership; no project terminal-level correction.
    env = make_env('test', protocol='upstream', correct_terminal_level=False)
    sim = env.unwrapped
    env.action_space.seed(seed)
    weights = sim.reward_coeff
    if policy_name == 'ppo':
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        controller = CheckpointPolicy(env.observation_space, env.action_space, checkpoint)
    records = []
    started = time.perf_counter()
    for episode in range(13):
        obs, info = env.reset(options={'rewind_profiles': episode == 0})
        year = sim.curr_year_data
        for day in range(365):
            storage_before = info['storage']
            if policy_name == 'random':
                action = env.action_space.sample()
            elif policy_name == 'mean':
                action = (env.action_space.high + env.action_space.low) / 2
            elif policy_name == 'max':
                action = env.action_space.high.copy()
            elif policy_name == 'ead':
                action = np.array([info['exponential_average_demand']], dtype=np.float32)
            elif policy_name == 'ppo':
                action = controller(obs)
            else:
                raise ValueError(f'Unknown policy: {policy_name}')
            obs, reward, terminated, truncated, info = env.step(action)
            weighted = info['weighted_reward']
            if not np.isclose(reward, sum(weighted.values())):
                raise RuntimeError('Reward-component accounting failed')
            deficit = max(info['demand'] - max(info['release'] - sim.lake.min_env_flow[day], 0), 0)
            records.append({
                'policy': policy_name, 'seed': seed, 'year': year, 'day': day + 1,
                'objective_reward': sum(weighted[k] for k in OBJECTIVE), 'full_reward': reward,
                'level_m': info['level'], 'release_m3_s': info['release'],
                'request_m3_s': float(np.asarray(info['action']).item()),
                'demand_m3_s': info['demand'], 'inflow_m3_s': sim.inflow[day],
                'deficit_m3': deficit * 86400,
                'flood': int(info['level'] > sim.flood_level),
                'low_water': int(info['level'] < sim.starving_level),
                'storage_m3': info['storage'],
                'balance_error_m3': info['storage'] - storage_before - 86400 * (sim.inflow[day] - info['release']),
                **{f'weighted_{k}': v for k, v in weighted.items()},
            })
            if terminated or truncated:
                if day != 364 or terminated or not truncated:
                    raise RuntimeError('Unexpected episode end')
                break
    env.close()
    frame = pd.DataFrame(records)
    expected = list(settings('test', protocol='upstream')['inflow'])
    if frame.year.drop_duplicates().tolist() != expected or len(frame) != 13 * 365:
        raise RuntimeError('Incorrect test membership or episode count')
    if not np.isfinite(frame.select_dtypes('number')).all().all():
        raise RuntimeError('Nonfinite result')
    return frame, time.perf_counter() - started


def summarize(frame):
    rows = []
    for (policy, seed, year), f in frame.groupby(['policy', 'seed', 'year'], sort=False):
        rows.append(dict(policy=policy, seed=seed, year=year,
                         objective_return=f.objective_reward.sum(), full_return=f.full_reward.sum(),
                         flood_days=int(f.flood.sum()), low_water_days=int(f.low_water.sum()),
                         deficit_million_m3=f.deficit_m3.sum() / 1e6,
                         max_level_m=f.level_m.max(), min_level_m=f.level_m.min(),
                         terminal_storage_m3=f.storage_m3.iloc[-1],
                         max_balance_error_m3=f.balance_error_m3.abs().max()))
    return pd.DataFrame(rows)


def plot_results(frame, annual, output):
    colors = ['#c18b19', '#268b77', '#d16a20', '#91608e', '#2876b0']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    values = []
    for policy, color in zip(POLICIES, colors):
        f = frame[frame.policy == policy]
        if f.empty:
            continue
        curves = np.array([g.objective_reward.cumsum().to_numpy() for _, g in f.groupby('year', sort=False)])
        mean = curves.mean(axis=0)
        # Upstream notebook uses population SD / sqrt(13), not a seed-level CI.
        ci = 1.96 * curves.std(axis=0, ddof=0) / np.sqrt(len(curves))
        x = np.arange(1, 366)
        axes[0].plot(x, mean, color=color, label=policy.upper() if policy in ('ead', 'ppo') else policy.title())
        axes[0].fill_between(x, mean-ci, mean+ci, color=color, alpha=.12)
        values.append(annual.loc[annual.policy == policy, 'objective_return'].to_numpy())
    names = [p.upper() if p in ('ead', 'ppo') else p.title() for p in POLICIES if p in frame.policy.unique()]
    axes[0].set(xlabel='Simulation day', ylabel='Cumulative three-term objective reward', title='Mean and upstream-style 95% bands over 13 years')
    axes[0].legend()
    boxes = axes[1].boxplot(values, tick_labels=names, patch_artist=True)
    for patch, color in zip(boxes['boxes'], [c for p, c in zip(POLICIES, colors) if p in frame.policy.unique()]):
        patch.set_facecolor(color); patch.set_alpha(.65)
    axes[1].set(ylabel='Annual three-term objective return', title='Distribution across the original test years')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.savefig(output / 'figure1_reproduction.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/reproduction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, times = [], {}
    for policy in POLICIES:
        frame, seconds = evaluate(policy, args.seed, args.checkpoint)
        frames.append(frame); times[policy] = seconds
        print(f'{policy}: 13 years complete ({seconds:.2f}s)', flush=True)
    frame = pd.concat(frames, ignore_index=True)
    annual = summarize(frame)
    aggregate = annual.groupby('policy', sort=False).agg(
        mean_objective_return=('objective_return','mean'), median_objective_return=('objective_return','median'),
        mean_full_return=('full_return','mean'), mean_flood_days=('flood_days','mean'),
        mean_low_water_days=('low_water_days','mean'), mean_deficit_million_m3=('deficit_million_m3','mean'))
    frame.to_csv(args.output / 'trajectories.csv', index=False)
    annual.to_csv(args.output / 'annual.csv', index=False)
    aggregate.to_csv(args.output / 'aggregate.csv')
    plot_results(frame, annual, args.output)
    provenance_files = [*sorted((UPSTREAM_ROOT/'gym4real/envs/dam').glob('*.yaml')),
                        *sorted((UPSTREAM_ROOT/'gym4real/data/dam').rglob('*.csv')),
                        UPSTREAM_ROOT/'gym4real/envs/dam/env.py', UPSTREAM_ROOT/'gym4real/envs/dam/lake.py',
                        UPSTREAM_ROOT/'gym4real/algorithms/dam/ppo_skrl.py',
                        UPSTREAM_ROOT/'examples/dam/benchmarks.ipynb']
    metadata = dict(protocol='upstream', split='test', years=list(settings('test', protocol='upstream')['inflow']),
                    seed=args.seed, checkpoint=str(args.checkpoint.relative_to(UPSTREAM_ROOT)) if args.checkpoint.is_relative_to(UPSTREAM_ROOT) else str(args.checkpoint),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(UPSTREAM_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files},
                    upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM_ROOT),'rev-parse','HEAD'],text=True).strip(),
                    versions={p:version(p) for p in ['torch','skrl','gymnasium','numpy','pandas','matplotlib']},
                    python=platform.python_version(), platform=platform.platform(), seconds_by_policy=times,
                    objective_terms=list(OBJECTIVE), full_reward_terms=list(settings('test', protocol='upstream')['reward_coeff']),
                    inference='CPU; saved observation scaler frozen; deterministic Gaussian mean in FP32 (no autocast); Gymnasium RescaleAction [-1,1] to physical bounds',
                    differences=['Random action space explicitly seeded once at 123; upstream notebook does not explicitly seed it.',
                                 'Original observations, annual resets, leap-year 365-step behavior retained.',
                                 'This command evaluates without training; consult training metadata for checkpoint origin.',
                                 'No digitized paper reference numbers; matching the printed figure is qualitative.',
                                 'Inference uses CPU FP32 rather than the source agent autocast context.'])
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(aggregate.round(2).to_string())
    print('Saved:',args.output)


if __name__ == '__main__':
    main()
