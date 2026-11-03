"""Staged SAC smoke/training/validation study. No test-year evaluation path.

uv run --group reproduction python -m lake_como.sac_study --stage prepare
Then --stage smoke, --stage train, and --stage evaluate. Existing outputs are kept.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from lake_como.compare import annual_metrics, verify_matched_inputs
from lake_como.environment import make_env, settings
from lake_como.inflow_observation import PastInflowObservation
from lake_como.mpc import MPCConfig, PlanningModel, SeasonalForecast
from lake_como.mpc_hindsight import digest
from lake_como.mpc_objective_closed_loop import level_severity
from lake_como.paths import PROJECT_ROOT, DAM_DATA, DAM_CONFIG
from lake_como.reproduce import evaluate
from lake_como.sac import SACAgent, SACConfig, ReplayBuffer
from lake_como.sac_loop import train_steps
from lake_como.safety_filter import FloodSafetyFilter, fit_error_limits
from lake_como.splits import split_years

CONFIG = PROJECT_ROOT / 'configs/sac.json'
FREEZE = PROJECT_ROOT / 'configs/experiments/control_protocol_v1.json'
METRICS = ['full_return', 'flood_days', 'low_water_days', 'deficit_million_m3',
           'flood_excess_m_days', 'low_excess_m_days']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_plan(root):
    plan = json.loads((root / 'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if digest(PROJECT_ROOT / name) != expected:
            raise ValueError(f'Frozen study source changed: {name}')
    for name, expected in plan['reference_sha256'].items():
        if digest(PROJECT_ROOT / name) != expected:
            raise ValueError(f'Frozen reference changed: {name}')
    for package, expected in plan['versions'].items():
        if version(package) != expected:
            raise ValueError(f'Package changed: {package}')
    if digest(root / 'normalization.json') != plan['normalization_sha256']:
        raise ValueError('Normalization changed')
    return plan


def prepare(root):
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose a new empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    freeze = json.loads(FREEZE.read_text())
    for name, expected in freeze['source_sha256'].items():
        if digest(PROJECT_ROOT / name) != expected:
            raise ValueError(f'Reference source changed: {name}')
    train = settings('train')
    fallback = float(SeasonalForecast.fit(train['inflow'], train['demand']).inflow[0])
    if fallback != freeze['past_inflow']['reset_fallback_m3_s']:
        raise ValueError('Fallback differs from frozen references')
    rng = np.random.default_rng(config['normalization']['seed'])
    observations = []
    started = time.perf_counter()
    for year in split_years('train'):
        env = PastInflowObservation(make_env('train', year), fallback)
        try:
            obs, _ = env.reset()
            for day in range(365):
                observations.append(obs.copy())
                request = (rng.uniform(float(env.action_space.low[0]), float(env.action_space.high[0]))
                           if rng.random() < .5 else np.clip(float(obs[3]),
                                                           env.action_space.low[0], env.action_space.high[0]))
                obs, reward, terminated, truncated, _ = env.step(np.array([request], np.float32))
                if terminated or truncated != (day == 364):
                    raise ValueError('Unexpected normalization boundary')
        finally:
            env.close()
    observations = np.asarray(observations, dtype=np.float64)
    if observations.shape != (15330, 5) or not np.isfinite(observations).all():
        raise ValueError('Invalid normalization sample')
    normalization = dict(mean=observations.mean(0).tolist(),
                         scale=np.maximum(observations.std(0), 1e-6).tolist(),
                         train_years=split_years('train'), interactions=len(observations),
                         seed=config['normalization']['seed'],
                         seconds=time.perf_counter() - started,
                         observation_sha256=__import__('hashlib').sha256(observations.tobytes()).hexdigest())
    write_json(root / 'normalization.json', normalization)
    files = [CONFIG, FREEZE, Path(__file__), PROJECT_ROOT / 'src/lake_como/sac.py',
             PROJECT_ROOT / 'src/lake_como/sac_loop.py', PROJECT_ROOT / 'src/lake_como/compare.py',
             PROJECT_ROOT / 'src/lake_como/splits.py', PROJECT_ROOT / 'src/lake_como/mpc_objective_closed_loop.py',
             PROJECT_ROOT / 'external/gym4real/gym4real/envs/dam/env.py',
             PROJECT_ROOT / 'external/gym4real/gym4real/envs/dam/lake.py']
    files += [PROJECT_ROOT / name for name in freeze['source_sha256']]
    files += list(DAM_DATA.rglob('*.csv')) + list(DAM_CONFIG.glob('*.yaml'))
    files = list(dict.fromkeys(files))
    plan = dict(config=config, fallback=fallback, training_years=split_years('train'),
                validation_years=split_years('validation'), test_evaluated=False,
                normalization_sha256=digest(root / 'normalization.json'),
                source_sha256={str(p.relative_to(PROJECT_ROOT)): digest(p) for p in files},
                reference_sha256=freeze['reference_results'],
                versions={p: version(p) for p in ['torch', 'numpy', 'gymnasium', 'scipy', 'skrl']},
                scope=config['budget_note'], device='cpu', precision='float32',
                checkpoint_selection='final only; no validation during training',
                normalizer_interactions=len(observations), environment_seed=config['environment_seed'])
    write_json(root / 'plan.json', plan)
    print('Prepared training-only normalization and frozen plan', normalization, flush=True)


class TrainingAudit:
    """Telemetry is checked after stepping; none is added to policy observations."""
    def __init__(self, env, agent, output, config, steps):
        self.env, self.agent, self.output, self.config, self.steps = env, agent, output, config, steps
        self.start = time.perf_counter()
        self.years = Counter()
        self.completed = []
        self.trace, self.window = [], []
        self.bounds = dict(reward=[float('inf'), -float('inf')], request=[float('inf'), -float('inf')],
                           normalized=[float('inf'), -float('inf')], level=[float('inf'), -float('inf')])
        self.max_gradient = dict(critic_gradient_norm=0., actor_gradient_norm=0., alpha_gradient_norm=0.)
        self.nonzero_gradients = Counter()
        self.max_balance_error, self.reset_count = 0., 0
        self.storage = None

    def __call__(self, event):
        sim = self.env.unwrapped
        year, day = int(sim.curr_year_data), event['episode_length']
        if year not in split_years('train') or day != sim.current_step:
            raise ValueError('Non-training profile or wrong step')
        self.years[year] += 1
        obs, following, info = event['observation'], event['next_observation'], event['info']
        if day == 1:
            self.reset_count += 1
            self.storage = sim.lake.level_to_storage(.35)
            np.testing.assert_allclose(obs, [.35, 0, 1, 0, self.env.first_day_fallback], atol=1e-5)
        if not np.isfinite(np.r_[obs, following, event['request'], event['reward']]).all():
            raise FloatingPointError('Nonfinite training transition')
        if (np.abs(event['normalized_action']) > 1).any() or not self.env.action_space.contains(event['request']):
            raise ValueError('Action outside bounds')
        np.testing.assert_allclose(following[-1], sim.inflow[day - 1], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(following[0], info['level'], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sum(info['weighted_reward'].values()), event['reward'], rtol=1e-10, atol=1e-7)
        error = abs(info['storage'] - self.storage - 86400 * (sim.inflow[day - 1] - info['release']))
        if error >= 1e-4:
            raise ValueError('Training water balance failure')
        self.max_balance_error = max(self.max_balance_error, error)
        self.storage = info['storage']
        if event['terminated'] or event['truncated'] != (day == 365):
            raise ValueError('Unexpected annual boundary')
        if event['truncated']:
            self.completed.append(dict(step=event['step'], year=year, length=day,
                                       full_return=event['episode_return'], final_level=info['level']))
        for name, value in dict(reward=event['reward'], request=float(event['request'][0]),
                                normalized=float(event['normalized_action'][0]), level=info['level']).items():
            self.bounds[name][0] = min(self.bounds[name][0], value)
            self.bounds[name][1] = max(self.bounds[name][1], value)
        metrics = event['diagnostics']
        if metrics is not None:
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Nonfinite training metrics')
            self.window.append(metrics)
            for name in self.max_gradient:
                self.max_gradient[name] = max(self.max_gradient[name], metrics[name])
                self.nonzero_gradients[name] += int(metrics[name] > 0)
        step = event['step']
        if step % self.config['trace_every_steps'] == 0 or step == self.steps:
            mean = {k: float(np.mean([row[k] for row in self.window])) for k in self.window[0]} if self.window else {}
            row = dict(step=step, updates=self.agent.updates, elapsed_seconds=time.perf_counter() - self.start,
                       latest_episode_return=self.completed[-1]['full_return'] if self.completed else None, **mean)
            self.trace.append(row)
            self.window.clear()
            pd.DataFrame(self.trace).to_csv(self.output / 'learning_trace.csv', index=False)
            pd.DataFrame(self.completed).to_csv(self.output / 'training_episodes.csv', index=False)
        if step % self.config['checkpoint_every_steps'] == 0 or step == self.steps:
            self.agent.save(self.output / f'step_{step}.pt')
        if step % 10000 == 0 or step == self.steps:
            print(f"{self.output.name}: {step}/{self.steps}, updates={self.agent.updates}, "
                  f"elapsed={time.perf_counter()-self.start:.1f}s", flush=True)

    def result(self):
        return dict(passed=True, finite_actions_rewards_losses_gradients=True,
                    correct_terminal_observations=True, no_bootstrap_across_annual_boundary=True,
                    training_year_transition_counts=dict(self.years), complete_episodes=len(self.completed),
                    reset_count=self.reset_count, all_complete_episode_lengths_365=True,
                    ranges=self.bounds, maximum_gradient_norms=self.max_gradient,
                    updates_with_nonzero_gradient=dict(self.nonzero_gradients),
                    max_water_balance_error_m3=self.max_balance_error)


def new_agent(root, plan, seed):
    norm = json.loads((root / 'normalization.json').read_text())
    return SACAgent(SACConfig(**plan['config']['agent']), observation_mean=norm['mean'],
                    observation_scale=norm['scale'], seed=seed)


def checker(plan):
    from gymnasium.utils.env_checker import check_env
    env = PastInflowObservation(make_env('train', 1946), plan['fallback'])
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                check_env(env, skip_render_check=True)
                result = dict(passed=True)
            except AssertionError as exc:
                # Upstream DamEnv does not call super().reset(seed=seed).
                if 'random number generator' not in str(exc) and '_np_random' not in str(exc):
                    raise
                result = dict(passed=False, upstream_limitation=str(exc))
            result['warnings'] = [str(w.message) for w in captured]
        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        result['manual_seeded_reset_observation_check'] = True
        return result
    finally:
        env.close()


def train_one(root, *, smoke=False, seed=None):
    plan = verify_plan(root)
    config = plan['config']
    torch.set_num_threads(config['torch_threads_per_worker'])
    seed = config['smoke']['seed'] if smoke else seed
    if not smoke and seed not in config['learner_seeds']:
        raise ValueError('Unknown seed')
    if not smoke:
        gate = json.loads((root / 'smoke/run.json').read_text())
        if not gate['complete'] or not gate['audit']['passed']:
            raise ValueError('Smoke gate not passed')
    output = root / ('smoke' if smoke else f'seed_{seed}')
    if output.exists():
        raise ValueError(f'Preserve existing output: {output}')
    output.mkdir()
    if smoke:
        write_json(output / 'environment_checker.json', checker(plan))
    agent = new_agent(root, plan, seed)
    env = PastInflowObservation(make_env('train'), plan['fallback'])
    steps = config['smoke']['steps'] if smoke else config['steps_per_seed']
    warmup = config['smoke']['warmup_steps'] if smoke else config['warmup_steps']
    replay = ReplayBuffer(config['replay_capacity'], 5, 1, seed=seed)
    audit = TrainingAudit(env, agent, output, config, steps)
    started = time.perf_counter()
    try:
        result = train_steps(env, agent, replay, steps=steps, warmup_steps=warmup,
                             batch_size=config['batch_size'], updates_per_step=config['updates_per_step'],
                             seed=seed, environment_seed=config['environment_seed'], on_step=audit)
    finally:
        env.close()
    if result['updates'] != (steps - warmup) * config['updates_per_step']:
        raise ValueError('Unexpected gradient update count')
    if not all(audit.nonzero_gradients[k] > 0 for k in audit.max_gradient):
        raise ValueError('A learner component received no nonzero gradients')
    if smoke and (len(audit.completed) != 20 or replay.data['ends'][:steps].sum() != 20):
        raise ValueError('Smoke annual boundary audit failed')
    agent.save(output / 'final.pt')
    restored = SACAgent.load(output / 'final.pt')
    sample = np.asarray(json.loads((root / 'normalization.json').read_text())['mean'], np.float32)
    np.testing.assert_array_equal(agent.act(sample, deterministic=True), restored.act(sample, deterministic=True))
    write_json(output / 'run.json', dict(complete=True, smoke=smoke, seed=seed, steps=steps,
               updates=agent.updates, training_seconds=time.perf_counter()-started, test_evaluated=False,
               audit=audit.result(), checkpoint_sha256=digest(output / 'final.pt'),
               plan_sha256=digest(root / 'plan.json'), result=result))
    print('COMPLETE', seed, steps, 'seconds', time.perf_counter()-started, flush=True)


class SACController:
    """Adapter for the unchanged common four-observation evaluation runner."""
    def __init__(self, agent, fallback, safety_filter=None):
        self.agent, self.fallback, self.filter = agent, fallback, safety_filter
        self.previous_storage, self.previous_day = None, None

    def __call__(self, observation, info):
        obs = np.asarray(observation, np.float32)
        day = int(np.rint((np.arctan2(obs[1], obs[2]) % (2*np.pi)) * 365/(2*np.pi))) % 365 + 1
        storage = float(info['storage'])
        if day == 1:
            self.previous_storage = None
        elif self.previous_day != day - 1:
            raise ValueError('Consecutive observations required')
        past = self.fallback if self.previous_storage is None else max(0., (storage-self.previous_storage)/86400 + float(info['release']))
        state = np.r_[obs, past].astype(np.float32)
        action = self.agent.act(state, deterministic=True)
        request = self.agent.physical_action(torch.from_numpy(action)).numpy()
        self.previous_storage, self.previous_day = storage, day
        return self.filter(request, state, info) if self.filter is not None else request


def evaluate_study(root):
    plan = verify_plan(root)
    if (root / 'run.json').exists():
        raise ValueError('Completed study exists; preserve it')
    torch.set_num_threads(1)
    config = plan['config']
    params = settings('train')
    seasonal = SeasonalForecast.fit(params['inflow'], params['demand'])
    model = PlanningModel(params, MPCConfig.load())
    errors = fit_error_limits(params['inflow'], seasonal.inflow)
    daily_frames, annual_frames, filter_frames = [], [], []
    for seed in config['learner_seeds']:
        output = root / f'seed_{seed}'
        metadata = json.loads((output / 'run.json').read_text())
        if (not metadata['complete'] or metadata['steps'] != config['steps_per_seed']
                or metadata['checkpoint_sha256'] != digest(output / 'final.pt')
                or metadata['plan_sha256'] != digest(root / 'plan.json')):
            raise ValueError('Incomplete/mismatched training run')
        agent = SACAgent.load(output / 'final.pt')
        before = {k: v.clone() for k, v in agent.state_dict().items()}
        updates_before = agent.updates
        for variant in ['plain', 'shared_filter']:
            filt = FloodSafetyFilter(model, plan['fallback'], errors) if variant == 'shared_filter' else None
            controller = SACController(agent, plan['fallback'], filt)
            frame, _ = evaluate('sac', split='validation', protocol='chronological',
                                correct_terminal_level=True, sequential_years=True, measure_decisions=True,
                                controller_factory=lambda env: controller)
            frame['controller'] = f'sac_{seed}_{variant}'
            frame['training_seed'], frame['variant'] = seed, variant
            annual = annual_metrics(frame)
            severity = level_severity(frame.assign(objective=variant)).reset_index().drop(
                columns=['objective', 'max_level_m', 'min_level_m'])
            annual = annual.merge(severity, on='year', validate='one_to_one')
            annual['training_seed'], annual['variant'] = seed, variant
            annual_frames.append(annual)
            daily_frames.append(frame)
            if filt is not None:
                diag = pd.DataFrame(filt.diagnostics)
                diag['training_seed'] = seed
                filter_frames.append(diag)
            print('Evaluated SAC', seed, variant, flush=True)
        if updates_before != agent.updates or any(not torch.equal(v, agent.state_dict()[k]) for k, v in before.items()):
            raise ValueError('Learner changed during evaluation')
    daily = pd.concat(daily_frames, ignore_index=True)
    annual = pd.concat(annual_frames, ignore_index=True)
    # Include saved references to check exact annual forcing/reset pairing.
    reference = pd.read_csv(PROJECT_ROOT / 'results/safety/common_filter_v1/trajectories.csv')
    verify_matched_inputs(pd.concat([daily, reference], ignore_index=True), split_years('validation'))
    means = annual.groupby(['variant', 'training_seed'])[METRICS].mean().reset_index()
    summary = means.groupby('variant')[METRICS].mean().reset_index()
    variability = means.groupby('variant').full_return.agg(seed_sd='std', weakest_seed='min', best_seed='max').reset_index()
    references = pd.read_csv(PROJECT_ROOT / 'results/safety/common_filter_v1/annual.csv')
    paired = []
    for variant in ['plain', 'shared_filter']:
        ref_variant = 'unfiltered' if variant == 'plain' else 'filtered'
        for family in ['ppo', 'mpc']:
            ref = references[(references.family == family) & (references.variant == ref_variant)]
            # Pair PPO by learner-seed label, MPC by year; seeds are not independent years.
            on = ['year', 'training_seed'] if family == 'ppo' else ['year']
            left = annual[annual.variant == variant]
            merged = left.merge(ref[on + METRICS], on=on, suffixes=('_sac', '_reference'), validate='one_to_one' if family == 'ppo' else 'many_to_one')
            record = merged[['year', 'training_seed']].copy()
            record['variant'], record['reference'] = variant, family
            for metric in METRICS:
                record[f'delta_{metric}'] = merged[f'{metric}_sac'] - merged[f'{metric}_reference']
            paired.append(record)
    for name, table in [('trajectories', daily), ('annual', annual), ('seed_means', means),
                        ('summary', summary), ('variability', variability),
                        ('paired', pd.concat(paired, ignore_index=True)),
                        ('filter_diagnostics', pd.concat(filter_frames, ignore_index=True))]:
        table.to_csv(root / f'{name}.csv', index=False)
    plot_study(root, summary)
    verify_plan(root)
    write_json(root / 'run.json', dict(complete=True, test_evaluated=False, annual_evaluation_episodes=100,
                training_steps_per_seed=config['steps_per_seed'], learner_seeds=config['learner_seeds'],
                normalization_interactions=15330, smoke_interactions=config['smoke']['steps'],
                unchanged_references=True, no_intermediate_checkpoint_selection=True))
    print(summary.to_string(index=False), flush=True)
    print(variability.to_string(index=False), flush=True)


def plot_study(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for seed in json.loads((root / 'plan.json').read_text())['config']['learner_seeds']:
        frame = pd.read_csv(root / f'seed_{seed}/learning_trace.csv')
        episodes = pd.read_csv(root / f'seed_{seed}/training_episodes.csv')
        axes[0, 0].plot(episodes.step, episodes.full_return.rolling(20, min_periods=1).mean(), label=str(seed))
        for ax, metric in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ['alpha', 'critic_loss', 'actor_gradient_norm']):
            ax.plot(frame.step, frame[metric], label=str(seed))
    for ax, title in zip(axes.flat, ['Training return (20-episode rolling mean; changing years)', 'Entropy temperature', 'Critic loss', 'Actor gradient norm']):
        ax.set(title=title, xlabel='Training interactions')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(title='Learner seed')
    fig.suptitle('SAC first 200k-step study: training diagnostics, not validation curves')
    fig.savefig(root / 'learning_curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'train', 'evaluate', 'worker'], required=True)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/sac/baseline_200k_v1')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == 'prepare':
        prepare(root)
    elif args.stage == 'smoke':
        train_one(root, smoke=True)
    elif args.stage == 'worker':
        train_one(root, seed=args.seed)
    elif args.stage == 'evaluate':
        evaluate_study(root)
    else:
        plan = verify_plan(root)
        def launch(seed):
            with (root / f'console_{seed}.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'lake_como.sac_study', '--stage', 'worker',
                                '--output', str(root), '--seed', str(seed)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, plan['config']['learner_seeds']))


if __name__ == '__main__':
    main()
