"""Paired chronological controller evaluation; validation by default, no training."""
import argparse
from itertools import combinations
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, UPSTREAM_ROOT, DAM_CONFIG, DAM_DATA
from lake_como.reproduce import evaluate, summarize
from lake_como.splits import split_years

RULES = ('random', 'mean', 'max', 'ead', 'constant')
SEEDS = tuple(range(123, 128))
METRICS = ('full_return', 'objective_return', 'flood_days', 'low_water_days',
           'deficit_million_m3', 'terminal_storage_m3')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint(seed, checkpoint):
    """Reject unknown training membership or modified checkpoints before evaluation."""
    checkpoint = checkpoint.resolve()
    path = checkpoint.parent / 'run.json'
    metadata = json.loads(path.read_text())
    if not metadata.get('complete') or metadata['job']['seed'] != seed:
        raise ValueError(f'Incomplete checkpoint or incorrect learner seed: {checkpoint}')
    if metadata['training_years'] != split_years('train'):
        raise ValueError('Checkpoint must use the chronological training membership')
    common = metadata['job']['common']
    if common['protocol'] != 'chronological' or not common['correct_terminal_level']:
        raise ValueError('Checkpoint protocol differs from the comparison')
    hashes = metadata['checkpoint_sha256']
    expected = hashes.get(checkpoint.stem.removeprefix('step_')) if isinstance(hashes, dict) else hashes
    if digest(checkpoint) != expected:
        raise ValueError(f'Checkpoint hash differs from training metadata: {checkpoint}')
    return dict(training_seed=seed, path=str(checkpoint), sha256=expected,
                training_metadata_path=str(path), training_metadata_sha256=digest(path),
                training_years=metadata['training_years'], common=common,
                budgets=metadata['job'].get('budgets'), training_versions=metadata['versions'])


def annual_metrics(frame):
    annual = summarize(frame)
    weighted = [c for c in frame if c.startswith('weighted_')]
    keys = ['policy', 'seed', 'year']
    components = frame.groupby(keys, sort=False)[weighted].sum().reset_index()
    result = annual.merge(components, on=keys, validate='one_to_one')
    timings = frame.groupby(keys, sort=False).decision_ns.agg(
        decision_mean_us=lambda s: s.mean() / 1000,
        decision_p50_us=lambda s: s.quantile(.5) / 1000,
        decision_p95_us=lambda s: s.quantile(.95) / 1000).reset_index()
    return result.merge(timings, on=keys, validate='one_to_one')


def paired_metrics(annual, metrics=METRICS):
    """Candidate minus reference for identical years; never pair rows by position."""
    if annual.duplicated(['controller', 'year']).any():
        raise ValueError('Duplicate controller/year results')
    groups = {name: group.set_index('year') for name, group in annual.groupby('controller', sort=False)}
    memberships = [set(group.index) for group in groups.values()]
    if not memberships or any(years != memberships[0] for years in memberships):
        raise ValueError('Controllers must have identical year membership')
    rows = []
    for reference, candidate in combinations(groups, 2):
        delta = groups[candidate][list(metrics)].sort_index() - groups[reference][list(metrics)].sort_index()
        delta = delta.rename(columns={metric: f'delta_{metric}' for metric in metrics})
        delta['reference'] = reference
        delta['candidate'] = candidate
        rows.append(delta.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['year', 'reference', 'candidate', *[f'delta_{m}' for m in metrics]])


def summaries(annual):
    """Keep year variability distinct from variability of PPO learner-seed means."""
    metrics = list(METRICS)
    per_controller = annual.groupby(['family', 'controller'], sort=False)[metrics].mean().reset_index()
    family_years = annual.groupby(['family', 'year'], sort=False)[metrics].mean().reset_index()
    rows = []
    for family, group in per_controller.groupby('family', sort=False):
        year_values = family_years[family_years.family == family]
        row = dict(family=family, controllers=len(group), years=len(year_values))
        for metric in metrics:
            row[f'mean_{metric}'] = group[metric].mean()
            row[f'seed_sd_{metric}'] = group[metric].std() if family == 'ppo' and len(group) > 1 else np.nan
            row[f'year_sd_{metric}'] = year_values[metric].std()
        row['weakest_seed_full_return'] = group.full_return.min() if family == 'ppo' else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    return per_controller, family_years, summary


def verify_matched_inputs(trajectories, years):
    """All controllers must see the same forcing and annual reset conditions."""
    reference = None
    for controller, frame in trajectories.groupby('controller', sort=False):
        if frame.year.drop_duplicates().tolist() != years:
            raise ValueError(f'Incorrect split coverage: {controller}')
        if frame.duplicated(['year', 'day']).any() or not frame.groupby('year').size().eq(365).all():
            raise ValueError(f'Incomplete annual trajectory: {controller}')
        forcing = frame.set_index(['year', 'day'])[['inflow_m3_s', 'demand_m3_s']].sort_index()
        initial = frame.loc[frame.day == 1, ['year', 'level_before_m', 'storage_before_m3']].set_index('year').sort_index()
        if reference is None:
            reference = forcing, initial
        else:
            pd.testing.assert_frame_equal(forcing, reference[0], check_exact=True)
            pd.testing.assert_frame_equal(initial, reference[1], check_exact=True)
        weighted = frame.filter(regex='^weighted_').sum(axis=1)
        np.testing.assert_allclose(weighted, frame.full_reward, rtol=1e-10, atol=1e-7)
        if frame.balance_error_m3.abs().max() >= 1e-4:
            raise ValueError('Water balance check failed')


def plot_results(annual, family_years, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(family_years.family.drop_duplicates())
    years = sorted(family_years.year.unique())
    colors = dict(zip(names, plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for ax, metric, title in zip(axes.flat,
                                ['full_return', 'flood_days', 'low_water_days', 'deficit_million_m3'],
                                ['Full annual return (higher is better)', 'Flood days', 'Low-water days', 'Unmet demand (million m³)']):
        for family in names:
            f = family_years[family_years.family == family].sort_values('year')
            ax.plot(f.year, f[metric], marker='o', ms=3, label=family.upper(), color=colors[family])
            if family == 'ppo':
                seeds = annual[annual.family == 'ppo'].groupby('year')[metric].agg(['min', 'max']).reindex(years)
                ax.fill_between(years, seeds['min'], seeds['max'], color=colors[family], alpha=.15)
        ax.set(title=title, xlabel='Year')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(ncol=3, fontsize=9)
    ppo_count = annual.loc[annual.family == 'ppo', 'controller'].nunique()
    subtitle = f'PPO averages {ppo_count} seeds; shading is seed range' if ppo_count else 'Identical annual forcing and resets'
    fig.suptitle(f'Same-year controller comparison · {subtitle}')
    fig.savefig(output / 'annual_comparison.png', dpi=170)
    plt.close(fig)
    ppo = annual[annual.family == 'ppo']
    if ppo.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['full_return', 'deficit_million_m3'],
                                ['PPO minus reference: full return (positive is better)', 'PPO minus reference: unmet demand (negative is better)']):
        ax.axhline(0, color='black', lw=1)
        for family in names:
            if family == 'ppo' or ppo.empty:
                continue
            rule = annual[annual.family == family].set_index('year')[metric]
            deltas = ppo.assign(delta=ppo[metric] - ppo.year.map(rule)).groupby('year').delta.agg(['mean', 'min', 'max']).reindex(years)
            ax.plot(years, deltas['mean'], marker='o', ms=3, label=f'vs {family.upper()}', color=colors[family])
            ax.fill_between(years, deltas['min'], deltas['max'], color=colors[family], alpha=.10)
        ax.set(title=title, xlabel='Year')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=9)
    fig.suptitle('Paired annual differences · mean and range over PPO seeds, not confidence intervals')
    fig.savefig(output / 'paired_ppo_vs_rules.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--controllers', nargs='+', choices=[*RULES, 'ppo', 'mpc'], default=['random', 'mean', 'max', 'ead', 'ppo'])
    parser.add_argument('--ppo-checkpoint', action='append', metavar='SEED=PATH', help='Repeat for each PPO checkpoint; defaults to all five selected 1M seeds')
    parser.add_argument('--action-seed', type=int, default=123, help='One seeded random-policy sequence across the whole split')
    parser.add_argument('--mpc-config', type=Path, default=PROJECT_ROOT/'configs/mpc.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(set(args.controllers)) != len(args.controllers):
        parser.error('Controller names must be unique')
    checkpoint_paths = {}
    try:
        for value in args.ppo_checkpoint or []:
            seed, path = value.split('=', 1)
            seed = int(seed)
            if seed in checkpoint_paths:
                raise ValueError('Duplicate PPO seed')
            checkpoint_paths[seed] = Path(path)
    except ValueError as error:
        parser.error(f'Invalid checkpoint argument: {error}')
    if not checkpoint_paths:
        selected = json.loads((PROJECT_ROOT / 'configs/ppo.json').read_text())
        checkpoint_paths = {seed: PROJECT_ROOT / selected['checkpoint_template'].format(seed=seed)
                            for seed in selected['learner_seeds']}
    specs = []
    for family in args.controllers:
        if family == 'ppo':
            for seed, path in checkpoint_paths.items():
                provenance = validate_checkpoint(seed, path)
                specs.append(dict(controller=f'ppo_{seed}', family=family, training_seed=seed, checkpoint=path.resolve(), provenance=provenance))
        else:
            specs.append(dict(controller=family, family=family, training_seed=None, checkpoint=None, provenance=None))
    if 'ppo' in args.controllers:
        import torch
        torch.set_num_threads(1)
    years = split_years(args.split)
    output = args.output or PROJECT_ROOT / f'results/comparison/{args.split}_v1'
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output to preserve existing results')
    output.mkdir(parents=True, exist_ok=True)
    frames, rows, timings = [], [], []
    mpc_records = []
    for spec in specs:
        kwargs = dict(checkpoint=spec['checkpoint']) if spec['checkpoint'] is not None else {}
        mpc = None
        if spec['family'] == 'mpc':
            from lake_como.mpc import DeterministicMPC, MPCConfig
            from dataclasses import asdict
            mpc = DeterministicMPC.from_training(MPCConfig.load(args.mpc_config))
            kwargs['controller_factory'] = lambda env: mpc
            spec['provenance'] = dict(config=asdict(mpc.config), training_years=list(mpc.forecast.training_years),
                                      forecast='Training-only inflow/demand climatology', config_sha256=digest(args.mpc_config))
        frame, seconds = evaluate(spec['family'], seed=args.action_seed, split=args.split,
                                  protocol='chronological', correct_terminal_level=True,
                                  sequential_years=True, measure_decisions=True, **kwargs)
        if mpc is not None:
            from lake_como.mpc import save_diagnostics
            diagnostics = save_diagnostics(mpc, frame, output)
            mpc_records.append(diagnostics)
        annual = annual_metrics(frame)
        for table in [frame, annual]:
            table['controller'] = spec['controller']
            table['family'] = spec['family']
            table['training_seed'] = spec['training_seed']
        frames.append(frame)
        rows.append(annual)
        timings.append(dict(controller=spec['controller'], evaluation_seconds=seconds,
                            decisions=len(frame), decision_mean_us=frame.decision_ns.mean() / 1000,
                            decision_p50_us=frame.decision_ns.quantile(.5) / 1000,
                            decision_p95_us=frame.decision_ns.quantile(.95) / 1000))
        print(f"{spec['controller']}: {len(years)} years complete ({seconds:.2f}s)", flush=True)
    trajectories = pd.concat(frames, ignore_index=True)
    annual = pd.concat(rows, ignore_index=True)
    verify_matched_inputs(trajectories, years)
    paired = paired_metrics(annual)
    per_controller, family_years, summary = summaries(annual)
    family_pairs = paired_metrics(family_years.rename(columns={'family': 'controller'}))
    trajectories.to_csv(output / 'trajectories.csv', index=False)
    annual.to_csv(output / 'annual.csv', index=False)
    paired.to_csv(output / 'paired_annual.csv', index=False)
    family_pairs.to_csv(output / 'paired_family_annual.csv', index=False)
    per_controller.to_csv(output / 'controller_means.csv', index=False)
    family_years.to_csv(output / 'family_years.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(timings).to_csv(output / 'timings.csv', index=False)
    plot_results(annual, family_years, output)
    if mpc_records:
        pd.concat(mpc_records, ignore_index=True).to_csv(output/'mpc_diagnostics.csv', index=False)
    files = [Path(__file__), PROJECT_ROOT / 'src/lake_como/reproduce.py', PROJECT_ROOT / 'src/lake_como/environment.py',
             PROJECT_ROOT / 'configs/splits.json', PROJECT_ROOT / 'configs/ppo.json', PROJECT_ROOT / 'uv.lock',
             PROJECT_ROOT / 'src/lake_como/splits.py', PROJECT_ROOT / 'pyproject.toml',
             UPSTREAM_ROOT / 'gym4real/algorithms/dam/ppo_skrl.py',
             *sorted(DAM_CONFIG.glob('*.yaml')), *sorted(DAM_DATA.rglob('*.csv')),
             *sorted((DAM_CONFIG).glob('*.py'))]
    if 'mpc' in args.controllers:
        files.extend([PROJECT_ROOT/'src/lake_como/mpc.py', PROJECT_ROOT/'src/lake_como/run_mpc.py', args.mpc_config.resolve()])
    metadata = dict(protocol='chronological', split=args.split, years=years,
                    test_evaluated=args.split == 'test', action_seed=args.action_seed,
                    controllers=[{k: str(v) if isinstance(v, Path) else v for k, v in spec.items()} for spec in specs],
                    calendar='Leap years included with unchanged 365-step upstream approximation',
                    reward_coeff=settings(args.split)['reward_coeff'],
                    smooth_daily_deficit_coeff=settings(args.split)['smooth_daily_deficit_coeff'],
                    simulation=dict(daily_actions=365, hourly_substeps=24, sequential_years=True),
                    initialization='Independent annual resets to level 0.35m, demand average 0; identical forcing checked',
                    inference='PPO deterministic mean, frozen observation scaler, CPU FP32, one torch thread',
                    timing='CPU decision only, after per-year PPO warmup; excludes env.step, rescaling, initialization, CSV; microsecond overhead/noise is material',
                    paired_direction='candidate minus reference; higher returns / lower physical penalties are better',
                    uncertainty=f'PPO seed SD uses learner-seed annual means; year SD uses {len(years)} yearly seed averages; rules have one action-seed run, no learner-seed SD',
                    versions={p: version(p) for p in ['gymnasium', 'numpy', 'pandas', 'matplotlib'] + (['torch', 'skrl'] if 'ppo' in args.controllers else []) + (['scipy'] if 'mpc' in args.controllers else [])},
                    python=platform.python_version(), platform=platform.platform(),
                    upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM_ROOT),'rev-parse','HEAD'], text=True).strip(),
                    hashes={(str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)): digest(path) for path in files}, complete=True)
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary[['family', *[f'mean_{m}' for m in METRICS]]].round(2).to_string(index=False))
    print('Saved:', output)


if __name__ == '__main__':
    main()
