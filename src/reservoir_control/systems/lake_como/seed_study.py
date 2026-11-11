"""Fixed-budget PPO seed study, reusing the original seed-123 run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT

SEEDS = (123, 124, 125, 126, 127)


def check_training(metadata, seed):
    config = metadata['config']
    expected = dict(training_timesteps=200000, mini_batches=32, learning_epochs=10,
                    rollouts=2048, discount_factor=.995, ent_coef=0., learning_rate=8e-6,
                    net_arch=[16,16], initial_log_std=-.5, normalize_observations=True)
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f'Seed {seed}: configuration mismatch for {key}')
    if config['seed'] != seed or metadata['protocol'] != 'upstream' or metadata['device'] != 'cpu':
        raise ValueError(f'Incorrect seed/protocol/device for seed {seed}')
    checkpoint = PROJECT_ROOT / metadata['final_checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch for seed {seed}')
    return checkpoint


def run_seed(seed, output):
    directory = output / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    training_dir = PROJECT_ROOT / 'results/reproduction/training' if seed == 123 else directory / 'training'
    metadata_path = training_dir / 'training_run.json'
    if not metadata_path.exists() or 'final_checkpoint' not in json.loads(metadata_path.read_text()):
        if seed == 123:
            raise ValueError('Original complete seed-123 training metadata is required for reuse')
        print(f'Seed {seed}: starting 200,000-step training', flush=True)
        with (directory / 'training.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'reservoir_control.systems.lake_como.train_reference', '--seed', str(seed),
                            '--output', str(training_dir)], stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = json.loads(metadata_path.read_text())
    checkpoint = check_training(metadata, seed)
    # Keep the same evaluation random seed and baseline trajectories for every learner seed.
    with (directory / 'evaluation.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'reservoir_control.systems.lake_como.reproduce', '--seed', '123',
                        '--checkpoint', str(checkpoint), '--output', str(directory / 'evaluation')],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Seed {seed}: trained checkpoint evaluated', flush=True)
    return dict(training_seed=seed, training_metadata=str(metadata_path.relative_to(PROJECT_ROOT)),
                checkpoint=str(checkpoint.relative_to(PROJECT_ROOT)), checkpoint_sha256=metadata['checkpoint_sha256'],
                training_seconds=metadata['elapsed_seconds'], evaluation_dir=str((directory/'evaluation').relative_to(PROJECT_ROOT)),
                source_sha256=metadata['source_sha256'], versions=metadata['versions'])


def summarize(output, manifest):
    frames = []
    baselines = None
    for run in manifest:
        frame = pd.read_csv(PROJECT_ROOT / run['evaluation_dir'] / 'annual.csv')
        baseline = frame[frame.policy != 'ppo'].reset_index(drop=True)
        if baselines is None:
            baselines = baseline
        else:
            pd.testing.assert_frame_equal(baseline, baselines, check_exact=False, atol=1e-7, rtol=1e-10)
        frames.append(frame[frame.policy == 'ppo'].assign(training_seed=run['training_seed']))
    annual = pd.concat(frames, ignore_index=True).drop(columns='seed')
    if annual.groupby('training_seed').size().tolist() != [13] * 5:
        raise ValueError('Expected five seeds with 13 years each')
    metrics = ['objective_return','full_return','flood_days','low_water_days','deficit_million_m3']
    seed_means = annual.groupby('training_seed')[metrics].mean()
    seed_means['training_seconds'] = [r['training_seconds'] for r in sorted(manifest,key=lambda r:r['training_seed'])]
    aggregate = seed_means[metrics].agg(['mean','std','min','max']).T
    # SD is between five annual-test-averaged learner seeds, not between 65 independent observations.
    paired = []
    for seed, rows in annual.groupby('training_seed'):
        scores = rows.set_index('year').objective_return
        for policy, baseline in baselines.groupby('policy'):
            difference = scores - baseline.set_index('year').objective_return
            paired.append(dict(training_seed=seed, baseline=policy,
                               mean_objective_advantage=difference.mean(),
                               years_with_higher_objective=int((difference > 0).sum()),
                               total_years=len(difference)))
    paired = pd.DataFrame(paired)
    annual.to_csv(output / 'annual_ppo.csv', index=False)
    seed_means.to_csv(output / 'seed_means.csv')
    aggregate.to_csv(output / 'aggregate_across_seeds.csv')
    paired.to_csv(output / 'paired_comparisons.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12,5),layout='constrained')
    names = ['Random','Mean','Max','EAD','PPO']
    means = [baselines[baselines.policy==p].objective_return.mean() for p in ('random','mean','max','ead')]
    means.append(seed_means.objective_return.mean())
    axes[0].barh(names, means, color=['#bbc7d1']*4+['#267ead'])
    axes[0].scatter(seed_means.objective_return, [4]*5, color='#15354f', zorder=3,
                    label='Each dot: one seed, averaged over 13 years')
    axes[0].invert_yaxis();axes[0].set(xlabel='Mean annual objective return (higher is better)',title='Fixed rule references and five trained PPO policies');axes[0].legend(loc='lower left',fontsize=8)
    years = sorted(annual.year.unique())
    curves = annual.pivot(index='training_seed',columns='year',values='objective_return')[years]
    for seed, row in curves.iterrows():
        axes[1].plot(range(13),row.to_numpy(),marker='o',markersize=3,alpha=.65,label=str(seed))
    axes[1].set_xticks(range(13),years,rotation=60)
    axes[1].set(xlabel='Original held-out year',ylabel='Annual objective return',title='Same years; different PPO training seeds')
    axes[1].legend(title='Training seed',fontsize=8,ncol=2)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(output / 'seed_study.png',dpi=170);plt.close(fig)
    print(seed_means.round(2).to_string(),flush=True)
    print('Across-seed summaries:\n'+aggregate.round(2).to_string(),flush=True)
    return seed_means, aggregate, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2,choices=[1,2,3,4])
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/reproduction/seeds')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        manifest = list(pool.map(lambda seed:run_seed(seed,args.output),SEEDS))
    if len({r['source_sha256'] for r in manifest}) != 1:
        raise ValueError('Training source differs between seeds')
    if len({json.dumps(r['versions'],sort_keys=True) for r in manifest}) != 1:
        raise ValueError('Training library versions differ between seeds')
    record=dict(seeds=list(SEEDS),runs=manifest,workers=args.workers,
                learner_seed_varies=True,environment_year_sampling_seed=123,evaluation_random_seed=123,
                scope='Fixed configuration, CPU/one thread per run, original training/test memberships. Only learner RNG seed varies; hydrological training-year sampling order is held fixed.',
                selection='Final 200,000-step checkpoint for every seed; all seeds retained',
                uncertainty='SD/min/max across five seed means, each averaged over the same 13 years; years and seeds are not 65 independent replicates')
    (args.output/'study.json').write_text(json.dumps(record,indent=2)+'\n')
    summarize(args.output,manifest)


if __name__=='__main__':
    main()
