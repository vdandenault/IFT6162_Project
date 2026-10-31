"""Run the authors' fixed PPO training configuration on their original split."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, UPSTREAM_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/reproduction/training')
    args = parser.parse_args()
    import torch
    from gym4real.algorithms.dam.ppo_skrl import train_ppo
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(training_timesteps=args.steps, mini_batches=32, learning_epochs=10,
                  rollouts=2048, discount_factor=0.995, ent_coef=0.0,
                  learning_rate=8e-6, net_arch=[16,16], initial_log_std=-0.5,
                  seed=args.seed, normalize_observations=True, save_dir=str(args.output))
    metadata = dict(config=config, device='cpu', threads=args.threads,
                    protocol='upstream', environment_seed=123, calendar='original 365-step behavior including leap years',
                    observations='original uncorrected terminal observation, as in upstream training',
                    train_years=list(settings('train', protocol='upstream')['inflow']),
                    test_years=list(settings('test', protocol='upstream')['inflow']),
                    versions={p:version(p) for p in ['torch','skrl','gymnasium','numpy']},
                    platform=platform.platform(), python=platform.python_version(),
                    source_sha256=hashlib.sha256((UPSTREAM_ROOT/'gym4real/algorithms/dam/ppo_skrl.py').read_bytes()).hexdigest(),
                    checkpoint_selection='Final requested training step, not best test score',
                    notes=['Exact source function/config; CPU instead of CUDA; one training seed.',
                           'Source scheduler changes learning rate from its initial value.',
                           'Source function evaluates on its original test split only after training.'])
    (args.output/'training_run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    started=time.perf_counter()
    train_ppo(settings('train', protocol='upstream'), config,
              settings('test', protocol='upstream'), device='cpu')
    metadata['elapsed_seconds']=time.perf_counter()-started
    checkpoints=list(args.output.rglob(f'agent_{args.steps}.pt'))
    if len(checkpoints)!=1:
        raise RuntimeError(f'Expected one final checkpoint, found {checkpoints}')
    metadata['final_checkpoint']=str(checkpoints[0].relative_to(PROJECT_ROOT))
    metadata['checkpoint_sha256']=hashlib.sha256(checkpoints[0].read_bytes()).hexdigest()
    (args.output/'training_run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Final checkpoint:',checkpoints[0])
    print('Training seconds:',metadata['elapsed_seconds'])


if __name__=='__main__':
    main()
