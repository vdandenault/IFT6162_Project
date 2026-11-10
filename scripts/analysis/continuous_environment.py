"""Small training-only continuous-environment check; no reference changes.

Default: 730 constant-release transitions, zero updates.
--train-smoke: fresh SAC, 730 transitions including 64 warmup, 666 updates.
This checks integration/boundaries, not performance or stabilization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.continuous_environment import make_continuous_env
from lake_como.mpc_hindsight import digest
from lake_como.paths import DAM_CONFIG, DAM_DATA, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    name = 'training_smoke_v1' if args.train_smoke else 'continuity_check_v1'
    root = args.output or PROJECT_ROOT / 'results/simulator' / name
    if root.exists() and any(root.iterdir()):
        parser.error('Choose an empty output directory; existing results are preserved')
    root.mkdir(parents=True, exist_ok=True)
    env = make_continuous_env('train', (1946, 1947))
    rows = []
    files = [Path(__file__), PROJECT_ROOT / 'src/lake_como/continuous_environment.py',
             PROJECT_ROOT / 'src/lake_como/environment.py',
             PROJECT_ROOT / 'src/lake_como/inflow_observation.py',
             PROJECT_ROOT / 'configs/splits.json', DAM_CONFIG / 'env.py', DAM_CONFIG / 'lake.py',
             DAM_CONFIG / 'world_train.yaml', DAM_CONFIG / 'lake.yaml', DAM_DATA / 'MEF.csv']
    files += [DAM_DATA / signal / f'water_{signal}_{part}.csv'
              for signal in ['demand', 'inflow'] for part in ['train', 'test']]
    if args.train_smoke:
        files += [PROJECT_ROOT / 'src/lake_como/sac.py', PROJECT_ROOT / 'src/lake_como/sac_loop.py']
    hashes = {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in files}
    declaration = dict(protocol='continuous-model-years-v1', years=[1946, 1947],
        calendar='365 samples per model year; original leap approximation unchanged',
        observation='Existing five features: level, sin/cos season, demand EMA, completed-day inflow',
        steps=730, seed=901, train_smoke=args.train_smoke,
        warmup_steps=64 if args.train_smoke else 0,
        updates=666 if args.train_smoke else 0,
        end_rule='No end at intermediate year; original SAC loop masks final record truncation only',
        normalization='Fixed unit scales [1,1,1,100,100], zero means; no fitted statistics or extra collection',
        purpose='Correctness smoke, not a new performance benchmark; no evaluation or checkpoint promotion',
        source_sha256=hashes, validation_evaluated=False, test_evaluated=False)
    (root / 'plan.json').write_text(json.dumps(declaration, indent=2) + '\n')

    def collect(observation, following, reward, terminated, truncated, info):
        rows.append(dict(step=info['elapsed_steps'], year=info['transition_year'], day=info['transition_day'],
            storage_before_m3=info['storage_before'], storage_m3=info['storage'],
            level_before_m=float(observation[0]), level_m=float(following[0]),
            past_inflow_m3_s=float(observation[-1]), next_past_inflow_m3_s=float(following[-1]),
            reward=float(reward), release_m3_s=info['release'],
            year_boundary=info['year_boundary'], terminated=terminated, truncated=truncated,
            balance_error_m3=info['balance_error_m3']))
    try:
        if args.train_smoke:
            import torch
            from lake_como.sac import SACAgent, SACConfig, ReplayBuffer
            from lake_como.sac_loop import train_steps
            torch.set_num_threads(1)
            agent = SACAgent(SACConfig(observation_dim=5),
                observation_mean=np.zeros(5), observation_scale=[1, 1, 1, 100, 100],
                action_low=env.action_space.low, action_high=env.action_space.high, seed=901)
            replay = ReplayBuffer(1024, 5, 1, seed=901)
            def on_step(event):
                collect(event['observation'], event['next_observation'], event['reward'],
                        event['terminated'], event['truncated'], event['info'])
            result = train_steps(env, agent, replay, steps=730, warmup_steps=64,
                                 batch_size=32, seed=901, on_step=on_step)
            assert result['updates'] == 666
            assert len(result['episodes']) == 1 and result['episodes'][0]['length'] == 730
            assert len(replay) == 730
            agent.save(root / 'smoke_only.pt')
            updates = result['updates']
        else:
            obs, _ = env.reset(seed=901)
            for _ in range(730):
                following, reward, term, trunc, info = env.step(np.array([100.], np.float32))
                collect(obs, following, reward, term, trunc, info)
                obs = following
            updates = 0
        frame = pd.DataFrame(rows)
        assert len(frame) == 730 and frame.year.unique().tolist() == [1946, 1947]
        assert frame.terminated.sum() == 0 and frame.truncated.sum() == 1
        assert not frame.iloc[364].truncated and frame.iloc[729].truncated
        assert frame.iloc[364].storage_m3 == frame.iloc[365].storage_before_m3
        assert frame.balance_error_m3.abs().max() < 1e-6
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
        frame.to_csv(root / 'transitions.csv', index=False)
        for name, expected in hashes.items():
            assert digest(PROJECT_ROOT / name) == expected, name
        report = dict(complete=True, transitions=730, updates=updates,
            year_boundary_preserves_storage=True, intermediate_year_not_terminal=True,
            episode_ends_only_at_record_end=True, finite_values=True,
            max_balance_error_m3=float(frame.balance_error_m3.abs().max()),
            test_evaluated=False, reference_changed=False)
        (root / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
