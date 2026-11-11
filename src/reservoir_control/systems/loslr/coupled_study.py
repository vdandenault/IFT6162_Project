"""Frozen coupled pilot: rule curve, external MPC, nested bilevel MPC and PPO."""
from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from .coupled import (CONFIG, ROOT, load_protocol, start_states, CoupledSession,
                      operating_cost)
from .data import load_inputs
from .model import OntarioModel
from .native import NativeReference, sha256


def annual_summary(rows, c):
    records = []
    for (controller, seed, year), data in rows.groupby(['controller', 'seed', 'year']):
        if data.quarter_month.tolist() != list(range(1, 49)):
            raise ValueError('Every comparison trajectory requires all 48 periods')
        upstream = (data.ontario_end_m-c['ontario_flood_cap_m']).clip(lower=0)
        downstream = (data.pointe_claire_m-data.downstream_target_m).clip(lower=0)
        low = (c['ontario_low_water_m']-data.ontario_end_m).clip(lower=0)
        records.append(dict(controller=controller, seed=seed, year=year,
            annual_cost=float(data.cost.sum()), annual_score=-float(data.cost.sum()),
            ontario_peak_m=float(data.ontario_end_m.max()),
            pointe_claire_peak_m=float(data.pointe_claire_m.max()),
            ontario_excess_m_periods=float(upstream.sum()),
            pointe_claire_excess_m_periods=float(downstream.sum()),
            ontario_periods_above=int((upstream > 1e-9).sum()),
            pointe_claire_periods_above=int((downstream > 1e-9).sum()),
            low_water_excess_m_periods=float(low.sum()),
            end_level_m=float(data.ontario_end_m.iloc[-1]),
            ottawa_end_storage_m3=float(data.ottawa_storage_m3.iloc[-1]),
            ottawa_min_storage_m3=float(data.ottawa_storage_m3.min()),
            ottawa_max_storage_m3=float(data.ottawa_storage_m3.max()),
            ottawa_service_shortfall_million_m3=float(data.follower_service_shortfall_m3_s.sum()*365*86400/48/1e6),
            ottawa_safety_adjustments=int((data.ottawa_safety_adjustment_m3_s.abs()>1e-7).sum()),
            max_balance_error_m3=float(data.ottawa_balance_error_m3.abs().max()),
            max_kkt_residual=float(data.follower_kkt_residual.max()),
            follower_solver_warnings=int((~data.follower_solver_success.astype(bool)).sum()),
            fallback_periods=int(data.fallback.sum()),
            mean_abs_release_change_m3_s=float(data.release_change_m3_s.mean()),
            total_runtime_seconds=float(data.runtime_seconds.sum())))
    return pd.DataFrame(records)


def evaluate(model, frame, states, c, output, agents):
    from .gym_env import OntarioOttawaEnv
    records = []
    for year in c['pilot_years']:
        for controller in ('rule_curve', 'external_mpc', 'bilevel_mpc'):
            sim = CoupledSession(model, frame, states[year], c)
            for _ in range(48):
                started = perf_counter()
                decision, time = sim.information()
                if controller == 'rule_curve':
                    schedule = sim.adapter.neutral(sim.session.state, decision, time)
                    plan = sim.adapter.evaluate(sim.session.state, sim.storage_m3, decision, time, schedule)
                else:
                    plan = sim.adapter.plan(sim.session.state, sim.storage_m3, decision, time,
                                            anticipating=controller == 'bilevel_mpc')
                    schedule = plan.requests
                row = sim.step(schedule)
                row.update(controller=controller, seed=-1, fallback=not plan.feasible,
                           runtime_seconds=perf_counter()-started)
                records.append(row)
            print(f'{controller} {year}: complete', flush=True)
        for seed, agent in agents.items():
            env = OntarioOttawaEnv(model, frame, states, c, [year])
            obs, _ = env.reset(seed=seed, options={'year': year})
            for _ in range(48):
                started = perf_counter()
                action = agent.act(obs, deterministic=True)[0]
                decision, time = env.simulation.information()
                plan = env.simulation.adapter.evaluate(env.simulation.session.state,
                    env.simulation.storage_m3, decision, time, env.schedule(action))
                obs, _, _, _, row = env.step(action)
                row.update(controller='ppo', seed=seed, fallback=not plan.feasible,
                           runtime_seconds=perf_counter()-started)
                records.append(row)
            print(f'PPO seed {seed} {year}: complete', flush=True)
    trajectories = pd.DataFrame(records)
    trajectories.to_csv(output/'trajectories.csv', index=False)
    annual = annual_summary(trajectories, c)
    annual.to_csv(output/'annual.csv', index=False)
    return annual


def train(model, frame, states, c, output):
    import torch
    from reservoir_control.controllers.ppo.agent import PPOAgent, PPOConfig
    from .gym_env import OntarioOttawaEnv
    torch.set_num_threads(1)
    config = PPOConfig(observation_dim=64, hidden=32, rollout=256, epochs=4,
                       batch_size=64, learning_rate=1e-4, reward_scale=1.)
    years = list(range(c['training_years'][0], c['training_years'][1]+1))
    agents = {}
    for seed in c['seeds']:
        folder = output/f'ppo_seed_{seed}'; folder.mkdir()
        env = OntarioOttawaEnv(model, frame, states, c, years)
        obs, _ = env.reset(seed=seed)
        agent = PPOAgent(config, action_low=-2000., action_high=2000., seed=seed)
        logs = []
        while agent.steps < c['training_decisions_per_seed']:
            rollout = {k: [] for k in ('observation', 'latent', 'log_probability',
                'value', 'reward', 'next_value', 'terminated', 'truncated')}
            actions, rewards, endings = [], [], 0
            for _ in range(min(config.rollout, c['training_decisions_per_seed']-agent.steps)):
                action, latent, logp, value = agent.act(obs)
                following, reward, terminated, truncated, info = env.step(action)
                values = (obs.copy(), latent, logp, value, reward*config.reward_scale,
                          0. if terminated else agent.value(following), terminated, truncated)
                for key, val in zip(rollout, values, strict=True): rollout[key].append(val)
                actions.append(float(action[0])); rewards.append(reward)
                endings += int(terminated); agent.steps += 1
                obs = env.reset()[0] if terminated or truncated else following
            log = agent.update(rollout)
            log.update(steps=agent.steps, action_min=min(actions), action_max=max(actions),
                       reward_min=min(rewards), reward_max=max(rewards), episode_ends=endings)
            if not all(np.isfinite(v) for v in log.values()):
                raise RuntimeError('Nonfinite training diagnostic')
            logs.append(log)
        pd.DataFrame(logs).to_csv(folder/'training.csv', index=False)
        agent.save(folder/'final.pt')
        # Evaluate the saved endpoint, not an in-memory or selected checkpoint.
        agents[seed] = PPOAgent.load(folder/'final.pt')
        print(f'PPO seed {seed}: {agent.steps} training-only decisions complete', flush=True)
    return agents


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-decisions', type=int,
                        help='Freeze a different budget in this new study plan')
    parser.add_argument('--skip-ppo', action='store_true', help='Controller-only diagnostic, labelled in plan')
    args = parser.parse_args()
    c = load_protocol(args.config)
    if args.training_decisions is not None: c['training_decisions_per_seed'] = args.training_decisions
    if c['training_decisions_per_seed'] < 256: parser.error('At least one 256-step rollout required')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    reference = NativeReference(); frame = load_inputs(reference); model = OntarioModel(reference)
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((ROOT/'src/reservoir_control/systems/ottawa').glob('*.py'))
    sources += list((ROOT/'src/reservoir_control/controllers/mpc').glob('*.py'))
    sources += [ROOT/'src/reservoir_control/controllers/ppo/agent.py', ROOT/'uv.lock', args.config]
    plan = dict(protocol=c, ppo_included=not args.skip_ppo, test_evaluated=False,
                upstream=reference.provenance(),
                sources_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in sources},
                ppo_config=dict(observation_dim=64, hidden=32, rollout=256, epochs=4,
                                batch_size=64, learning_rate=1e-4, reward_scale=1.),
                selection='Final endpoint only; no validation-based checkpoint selection',
                limitations='Synthetic controlled/uncontrolled SLON decomposition; zero-lag routing; finite nested leader search; short PPO pilot')
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    train_years = list(range(c['training_years'][0], c['training_years'][1]+1)) if not args.skip_ppo else []
    states = start_states(model, frame, sorted(set(train_years+c['pilot_years'])))
    agents = {} if args.skip_ppo else train(model, frame, states, c, args.output)
    annual = evaluate(model, frame, states, c, args.output, agents)
    means = annual.groupby('controller').mean(numeric_only=True).drop(columns=['seed', 'year'])
    means.to_csv(args.output/'comparison.csv')
    summary = dict(complete=True, test_evaluated=False, default_changed=False,
                   training_seeds=len(agents), annual_records=len(annual),
                   means=means.to_dict('index'),
                   output_sha256={str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(means[['annual_cost', 'ontario_excess_m_periods', 'pointe_claire_excess_m_periods']].to_string())


if __name__ == '__main__': main()
