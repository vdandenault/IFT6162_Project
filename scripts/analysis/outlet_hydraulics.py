"""Bounded training-only outlet ablation, physical checks and fresh SAC smoke."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT, DAM_DATA, DAM_CONFIG
from reservoir_control.systems.lake_como.simulation.outlet import (
    OutletSpec, OutletLake, OutletDomainError, make_outlet_env, OutletTrainingEpisodes,
)
from reservoir_control.systems.lake_como.simulation.protocol import TaskProtocol
from reservoir_control.systems.lake_como.simulation.training import train_operational


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT/'results/simulator/outlet_ablation_v2')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    params = settings('train')
    spec = OutletSpec()
    variants = {name: OutletLake(params['lake_params'], replace(spec, capacity=name))
                for name in ['legacy', 'published_full_open']}
    files = list((PROJECT_ROOT/'src/reservoir_control/systems/lake_como/simulation').glob('*.py'))
    files += [Path(__file__), PROJECT_ROOT/'src/reservoir_control/controllers/sac/agent.py',
              PROJECT_ROOT/'src/reservoir_control/systems/lake_como/environment.py',
              PROJECT_ROOT/'configs/splits.json', DAM_CONFIG/'lake.py', DAM_CONFIG/'lake.yaml']
    files += [DAM_DATA/k/f'water_{k}_{part}.csv'
              for k in ['level', 'inflow', 'demand'] for part in ['train','test']]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    save(root/'plan.json', dict(outlet=asdict(spec), area_comparison_m2=[145900000.,145033000.,145152000.],
         source_sha256=hashes, test_evaluated=False, objective='physical (existing default weights)',
         unchanged='Eight observations, nominal action range, target/reward code, minimum-flow/high-water opening rules.',
         changed='Outlet maximum capacity only in primary comparison; area sensitivity separately.',
         envelope='Declared numerical evaluation range, not certified hydraulic validity; stop outside.',
         interpretation='Source-based research ablation; no independent operational calibration or controller ranking.',
         source='Ranzi et al. 2021, Hydrometry of the Adda River at Lecco, printed pp.15–16; docs/CALIBRATION.md',
         controls=dict(years=[1946,1947], policies=['constant_100','causal_feedback']),
         sac_smoke=dict(steps=96, seed=901, warmup=32, episode_years=1, burn_in_days=30)))

    curves = []
    levels = np.unique(np.r_[np.linspace(-.705, 2.64, 1001), -.6, -.5, -.4, 1.25])
    for name, lake in variants.items():
        for level in levels:
            storage = lake.level_to_storage(level)
            curves.append(dict(variant=name, level_m=level,
                capacity_m3_s=lake.max_release(storage, 1),
                operating_minimum_m3_s=lake.min_release(storage, 1),
                requested_100_actual_m3_s=lake.actual_release(100, storage, 1)))
    pd.DataFrame(curves).to_csv(root/'release_curves.csv', index=False)

    # Raw within-year samples include leap days; date/flow alignment remains
    # conditional. No proxy is relabeled as independently measured discharge.
    # Upstream filenames use a different split from our chronological protocol.
    # Filter each source before any scoring; final project test years stay unused.
    raw = pd.concat([pd.read_csv(DAM_DATA/f'level/water_level_{part}.csv', index_col='year')
                     .loc[lambda frame: frame.index.isin(params['inflow'])]
                     for part in ['train','test']])
    observations = []
    for year in sorted(params['inflow']):
        h = raw.loc[year].dropna().to_numpy(float)
        targets = np.asarray(params['demand'][year], float)
        for i in range(len(h)-1):
            capacities = [lake.max_release(lake.level_to_storage(h[i]), 1) for lake in variants.values()]
            observations.append(dict(year=year, ordinal=i, level_m=h[i], next_level_m=h[i+1],
                target_proxy_m3_s=targets[i], touches_below_minus_half=bool(min(h[i],h[i+1])<-.5),
                legacy_capacity_m3_s=capacities[0], candidate_capacity_m3_s=capacities[1],
                legacy_proxy_above_capacity=bool(targets[i]>capacities[0]+1e-8),
                candidate_proxy_above_capacity=bool(targets[i]>capacities[1]+1e-8)))
    observed = pd.DataFrame(observations)
    observed.to_csv(root/'historical_capacity_consistency.csv', index=False)
    low = observed[observed.touches_below_minus_half]
    historical = dict(transitions=len(observed), low_water_transitions=len(low),
        low_water_positive_proxy=int((low.target_proxy_m3_s>0).sum()),
        low_water_legacy_proxy_above_capacity=int(low.legacy_proxy_above_capacity.sum()),
        low_water_candidate_proxy_above_capacity=int(low.candidate_proxy_above_capacity.sum()),
        all_legacy_proxy_above_capacity=int(observed.legacy_proxy_above_capacity.sum()),
        all_candidate_proxy_above_capacity=int(observed.candidate_proxy_above_capacity.sum()),
        caveat='Daily proxy versus point-level capacity is conditional consistency, not a hydraulic violation certificate.')
    save(root/'historical_summary.json', historical)

    convergence = []
    for name, lake in variants.items():
        for initial in [-.6, -.45, 1.3]:
            storage = lake.level_to_storage(initial)
            ref = lake.integrate_interval(storage, 400, 20, 1, substeps=1536)
            for steps in [24,48,96,192,384]:
                result = lake.integrate_interval(storage, 400, 20, 1, substeps=steps)
                convergence.append(dict(variant=name, initial_level_m=initial, substeps=steps,
                    error_vs_fine_level_mm=abs(result['storage_m3']-ref['storage_m3'])/lake.surface*1000,
                    **result))
    pd.DataFrame(convergence).to_csv(root/'integration_convergence.csv', index=False)

    area_rows = []
    for area in [145900000.,145033000.,145152000.]:
        lake = OutletLake(params['lake_params'], replace(spec, area_m2=area))
        start = lake.level_to_storage(.35)
        result = lake.integrate_interval(start, 100, 120, 1)
        area_rows.append(dict(area_m2=area, initial_level_m=.35,
                             final_level_m=lake.storage_to_level(result['storage_m3']), **result))
    pd.DataFrame(area_rows).to_csv(root/'area_sensitivity.csv', index=False)

    controls = []
    for name in variants:
        for policy in ['constant_100', 'causal_feedback']:
            env = make_outlet_env(years=(1946,1947), outlet=replace(spec,capacity=name))
            obs, info = env.reset(seed=901)
            trace = []; rewards = []; error = None
            try:
                for day in range(730):
                    request = 100 if policy=='constant_100' else np.clip(float(obs[3])+20+150*(float(obs[0])-.35),0,env.model.high)
                    obs, reward, term, trunc, info = env.step(np.array([request],np.float32))
                    trace.extend(info['hourly']); rewards.append(reward)
                    if term or trunc: break
            except OutletDomainError as exc:
                error = str(exc)
            frame = pd.DataFrame(trace)
            frame.to_csv(root/f'control_{name}_{policy}.csv', index=False)
            controls.append(dict(variant=name, policy=policy, completed_days=len(rewards),
                complete=len(rewards)==730, domain_stop=error, return_=sum(rewards),
                maximum_balance_error_m3=float(frame.lake_balance_error_m3.abs().max()),
                minimum_level_m=float(frame.level_m.min()),maximum_level_m=float(frame.level_m.max()),
                below_reference_hours=int(frame.below_active_storage_datum.sum()),
                flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),
                low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum())))
            env.close()
    save(root/'controls.json', controls)

    protocol = TaskProtocol(episode_years=1)
    training = OutletTrainingEpisodes(protocol, outlet=spec)
    try:
        agent, learning, accounting = train_operational(training,steps=96,seed=901,warmup=32)
        assert all(np.isfinite(parameter.detach().numpy()).all() for parameter in agent.parameters())
        pd.DataFrame(learning).to_csv(root/'sac_smoke.csv',index=False)
        agent.save(root/'sac_smoke.pt')
        save(root/'sac_smoke.json',dict(complete=True, **accounting,
             purpose='Finite-gradient/observation/action/reward integration check, not performance evidence.'))
    except OutletDomainError as exc:
        save(root/'sac_smoke.json',dict(complete=False,domain_stop=str(exc),
            agent_transitions=training.agent_transitions,burn_in_transitions=training.burn_in_transitions))
    finally:
        training.close()
    for name, sha in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha, name
    save(root/'run.json',dict(complete=True, source_hashes_verified=True,test_evaluated=False,
         operationally_calibrated=False, controls_complete=all(row['complete'] for row in controls),
         sac_smoke=json.loads((root/'sac_smoke.json').read_text())))
    print(json.dumps(dict(output=str(root),historical=historical,controls=controls,
                         sac=json.loads((root/'sac_smoke.json').read_text())),indent=2))


if __name__=='__main__':
    main()
