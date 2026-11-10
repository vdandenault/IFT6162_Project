"""Frozen 2x2 MPC service-precision/tie experiment at both 1995 storage starts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, DAM_CONFIG, DAM_DATA
from .carryover_check import BoundaryHistory, restore_branch, restore_memory, take_step, FIELDS
from .precision_mpc import PrecisionMPC
from .protected_mpc import CausalForecastBank, ProtectedMPCConfig
from .model import PhysicalWeights
from .outlet import OutletSpec, OutletDomainError
from .low_water import LowWaterRule
from .reserve_use_check import save, verify, inventory_trace

DEFAULT = PROJECT_ROOT / 'results/simulator/precision_check_v1'
REFERENCE = PROJECT_ROOT / 'results/simulator/reserve_use_check_v2'
BOUNDARY = PROJECT_ROOT / 'results/simulator/carryover_check_v1/boundary.json'
VARIANTS = (('original', False, False), ('precision', True, False),
            ('ties', False, True), ('combined', True, True))


def make_controller(training, bank, precision, ties):
    physical = {k: training[k] for k in ['lake_params', 'action', 'flood_level', 'starving_level']}
    return PrecisionMPC(physical, bank, precision_safe=precision, conservative=ties,
        forecast='conditional', ema_coefficient=training['exponential_average_coeff'])


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = list(Path(__file__).parent.glob('*.py')) + [BOUNDARY, REFERENCE/'plan.json',
        REFERENCE/'summary.csv', PROJECT_ROOT/'src/lake_como/environment.py',
        PROJECT_ROOT/'src/lake_como/hourly_telemetry.py', PROJECT_ROOT/'src/lake_como/splits.py',
        PROJECT_ROOT/'configs/splits.json', PROJECT_ROOT/'pyproject.toml', PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py', DAM_CONFIG/'lake.yaml', DAM_CONFIG/'world_train.yaml',
        DAM_CONFIG/'world_test.yaml', DAM_DATA/'MEF.csv']
    paths += [DAM_DATA/s/f'water_{s}_{p}.csv' for s in ['inflow', 'demand'] for p in ['train', 'test']]
    for storage in ['low', 'high']:
        record_path = REFERENCE/('causal_'+storage)/'run.json'
        record = json.loads(record_path.read_text())
        paths += [record_path, Path(record['data_directory'])/'daily.csv',
                  Path(record['data_directory'])/'decisions.jsonl']
    cases = [dict(name=f'{variant}_{storage}', variant=variant, storage=storage,
                  precision_safe=precision, conservative_ties=ties)
             for variant, precision, ties in VARIANTS for storage in ['low', 'high']]
    save(root/'plan.json', dict(cases=cases,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        fit_years=list(range(1946, 1988)), evaluation_years=[1995],
        mpc=asdict(ProtectedMPCConfig()), weights=asdict(PhysicalWeights()),
        outlet=asdict(OutletSpec()), rule=asdict(LowWaterRule()),
        precision='Replace nearest float32 service with smallest sufficient float32 request, capped at representable action ceiling. No extra volume margin.',
        tie_rule='Exact finite cost minima only: smallest first request, then smallest sum of requests, then original candidate order. No cost tolerance.',
        held_fixed='Original two-year clock, 1995 initial states/history/carried plan, prior-only causal target and inflow forecasts, hourly physics, costs, grid and search structure.',
        outcomes='Full-year cost, shortage, flood/low-water days and severity, environmental shortage, ending inventory, early river flow and marginal storage accounting.',
        caveats='Selected validation case and one forecast seed; conditional boundary-stock interventions are not free-water policies. Tie preference is not a safety guarantee or general MPC improvement.',
        test_evaluated=False, model_or_costs_changed=False, default_changed=False))


def run_case(root, spec, training, bank, boundary):
    folder = root/spec['name']
    folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():
        return
    attempt = 1
    while (folder/f'attempt_{attempt}').exists():
        attempt += 1
    data = folder/f'attempt_{attempt}'
    data.mkdir()
    history = BoundaryHistory(**boundary['common_history'])
    storage = boundary[spec['storage']+'_storage_m3']
    env, obs, info = restore_branch(bank, history, storage)
    initial_obs = obs.tolist()
    controller = make_controller(training, bank, spec['precision_safe'], spec['conservative_ties'])
    restore_memory(controller, history)
    records, failure = [], None
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(365):
                obs, info, row, term, trunc = take_step(env, controller, obs, info)
                records.append(row)
                log.write(json.dumps(controller.diagnostics[-1], allow_nan=False)+'\n')
                log.flush()
                if len(records) % 100 == 0:
                    print(spec['name'], len(records), 'days', flush=True)
                if term or trunc:
                    break
        except OutletDomainError as error:
            failure = dict(error=str(error), elapsed_hour=env.hour)
        finally:
            final_storage = env.state.storage
            env.close()
    frame = pd.DataFrame(records, columns=FIELDS)
    frame.to_csv(data/'daily.csv', index=False)
    complete = len(frame) == 365 and failure is None
    if spec['variant'] == 'original' and complete:
        record = json.loads((REFERENCE/('causal_'+spec['storage'])/'run.json').read_text())
        original = pd.read_csv(Path(record['data_directory'])/'daily.csv', float_precision='round_trip')
        np.testing.assert_array_equal(frame.to_numpy(float), original[list(FIELDS)].to_numpy(float))
    save(folder/'run.json', dict(**spec, complete=complete, observed_days=len(frame), failure=failure,
        data_directory=str(data), initial_observation=initial_obs,
        initial_storage_m3=storage, final_storage_m3=final_storage,
        annual_cost=float(-frame.reward.sum()) if complete else None,
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6),
        delivered_target_million_m3=float(frame.delivered_target_m3.sum()/1e6),
        remaining_river_million_m3=float(frame.remaining_river_m3.sum()/1e6),
        first_20_days_river_million_m3=float(frame.remaining_river_m3.iloc[:20].sum()/1e6),
        environmental_shortage_million_m3=float(frame.environmental_shortage_m3.sum()/1e6),
        flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),
        low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
        flood_days=int((frame.flood_excess_m_hours > 0).sum()),
        low_water_days=int((frame.low_water_excess_m_hours > 0).sum()),
        minimum_level_m=float(min(frame.minimum_level_m.min(), frame.level_before_m.min())),
        maximum_level_m=float(max(frame.maximum_level_m.max(), frame.level_before_m.max())),
        maximum_balance_error_m3=float(frame.balance_error_m3.abs().max()) if len(frame) else None,
        recovery_days=sum(d['domain_recovery'] for d in controller.diagnostics),
        tie_changed_selection_days=sum(d['tie_changed_selection'] for d in controller.diagnostics)))
    print('FINISHED', spec['name'], 'cost', -frame.reward.sum(), 'failure', failure, flush=True)


def aggregate(root, plan, training):
    records, frames, decisions = {}, {}, {}
    for spec in plan['cases']:
        name = spec['name']
        record = json.loads((root/name/'run.json').read_text())
        records[name] = record
        data = Path(record['data_directory'])
        frames[name] = pd.read_csv(data/'daily.csv', float_precision='round_trip')
        decisions[name] = [json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines()]
    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ['failure', 'initial_observation']} for r in records.values()])
    summary.to_csv(root/'summary.csv', index=False)
    if not all(r['complete'] for r in records.values()):
        save(root/'run.json', dict(complete=True, all_branches_complete=False,
            interpretation='Incomplete branches: suppress full-year paired attribution.', test_evaluated=False))
        return
    base = frames['original_low']
    exogenous = ['elapsed_hour', 'year', 'model_day', 'inflow_m3_s', 'target_m3_s', 'past_target_ema', 'past_inflow_m3_s']
    for name, frame in frames.items():
        np.testing.assert_array_equal(frame[exogenous], base[exogenous])
        assert set(frame.year) == {1995}
        np.testing.assert_allclose(-frame.reward, frame.unmet_target_m3/1e6 + frame.flood_excess_m_hours
                                  + frame.low_water_excess_m_hours, rtol=0, atol=1e-12)
        for original, changed in zip(decisions['original_low'], decisions[name], strict=True):
            np.testing.assert_array_equal(original['forecast_inflows'], changed['forecast_inflows'])
            np.testing.assert_array_equal(original['forecast_target'], changed['forecast_target'])
    for storage in ['low', 'high']:
        for variant, _, _ in VARIANTS:
            np.testing.assert_array_equal(records['original_'+storage]['initial_observation'],
                                          records[variant+'_'+storage]['initial_observation'])
    budgets, paired = [], []
    for variant, _, _ in VARIANTS:
        delta = inventory_trace(frames[variant+'_low'], frames[variant+'_high'], training['lake_params']['surface'])
        delta.to_csv(root/(variant+'_inventory_trace.csv'), index=False)
        budgets.append(dict(variant=variant,
            extra_initial_million_m3=float(delta.extra_storage_before_m3.iloc[0]/1e6),
            extra_delivery_million_m3=float(delta.cumulative_extra_delivery_m3.iloc[-1]/1e6),
            extra_remaining_river_million_m3=float(delta.cumulative_extra_remaining_river_m3.iloc[-1]/1e6),
            extra_final_storage_million_m3=float(delta.extra_storage_after_m3.iloc[-1]/1e6)))
        for storage in ['low', 'high']:
            original = records['original_'+storage]
            changed = records[variant+'_'+storage]
            paired.append(dict(variant=variant, storage=storage,
                cost_minus_original=changed['annual_cost']-original['annual_cost'],
                final_storage_minus_original_million_m3=(changed['final_storage_m3']-original['final_storage_m3'])/1e6,
                first20_river_minus_original_million_m3=changed['first_20_days_river_million_m3']-original['first_20_days_river_million_m3']))
    pd.DataFrame(budgets).to_csv(root/'marginal_inventory.csv', index=False)
    pd.DataFrame(paired).to_csv(root/'paired.csv', index=False)
    save(root/'run.json', dict(complete=True, all_branches_complete=True,
        original_replays_bitwise_match_saved_trajectories=True,
        all_forecasts_and_exogenous_history_identical=True, cost_identity_and_marginal_ledgers_verified=True,
        test_evaluated=False, model_or_costs_changed=False, default_changed=False,
        interpretation='One selected validation case, two conditional storage starts, one forecast seed. No general controller promotion.'))
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(budgets).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--stage', choices=['all', 'prepare', 'run', 'aggregate'], default='all')
    args = parser.parse_args()
    root = args.output
    if args.stage == 'prepare' or (args.stage == 'all' and not root.exists()):
        prepare(root)
    plan = verify(root)
    training = settings('train')
    if args.stage in ['all', 'run']:
        bank = CausalForecastBank(training['inflow'], training['demand'], before_year=1988)
        boundary = json.loads(BOUNDARY.read_text())
        for spec in plan['cases']:
            run_case(root, spec, training, bank, boundary)
        verify(root)
    if args.stage in ['all', 'aggregate']:
        aggregate(root, plan, training)
    verify(root)


if __name__ == '__main__':
    main()
