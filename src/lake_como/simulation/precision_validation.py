"""Paired two-year validation of precision/tie MPC over fixed forecast seeds.

No tuning, final-test evaluation or default promotion. Seed repetitions share
historical forcing; summaries keep seed sensitivity separate from hydrology.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, DAM_CONFIG, DAM_DATA
from .carryover_check import take_step, FIELDS
from .low_water import make_protected_outlet_env, LowWaterRule
from .model import PhysicalWeights
from .outlet import OutletSpec, OutletDomainError
from .precision_mpc import PrecisionMPC
from .protected_mpc import CausalForecastBank, ProtectedMPCConfig
from .reserve_use_check import save, verify as verify_sources
from .target_year_check import validation_pairs, annual_metrics

DEFAULT = PROJECT_ROOT/'results/simulator/precision_validation_v1'
REFERENCE = PROJECT_ROOT/'results/simulator/target_year_check_v1'
SEEDS = (901, 902, 903, 904, 905)
ARMS = ('original', 'combined')
SCOPES = {
    'all_validation': tuple(range(1988, 1998)),
    'diagnosed_1994_1995': (1994, 1995),
    'other_validation': tuple(y for y in range(1988, 1998) if y not in (1994, 1995)),
}
METRICS = ('annual_cost', 'unmet_target_million_m3', 'environmental_shortage_million_m3',
           'flood_excess_m_hours', 'low_water_excess_m_hours', 'flood_days', 'low_water_days',
           'end_storage_million_m3')


def specs():
    return [dict(name=f'{a}_{b}__seed{seed}__{arm}', years=[a, b], seed=seed, arm=arm)
            for a, b in validation_pairs() for seed in SEEDS for arm in ARMS]


def make_controller(training, bank, arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Declared arm and forecast seed required')
    physical = {k: training[k] for k in ['lake_params', 'action', 'flood_level', 'starving_level']}
    return PrecisionMPC(physical, bank, precision_safe=arm == 'combined', conservative=arm == 'combined',
        forecast='conditional', config=ProtectedMPCConfig(seed=seed),
        ema_coefficient=training['exponential_average_coeff'])


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = list(Path(__file__).parent.glob('*.py')) + [
        PROJECT_ROOT/'src/lake_como/environment.py', PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/lake_como/splits.py', PROJECT_ROOT/'configs/splits.json',
        PROJECT_ROOT/'pyproject.toml', PROJECT_ROOT/'uv.lock', DAM_CONFIG/'lake.py',
        DAM_CONFIG/'lake.yaml', DAM_CONFIG/'world_train.yaml', DAM_CONFIG/'world_test.yaml',
        DAM_DATA/'MEF.csv', REFERENCE/'plan.json', REFERENCE/'annual.csv',
        PROJECT_ROOT/'results/simulator/precision_check_v1/plan.json',
        PROJECT_ROOT/'results/simulator/precision_check_v1/summary.csv',
    ]
    paths += [DAM_DATA/s/f'water_{s}_{part}.csv' for s in ['inflow', 'demand'] for part in ['train', 'test']]
    for a, b in validation_pairs():
        path = REFERENCE/f'{a}_{b}__ema_restricted/run.json'
        record = json.loads(path.read_text())
        paths += [path, Path(record['data_directory'])/'daily.csv', Path(record['data_directory'])/'decisions.jsonl']
    training = settings('train')
    quartiles = np.quantile([np.mean(training['inflow'][y][:365]) for y in sorted(training['inflow'])], [.25, .75])
    save(root/'plan.json', dict(cases=specs(), seeds=list(SEEDS), arms=list(ARMS),
        scopes={k: list(v) for k, v in SCOPES.items()},
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        training_years=list(range(1946, 1988)), training_inflow_quartiles_m3_s=quartiles.tolist(),
        mpc=asdict(ProtectedMPCConfig()), weights=asdict(PhysicalWeights()),
        outlet=asdict(OutletSpec()), rule=asdict(LowWaterRule()),
        protocol='Five nonoverlapping two-year validation episodes, reset at .35 m with zero target EMA and prior inflow climatology; carry physical state, history and plan through the boundary. First 365 samples per model year unchanged.',
        seeds_protocol='Five declared forecast seeds, seed + elapsed hour at every decision; environmental reset seed fixed at 901. All five included; no best-seed selection.',
        intervention='Only precision-safe service proposals and conservative exact-cost ties, together, versus both disabled. Strict primary cost unchanged; no near-tie tolerance.',
        fit='All forecasts fitted only on 1946–1987, frozen for every block; no final test control or fitting.',
        outcomes='Paired full-year and full-block costs, physical shortages/exposures, annual ending stock, regime and episode position. Separate diagnosed 1994–1995 from other validation years.',
        failures='No full-year or block score for incomplete trajectories; required-scope averages suppressed if any member incomplete. Report completion and matched-prefix costs separately.',
        statistics='Paired differences by seed and block; equal seed weighting. Seeds share forcing, years within each episode are dependent, and validation has been used for development. No p-values or independent-sample confidence claim.',
        promotion='No default change or final-test use as part of this study; evaluate flood/low-water tradeoffs as well as scalar cost.',
        test_evaluated=False, model_or_costs_changed=False, default_changed=False))


def verify(root):
    plan = verify_sources(root)
    if plan['cases'] != specs() or plan['seeds'] != list(SEEDS):
        raise ValueError('Declared cases and seeds must remain fixed')
    return plan


def run_case(root, spec):
    folder = root/spec['name']
    folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():
        return spec['name']+' already recorded'
    attempt = 1
    while (folder/f'attempt_{attempt}').exists():
        attempt += 1
    data = folder/f'attempt_{attempt}'
    data.mkdir()
    training = settings('train')
    bank = CausalForecastBank(training['inflow'], training['demand'], before_year=1988)
    env = make_protected_outlet_env('validation', tuple(spec['years']))
    env.fallback = bank.initial_last_inflow
    obs, info = env.reset(seed=901)
    initial_obs = obs.tolist()
    initial_storage = env.state.storage
    controller = make_controller(training, bank, spec['arm'], spec['seed'])
    records, failure = [], None
    started = time.monotonic()
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(730):
                attempted_hour = env.hour
                obs, info, row, term, trunc = take_step(env, controller, obs, info)
                records.append(row)
                log.write(json.dumps(controller.diagnostics[-1], allow_nan=False)+'\n')
                log.flush()
                if term or trunc:
                    break
        except OutletDomainError as error:
            failure = dict(error=str(error), attempted_decision_hour=attempted_hour, elapsed_hour=env.hour)
            if controller.diagnostics:
                save(data/'failed_decision.json', controller.diagnostics[-1])
        finally:
            final_storage = env.state.storage
            env.close()
    frame = pd.DataFrame(records, columns=FIELDS)
    frame.to_csv(data/'daily.csv', index=False)
    complete = len(frame) == 730 and failure is None
    if spec['arm'] == 'original' and spec['seed'] == 901 and complete:
        a, b = spec['years']
        ref = json.loads((REFERENCE/f'{a}_{b}__ema_restricted/run.json').read_text())
        old = pd.read_csv(Path(ref['data_directory'])/'daily.csv', float_precision='round_trip')
        common = [k for k in old.columns if k in frame.columns]
        np.testing.assert_array_equal(frame[common].to_numpy(float), old[common].to_numpy(float))
    save(folder/'run.json', dict(**spec, complete=complete, expected_days=730, observed_days=len(frame),
        data_directory=str(data), failure=failure, initial_observation=initial_obs,
        initial_storage_m3=initial_storage,
        final_storage_m3=final_storage,
        two_year_cost=float(-frame.reward.sum()) if complete else None,
        observed_prefix_cost=float(-frame.reward.sum()) if len(frame) else None,
        domain_recoveries=sum(d['domain_recovery'] for d in controller.diagnostics),
        tie_changed_selection_days=sum(d['tie_changed_selection'] for d in controller.diagnostics),
        elapsed_seconds=time.monotonic()-started))
    return f"{spec['name']}: {len(frame)}/730 days; cost={-frame.reward.sum():.3f}; failure={failure}"


def annual_records(frame, spec, quartiles, area):
    rows = annual_metrics(frame, spec['years'], spec['arm'], quartiles)
    for row in rows:
        group = frame.loc[frame.year.eq(row['year'])]
        row.update(seed=spec['seed'], name=spec['name'],
            end_storage_million_m3=float(area*(group.level_m.iloc[-1]+.5)/1e6) if row['complete'] else None,
            delivered_target_million_m3=float(group.delivered_target_m3.sum()/1e6) if row['complete'] else None,
            remaining_river_million_m3=float(group.remaining_river_m3.sum()/1e6) if row['complete'] else None)
    return rows


def paired_annual(annual):
    keys = ['seed', 'block_start', 'year', 'position_in_pair']
    before = annual[annual.arm.eq('original')].set_index(keys)
    after = annual[annual.arm.eq('combined')].set_index(keys)
    if not before.index.equals(after.index):
        after = after.reindex(before.index)
    paired = before[[]].copy()
    paired['complete'] = before.complete.fillna(False).astype(bool) & after.complete.fillna(False).astype(bool)
    for metric in METRICS:
        paired['original_'+metric] = before[metric]
        paired['combined_'+metric] = after[metric]
        paired['delta_'+metric] = (after[metric]-before[metric]).where(paired.complete)
    return paired.reset_index()


def scope_summary(paired, years, seeds):
    selected = paired[paired.year.isin(years) & paired.seed.isin(seeds)]
    expected = {(seed, year) for seed in seeds for year in years}
    present = set(zip(selected.seed, selected.year))
    complete = len(selected) == len(expected) and present == expected and bool(selected.complete.all())
    result = dict(expected_pairs=len(expected), complete_pairs=int(selected.complete.sum()),
        all_required_complete=complete,
        wins=int((selected.delta_annual_cost < -1e-8).sum()),
        ties=int((selected.delta_annual_cost.abs() <= 1e-8).sum()),
        losses=int((selected.delta_annual_cost > 1e-8).sum()))
    for metric in METRICS:
        for label in ['original', 'combined', 'delta']:
            result['mean_'+label+'_'+metric] = float(selected[label+'_'+metric].mean()) if complete else None
    return result


def aggregate(root, plan):
    annual, blocks, failures, prefixes = [], [], [], []
    training = settings('train')
    area = training['lake_params']['surface']
    for a, b in validation_pairs():
        for seed in SEEDS:
            matched_frames, matched_decisions = {}, {}
            for arm in ARMS:
                name = f'{a}_{b}__seed{seed}__{arm}'
                record = json.loads((root/name/'run.json').read_text())
                data = Path(record['data_directory'])
                frame = pd.read_csv(data/'daily.csv', float_precision='round_trip')
                decisions = [json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines()]
                spec = dict(name=name, years=[a, b], seed=seed, arm=arm)
                annual.extend(annual_records(frame, spec, plan['training_inflow_quartiles_m3_s'], area))
                blocks.append({k: record[k] for k in ['name', 'seed', 'arm', 'complete', 'observed_days',
                    'two_year_cost', 'final_storage_m3', 'domain_recoveries', 'tie_changed_selection_days']}
                    | dict(block_start=a))
                if record['failure']:
                    failures.append(dict(name=name, **record['failure']))
                matched_frames[arm] = frame
                matched_decisions[arm] = decisions
                if len(frame):
                    np.testing.assert_allclose(-frame.reward, frame.unmet_target_m3/1e6 + frame.flood_excess_m_hours
                        + frame.low_water_excess_m_hours, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(area*(frame.level_m-frame.level_before_m),
                        (frame.inflow_m3_s-frame.release_m3_s)*86400, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(frame.release_m3_s*86400,
                        frame.delivered_target_m3+frame.remaining_river_m3, rtol=0, atol=1e-6)
                if record['complete']:
                    assert len(decisions) == 730
                    assert decisions[365]['elapsed_hours'] == 8760 and decisions[365]['shifted_previous_plan']
                    assert decisions[365]['horizon_days'] == 7
                    assert decisions[-1]['horizon_days'] == 1
                    assert frame.year.tolist() == [a]*365+[b]*365
                    assert frame.level_before_m.iloc[365] == frame.level_m.iloc[364]
            original, combined = (matched_frames[arm] for arm in ARMS)
            n = min(len(original), len(combined))
            exogenous = ['elapsed_hour', 'year', 'model_day', 'inflow_m3_s', 'target_m3_s', 'past_target_ema', 'past_inflow_m3_s']
            np.testing.assert_array_equal(original[exogenous].iloc[:n], combined[exogenous].iloc[:n])
            for old, new in zip(matched_decisions['original'][:n], matched_decisions['combined'][:n], strict=True):
                np.testing.assert_array_equal(old['forecast_inflows'], new['forecast_inflows'])
                np.testing.assert_array_equal(old['forecast_target'], new['forecast_target'])
            prefixes.append(dict(block_start=a, seed=seed, matched_days=n, full_block=n == 730,
                cost_delta=float(-combined.reward.iloc[:n].sum()+original.reward.iloc[:n].sum()) if n else None))
    annual = pd.DataFrame(annual)
    annual.to_csv(root/'annual.csv', index=False)
    blocks = pd.DataFrame(blocks)
    blocks.to_csv(root/'blocks.csv', index=False)
    paired = paired_annual(annual)
    paired.to_csv(root/'paired_annual.csv', index=False)
    block_pairs = []
    for (seed, block_start), group in paired.groupby(['seed', 'block_start'], sort=True):
        complete = len(group) == 2 and bool(group.complete.all())
        block_pairs.append(dict(seed=int(seed), block_start=int(block_start), complete=complete,
            cost_delta=float(group.delta_annual_cost.sum()) if complete else None,
            original_cost=float(group.original_annual_cost.sum()) if complete else None,
            combined_cost=float(group.combined_annual_cost.sum()) if complete else None,
            final_stock_delta_million_m3=float(group.loc[group.position_in_pair.eq(2), 'delta_end_storage_million_m3'].iloc[0]) if complete else None))
    pd.DataFrame(block_pairs).to_csv(root/'paired_blocks.csv', index=False)
    summaries, by_seed, by_position = [], [], []
    for scope, years in SCOPES.items():
        summaries.append(dict(scope=scope, **scope_summary(paired, years, SEEDS)))
        for seed in SEEDS:
            by_seed.append(dict(scope=scope, seed=seed, **scope_summary(paired, years, [seed])))
        for position in [1, 2]:
            subset = [year for year in years if (year-1988)%2+1 == position]
            by_position.append(dict(scope=scope, position_in_pair=position,
                **scope_summary(paired, subset, SEEDS)))
    pd.DataFrame(summaries).to_csv(root/'scope_summary.csv', index=False)
    pd.DataFrame(by_seed).to_csv(root/'seed_summary.csv', index=False)
    pd.DataFrame(by_position).to_csv(root/'episode_position.csv', index=False)
    pd.DataFrame(prefixes).to_csv(root/'matched_prefixes.csv', index=False)
    save(root/'failures.json', failures)
    save(root/'run.json', dict(complete=True, all_arms_complete=bool(blocks.complete.all()),
        expected_episodes=50, completed_episodes=int(blocks.complete.sum()), domain_failures=len(failures),
        paired_forecasts_and_exogenous_history_verified=True, cost_and_water_ledgers_verified=True,
        cross_year_continuity_verified=True, source_hashes_verified=True,
        test_evaluated=False, default_changed=False, model_or_costs_changed=False,
        interpretation='Paired forecast-seed sensitivity on reused validation hydrology. No independent-final-test claim or automatic promotion.'))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--stage', choices=['all', 'prepare', 'run', 'aggregate'], default='all')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    root = args.output
    if args.stage == 'prepare' or (args.stage == 'all' and not root.exists()):
        prepare(root)
    plan = verify(root)
    if args.stage in ['all', 'run']:
        if args.workers == 1:
            for spec in plan['cases']:
                print(run_case(root, spec), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_case, root, spec) for spec in plan['cases']]
                for future in as_completed(futures):
                    print(future.result(), flush=True)
        verify(root)
    if args.stage in ['all', 'aggregate']:
        aggregate(root, plan)
    verify(root)


if __name__ == '__main__':
    main()
