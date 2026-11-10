"""Frozen 1993/1997 event replays, matched-state branches and forecast audits.

Uses validation records only. Controller costs, physical rules and defaults stay
fixed; relaxed-capacity and expanded-search checks are offline diagnostics.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT, DAM_CONFIG, DAM_DATA
from .carryover_check import take_step, FIELDS
from .feasibility_audit import audit_forecast
from .low_water import make_protected_outlet_env
from .model import WaterState
from .precision_validation import make_controller, SEEDS, ARMS
from .protected_mpc import CausalForecastBank
from .reserve_use_check import save, verify

DEFAULT = PROJECT_ROOT/'results/simulator/flood_diagnosis_v1'
REFERENCE = PROJECT_ROOT/'results/simulator/precision_validation_v1'
YEARS = (1993, 1997)
LEADS = (14, 7)


def source(year, seed, arm):
    if year not in YEARS or seed not in SEEDS or arm not in ARMS:
        raise ValueError('Declared validation events, seeds and controllers only')
    path = REFERENCE/f'{year-1}_{year}__seed{seed}__{arm}/run.json'
    record = json.loads(path.read_text())
    data = Path(record['data_directory'])
    frame = pd.read_csv(data/'daily.csv', float_precision='round_trip')
    decisions = [json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines()]
    return record, frame, decisions


def event_bounds(year, frames):
    days = sorted({int(day) for f in frames for day in f.loc[f.year.eq(year) & (f.flood_excess_m_hours > 0), 'model_day']})
    if not days or days != list(range(days[0], days[-1]+1)):
        raise ValueError('This diagnostic requires one contiguous union flood event per selected year')
    if days[0]-max(LEADS) < 1 or days[-1]+7 > 365:
        raise ValueError('Declared padding must fit inside the selected validation year')
    return dict(year=year, first_flood_day=days[0], last_flood_day=days[-1], end_day=days[-1]+7,
                anchors=[dict(lead=lead, start_day=days[0]-lead) for lead in LEADS])


def branch_specs(plan):
    rows = []
    for event in plan['events']:
        for seed in SEEDS:
            for anchor in event['anchors']:
                for origin in ARMS:
                    for controller in ARMS:
                        rows.append(dict(year=event['year'], seed=seed, lead=anchor['lead'],
                            start_day=anchor['start_day'], end_day=event['end_day'],
                            storage_origin=origin, controller=controller, memory='original'))
                rows.append(dict(year=event['year'], seed=seed, lead=anchor['lead'],
                    start_day=anchor['start_day'], end_day=event['end_day'],
                    storage_origin='combined', controller='combined', memory='combined'))
    for row in rows:
        row['name'] = f"{row['year']}_seed{row['seed']}_lead{row['lead']}__stock_{row['storage_origin']}__{row['controller']}__memory_{row['memory']}"
    return rows


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = list(Path(__file__).parent.glob('*.py')) + [REFERENCE/'plan.json', REFERENCE/'run.json',
        REFERENCE/'paired_annual.csv', PROJECT_ROOT/'src/lake_como/environment.py',
        PROJECT_ROOT/'src/lake_como/hourly_telemetry.py', PROJECT_ROOT/'src/lake_como/splits.py',
        PROJECT_ROOT/'configs/splits.json', PROJECT_ROOT/'pyproject.toml', PROJECT_ROOT/'uv.lock',
        DAM_CONFIG/'lake.py', DAM_CONFIG/'lake.yaml', DAM_CONFIG/'world_train.yaml',
        DAM_CONFIG/'world_test.yaml', DAM_DATA/'MEF.csv']
    paths += [DAM_DATA/s/f'water_{s}_{p}.csv' for s in ['inflow', 'demand'] for p in ['train', 'test']]
    events = []
    for year in YEARS:
        frames = []
        for seed in SEEDS:
            for arm in ARMS:
                record, frame, decisions = source(year, seed, arm)
                frames.append(frame)
                paths += [REFERENCE/f'{year-1}_{year}__seed{seed}__{arm}/run.json',
                    Path(record['data_directory'])/'daily.csv', Path(record['data_directory'])/'decisions.jsonl']
        events.append(event_bounds(year, frames))
    save(root/'plan.json', dict(events=events, seeds=list(SEEDS), leads=list(LEADS),
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        selection='Union of factual flood days across both controllers/all five seeds, padded by 14 days before and 7 after. Branch also 7 days before first flood; no anchor selected by outcome.',
        matched_state='Four branches per anchor/seed: each controller from each inherited storage. Common original command, exogenous history and original warm plan. Fifth branch uses combined own warm plan as factual/memory control.',
        restoration='Replay saved nominal commands from the original two-year reset to recover exact physical state and unrounded target EMA. Keep original full-record clock/remaining time, never shorten MPC horizon at event end.',
        diagnostic_search='Fixed augmented family: 17 constants; all 2^H low/high schedules; every nine-grid two-stage schedule with cuts 1..H-1; nearest/upward service with every max-first/service-tail cut; saved incumbent. No controller search changes.',
        bound='Offline full-capacity/reserve-cap relaxation ignores nominal request ceiling. Certified monotone hourly map lower-bounds any admissible level. Upper exit proves that forecast scenario cannot stay in range at this state; non-exit is inconclusive.',
        attribution='Conditional two-order storage/controller contrasts, with a separate warm-plan effect. No unique percentage attribution. Events and anchors are selected diagnostics, not a new controller ranking.',
        held_fixed='Prior-only causal forecasts, seed + original elapsed hour, target observations, physical rules, objective weights, daily request cadence and seven-day controller search.',
        test_evaluated=False, model_or_costs_changed=False, default_changed=False))


def forecast_bank():
    training = settings('train')
    return training, CausalForecastBank(training['inflow'], training['demand'], before_year=1988)


def capture(env, decisions):
    ordinal = env.hour//24
    return dict(hour=env.hour, state=asdict(env.state), target_ema=env.demand_ema,
        last_inflow=env.last_inflow, previous_hour=env.hour-24,
        previous_plan=decisions[ordinal-1]['selected_plan'])


def prefix(root, plan):
    folder = root/'snapshots'
    folder.mkdir(exist_ok=True)
    _, bank = forecast_bank()
    for event in plan['events']:
        year = event['year']
        for seed in SEEDS:
            for arm in ARMS:
                path = folder/f'{year}_seed{seed}_{arm}.json'
                if path.exists():
                    continue
                _, frame, decisions = source(year, seed, arm)
                wanted = {(365+a['start_day']-1)*24 for a in event['anchors']}
                wanted |= {d['elapsed_hours'] for d in decisions if d['domain_recovery']}
                env = make_protected_outlet_env('validation', (year-1, year))
                env.fallback = bank.initial_last_inflow
                obs, info = env.reset(seed=901)
                snapshots, replayed = {}, []
                try:
                    for ordinal in range(max(wanted)//24+1):
                        if env.hour in wanted:
                            snapshots[str(env.hour)] = capture(env, decisions)
                        if env.hour == max(wanted):
                            break
                        request = np.float32(frame.request_m3_s.iloc[ordinal])
                        obs, info, row, _, _ = take_step(env, lambda o, i: np.array([request], np.float32), obs, info)
                        replayed.append(list(row.values()))
                    np.testing.assert_array_equal(np.array(replayed), frame[list(FIELDS)].iloc[:len(replayed)].to_numpy(float))
                finally:
                    env.close()
                save(path, dict(year=year, seed=seed, arm=arm, prefix_bitwise_matches=True, snapshots=snapshots))
    print('Exact prefix snapshots complete', flush=True)


def get_snapshot(root, year, seed, origin, hour):
    path = root/'snapshots'/f'{year}_seed{seed}_{origin}.json'
    return json.loads(path.read_text())['snapshots'][str(hour)]


def restore(year, bank, common, storage_snapshot):
    if year not in YEARS or common['hour'] != storage_snapshot['hour'] or common['hour']%24:
        raise ValueError('Aligned declared event clock required')
    for key in ['target_ema', 'last_inflow']:
        if common[key] != storage_snapshot[key]:
            raise ValueError('Shared exogenous history required')
    env = make_protected_outlet_env('validation', (year-1, year))
    env.fallback = bank.initial_last_inflow
    env.reset(seed=901)
    env.hour = common['hour']
    env.state = WaterState(storage_snapshot['state']['storage'], common['state']['command'], 0.)
    env.model.lake.check_domain(env.model.lake.storage_to_level(env.state.storage))
    env.demand_ema, env.last_inflow = common['target_ema'], common['last_inflow']
    return env, env._observation(), env._info()


def restore_plan(controller, snapshot):
    controller.reset()
    controller.previous_hour = snapshot['previous_hour']
    controller.previous_plan = np.asarray(snapshot['previous_plan'], np.float32)


def run_branch(root, spec, training, bank):
    folder = root/spec['name']
    folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():
        return
    attempt = 1
    while (folder/f'attempt_{attempt}').exists():
        attempt += 1
    data = folder/f'attempt_{attempt}'
    data.mkdir()
    hour = (365+spec['start_day']-1)*24
    common = get_snapshot(root, spec['year'], spec['seed'], 'original', hour)
    stock = get_snapshot(root, spec['year'], spec['seed'], spec['storage_origin'], hour)
    memory = get_snapshot(root, spec['year'], spec['seed'], spec['memory'], hour)
    env, obs, info = restore(spec['year'], bank, common, stock)
    initial_obs = obs.tolist()
    controller = make_controller(training, bank, spec['controller'], spec['seed'])
    restore_plan(controller, memory)
    rows, failure = [], None
    expected = spec['end_day']-spec['start_day']+1
    from .outlet import OutletDomainError
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(expected):
                obs, info, row, term, trunc = take_step(env, controller, obs, info)
                rows.append(row)
                log.write(json.dumps(controller.diagnostics[-1], allow_nan=False)+'\n')
                if term or trunc:
                    raise AssertionError('Event boundary must not end the original episode')
        except OutletDomainError as error:
            failure = dict(error=str(error), elapsed_hour=env.hour)
        finally:
            end_storage = env.state.storage
            env.close()
    frame = pd.DataFrame(rows, columns=FIELDS)
    frame.to_csv(data/'daily.csv', index=False)
    complete = len(frame) == expected and failure is None
    factual = (spec['storage_origin'] == spec['controller'] == 'original' or
               spec['storage_origin'] == spec['controller'] == spec['memory'] == 'combined')
    if complete and factual:
        _, original, _ = source(spec['year'], spec['seed'], spec['controller'])
        reference = original[original.year.eq(spec['year']) & original.model_day.between(spec['start_day'], spec['end_day'])]
        np.testing.assert_array_equal(frame.to_numpy(float), reference[list(FIELDS)].to_numpy(float))
    save(folder/'run.json', dict(**spec, complete=complete, observed_days=len(frame), expected_days=expected,
        data_directory=str(data), failure=failure, factual_reference=factual,
        initial_observation=initial_obs, initial_storage_m3=stock['state']['storage'], final_storage_m3=end_storage,
        window_cost=float(-frame.reward.sum()) if complete else None,
        flood_severity=float(frame.flood_excess_m_hours.sum()) if complete else None,
        low_water_severity=float(frame.low_water_excess_m_hours.sum()) if complete else None,
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6) if complete else None,
        flood_days=int((frame.flood_excess_m_hours>0).sum()) if complete else None,
        peak_level_m=float(max(frame.maximum_level_m.max(), frame.level_before_m.max())) if complete else None,
        initial_headroom_million_m3=(controller.model.flood-float(initial_obs[0]))*controller.model.lake.surface/1e6,
        domain_recoveries=sum(d['domain_recovery'] for d in controller.diagnostics)))


def forecast_audits(root, plan):
    out = root/'forecast_audits'
    out.mkdir(exist_ok=True)
    training, bank = forecast_bank()
    rows, skill = [], []
    for event in plan['events']:
        year = event['year']
        for seed in SEEDS:
            _, original_frame, original_decisions = source(year, seed, 'original')
            selected = original_frame[original_frame.year.eq(year) & original_frame.model_day.between(event['first_flood_day']-14, event['end_day'])]
            for row in selected.itertuples():
                d = original_decisions[int(row.elapsed_hour)//24]
                n = d['horizon_days']
                actual = original_frame.inflow_m3_s.iloc[int(row.elapsed_hour)//24:int(row.elapsed_hour)//24+n].to_numpy()
                flows = np.asarray(d['forecast_inflows'])
                sums = flows.sum(axis=1)*86400/1e6
                skill.append(dict(year=year, seed=seed, model_day=int(row.model_day),
                    actual_inflow_m3_s=float(actual[0]), forecast_first_min=float(flows[:,0].min()),
                    forecast_first_max=float(flows[:,0].max()), forecast_first_mean=float(flows[:,0].mean()),
                    actual_seven_day_million_m3=float(actual.sum()*86400/1e6),
                    forecast_seven_day_min_million_m3=float(sums.min()), forecast_seven_day_max_million_m3=float(sums.max())))
            for arm in ARMS:
                _, frame, decisions = source(year, seed, arm)
                for d in decisions:
                    if not d['domain_recovery']:
                        continue
                    hour = d['elapsed_hours']
                    snapshot = get_snapshot(root, year, seed, arm, hour)
                    env, obs, info = restore(year, bank, snapshot, snapshot)
                    controller = make_controller(training, bank, arm, seed)
                    restore_plan(controller, snapshot)
                    controller(obs, info)
                    np.testing.assert_array_equal(controller.diagnostics[-1]['selected_plan'], d['selected_plan'])
                    np.testing.assert_array_equal(controller.diagnostics[-1]['forecast_inflows'], d['forecast_inflows'])
                    assert controller.diagnostics[-1]['domain_recovery']
                    # The planner uses the float32 observation, not hidden exact storage.
                    state = WaterState(controller.model.lake.level_to_storage(float(obs[0])), float(obs[5]), float(obs[6]))
                    days = (hour//24+np.arange(d['horizon_days']))%365+1
                    audit = audit_forecast(controller.model, state, d, days)
                    actual_paths = frame.inflow_m3_s.iloc[hour//24:hour//24+d['horizon_days']].to_numpy()
                    result = dict(year=year, seed=seed, arm=arm, model_day=hour//24%365+1,
                        elapsed_hour=hour, planning_level_m=float(obs[0]), actual_future_inflows=actual_paths.tolist(),
                        forecast_inflows=d['forecast_inflows'], **audit)
                    path = out/f'{year}_seed{seed}_{arm}_hour{hour}.json'
                    save(path, result)
                    rows.append({k: result[k] for k in ['year', 'seed', 'arm', 'model_day', 'elapsed_hour',
                        'planning_level_m', 'candidate_count', 'feasible_candidates', 'certified_infeasible_forecast']})
                    env.close()
    pd.DataFrame(rows).to_csv(root/'feasibility_summary.csv', index=False)
    pd.DataFrame(skill).to_csv(root/'forecast_skill.csv', index=False)
    print('Offline forecast audits complete', flush=True)


def contrasts(values):
    """Two conditional decompositions plus the separately controlled plan effect."""
    oo, oc, co, cc, own = (values[k] for k in ['oo', 'oc', 'co', 'cc', 'own'])
    observed = own-oo
    memory = own-cc
    result = dict(observed_factual_delta=observed, memory_effect=memory,
        storage_effect_original_controller=co-oo, controller_effect_at_combined_stock=cc-co,
        controller_effect_at_original_stock=oc-oo, storage_effect_combined_controller=cc-oc)
    assert abs(observed-((co-oo)+(cc-co)+memory)) < 1e-7
    assert abs(observed-((oc-oo)+(cc-oc)+memory)) < 1e-7
    return result


def aggregate(root, plan):
    records = [json.loads((root/spec['name']/'run.json').read_text()) for spec in branch_specs(plan)]
    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ['failure', 'initial_observation']} for r in records])
    summary.to_csv(root/'branches.csv', index=False)
    attribution = []
    for (year, seed, lead), group in summary.groupby(['year', 'seed', 'lead']):
        if not group.complete.all():
            continue
        values = {}
        for row in group.itertuples():
            key = 'own' if row.memory == 'combined' else row.storage_origin[0]+row.controller[0]
            # Original/combined initials are unambiguous, but normalize to o/c.
            values[key] = row
        for metric in ['flood_severity', 'window_cost', 'unmet_target_million_m3']:
            attribution.append(dict(year=int(year), seed=int(seed), lead=int(lead), metric=metric,
                **contrasts({key: getattr(values[key], metric) for key in ['oo', 'oc', 'co', 'cc', 'own']})))
    pd.DataFrame(attribution).to_csv(root/'attribution.csv', index=False)
    # Same initial observation and information for each matched-state pair.
    by_name = {r['name']: r for r in records}
    for event in plan['events']:
        for seed in SEEDS:
            for anchor in event['anchors']:
                selected = [r for r in records if r['year']==event['year'] and r['seed']==seed and r['lead']==anchor['lead']]
                for origin in ARMS:
                    pair = [r for r in selected if r['storage_origin']==origin and r['memory']=='original']
                    np.testing.assert_array_equal(pair[0]['initial_observation'], pair[1]['initial_observation'])
                frames, decisions = [], []
                for r in selected:
                    data = Path(r['data_directory'])
                    f = pd.read_csv(data/'daily.csv', float_precision='round_trip')
                    frames.append(f)
                    decisions.append([json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines()])
                    if len(f):
                        np.testing.assert_allclose(-f.reward, f.unmet_target_m3/1e6+f.flood_excess_m_hours+f.low_water_excess_m_hours, atol=1e-12, rtol=0)
                        np.testing.assert_allclose(145900000*(f.level_m-f.level_before_m), (f.inflow_m3_s-f.release_m3_s)*86400, atol=1e-6, rtol=0)
                n = min(len(f) for f in frames)
                for f, ds in zip(frames[1:], decisions[1:]):
                    exogenous = ['elapsed_hour', 'year', 'model_day', 'inflow_m3_s', 'target_m3_s', 'past_target_ema', 'past_inflow_m3_s']
                    np.testing.assert_array_equal(frames[0][exogenous].iloc[:n], f[exogenous].iloc[:n])
                    for first, second in zip(decisions[0][:n], ds[:n], strict=True):
                        np.testing.assert_array_equal(first['forecast_inflows'], second['forecast_inflows'])
                        np.testing.assert_array_equal(first['forecast_target'], second['forecast_target'])
                        assert first['horizon_days']==second['horizon_days']==7
    failures = [dict(name=r['name'], **r['failure']) for r in records if r['failure']]
    save(root/'failures.json', failures)
    audit = pd.read_csv(root/'feasibility_summary.csv')
    save(root/'run.json', dict(complete=True, expected_branches=100, completed_branches=int(summary.complete.sum()),
        all_branches_complete=bool(summary.complete.all()), branch_domain_failures=len(failures),
        audited_failed_decisions=len(audit), certified_infeasible_decisions=int(audit.certified_infeasible_forecast.sum()),
        expanded_search_feasible_decisions=int((audit.feasible_candidates>0).sum()),
        paired_information_and_physics_verified=True, source_hashes_verified=True,
        test_evaluated=False, model_or_costs_changed=False, default_changed=False))
    print(summary.groupby(['year', 'lead', 'storage_origin', 'controller', 'memory'])[['flood_severity', 'window_cost']].mean().to_string(), flush=True)
    print(pd.DataFrame(attribution).groupby(['year', 'lead', 'metric']).mean(numeric_only=True).to_string(), flush=True)
    print(audit.groupby('arm')[['certified_infeasible_forecast', 'feasible_candidates']].sum().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--stage', choices=['all', 'prepare', 'prefix', 'branches', 'audit', 'aggregate'], default='all')
    args = parser.parse_args()
    root = args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):
        prepare(root)
    plan = verify(root)
    if args.stage in ['all', 'prefix']:
        prefix(root, plan)
    if args.stage in ['all', 'branches']:
        training, bank = forecast_bank()
        for index, spec in enumerate(branch_specs(plan), 1):
            run_branch(root, spec, training, bank)
            if index%10==0:
                print('Branches', index, '/ 100', flush=True)
    if args.stage in ['all', 'audit']:
        forecast_audits(root, plan)
    if args.stage in ['all', 'aggregate']:
        aggregate(root, plan)
    verify(root)


if __name__ == '__main__':
    main()
