"""Full two-year validation of a frozen weighted-tail forecast/controller.

Prior history forecast versus the selected explicit-probability mixture, using
all validation pairs and declared forecast seeds. No fitting to validation or
automatic promotion. Hydrological regime labels are retrospective context only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_CONFIG,DAM_DATA
from .weighted_study import DEFAULT as REFERENCE,make_controller
from .weighted_forecast import CONFIGS
from .precision_validation import SEEDS,annual_records
from .target_year_check import validation_pairs
from .forecast_study import controller as reference_controller
from .carryover_check import take_step,FIELDS
from .low_water import make_protected_outlet_env
from .outlet import OutletDomainError
from .tail_diagnosis import forecast_infeasible
from .reserve_use_check import save,verify as verify_sources

DEFAULT=PROJECT_ROOT/'results/simulator/weighted_validation_v1'
ARMS=('legacy','weighted')
SCOPES={'all_validation':tuple(range(1988,1998)),
        'previous_event_years':(1993,1997),
        'other_validation':tuple(y for y in range(1988,1998) if y not in (1993,1997))}
METRICS=('annual_cost','unmet_target_million_m3','delivered_target_million_m3','remaining_river_million_m3',
         'flood_excess_m_hours','low_water_excess_m_hours','flood_days','low_water_days',
         'environmental_shortage_million_m3','end_storage_million_m3','fallback_days')


def specs(selected):
    if selected not in CONFIGS:raise ValueError('Frozen forecast variant required')
    return [dict(name=f'{a}_{b}__seed{seed}__{arm}',years=[a,b],seed=seed,arm=arm,
                 variant='legacy' if arm=='legacy' else selected)
            for a,b in validation_pairs() for seed in SEEDS for arm in ARMS]


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    selection=json.loads((REFERENCE/'selection.json').read_text());selected=selection['selected_variant']
    paths=list(Path(__file__).parent.glob('*.py'))+[
        REFERENCE/'plan.json',REFERENCE/'run.json',REFERENCE/'selection.json',REFERENCE/'event_plan.json',
        REFERENCE/'development_summary.csv',REFERENCE/'late_training_check_summary.csv',
        PROJECT_ROOT/'src/lake_como/environment.py',PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/lake_como/splits.py',PROJECT_ROOT/'configs/splits.json',PROJECT_ROOT/'pyproject.toml',
        PROJECT_ROOT/'uv.lock',DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',DAM_CONFIG/'world_train.yaml',
        DAM_CONFIG/'world_test.yaml',DAM_DATA/'MEF.csv']
    paths += [DAM_DATA/s/f'water_{s}_{part}.csv' for s in ['inflow','demand'] for part in ['train','test']]
    training=settings('train')
    quartiles=np.quantile([np.mean(training['inflow'][y][:365]) for y in sorted(training['inflow'])],[.25,.75])
    save(root/'plan.json',dict(cases=specs(selected),arms=list(ARMS),seeds=list(SEEDS),
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        selected_variant=selected,selected_config=CONFIGS[selected],
        selection_sha256=hashlib.sha256((REFERENCE/'selection.json').read_bytes()).hexdigest(),
        fit_years=list(range(1946,1988)),training_inflow_quartiles_m3_s=quartiles.tolist(),
        scopes={k:list(v) for k,v in SCOPES.items()},
        protocol='Five nonoverlapping two-year validation episodes from reset, .35 m initial level, target EMA zero, prior inflow climatology. State/history/plan continue across the year boundary. First 365 samples per model year retained.',
        forecast='Legacy is the prior scaled/stratified seven-observation history forecast. Weighted is the frozen training-selected mixture and band probabilities. Both banks fit only 1946–1987 and remain fixed. Seed + original elapsed hour; environment reset seed always 901.',
        held_fixed='Combined precision-safe service/exact-cost tie controller, physical reward weights, target forecaster, observations, action grid, seven-day horizon, outlet physics and all-path domain rule. No paths discarded or infeasible costs renormalized.',
        regimes='Annual mean inflow compared with training Q25/Q75: dry/middle/wet. Labels are ex-post descriptive context, never policy inputs. Report unique years and their five seed repeats separately.',
        reference_control='One additional full legacy 1988–1989 seed-901 replay with the previous HistoryMPC implementation must match bit for bit. Not a primary comparison or extra independent sample.',
        failures='Incomplete years/blocks retain status and observed prefix costs only. Required-scope means suppressed when any required seed-year is absent/incomplete. Matched prefixes are not full-horizon scores.',
        statistics='Paired year and block differences, seed ranges, first/second-year split, regime means. Seeds share historical forcing; years within episodes and reused validation are dependent. No p-value, confidence interval or final-test claim.',
        promotion='Describe scalar-objective and physical tradeoffs; no default change or operator-priority assumption.',
        test_evaluated=False,model_or_reward_weights_changed=False,feasibility_rule_changed=False,default_changed=False))


def verify(root):
    plan=verify_sources(root)
    if plan['cases']!=specs(plan['selected_variant']):raise ValueError('Changed validation grid')
    if plan['selected_config']!=CONFIGS[plan['selected_variant']]:raise ValueError('Changed frozen probabilities')
    return plan


def reference_replay(training,years):
    c,bank=reference_controller(training,'history_scaled_stratified',901)
    env=make_protected_outlet_env('validation',tuple(years));env.fallback=bank.initial_last_inflow
    obs,info=env.reset(seed=901);rows=[]
    try:
        for _ in range(730):
            obs,info,row,term,trunc=take_step(env,c,obs,info);rows.append(list(row.values()))
            if term or trunc:break
    finally:env.close()
    return np.asarray(rows,float)


def run_case(root,spec):
    folder=root/spec['name'];folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():return spec['name']+' already recorded'
    attempt=1
    while (folder/f'attempt_{attempt}').exists():attempt+=1
    data=folder/f'attempt_{attempt}';data.mkdir()
    training=settings('train');c,bank=make_controller(training,spec['variant'],spec['seed'])
    env=make_protected_outlet_env('validation',tuple(spec['years']));env.fallback=bank.initial_last_inflow
    obs,info=env.reset(seed=901);initial_obs=obs.tolist();initial_storage=env.state.storage
    rows=[];failure=None;started=time.monotonic()
    with (data/'decisions.jsonl').open('w') as log:
        try:
            for _ in range(730):
                attempted_hour=env.hour
                obs,info,row,term,trunc=take_step(env,c,obs,info);rows.append(row)
                log.write(json.dumps(c.diagnostics[-1],allow_nan=False)+'\n');log.flush()
                if term or trunc:break
        except OutletDomainError as error:
            failure=dict(error=str(error),attempted_decision_hour=attempted_hour,elapsed_hour=env.hour)
            if c.diagnostics and c.diagnostics[-1]['elapsed_hours']==attempted_hour:
                save(data/'failed_decision.json',c.diagnostics[-1])
        finally:
            final_storage=env.state.storage;env.close()
    frame=pd.DataFrame(rows,columns=FIELDS);frame.to_csv(data/'daily.csv',index=False)
    complete=len(frame)==730 and failure is None;reference_match=None
    if complete and spec['years']==[1988,1989] and spec['seed']==901 and spec['arm']=='legacy':
        np.testing.assert_array_equal(frame.to_numpy(float),reference_replay(training,spec['years']))
        reference_match=True
    save(folder/'run.json',dict(**spec,complete=complete,observed_days=len(frame),expected_days=730,
        failure=failure,data_directory=str(data),initial_observation=initial_obs,
        initial_storage_m3=initial_storage,final_storage_m3=final_storage,
        two_year_cost=float(-frame.reward.sum()) if complete else None,
        observed_prefix_cost=float(-frame.reward.sum()) if len(frame) else None,
        domain_recoveries=sum(d['domain_recovery'] for d in c.diagnostics),
        reference_bitwise_match=reference_match,elapsed_seconds=time.monotonic()-started))
    return f"{spec['name']}: {len(frame)}/730 days; cost={-frame.reward.sum():.3f}; failure={failure}"


def pair_annual(annual):
    keys=['seed','block_start','year','position_in_pair']
    a=annual[annual.arm.eq('legacy')].set_index(keys)
    b=annual[annual.arm.eq('weighted')].set_index(keys).reindex(a.index)
    result=a[[]].copy();result['regime']=a.regime
    result['complete']=a.complete.fillna(False).astype(bool)&b.complete.fillna(False).astype(bool)
    for metric in METRICS:
        result['legacy_'+metric]=a[metric]
        result['weighted_'+metric]=b[metric]
        result['delta_'+metric]=(b[metric]-a[metric]).where(result.complete)
    return result.reset_index()


def scope_summary(paired,years,seeds):
    selected=paired[paired.year.isin(years)&paired.seed.isin(seeds)]
    expected={(seed,year) for seed in seeds for year in years}
    complete=(bool(expected) and len(selected)==len(expected) and
              set(zip(selected.seed,selected.year))==expected and bool(selected.complete.all()))
    result=dict(unique_years=len(years),forecast_seeds=len(seeds),expected_pairs=len(expected),
        complete_pairs=int(selected.complete.sum()),all_required_complete=complete,
        cost_wins=int((selected.delta_annual_cost<-1e-8).sum()),
        cost_ties=int((selected.delta_annual_cost.abs()<=1e-8).sum()),
        cost_losses=int((selected.delta_annual_cost>1e-8).sum()))
    for metric in METRICS:
        for arm in ['legacy','weighted','delta']:
            result['mean_'+arm+'_'+metric]=float(selected[arm+'_'+metric].mean()) if complete else None
    return result


def aggregate(root,plan):
    training=settings('train');area=training['lake_params']['surface']
    annual=[];blocks=[];prefixes=[];failures=[];fallbacks=[];regime_years={k:set() for k in ['dry','middle','wet']}
    model=make_controller(training,'legacy',901)[0].model
    source_fields=['elapsed_hour','year','model_day','inflow_m3_s','target_m3_s','past_target_ema','past_inflow_m3_s']
    for a,b in validation_pairs():
        for seed in SEEDS:
            frames={};decisions={};initial=[]
            for arm in ARMS:
                name=f'{a}_{b}__seed{seed}__{arm}';record=json.loads((root/name/'run.json').read_text())
                data=Path(record['data_directory']);f=pd.read_csv(data/'daily.csv',float_precision='round_trip')
                ds=[json.loads(line) for line in (data/'decisions.jsonl').read_text().splitlines()]
                spec=dict(name=name,years=[a,b],seed=seed,arm=arm)
                yearly=annual_records(f,spec,plan['training_inflow_quartiles_m3_s'],area)
                for row in yearly:
                    n=0 if row['year']==a else 365
                    if row['complete']:
                        row['fallback_days']=sum(d['domain_recovery'] for d in ds[n:n+365])
                    else:row['fallback_days']=None
                    annual.append(row)
                for year in [a,b]:
                    flow=float(np.mean(settings('validation')['inflow'][year][:365]))
                    lo,hi=plan['training_inflow_quartiles_m3_s']
                    regime='dry' if flow<lo else 'wet' if flow>hi else 'middle'
                    regime_years[regime].add(year)
                blocks.append({k:record[k] for k in ['name','seed','arm','variant','complete','observed_days',
                    'two_year_cost','final_storage_m3','domain_recoveries','reference_bitwise_match']}|dict(block_start=a))
                if record['failure']:failures.append(dict(name=name,**record['failure']))
                initial.append(record['initial_observation']);frames[arm]=f;decisions[arm]=ds
                if len(f):
                    np.testing.assert_allclose(-f.reward,f.unmet_target_m3/1e6+f.flood_excess_m_hours+f.low_water_excess_m_hours,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(area*(f.level_m-f.level_before_m),(f.inflow_m3_s-f.release_m3_s)*86400,atol=1e-6,rtol=0)
                    np.testing.assert_allclose(f.release_m3_s*86400,f.delivered_target_m3+f.remaining_river_m3,atol=1e-6,rtol=0)
                for d,row in zip(ds,f.itertuples(),strict=True):
                    p=np.array(d['applied_probabilities']);assert (p>0).all() and abs(p.sum()-1)<1e-12
                    np.testing.assert_array_equal(p,d['forecast_metadata']['scenario_probabilities'])
                    assert d['forecast_metadata']['training_years']==list(range(1946,1988))
                    if d['domain_recovery']:
                        assert forecast_infeasible(model,row.level_before_m,d['forecast_inflows'])
                        fallbacks.append(dict(name=name,year=int(row.year),seed=seed,arm=arm,model_day=int(row.model_day)))
                if record['complete']:
                    assert len(ds)==730 and f.year.tolist()==[a]*365+[b]*365
                    assert f.level_before_m.iloc[365]==f.level_m.iloc[364]
                    assert ds[365]['elapsed_hours']==8760 and ds[365]['shifted_previous_plan'] and ds[365]['horizon_days']==7
                    assert ds[-1]['horizon_days']==1
                    assert len(ds[365]['forecast_metadata']['observed_history'])==7
            np.testing.assert_array_equal(initial[0],initial[1])
            n=min(len(f) for f in frames.values())
            np.testing.assert_array_equal(frames['legacy'][source_fields].iloc[:n],frames['weighted'][source_fields].iloc[:n])
            for first,second in zip(decisions['legacy'][:n],decisions['weighted'][:n],strict=True):
                np.testing.assert_array_equal(first['forecast_target'],second['forecast_target'])
            prefixes.append(dict(block_start=a,seed=seed,matched_days=n,full_block=n==730,
                cost_delta=float(-frames['weighted'].reward.iloc[:n].sum()+frames['legacy'].reward.iloc[:n].sum()) if n else None))
    annual=pd.DataFrame(annual);annual.to_csv(root/'annual.csv',index=False)
    blocks=pd.DataFrame(blocks);blocks.to_csv(root/'blocks.csv',index=False)
    paired=pair_annual(annual);paired.to_csv(root/'paired_annual.csv',index=False)
    paired_blocks=[]
    for (seed,block),g in paired.groupby(['seed','block_start']):
        complete=len(g)==2 and bool(g.complete.all())
        end=g[g.position_in_pair.eq(2)]
        paired_blocks.append(dict(seed=int(seed),block_start=int(block),complete=complete,
            legacy_cost=float(g.legacy_annual_cost.sum()) if complete else None,
            weighted_cost=float(g.weighted_annual_cost.sum()) if complete else None,
            cost_delta=float(g.delta_annual_cost.sum()) if complete else None,
            legacy_final_stock_million_m3=float(end.legacy_end_storage_million_m3.iloc[0]) if complete else None,
            weighted_final_stock_million_m3=float(end.weighted_end_storage_million_m3.iloc[0]) if complete else None,
            final_stock_delta_million_m3=float(end.delta_end_storage_million_m3.iloc[0]) if complete else None))
    pd.DataFrame(paired_blocks).to_csv(root/'paired_blocks.csv',index=False)
    scope_rows=[];seeds=[];positions=[]
    for scope,years in SCOPES.items():
        scope_rows.append(dict(scope=scope,**scope_summary(paired,years,SEEDS)))
        for seed in SEEDS:seeds.append(dict(scope=scope,seed=seed,**scope_summary(paired,years,[seed])))
        for position in [1,2]:
            selected=[y for y in years if (y-1988)%2+1==position]
            positions.append(dict(scope=scope,position_in_pair=position,**scope_summary(paired,selected,SEEDS)))
    regimes=[]
    for regime,years in regime_years.items():
        regimes.append(dict(regime=regime,years=';'.join(map(str,sorted(years))),
            **scope_summary(paired,sorted(years),SEEDS)))
    pd.DataFrame(scope_rows).to_csv(root/'scope_summary.csv',index=False)
    pd.DataFrame(seeds).to_csv(root/'seed_summary.csv',index=False)
    pd.DataFrame(positions).to_csv(root/'episode_position.csv',index=False)
    pd.DataFrame(regimes).to_csv(root/'regime_summary.csv',index=False)
    pd.DataFrame(prefixes).to_csv(root/'matched_prefixes.csv',index=False)
    pd.DataFrame(fallbacks,columns=['name','year','seed','arm','model_day']).to_csv(root/'fallbacks.csv',index=False)
    save(root/'failures.json',failures)
    save(root/'run.json',dict(complete=True,expected_episodes=50,completed_episodes=int(blocks.complete.sum()),
        all_arms_complete=bool(blocks.complete.all()),domain_failures=len(failures),
        frozen_selected_variant=plan['selected_variant'],source_hashes_verified=True,
        paired_exogenous_history_and_target_forecasts_verified=True,cross_year_continuity_verified=True,
        physical_ledgers_verified=True,all_logged_fallbacks_capacity_certified=True,
        full_legacy_reference_bitwise_match=bool(blocks.reference_bitwise_match.fillna(False).any()),
        test_evaluated=False,model_or_reward_weights_changed=False,feasibility_rule_changed=False,default_changed=False))
    print(pd.DataFrame(scope_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(regimes).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['all','prepare','run','aggregate'],default='all');p.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=p.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    plan=verify(root)
    if args.stage in ['all','run']:
        if args.workers==1:
            for spec in plan['cases']:print(run_case(root,spec),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_case,root,spec) for spec in plan['cases']]
                for future in as_completed(futures):print(future.result(),flush=True)
        verify(root)
    if args.stage in ['all','aggregate']:aggregate(root,plan)
    verify(root)


if __name__=='__main__':main()
