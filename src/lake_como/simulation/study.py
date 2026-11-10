"""Declared implementation/validation suite for six simulator-development tracks.

No frozen controller is replaced; no test-year controller evaluation occurs.
Results distinguish mechanisms, conditional diagnostics, and uncalibrated stress.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lake_como.paths import PROJECT_ROOT,DAM_DATA,DAM_CONFIG
from lake_como.environment import settings
from lake_como.mpc_hindsight import digest
from .environment import OperationalEnv,make_operational_env
from .model import Operations,PhysicalWeights
from .protocol import TaskProtocol,TrainingEpisodes
from .scenarios import HistoricalScenarios,stress_profiles
from .validation import observed_level_consistency,rating_curve_audit

ROOT=PROJECT_ROOT/'results/simulator/operational_v2'


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError(f'Changed declared source: {name}')
    return plan


def prepare(root):
    if root.exists() and any(root.iterdir()):raise ValueError('Choose an empty output; preserve all prior studies')
    root.mkdir(parents=True,exist_ok=True)
    config=json.loads((PROJECT_ROOT/'configs/simulator.json').read_text())
    files=list((PROJECT_ROOT/'src/lake_como/simulation').glob('*.py'))
    files += [PROJECT_ROOT/'configs/simulator.json',PROJECT_ROOT/'configs/splits.json',
        PROJECT_ROOT/'src/lake_como/environment.py',PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/lake_como/sac.py',PROJECT_ROOT/'src/lake_como/mpc.py']
    files += sorted(DAM_DATA.rglob('*.csv'))+[DAM_CONFIG/'lake.py',DAM_CONFIG/'env.py',DAM_CONFIG/'lake.yaml',DAM_CONFIG/'world_train.yaml',DAM_DATA/'metadata_dam.json']
    save(root/'plan.json',dict(config=config,source_sha256={str(f.relative_to(PROJECT_ROOT)):digest(f) for f in files},
        scope='Six-track development suite: task protocol, conditional physical validity, causal scenarios, operational mechanisms, physical objective/filter, subdaily control.',
        validation='Observed-level timing/envelope diagnostics use training years only and do not fit or select offsets. No measured-release calibration is claimed.',
        forecast='Training-only historical conditional blocks; chronological validation years used only for declared forecast scoring.',
        stress='Fixed 1946–1947 training model years, 30-day fixed-action spin-up, constant/causal feedback rules, declared synthetic regimes and sensitivity operations; no method ranking.',
        training='Fresh eight-feature SAC smokes only; all declared seeds retained. Unit normalization, gamma adjusted by decision interval, finite termination masks; no checkpoint promotion.',
        constraints='Finite sampled-scenario and discrete-action witnesses; explicit recovery. No proof for unseen inflows or unsampled actions.',
        calibration='Ramping 5 m3/s/hour, routing 6 h, downstream reference 300 m3/s and physical weights are illustrative uncalibrated inputs.',
        test_evaluated=False,reference_changed=False))


def validate(root):
    verify(root)
    frame,summary=observed_level_consistency();frame.to_csv(root/'observed_level_checks.csv',index=False);summary.to_csv(root/'observed_level_summary.csv',index=False)
    curves=rating_curve_audit();curves.to_csv(root/'rating_curve_checks.csv',index=False)
    save(root/'physical_validation.json',dict(complete=True,scope='Conditional training-data diagnostics, not historical operation replay or calibrated validation',
        rating_grid_rows=len(curves),all_rating_values_finite_nonnegative=bool(curves.finite_nonnegative.all()),
        minimum_above_capacity_cases=int(curves.environmental_minimum_above_hydraulic_capacity.sum()),
        timing_assumptions_reported=[-1,0,1],timing_selected=False,
        unresolved=['Source timestamps/date anchor','Level datum mapped to pinned model','Measured aligned outlet releases','Ramping/routing/channel parameters'],
        public_data_leads={'operator_publications':'https://www.addaconsorzio.it/attivita/',
            'arpa_hydrology':'https://www.arpalombardia.it/temi-ambientali/acqua/acque-superficiali/quantita-idrologia/'},
        test_evaluated=False))
    print('Physical consistency diagnostics complete',flush=True)


def forecast(root):
    verify(root);training=settings('train');bank=HistoricalScenarios(training['inflow'],training['demand']);validation=settings('validation');records=[]
    for year,values in validation['inflow'].items():
        for day0 in range(0,359,14):
            last=bank.climatology[0] if day0==0 else values[day0-1]
            paths,_=bank.paths(day0+1,last,horizon=7,count=32,seed=901+year*1000+day0)
            lower,upper=np.quantile(paths,[.1,.9],axis=0)
            for lead in range(7):
                truth=values[day0+lead]
                records.append(dict(year=year,day=day0+1,lead=lead+1,truth=truth,
                    scenario_mean=float(paths[:,lead].mean()),persistence=last,climatology=float(bank.climatology[day0+lead]),
                    lower=float(lower[lead]),upper=float(upper[lead]),covered=bool(lower[lead]<=truth<=upper[lead])))
    frame=pd.DataFrame(records);frame.to_csv(root/'forecast_predictions.csv',index=False)
    summary=[]
    for name in ['scenario_mean','persistence','climatology']:
        errors=frame[name]-frame.truth
        summary.append(dict(method=name,mae_m3_s=float(abs(errors).mean()),rmse_m3_s=float(np.sqrt((errors**2).mean()))))
    pd.DataFrame(summary).to_csv(root/'forecast_summary.csv',index=False)
    save(root/'forecast_run.json',dict(complete=True,forecast_points=len(frame),empirical_10_90_coverage=float(frame.covered.mean()),
        mean_interval_width_m3_s=float((frame.upper-frame.lower).mean()),validation_evaluated=True,test_evaluated=False,
        caveat='Conditional historical scenario ranges, not calibrated prediction intervals; no method selected from this check'))
    print('Causal scenario forecast check complete',flush=True)


def stress(root):
    plan=verify(root);config=plan['config'];training=settings('train');bank=HistoricalScenarios(training['inflow'],training['demand'])
    weights=PhysicalWeights(**config['weights']);summaries=[];monthly=[]
    for case_id,case in enumerate(config['stress_cases']):
        params,hourly=stress_profiles(training,(1946,1947),case['forcing'])
        ops=Operations(**config['sensitivity_operations']) if case['operational_sensitivity'] else Operations()
        env=OperationalEnv(params,(1946,1947),decision_hours=case['decision_hours'],operations=ops,weights=weights,hourly_inflow=hourly)
        obs,info=env.reset(seed=901);scored=[];burn_in_hours=config['protocol']['burn_in_days']*24
        try:
            while True:
                if info['elapsed_hours']<burn_in_hours or case['controller']=='constant':request=100.
                else:
                    day=info['next_day'];service=bank.demand_climatology[day-1]+env.model.lake.min_env_flow[day-1]
                    reserve=env.model.lake.surface*(float(obs[0])-.35)/(7*86400)
                    request=float(np.clip(max(service,float(obs[4])+reserve),env.model.low,env.model.high))
                obs,reward,term,trunc,info=env.step(np.array([request],np.float32))
                if info['transition_start_hour']>=burn_in_hours:
                    scored.extend(dict(case_id=case_id,**row) for row in info['hourly'])
                if term or trunc:break
            frame=pd.DataFrame(scored)
            assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
            assert frame.lake_balance_error_m3.abs().max()<1e-6 and frame.river_balance_error_m3.abs().max()<1e-6
            row=dict(case_id=case_id,**case,scored_hours=len(frame),burn_in_hours=burn_in_hours,
                unmet_million_m3=float(frame.unmet_demand_m3.sum()/1e6),
                flood_hours=float(frame.flood_hours.sum()),low_water_hours=float(frame.low_water_hours.sum()),
                flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),low_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
                downstream_excess_million_m3=float(frame.downstream_excess_m3.sum()/1e6),
                terminal_storage_m3=env.state.storage,terminal_river_storage_m3=env.state.river_storage,
                below_active_storage_datum_hours=int(frame.below_active_storage_datum.sum()),
                max_lake_balance_error_m3=float(frame.lake_balance_error_m3.abs().max()),max_river_balance_error_m3=float(frame.river_balance_error_m3.abs().max()))
            summaries.append(row)
            frame['month']=pd.to_datetime('2001-01-01')+pd.to_timedelta(frame.model_day-1,unit='D')
            frame['month']=frame['month'].dt.month
            table=frame.groupby(['case_id','month'])[['unmet_demand_m3','flood_hours','low_water_hours','flood_excess_m_hours','low_water_excess_m_hours']].sum().reset_index();monthly.append(table)
            # Keep full traces for the declared front-loaded comparisons and parity reference.
            if case['forcing']=='front_loaded' or case_id<3:frame.drop(columns='month').to_csv(root/f'stress_case_{case_id}_hourly.csv',index=False)
            print('Stress case complete',case_id,case,flush=True)
        finally:env.close()
    summary=pd.DataFrame(summaries);summary.to_csv(root/'stress_summary.csv',index=False);pd.concat(monthly).to_csv(root/'stress_monthly.csv',index=False)
    reference=summary.iloc[:3]
    for col in ['unmet_million_m3','flood_hours','low_water_hours','terminal_storage_m3']:
        np.testing.assert_allclose(reference[col],reference[col].iloc[0],rtol=1e-12,atol=1e-8)
    save(root/'stress_run.json',dict(complete=True,cases=len(summaries),fixed_request_subdaily_physics_match=True,
        total_simulated_hours=sum((r['scored_hours']+r['burn_in_hours']) for r in summaries),
        source_years=[1946,1947],test_evaluated=False,calibrated=False))


def controls(root):
    from .controllers import ContinuousMPC,ScenarioPlanner,ScenarioConstraintFilter
    plan=verify(root);config=plan['config'];training=settings('train');bank=HistoricalScenarios(training['inflow'],training['demand']);records=[]
    # Cross-year warm start: 360-day shared preparation, then 14 actual decisions.
    env=make_operational_env();obs,info=env.reset()
    for _ in range(360):obs,_,_,_,info=env.step(np.array([100],np.float32))
    base=ContinuousMPC(training);base.reset()
    for _ in range(14):obs,_,_,_,info=env.step(base(obs,info))
    assert base.resets==1 and any(d['days'][0]==1 and d['used_shifted_plan'] for d in base.diagnostics)
    pd.DataFrame(base.diagnostics).to_csv(root/'continuous_mpc_boundary.csv',index=False);env.close()
    for interval in [24,6,1]:
        env=make_operational_env(decision_hours=interval,operations=Operations(**config['sensitivity_operations']));obs,info=env.reset()
        for _ in range(30*24//interval):obs,_,_,_,info=env.step(np.array([100],np.float32))
        planner=ScenarioPlanner(env.model,bank,PhysicalWeights(**config['weights']),horizon_days=2,scenarios=3,grid_points=5)
        filt=ScenarioConstraintFilter(env.model,bank,PhysicalWeights(**config['weights']),scenarios=3,grid_points=9)
        for _ in range(8):
            proposal=planner(obs,info);request=filt(proposal,obs,info)
            obs,reward,_,_,info=env.step(request)
            records.append(dict(decision_hours=interval,reward=reward,**filt.diagnostics[-1]))
        env.close()
    pd.DataFrame(records).to_csv(root/'scenario_control_checks.csv',index=False)
    save(root/'control_run.json',dict(complete=True,mpc_cross_year_warm_start=True,mpc_decisions=14,
        scenario_planner_filter_decisions=len(records),test_evaluated=False,guarantee=False))
    print('Continuous and subdaily controller integration checks complete',flush=True)


def train(root):
    from .training import train_operational
    plan=verify(root);c=plan['config'];rows=[]
    for seed in c['training_smoke_seeds']:
        out=root/f'training_{seed}';out.mkdir()
        env=TrainingEpisodes(TaskProtocol(**c['protocol']))
        try:
            agent,trace,report=train_operational(env,steps=c['training_smoke_steps'],seed=seed)
            assert len(report['episodes'])==1 and report['episodes'][0]['length']==700
            assert report['updates']==636 and report['burn_in_transitions']==30
            pd.DataFrame(trace).to_csv(out/'trace.csv',index=False);agent.save(out/'smoke_only.pt');save(out/'run.json',report)
            rows.append(dict(seed=seed,decision_hours=24,**report));print('Training smoke complete',seed,flush=True)
        finally:env.close()
    for mode in ['finite_horizon','continuing_record']:
        out=root/f'subdaily_{mode}';out.mkdir()
        protocol=TaskProtocol(**dict(c['protocol'],decision_hours=c['subdaily_smoke_decision_hours'],terminal_mode=mode))
        env=TrainingEpisodes(protocol)
        try:
            agent,trace,report=train_operational(env,steps=c['subdaily_smoke_steps'],seed=901)
            pd.DataFrame(trace).to_csv(out/'trace.csv',index=False);save(out/'run.json',report)
            rows.append(dict(seed=901,decision_hours=protocol.decision_hours,terminal_mode=mode,**report))
        finally:env.close()
    save(root/'training_run.json',dict(complete=True,smokes=rows,performance_claim=False,test_evaluated=False))


def aggregate(root):
    verify(root)
    required=['physical_validation','forecast_run','stress_run','control_run','training_run']
    for name in required:assert json.loads((root/f'{name}.json').read_text())['complete']
    save(root/'run.json',dict(complete=True,tracks_implemented=6,test_evaluated=False,reference_changed=False,
        calibrated_operations=False,real_calendar_confirmed=False,
        remaining='Measured data alignment/datum and calibrated actuator/river parameters remain external validation requirements'))
    print('All six implementation tracks complete; calibration limits retained.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','prepare','validate','forecast','stress','controls','train','aggregate'],default='all')
    p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args();root=args.output.resolve()
    stages=['prepare','validate','forecast','stress','controls','train','aggregate'] if args.stage=='all' else [args.stage]
    for stage in stages:globals()[stage](root)

if __name__=='__main__':main()
