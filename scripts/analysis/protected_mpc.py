"""Fixed-cost causal protected-MPC screen, resumable by completed arm.

Event training replay, small chronological validation screen, and a same-state
event branch. No controller tuning or final-test evaluation is performed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_DATA,DAM_CONFIG
from lake_como.simulation.low_water import LowWaterRule,make_protected_outlet_env
from lake_como.simulation.model import PhysicalWeights,WaterState
from lake_como.simulation.outlet import OutletSpec,OutletDomainError
from lake_como.simulation.protected_mpc import CausalForecastBank,ProtectedMPC,ProtectedMPCConfig

DEFAULT=PROJECT_ROOT/'results/simulator/protected_mpc_v1'
REFERENCE=PROJECT_ROOT/'results/simulator/high_water_diagnosis_v1'


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Changed declared source: '+name)
    return plan


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    anchor=json.loads((REFERENCE/'anchor_60d.json').read_text())
    event=json.loads((REFERENCE/'event.json').read_text())
    arms=['constant_100','causal_feedback']+[f'{mode}_{h}d' for mode in
        ['persistence','seasonal','conditional'] for h in [1,7]]
    cases=[dict(name='event_1986_1987',split='train',years=[1986,1987],before_year=1986,start_hour=0,end_hour=17520),
        dict(name='validation_1988_1989',split='validation',years=[1988,1989],before_year=1988,start_hour=0,end_hour=17520),
        dict(name='same_state_event',split='train',years=[1986,1987],before_year=1986,
             start_hour=anchor['hour'],end_hour=int(event['elapsed_end_hour'])+7*24,anchor=anchor)]
    files=list((PROJECT_ROOT/'src/lake_como/simulation').glob('*.py'))
    files += [Path(__file__),PROJECT_ROOT/'src/lake_como/environment.py',PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
              PROJECT_ROOT/'src/lake_como/splits.py',PROJECT_ROOT/'configs/splits.json',
              DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml',REFERENCE/'anchor_60d.json',REFERENCE/'event.json']
    files += [DAM_DATA/k/f'water_{k}_{part}.csv' for k in ['inflow','demand'] for part in ['train','test']]
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    save(root/'plan.json',dict(cases=cases,arms=arms,rule=asdict(LowWaterRule()),outlet=asdict(OutletSpec()),
        weights=asdict(PhysicalWeights()),mpc_defaults=asdict(ProtectedMPCConfig()),source_sha256=hashes,
        search='Nine constant-request schedules, service schedule, current feedback request; grid/feedback first day then service; shifted previous plan. Native float32 bounded requests.',
        objective='Exact existing physical shortage/flood/low-water exposure cost; no discount, reward change or added terminal cost.',
        feasibility='Reject schedules leaving model domain in any sampled path. If none remains, maximum nominal request is an explicitly uncertified recovery.',
        forecast='Only training years before each replay; seven-day source blocks fixed across one/seven-day horizons. Target forecast is the same prior seasonal mean in all modes.',
        initialization='Same initial physical state in every arm; initial last-inflow feature uses prior Dec-31 model-day climatology. Event branch receives the same saved state and measured previous-day inflow.',
        validation='One two-year chronological validation screen; no selection or retuning on its results. Not a full-split algorithm ranking.',
        event='A previously investigated training event; same-state branch timing is selected in hindsight, actions/forecasts remain causal.',
        physics='Batch planning kernel tested against protected model; every selected first-day scenario is checked against scalar protected rollout.',
        test_evaluated=False,operationally_calibrated=False,
        numpy_version=np.__version__,pandas_version=pd.__version__))


def aggregate_metrics(frame):
    return dict(completed_days=len(frame),return_=float(frame.reward.sum()),
        minimum_level_m=float(frame.minimum_level_m.min()),maximum_level_m=float(frame.maximum_level_m.max()),
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6),
        environmental_shortage_million_m3=float(frame.environmental_shortage_m3.sum()/1e6),
        flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),
        low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
        intervention_hours=int(frame.intervention_hours.sum()),
        maximum_balance_error_m3=float(frame.balance_error_m3.max()))


def run_arm(root,case,arm,training):
    folder=root/(case['name']+'__'+arm);folder.mkdir(exist_ok=True)
    if (folder/'run.json').exists():return
    # Preserve interrupted attempts; a fresh attempt uses the same frozen plan.
    number=1
    while (folder/f'attempt_{number}').exists():number+=1
    attempt=folder/f'attempt_{number}';attempt.mkdir()
    bank=CausalForecastBank(training['inflow'],training['demand'],before_year=case['before_year'])
    env=make_protected_outlet_env(case['split'],tuple(case['years']))
    env.fallback=bank.initial_last_inflow
    obs,info=env.reset(seed=901)
    if case['start_hour']:
        anchor=case['anchor'];env.hour=anchor['hour'];env.state=WaterState(**anchor['state'])
        env.demand_ema=anchor['demand_ema']
        previous=env.hour//24-1;year=case['years'][previous//365]
        env.last_inflow=float(env.forcing[year][previous%365].mean())
        obs=env._observation();info=env._info()
    if arm in ('constant_100','causal_feedback'):controller=None
    else:
        forecast,horizon=arm.rsplit('_',1)
        # Planner receives physical parameters only, never a live record.
        physical={k:training[k] for k in ['lake_params','action','flood_level','starving_level']}
        controller=ProtectedMPC(physical,bank,forecast=forecast,
            config=ProtectedMPCConfig(horizon_days=int(horizon[:-1])))
    rows=[];failure=None;start=time.monotonic()
    decisions=attempt/'decisions.jsonl'
    with decisions.open('w') as log:
        try:
            while env.hour<case['end_hour']:
                hour=env.hour
                if controller is None:
                    request=100 if arm=='constant_100' else np.clip(float(obs[3])+20+150*(float(obs[0])-.35),0,env.model.high)
                    action=np.array([request],np.float32)
                else:
                    action=controller(obs,info)
                    log.write(json.dumps(controller.diagnostics[-1],allow_nan=False)+'\n');log.flush()
                obs,reward,term,trunc,info=env.step(action)
                hourly=info['hourly']
                row=dict(elapsed_hour=hour,year=hourly[0]['year'],model_day=hourly[0]['model_day'],
                    level_before_m=hourly[0]['level_before_m'],level_m=hourly[-1]['level_m'],
                    minimum_level_m=min(r['level_m'] for r in hourly),maximum_level_m=max(r['level_m'] for r in hourly),
                    request_m3_s=float(action[0]),release_m3_s=float(np.mean([r['release_m3_s'] for r in hourly])),
                    inflow_m3_s=float(np.mean([r['inflow_m3_s'] for r in hourly])),
                    target_m3_s=hourly[0]['demand_m3_s'],reward=reward,
                    unmet_target_m3=sum(r['unmet_demand_m3'] for r in hourly),
                    environmental_shortage_m3=sum(r['environmental_shortage_m3'] for r in hourly),
                    flood_excess_m_hours=sum(r['flood_excess_m_hours'] for r in hourly),
                    low_water_excess_m_hours=sum(r['low_water_excess_m_hours'] for r in hourly),
                    intervention_hours=sum(r['low_water_intervened'] for r in hourly),
                    balance_error_m3=max(abs(r['lake_balance_error_m3']) for r in hourly))
                rows.append(row)
                if len(rows)%100==0:print(case['name'],arm,len(rows),'days',flush=True)
                if term or trunc:break
        except OutletDomainError as error:
            failure=dict(error=str(error),attempted_decision_hour=hour,elapsed_hour=env.hour,
                last_valid_level_m=env.model.lake.storage_to_level(env.state.storage))
        finally:env.close()
    frame=pd.DataFrame(rows)
    if frame.empty:raise RuntimeError('No completed day; inspect preserved attempt')
    frame.to_csv(attempt/'daily.csv',index=False)
    diagnostics=[] if controller is None else controller.diagnostics
    errors=[r['first_day_parity_error_m'] for r in diagnostics if r['first_day_parity_error_m'] is not None]
    save(folder/'run.json',dict(case=case['name'],arm=arm,attempt=f'attempt_{number}',
        complete=failure is None and len(frame)==(case['end_hour']-case['start_hour'])//24,
        failure=failure,expected_days=(case['end_hour']-case['start_hour'])//24,
        fit_years=list(bank.years),domain_recoveries=sum(r['domain_recovery'] for r in diagnostics),
        maximum_planning_parity_error_m=max(errors) if errors else None,
        elapsed_seconds=time.monotonic()-start,**aggregate_metrics(frame)))
    print('FINISHED',case['name'],arm,'days',len(frame),'failure',failure,'seconds',round(time.monotonic()-start,2),flush=True)


def aggregate(root,plan):
    summaries=[];pairs=[];skills=[]
    for case in plan['cases']:
        records={};frames={}
        for arm in plan['arms']:
            folder=root/(case['name']+'__'+arm)
            if not (folder/'run.json').exists():raise ValueError('Missing arm '+str(folder))
            record=json.loads((folder/'run.json').read_text());records[arm]=record
            frame=pd.read_csv(folder/record['attempt']/'daily.csv');frames[arm]=frame
            summaries.append({k:v for k,v in record.items() if k not in ['failure','fit_years']})
            if arm in ('constant_100','causal_feedback'):continue
            # Ex-post scoring only, after all decisions in this arm are frozen.
            actual=settings(case['split'])['inflow']
            log=folder/record['attempt']/'decisions.jsonl'
            for line in log.read_text().splitlines():
                decision=json.loads(line);paths=np.asarray(decision['forecast_inflows']);forecast=paths.mean(axis=0)
                start_day=decision['elapsed_hours']//24
                for lead,predicted in enumerate(forecast):
                    ordinal=start_day+lead;year=case['years'][ordinal//365]
                    observed=float(actual[year][ordinal%365])
                    skills.append(dict(case=case['name'],arm=arm,elapsed_hour=decision['elapsed_hours'],
                        lead_days=lead+1,absolute_error_m3_s=abs(predicted-observed),
                        predicted_inflow_m3_s=float(predicted),actual_inflow_m3_s=observed))
        for arm in plan['arms']:
            if arm=='causal_feedback':continue
            n=min(len(frames[arm]),len(frames['causal_feedback']))
            a=aggregate_metrics(frames[arm].iloc[:n]);b=aggregate_metrics(frames['causal_feedback'].iloc[:n])
            pairs.append(dict(case=case['name'],arm=arm,matched_days=n,
                full_window=records[arm]['complete'] and records['causal_feedback']['complete'],
                **{f'delta_{k}':a[k]-b[k] for k in a if k!='completed_days'}))
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(root/'paired_vs_feedback.csv',index=False)
    detail=pd.DataFrame(skills);detail.to_csv(root/'forecast_errors.csv',index=False)
    detail.groupby(['case','arm','lead_days']).absolute_error_m3_s.agg(['count','mean']).reset_index().to_csv(root/'forecast_skill.csv',index=False)
    save(root/'run.json',dict(complete=True,test_evaluated=False,model_or_costs_changed=False,
        source_hashes_verified=True,operationally_calibrated=False,
        interpretation='Completed study includes explicitly recorded controller/domain failures; not all arms necessarily complete.'))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','prepare','run','aggregate'],default='all')
    args=parser.parse_args();root=args.output
    if args.stage=='prepare' or (args.stage=='all' and not root.exists()):prepare(root)
    plan=verify(root)
    if args.stage in ('all','run'):
        training=settings('train')
        for case in plan['cases']:
            for arm in plan['arms']:run_arm(root,case,arm,training)
        verify(root)
    if args.stage in ('all','aggregate'):aggregate(root,plan)
    verify(root)


if __name__=='__main__':main()
