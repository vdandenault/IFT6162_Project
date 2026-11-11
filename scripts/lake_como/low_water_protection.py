"""Predeclared training-only low-water operating-rule comparison and SAC smokes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.paths import PROJECT_ROOT, DAM_DATA, DAM_CONFIG
from reservoir_control.systems.lake_como.simulation.low_water import (
    LowWaterRule, make_protected_outlet_env, ProtectedOutletTrainingEpisodes,
)
from reservoir_control.systems.lake_como.simulation.outlet import OutletSpec, OutletDomainError
from reservoir_control.systems.lake_como.simulation.protocol import TaskProtocol
from reservoir_control.systems.lake_como.simulation.training import train_operational


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(frame):
    return dict(days=len(frame), return_=float(frame.reward.sum()),
        unmet_target_million_m3=float(frame.unmet_target_m3.sum()/1e6),
        environmental_shortage_million_m3=float(frame.environmental_shortage_m3.sum()/1e6),
        flood_excess_m_hours=float(frame.flood_excess_m_hours.sum()),
        low_water_excess_m_hours=float(frame.low_water_excess_m_hours.sum()),
        low_water_hours=float(frame.low_water_hours.sum()),
        minimum_level_m=float(frame.minimum_level_m.min()),
        maximum_level_m=float(frame.maximum_level_m.max()),
        intervention_hours=int(frame.intervention_hours.sum()),
        curtailed_release_million_m3=float(frame.curtailed_release_m3.sum()/1e6),
        minimum_operating_flow_shortfall_million_m3=float(frame.minimum_operating_flow_shortfall_m3.sum()/1e6),
        maximum_balance_error_m3=float(frame.balance_error_m3.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/simulator/low_water_protection_v1')
    root=parser.parse_args().output
    root.mkdir(parents=True,exist_ok=False)
    variants={
        'legacy':dict(outlet=OutletSpec(capacity='legacy'),rule=None),
        'candidate':dict(outlet=OutletSpec(),rule=None),
        'protected':dict(outlet=OutletSpec(),rule=LowWaterRule()),
    }
    windows=[(1946,1947),(1966,1967),(1986,1987)]
    policies=['constant_100','causal_feedback']
    seeds=[901,902,903]
    files=list((PROJECT_ROOT/'src/reservoir_control/systems/lake_como/simulation').glob('*.py'))
    files += [Path(__file__),PROJECT_ROOT/'src/reservoir_control/controllers/sac/agent.py',
        PROJECT_ROOT/'src/reservoir_control/systems/lake_como/environment.py',PROJECT_ROOT/'configs/splits.json',
        DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml']
    files += [DAM_DATA/k/f'water_{k}_{part}.csv' for k in ['inflow','demand'] for part in ['train','test']]
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    save(root/'plan.json',dict(rule=asdict(LowWaterRule()),windows=windows,policies=policies,seeds=seeds,
        objective='Unchanged physical objective/default weights; report environmental shortfalls separately.',
        variants={name:dict(outlet=asdict(kw['outlet']),rule=None if kw['rule'] is None else asdict(kw['rule'])) for name,kw in variants.items()},
        guarantee_scope='Hourly integration, initial level >= floor, nonnegative inflow, zero evaporation, rule applied hourly; no flood guarantee.',
        priority='Low-water reserve overrides operating minimum release; shortages logged, not hidden.',
        knowledge='Current simulated level only; no realized or future inflow used by protection.',
        interpretation='Operating design sensitivity, not calibrated operations or learned-controller performance ranking.',
        sac_smoke=dict(steps=96,warmup=32,episode_years=1,burn_in_days=30),
        test_evaluated=False,source_sha256=hashes))

    summaries=[];pairs=[]
    for years in windows:
        for policy in policies:
            frames={}
            for variant,kwargs in variants.items():
                env=make_protected_outlet_env(years=years,**kwargs)
                obs,info=env.reset(seed=901);days=[];stop=None
                try:
                    for _ in range(730):
                        request=100 if policy=='constant_100' else np.clip(float(obs[3])+20+150*(float(obs[0])-.35),0,env.model.high)
                        obs,reward,term,trunc,info=env.step(np.array([request],np.float32))
                        rows=info['hourly']
                        days.append(dict(day=len(days)+1,reward=reward,
                            unmet_target_m3=sum(r['unmet_demand_m3'] for r in rows),
                            environmental_shortage_m3=sum(r['environmental_shortage_m3'] for r in rows),
                            flood_excess_m_hours=sum(r['flood_excess_m_hours'] for r in rows),
                            low_water_excess_m_hours=sum(r['low_water_excess_m_hours'] for r in rows),
                            low_water_hours=sum(r['low_water_hours'] for r in rows),
                            minimum_level_m=min(r['level_m'] for r in rows),
                            maximum_level_m=max(r['level_m'] for r in rows),
                            intervention_hours=sum(r.get('low_water_intervened',False) for r in rows),
                            curtailed_release_m3=sum(r.get('operationally_curtailed_m3',0) for r in rows),
                            minimum_operating_flow_shortfall_m3=sum(r.get('operating_minimum_shortfall_m3',0) for r in rows),
                            balance_error_m3=max(abs(r['lake_balance_error_m3']) for r in rows)))
                        if term or trunc:break
                except OutletDomainError as exc:
                    stop=str(exc)
                finally:env.close()
                frame=pd.DataFrame(days)
                if frame.empty:raise RuntimeError('No completed comparison interval')
                frames[variant]=frame
                frame.to_csv(root/f'{years[0]}_{policy}_{variant}_daily.csv',index=False)
                summaries.append(dict(start_year=years[0],policy=policy,variant=variant,
                                      complete=len(frame)==730,domain_stop=stop,**summarize(frame)))
            # Equal-length prefix only; failed variants never get a shorter
            # total return compared against a full run as if it were better.
            for baseline in ['legacy','candidate']:
                n=min(len(frames[baseline]),len(frames['protected']))
                before=summarize(frames[baseline].iloc[:n]);after=summarize(frames['protected'].iloc[:n])
                pairs.append(dict(start_year=years[0],policy=policy,baseline=baseline,
                    comparison='full matched window' if n==730 else 'matched prefix; not full-horizon performance',
                    **{f'delta_{k}':after[k]-before[k] for k in before if k!='days'},days=n))
    pd.DataFrame(summaries).to_csv(root/'controls.csv',index=False)
    pd.DataFrame(pairs).to_csv(root/'paired_deltas.csv',index=False)

    smokes=[]
    for seed in seeds:
        for variant,kwargs in variants.items():
            env=ProtectedOutletTrainingEpisodes(TaskProtocol(episode_years=1),**kwargs)
            try:
                agent,rows,accounting=train_operational(env,steps=96,seed=seed,warmup=32)
                assert all(np.isfinite(v.detach().numpy()).all() for v in agent.parameters())
                pd.DataFrame(rows).to_csv(root/f'sac_{variant}_{seed}.csv',index=False)
                smokes.append(dict(seed=seed,variant=variant,complete=True,**accounting))
            except OutletDomainError as exc:
                smokes.append(dict(seed=seed,variant=variant,complete=False,domain_stop=str(exc),
                    agent_transitions=env.agent_transitions,burn_in_transitions=env.burn_in_transitions,
                    updates=0 if env.agent_transitions<=32 else None))
            finally:env.close()
    save(root/'sac_smokes.json',smokes)
    for name,sha in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    save(root/'run.json',dict(complete=True,test_evaluated=False,source_hashes_verified=True,
        protected_control_windows_complete=all(row['complete'] for row in summaries if row['variant']=='protected'),
        protected_smokes_complete=all(row['complete'] for row in smokes if row['variant']=='protected'),
        operationally_calibrated=False))
    print(pd.DataFrame(summaries)[['start_year','policy','variant','days','return_','unmet_target_million_m3',
        'environmental_shortage_million_m3','low_water_excess_m_hours','intervention_hours']].to_string(index=False))
    print(json.dumps(smokes,indent=2))


if __name__=='__main__':main()
