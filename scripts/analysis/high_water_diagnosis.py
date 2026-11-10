"""Fixed-cost, fixed-rule diagnosis of the 1986–1987 training flood exit."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.paths import PROJECT_ROOT,DAM_DATA,DAM_CONFIG
from lake_como.simulation.high_water import flood_replay,capacity_volume_bound
from lake_como.simulation.low_water import LowWaterRule,ProtectedOutletModel
from lake_como.simulation.outlet import OutletSpec


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summary(trace,failure):
    valid=trace[trace.accepted]
    return dict(complete=failure is None,valid_hours=float(valid.step_seconds.sum()/3600),
        maximum_valid_level_m=float(valid.level_m.max()),minimum_valid_level_m=float(valid.level_m.min()),
        maximum_balance_error_m3=float(trace.balance_error_m3.abs().max()),
        flood_excess_m_hours=float(valid.flood_excess_m_hours.sum()),
        low_water_excess_m_hours=float(valid.low_water_excess_m_hours.sum()),
        unmet_target_million_m3=float(valid.unmet_target_m3.sum()/1e6),
        failure=failure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/simulator/high_water_diagnosis_v1')
    root=parser.parse_args().output;root.mkdir(parents=True,exist_ok=False)
    params=settings('train');years=(1986,1987)
    lookbacks=[1,3,7,14,30,60]
    refinements=[1,2,4,8,16,32]
    files=list((PROJECT_ROOT/'src/lake_como/simulation').glob('*.py'))
    files += [Path(__file__),PROJECT_ROOT/'src/lake_como/environment.py',PROJECT_ROOT/'src/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'configs/splits.json',DAM_CONFIG/'lake.py',DAM_CONFIG/'lake.yaml']
    files += [DAM_DATA/k/f'water_{k}_{part}.csv' for k in ['inflow','demand'] for part in ['train','test']]
    pdf=PROJECT_ROOT/'results/simulator/calibration_research/sources/ranzi_2021_historical_methods.pdf'
    files.append(pdf)
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    save(root/'plan.json',dict(years=years,policies=['constant_100','causal_feedback','maximum_nominal'],
        branch_lookback_days=lookbacks,refinement_substeps_per_hour=refinements,
        branch_end='Seven days after baseline exit, bounded by the same record end',
        source_sha256=hashes,rule=asdict(LowWaterRule()),outlet=asdict(OutletSpec()),
        costs='Unchanged; no optimization, reward-weight tuning or training.',
        causal='Policies use current level/demand history or constant maximum request. Branch timing is selected in hindsight.',
        refinement='Identical anchor and daily requests, hourly reserve ceiling; refine state-dependent hydraulic integration only.',
        domain='Stop after recording first rejected trial; never evaluate discharge outside the envelope.',
        source_note='Published full-open curve is not a gate-position calibration. Numeric extrema are research guards.',
        test_evaluated=False,operationally_calibrated=False))
    full=[];snapshots=None;baseline_failure=None
    for policy in ['constant_100','causal_feedback','maximum_nominal']:
        trace,anchors,failure=flood_replay(params,years,policy=policy)
        trace.to_csv(root/f'full_{policy}.csv',index=False)
        full.append(dict(policy=policy,**summary(trace,failure)))
        if policy=='constant_100':snapshots=anchors;baseline_failure=failure
    if baseline_failure is None or baseline_failure['boundary']!='upper':
        raise AssertionError('The declared baseline upper-domain exit was not reproduced')
    assert abs(baseline_failure['trial_level_m']-2.64157627)<1e-7
    event_day_start=baseline_failure['hour']//24*24
    end_hour=min(len(years)*365*24,math.ceil(baseline_failure['elapsed_end_hour'])+7*24)
    branches=[];bounds=[]
    for days in lookbacks:
        anchor=snapshots[event_day_start-days*24]
        bounds.append(dict(lookback_days=days,**capacity_volume_bound(params,years,anchor,math.ceil(baseline_failure['elapsed_end_hour']))))
        save(root/f'anchor_{days}d.json',dict(hour=anchor.hour,state=asdict(anchor.state),demand_ema=anchor.demand_ema))
        for policy in ['constant_100','causal_feedback','maximum_nominal']:
            trace,_,failure=flood_replay(params,years,policy=policy,anchor=anchor,end_hour=end_hour)
            trace.to_csv(root/f'branch_{days}d_{policy}.csv',index=False)
            branches.append(dict(lookback_days=days,policy=policy,start_hour=anchor.hour,end_hour=end_hour,
                                 **summary(trace,failure)))
    # The same local anchor is used for all resolutions. Every exit remains a
    # stop; we do not continue a finer trajectory through unvalidated levels.
    convergence=[]
    for count in refinements:
        trace,_,failure=flood_replay(params,years,policy='constant_100',
            anchor=snapshots[event_day_start-7*24],end_hour=end_hour,substeps_per_hour=count)
        trace.to_csv(root/f'refinement_{count}.csv',index=False)
        convergence.append(dict(substeps_per_hour=count,**summary(trace,failure)))
    # A bound at the final daily boundary tests whether last-day intervention
    # alone could succeed even with optimistic capacity at every lake level.
    bounds.append(dict(lookback_days=0,**capacity_volume_bound(params,years,snapshots[event_day_start],math.ceil(baseline_failure['elapsed_end_hour']))))
    model=ProtectedOutletModel(params);lake=model.lake;operating=[]
    for level in [1.249999,1.25,1.250001,2.,2.64]:
        storage=lake.level_to_storage(level)
        for request in [0.,100.,float(np.float32(model.high))]:
            operating.append(dict(level_m=level,request_m3_s=request,
                minimum_release_m3_s=lake.min_release(storage,1),capacity_m3_s=lake.max_release(storage,1),
                actual_release_m3_s=lake.actual_release(request,storage,1)))
    pd.DataFrame(operating).to_csv(root/'operating_rule_audit.csv',index=False)
    save(root/'full_replays.json',full);save(root/'branches.json',branches)
    save(root/'capacity_bounds.json',bounds);save(root/'integration_refinement.json',convergence)
    save(root/'event.json',baseline_failure)
    for name,sha in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    save(root/'run.json',dict(complete=True,test_evaluated=False,source_hashes_verified=True,
        model_or_costs_changed=False,operationally_calibrated=False))
    print(json.dumps(dict(full=full,branches=branches,refinement=convergence,bounds=bounds),indent=2))


if __name__=='__main__':main()
