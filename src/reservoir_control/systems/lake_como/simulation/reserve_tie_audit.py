"""Post-hoc prediction audit; never changes or reruns a controller trajectory.

Compare saved selected plans, existing service plans, and service plans rounded
up to a representable request. Costs, physics, states and forecast paths stay fixed.
The last proposal is diagnostic and was not in the frozen original search.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.model import WaterState
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank,evaluate_schedules
from reservoir_control.systems.lake_como.simulation.reserve_use_check import DEFAULT,make_controller,save,verify


def service_rounding(model,targets,days):
    exact=np.clip(targets+model.lake.min_env_flow[days-1],model.low,model.high)
    nearest=exact.astype(np.float32)
    upward=np.where(nearest.astype(float)<exact,np.nextafter(nearest,np.float32(np.inf)),nearest)
    return nearest,np.minimum(upward,np.float32(model.high))


def main():
    root=DEFAULT;verify(root)
    out=root/'posthoc_rounding_audit';out.mkdir(exist_ok=False)
    selected=json.loads((root/'selected_days.json').read_text())['days']
    days_to_check=sorted(set(selected)|set(range(1,21)))
    sources=[Path(__file__),root/'plan.json',root/'run.json',root/'selected_days.json']
    for mode in ['causal','perfect_target']:
        sources.append(root/(mode+'_high')/'attempt_1/decisions.jsonl')
    save(out/'plan.json',dict(source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        days=days_to_check,storage='high',
        purpose='Post-hoc expected-cost audit of numerical undershoot and finite-candidate tie ordering; no rollout, policy or cost changes.',
        proposals=['saved chosen plan','original service proposal','upward-representable service proposal (new diagnostic only)'],
        threshold_cost=1e-8,test_evaluated=False))
    training=settings('train');bank=CausalForecastBank(training['inflow'],training['demand'],before_year=1988)
    model=make_controller(training,bank,'causal',None).model;rows=[]
    for mode in ['causal','perfect_target']:
        decisions=[json.loads(line) for line in (root/(mode+'_high')/'attempt_1/decisions.jsonl').read_text().splitlines()]
        for day in days_to_check:
            d=decisions[day-1];obs=np.asarray(d['planning_observation']);targets=np.asarray(d['forecast_target'])
            horizon=len(targets);days=(d['elapsed_hours']//24+np.arange(horizon))%365+1
            state=WaterState(model.lake.level_to_storage(float(obs[0])),float(obs[5]),float(obs[6]))
            nearest,upward=service_rounding(model,targets,days)
            plans=np.array([d['selected_plan'],nearest,upward],np.float32)
            result=evaluate_schedules(model,state,plans,d['forecast_inflows'],targets,days)
            costs=result['scenario_costs'].mean(axis=1)
            first_releases=[]
            for plan in plans:
                releases=[]
                for path in d['forecast_inflows']:
                    _,hourly=model.rollout(state,np.repeat(plan[0],24),np.repeat(path[0],24),
                        np.repeat(targets[0],24),np.repeat(days[0],24))
                    releases.append(np.mean([r['release_m3_s'] for r in hourly]))
                first_releases.append(float(np.mean(releases)))
            rows.append(dict(mode=mode,model_day=day,selected_cost=float(costs[0]),
                nearest_service_cost=float(costs[1]),upward_service_cost=float(costs[2]),
                selected_first_request_m3_s=float(plans[0,0]),upward_service_first_request_m3_s=float(upward[0]),
                selected_expected_first_release_m3_s=first_releases[0],
                upward_service_expected_first_release_m3_s=first_releases[2],
                lower_release_with_equal_or_better_cost=bool(costs[2]<=costs[0]+1e-8 and first_releases[2]<first_releases[0]-1e-6),
                largest_rounding_change_m3_s=float(np.max(upward.astype(float)-nearest.astype(float))),
                upward_service_final_mean_level_m=float(result['daily_levels'][2,:,-1].mean())))
    frame=pd.DataFrame(rows);frame.to_csv(out/'audit.csv',index=False)
    save(out/'run.json',dict(complete=True,counts=frame.groupby('mode').lower_release_with_equal_or_better_cost.sum().astype(int).to_dict(),
        audited_days_per_mode=len(days_to_check),interpretation='Selected-state predicted alternatives, not evidence of improved realized annual performance.'))
    verify(out);verify(root)
    print(frame.to_string(index=False))


if __name__=='__main__':main()
