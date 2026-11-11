"""Record hourly physics for two training model years; zero learner updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.hourly_telemetry import make_hourly_env
from reservoir_control.systems.lake_como.mpc_hindsight import digest
from reservoir_control.systems.lake_como.paths import DAM_CONFIG, DAM_DATA, PROJECT_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROJECT_ROOT/'results/simulator/hourly_check_v1')
    root = p.parse_args().output
    if root.exists() and any(root.iterdir()):
        p.error('Choose an empty directory; prior runs are preserved')
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), PROJECT_ROOT/'src/reservoir_control/systems/lake_como/hourly_telemetry.py',
        PROJECT_ROOT/'src/reservoir_control/systems/lake_como/continuous_environment.py', PROJECT_ROOT/'src/reservoir_control/systems/lake_como/environment.py',
        PROJECT_ROOT/'src/reservoir_control/systems/lake_como/inflow_observation.py', PROJECT_ROOT/'configs/splits.json',
        DAM_CONFIG/'env.py', DAM_CONFIG/'lake.py', DAM_CONFIG/'lake.yaml', DAM_CONFIG/'world_train.yaml',
        DAM_DATA/'MEF.csv']
    sources += [DAM_DATA/s/f'water_{s}_{part}.csv' for s in ['inflow','demand'] for part in ['train','test']]
    hashes = {str(f.relative_to(PROJECT_ROOT)):digest(f) for f in sources}
    plan = dict(years=[1946,1947], split='train', protocol='continuous-model-years-v1',
        action='Constant 100 m3/s', daily_transitions=730, updates=0,
        telemetry='Original 24 hourly integration substeps; each day has one flat historical inflow',
        severity='Duration and excess integrate a piecewise-linear interpolation of hourly model levels',
        reward='Original endpoint/daily-mean reward unchanged; metrics are not reward replacements',
        calendar='365 samples per model year; no Gregorian timestamp claim',
        source_sha256=hashes, test_evaluated=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env = make_hourly_env('train',(1946,1947))
    daily=[];hourly=[]
    try:
        env.reset(seed=901)
        for _ in range(730):
            obs,reward,term,trunc,info=env.step(np.array([100.],np.float32))
            tag=dict(step=info['elapsed_steps'],year=info['transition_year'],model_day=info['transition_day'])
            rows=info['hourly'];metrics=info['physical_metrics']
            assert len(rows)==24 and rows[-1]['storage_m3']==info['storage']
            assert rows[0]['storage_before_m3']==info['storage_before']
            assert float(np.mean([r['release_m3_s'] for r in rows]))==info['release']
            assert abs(sum(r['balance_error_m3'] for r in rows))<1e-6
            daily.append(dict(**tag,reward=reward,level_m=info['level'],storage_m3=info['storage'],
                endpoint_flood=bool(info['flood']),endpoint_low_water=bool(info['level']<env.unwrapped.starving_level),
                terminated=term,truncated=trunc,**metrics))
            hourly.extend(dict(**tag,**row) for row in rows)
            assert env.observation_space.contains(obs)
        day_frame=pd.DataFrame(daily);hour_frame=pd.DataFrame(hourly)
        assert day_frame.truncated.sum()==1 and not day_frame.iloc[364].truncated
        assert len(hour_frame)==17520
        assert np.isfinite(hour_frame.select_dtypes(include='number').to_numpy()).all()
        assert day_frame.max_hourly_balance_error_m3.max()<1e-6
        day_frame.to_csv(root/'daily.csv',index=False);hour_frame.to_csv(root/'hourly.csv',index=False)
        report=dict(complete=True,environment_transitions=730,updates=0,hourly_rows=len(hour_frame),
            max_hourly_balance_error_m3=float(day_frame.max_hourly_balance_error_m3.max()),
            endpoint_flood_days=int(day_frame.endpoint_flood.sum()),
            days_with_interpolated_flood=int(day_frame.interpolated_flood_hours.gt(0).sum()),
            days_with_flood_missed_at_endpoint=int((day_frame.interpolated_flood_hours.gt(0)&~day_frame.endpoint_flood).sum()),
            source_hashes_verified=True,test_evaluated=False)
        for name,expected in hashes.items():assert digest(PROJECT_ROOT/name)==expected,name
        (root/'run.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
