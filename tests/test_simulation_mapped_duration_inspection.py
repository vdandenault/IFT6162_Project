import pandas as pd
import pytest
from reservoir_control.systems.lake_como.simulation.mapped_duration_inspection import episodes,select_period,trend_rows


def record():
    return dict(complete=True,prefix_verified=True,steps=400000,unfinished_length=690,
        episodes=[dict(length=730,physical_return=-100,years=[1946,1947]) for _ in range(547)])


def test_whole_episode_accounting_and_boundary_exclusion():
    frame=episodes(record());whole=select_period(frame,200000,400000);end=select_period(frame,200000,400000,contained=False)
    assert len(frame)==547 and frame.step.iloc[-1]==399310
    assert len(whole)==273 and len(end)==274
    assert whole.start.iloc[0]==200020
    assert end.iloc[0].start==199290 and end.iloc[0].step==200020
    assert all(whole.start>=200000) and all(whole.step<=400000)


def test_fixed_period_trends_and_accounting_guard():
    frame=episodes(record());rows=trend_rows(frame,123)
    assert len(rows)==8 and all(abs(row['change_per_100k'])<1e-12 for row in rows)
    bad=record();bad['unfinished_length']=0
    with pytest.raises(ValueError):episodes(bad)
