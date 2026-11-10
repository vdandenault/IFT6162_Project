import numpy as np
import pandas as pd
import pytest
from lake_como.simulation.ppo_execution import specs,noise_seed,pair_annual,METRICS
from lake_como.simulation.ppo_curve_audit import fixed_effect_slope,episode_frame,duration_signal


def test_case_counts_and_randomness_are_declared():
    cases=specs();assert len(cases)==300 and len({c['name'] for c in cases})==300
    assert sum(c['mode']=='deterministic' for c in cases)==50
    assert sum(c['mode']=='stochastic' for c in cases)==250
    assert all(min(c['years'])>=1988 and max(c['years'])<=1997 for c in cases)
    assert noise_seed(2101,1988)==19882101


def test_weather_adjustment_removes_window_composition():
    frame=pd.DataFrame(dict(step=[100,200,300,400,500,600],window=['dry','dry','wet','wet','wet','wet'],cost=[100,100,1000,1000,1000,1000]))
    assert fixed_effect_slope(frame)==0
    frame['cost']=frame.cost-.1*frame.step
    assert abs(fixed_effect_slope(frame)+.1)<1e-12
    assert fixed_effect_slope(frame.iloc[[0,2]]) is None


def test_pairing_averages_action_replicates_first():
    rows=[]
    for mode,values in [('deterministic',[10]),('stochastic',[8,9,10,11,12])]:
        for value in values:rows.append(dict(model='mapped',seed=123,block_start=1988,year=1988,mode=mode,end_storage_million_m3=value,**{m:value for m in METRICS}))
    paired=pair_annual(pd.DataFrame(rows));assert len(paired)==1 and paired.delta_annual_cost.iloc[0]==0
    with pytest.raises(ValueError):pair_annual(pd.DataFrame(rows+[rows[0]]))


def test_episode_prefix_exclusion_is_explicit():
    record=dict(complete=True,steps=200000,unfinished_length=710,
        episodes=[dict(length=730,physical_return=-100,years=[1946,1947]) for _ in range(273)])
    frame=episode_frame(record);assert len(frame)==273 and frame.step.iloc[-1]==199290
    bad=dict(record,unfinished_length=0)
    with pytest.raises(ValueError):episode_frame(bad)


def test_duration_signal_needs_cohort_progress_in_both_periods():
    rows=[]
    for seed in range(123,128):
        for period in ['late','last50k']:rows.append(dict(seed=seed,period=period,fraction_per_100k=-.03,change_per_100k=-30))
    assert duration_signal(pd.DataFrame(rows))['duration_probe_supported']
    for row in rows:
        if row['period']=='last50k':row['change_per_100k']=1
    assert not duration_signal(pd.DataFrame(rows))['duration_probe_supported']
