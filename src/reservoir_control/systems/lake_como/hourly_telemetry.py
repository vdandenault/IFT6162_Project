"""Opt-in hourly telemetry delegated to the unchanged pinned lake integrator.

Records actual integration states; never reruns the dynamics or changes policy
observations, rewards, episode boundaries, or the original environment factory.
"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gym4real.envs.dam.lake import Lake

from reservoir_control.systems.lake_como.continuous_environment import make_continuous_env


def linear_exposure(start, end, threshold, *, above=True, hours=1.):
    """Duration and positive excess integral under linear endpoint interpolation.

    Returns hours and m·hours. This is a stated interpolation of hourly model
    samples, not measured inundation duration or a damage estimate.
    """
    if not np.isfinite([start, end, threshold, hours]).all() or hours <= 0:
        raise ValueError('Finite levels and a positive duration required')
    sign = 1 if above else -1
    a, b = sign * (start - threshold), sign * (end - threshold)
    if a <= 0 and b <= 0:
        return 0., 0.
    if a >= 0 and b >= 0:
        return float(hours), float(.5 * (a + b) * hours)
    positive = max(a, b)
    duration = hours * positive / abs(b - a)
    return float(duration), float(.5 * positive * duration)


class RecordingLake(Lake):
    """Instrument the parent's integration calls without duplicating its equations."""

    @classmethod
    def from_lake(cls, lake):
        if type(lake) is not Lake:
            raise ValueError('Expected the pinned Lake implementation, without another instrument')
        result = cls.__new__(cls)
        result.__dict__ = deepcopy(lake.__dict__)
        result.last_hourly = []
        result._recording = False
        return result

    def actual_release(self, to_release, storage, cday):
        actual = super().actual_release(to_release, storage, cday)
        if self._recording:
            self.last_hourly.append(dict(storage_before_m3=float(storage),
                level_before_m=float(self.storage_to_level(storage)),
                requested_release_m3_s=float(to_release), release_m3_s=float(actual)))
        return actual

    def integration(self, step, init_storage, to_release, inflow, cday):
        if step != 24 or self.evaporation:
            raise ValueError('Hourly telemetry currently requires 24 substeps and no evaporation')
        self.last_hourly = []
        self._recording = True
        try:
            result = super().integration(step, init_storage, to_release, inflow, cday)
        finally:
            self._recording = False
        final_storage, _ = result
        assert len(self.last_hourly) == 24
        # Each following pre-state is the preceding post-state. The final one is
        # the exact state returned by the original integrator, not a recomputation.
        ends = [row['storage_before_m3'] for row in self.last_hourly[1:]] + [float(final_storage)]
        for hour, (row, storage) in enumerate(zip(self.last_hourly, ends), 1):
            row.update(hour=hour, duration_seconds=3600., storage_m3=storage,
                level_m=float(self.storage_to_level(storage)), inflow_m3_s=float(inflow),
                balance_error_m3=float(storage - row['storage_before_m3']
                    - 3600 * (inflow - row['release_m3_s'])))
        return result


class HourlyTelemetry(gym.Wrapper):
    """Add completed-day hourly records and physical metrics to info only."""

    def __init__(self, env):
        core = env.unwrapped
        if core.integ_step != 24 or core.lake.evaporation:
            raise ValueError('Expected the 24-substep, no-evaporation DamEnv setup')
        recorder = RecordingLake.from_lake(core.lake)
        super().__init__(env)
        core.lake = recorder
        self.recorder = recorder

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.recorder.last_hourly = []
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        core = self.env.unwrapped
        rows = deepcopy(self.recorder.last_hourly)
        metrics = dict(interpolated_flood_hours=0., interpolated_low_water_hours=0.,
                       interpolated_flood_excess_m_hours=0., interpolated_low_water_excess_m_hours=0.)
        # Info is returned after the transition. No forcing is exposed to the
        # policy's preceding observation, including at continuous-year rollover.
        day = int(info['transition_day']) if 'transition_day' in info else int(core.doy[-1])
        mef = float(core.lake.min_env_flow[day - 1])
        for row in rows:
            row.update(demand_m3_s=float(info['demand']), environmental_flow_m3_s=mef,
                flood_end_sample=bool(row['level_m'] > core.flood_level),
                low_water_end_sample=bool(row['level_m'] < core.starving_level))
            flood, flood_area = linear_exposure(row['level_before_m'], row['level_m'], core.flood_level)
            low, low_area = linear_exposure(row['level_before_m'], row['level_m'], core.starving_level, above=False)
            row.update(interpolated_flood_hours=flood, interpolated_low_water_hours=low,
                interpolated_flood_excess_m_hours=flood_area, interpolated_low_water_excess_m_hours=low_area,
                deficit_m3=max(float(info['demand']) - max(row['release_m3_s'] - mef, 0.), 0.) * 3600)
            for key in metrics:
                metrics[key] += row[key]
        metrics.update(hour_end_flood_samples=sum(r['flood_end_sample'] for r in rows),
            hour_end_low_water_samples=sum(r['low_water_end_sample'] for r in rows),
            max_sampled_level_m=max(rows[0]['level_before_m'], *(r['level_m'] for r in rows)),
            min_sampled_level_m=min(rows[0]['level_before_m'], *(r['level_m'] for r in rows)),
            hourly_unmet_demand_m3=sum(r['deficit_m3'] for r in rows),
            max_hourly_balance_error_m3=max(abs(r['balance_error_m3']) for r in rows))
        info = dict(info, hourly=rows, physical_metrics=metrics)
        return observation, reward, terminated, truncated, info


def make_hourly_env(split='train', years=None, *, past_inflow=True):
    """An explicitly logged continuous environment; no training starts here."""
    return HourlyTelemetry(make_continuous_env(split, years, past_inflow=past_inflow))
