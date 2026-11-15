"""Explicit volume-preserving integer-period plug-flow routing for synthetic demos.

A delay is an experiment assumption, not an Ottawa travel-time estimate.
Queue entries are volumes due in successive future periods, never flow rates.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TransitState:
    volumes_m3: tuple[float, ...] = ()

    def __post_init__(self):
        if not np.isfinite(self.volumes_m3).all() or any(x < 0 for x in self.volumes_m3):
            raise ValueError('Finite nonnegative transit volumes required')

    @property
    def total_m3(self): return float(sum(self.volumes_m3))

    def forecast(self, durations_seconds):
        dt = np.asarray(durations_seconds, float)
        if dt.ndim != 1 or len(dt) == 0 or not np.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError('Positive aligned routing durations required')
        h, delay = len(dt), len(self.volumes_m3)
        a, c = np.zeros((h, h)), np.zeros(h)
        if delay == 0: return np.eye(h), c
        for j in range(h):
            if j+delay < h: a[j+delay, j] = dt[j]/dt[j+delay]
        for j, volume in enumerate(self.volumes_m3[:h]): c[j] = volume/dt[j]
        return a, c

    def advance(self, release_m3_s, duration_seconds):
        if not np.isfinite([release_m3_s, duration_seconds]).all() or release_m3_s < 0 or duration_seconds <= 0:
            raise ValueError('Nonnegative finite release and positive duration required')
        outgoing_volume = release_m3_s*duration_seconds
        if not self.volumes_m3:
            return self, float(release_m3_s), 0.
        arriving_volume = self.volumes_m3[0]
        following = TransitState((*self.volumes_m3[1:], outgoing_volume))
        residual = following.total_m3-self.total_m3-outgoing_volume+arriving_volume
        return following, arriving_volume/duration_seconds, residual

    def after_plan(self, releases_m3_s, durations_seconds):
        if len(releases_m3_s) != len(durations_seconds): raise ValueError('Aligned routing plan required')
        state = self
        for q, dt in zip(releases_m3_s, durations_seconds): state = state.advance(float(q), float(dt))[0]
        return state
