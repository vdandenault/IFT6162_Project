from dataclasses import dataclass
import numpy as np
from reservoir_control.controllers.mpc.convex_response import solve_response


@dataclass(frozen=True)
class FollowerResult:
    release_m3_s: np.ndarray
    storage_m3: np.ndarray
    service_shortfall_m3_s: np.ndarray
    response: object
    announcement_derivative: np.ndarray


class OttawaFollower:
    def __init__(self, settings):
        self.settings = settings

    def problem(self, storage_m3, inflow_m3_s, uncontrolled_m3_s,
                announcement_m3_s, downstream_capacity_m3_s, duration_seconds):
        c = self.settings
        inflow, uncontrolled, announce, capacity = [np.asarray(v, float) for v in
            (inflow_m3_s, uncontrolled_m3_s, announcement_m3_s, downstream_capacity_m3_s)]
        n = len(inflow)
        if (n < 1 or any(v.shape != (n,) for v in (uncontrolled, announce, capacity))
                or not all(np.isfinite(v).all() for v in (inflow, uncontrolled, announce, capacity))
                or np.min(inflow) < 0 or duration_seconds <= 0
                or not 0 <= storage_m3 <= c['ottawa_capacity_m3']):
            raise ValueError('Aligned finite forecasts, positive duration and feasible storage required')
        # q is in 1000 m3/s; S in billion m3. Extra n variables are flow-excess slacks.
        d = duration_seconds*1000/1e9
        balance = -d*np.tril(np.ones((n, n)))
        free_storage = storage_m3/1e9 + d*np.cumsum(inflow/1000)
        target = c['ottawa_capacity_m3']/1e9*.5
        service = inflow/1000
        flood_offset = (uncontrolled+announce-capacity)/1000
        identity = np.eye(n)
        h = np.eye(2*n)*1e-8
        g = np.zeros(2*n)
        ws, wt = c['follower_storage_weight'], c['follower_terminal_weight']
        h[:n, :n] += 2*(c['follower_service_weight']*identity + ws*balance.T@balance
                                 + wt*np.outer(balance[-1], balance[-1]))
        g[:n] = (-2*c['follower_service_weight']*service
                 + 2*ws*balance.T@(free_storage-target)
                 + 2*wt*balance[-1]*(free_storage[-1]-target))
        # Squared downstream flow excess, epigraph slack >= q + offset.
        h[n:, n:] += 2*c['follower_flood_weight']*identity
        z = np.zeros((n, n))
        a = np.vstack([np.c_[identity, z], np.c_[-identity, z],
                       np.c_[balance, z], np.c_[-balance, z],
                       np.c_[identity, -identity], np.c_[z, -identity]])
        b = np.r_[np.full(n, c['ottawa_max_release_m3_s']/1000), np.zeros(n),
                  c['ottawa_capacity_m3']/1e9-free_storage, free_storage,
                  -flood_offset, np.zeros(n)]
        x0 = np.r_[service, np.maximum(0, service+flood_offset)]
        return h, g, a, b, x0, balance, free_storage

    def solve(self, storage_m3, inflow_m3_s, uncontrolled_m3_s,
              announcement_m3_s, downstream_capacity_m3_s, duration_seconds):
        h, g, a, b, x0, balance, free = self.problem(
            storage_m3, inflow_m3_s, uncontrolled_m3_s, announcement_m3_s,
            downstream_capacity_m3_s, duration_seconds)
        fit = solve_response(h, g, a, b, x0)
        n = len(inflow_m3_s)
        q = fit.x[:n]*1000
        storage = (free+balance@fit.x[:n])*1e9
        db = np.zeros((len(b), n))
        db[4*n:5*n] = -np.eye(n)/1000
        derivative = fit.sensitivity(h, np.zeros((2*n, n)), a, db)[:n]*1000
        return FollowerResult(q, storage, np.maximum(0, np.asarray(inflow_m3_s)-q),
                              fit, derivative)
