"""Opt-in service precision and exact-cost tie controls for protected MPC.

The original controller and its frozen studies are unchanged. Primary costs are
never relaxed: the conservative rule acts only on exactly equal finite minima.
"""
import numpy as np

from reservoir_control.systems.lake_como.simulation.model import WaterState, physical_cost
from reservoir_control.systems.lake_como.simulation.mpc_diagnosis import DiagnosticMPC
from reservoir_control.systems.lake_como.simulation.protected_mpc import evaluate_schedules


def service_requests(model, targets, days, *, precision_safe=False):
    exact = np.clip(targets + model.lake.min_env_flow[days - 1], model.low, model.high)
    requests = exact.astype(np.float32)
    if precision_safe:
        requests = np.where(
            requests.astype(float) < exact,
            np.nextafter(requests, np.float32(np.inf)), requests,
        )
        # A requested target above the representable action ceiling remains
        # infeasible to serve exactly; never cross that ceiling to erase cost.
        requests = np.minimum(requests, np.float32(model.high))
    return requests


def select_schedule(costs, schedules, *, conservative=False):
    """Exact cost first, then smallest first request, then smallest sum, then order.

    Smaller commands are a declared reserve preference, not a safety guarantee.
    All-infeasible recovery is handled by the caller, never treated as a tie.
    """
    costs = np.asarray(costs, float)
    finite = np.flatnonzero(np.isfinite(costs))
    if not len(finite):
        return None, 0
    best = float(costs[finite].min())
    ties = np.flatnonzero(costs == best)
    if not conservative:
        return int(ties[0]), len(ties)
    chosen = min(ties, key=lambda i: (
        float(schedules[i, 0]), float(np.asarray(schedules[i], float).sum()), int(i),
    ))
    return int(chosen), len(ties)


class PrecisionMPC(DiagnosticMPC):
    def __init__(self, params, bank, *, precision_safe=False, conservative=False, **kwargs):
        super().__init__(params, bank, target_mode='ema_residual', search='restricted', **kwargs)
        self.precision_safe = bool(precision_safe)
        self.conservative = bool(conservative)

    def __call__(self, observation, info):
        obs = np.asarray(observation, float)
        hour = int(info['elapsed_hours'])
        if (obs.shape != (8,) or not np.isfinite(obs).all() or hour % 24
                or info['decision_hours'] != 24):
            raise ValueError('Daily eight-feature observation required')
        horizon = min(self.config.horizon_days, int(info['remaining_hours']) // 24)
        if horizon < 1:
            raise ValueError('Do not plan from a completed record')
        day = hour // 24 % 365 + 1
        days = (day - 1 + np.arange(horizon)) % 365 + 1
        self.bank.context = (day, float(obs[3]), hour // 24)
        flows, targets, metadata = self.bank.forecast(
            self.forecast, day, float(obs[4]), horizon=horizon,
            count=self.config.scenarios, seed=self.config.seed + hour,
        )
        state = WaterState(self.model.lake.level_to_storage(obs[0]), float(obs[5]), float(obs[6]))
        grid = np.linspace(self.model.low, self.model.high, self.config.grid_points).astype(np.float32)
        service = service_requests(self.model, targets, days, precision_safe=self.precision_safe)
        feedback = np.float32(np.clip(obs[3] + 20 + 150 * (obs[0] - .35), self.model.low, self.model.high))
        candidates = [np.full(horizon, v, np.float32) for v in grid]
        candidates += [service.copy(), np.full(horizon, feedback, np.float32)]
        for first in [*grid, feedback]:
            sequence = service.copy()
            sequence[0] = first
            candidates.append(sequence)
        warm = self.previous_hour == hour - 24 and self.previous_plan is not None
        if warm:
            shifted = self.previous_plan[1:]
            if len(shifted):
                candidates.append(np.pad(shifted, (0, max(0, horizon - len(shifted))), mode='edge')[:horizon])
        schedules = np.array(list({tuple(row): row for row in candidates}.values()), np.float32)
        result = evaluate_schedules(self.model, state, schedules, flows, targets, days, self.weights)
        costs = result['scenario_costs'].mean(axis=1)
        index, ties = select_schedule(costs, schedules, conservative=self.conservative)
        recovery = index is None
        parity = []
        if recovery:
            plan = np.full(horizon, np.float32(self.model.high))
        else:
            plan = schedules[index].copy()
            for scenario, path in enumerate(flows):
                final, rows = self.model.rollout(state, np.repeat(plan[0], 24),
                    np.repeat(path[0], 24), np.repeat(targets[0], 24), np.repeat(days[0], 24))
                error = abs(self.model.lake.storage_to_level(final.storage) - result['daily_levels'][index, scenario, 0])
                cost_error = abs(physical_cost(rows, self.weights) - result['daily_costs'][index, scenario, 0])
                if error > 1e-8 or cost_error > 1e-6:
                    raise AssertionError('Selected schedule differs from scalar simulator')
                parity.append(error)
        self.previous_hour = hour
        self.previous_plan = plan.copy()
        original_index, _ = select_schedule(costs, schedules)
        self.diagnostics.append(dict(
            elapsed_hours=hour, horizon_days=horizon, forecast_kind=self.forecast,
            forecast_inflows=flows.tolist(), forecast_target=targets.tolist(), forecast_metadata=metadata,
            request_m3_s=float(plan[0]), candidate_count=len(schedules),
            feasible_candidates=int(np.isfinite(costs).sum()), domain_recovery=recovery,
            expected_cost=None if recovery else float(costs[index]),
            predicted_maximum_level_m=None if recovery else float(result['maximum_levels'][index].max()),
            predicted_minimum_level_m=None if recovery else float(result['minimum_levels'][index].min()),
            shifted_previous_plan=warm, first_day_parity_error_m=max(parity) if parity else None,
            precision_safe=self.precision_safe, conservative_ties=self.conservative,
            exact_minimum_tie_count=ties,
            tie_changed_selection=bool(not recovery and index != original_index),
            selected_plan=plan.tolist(),
        ))
        return np.array([plan[0]], np.float32)
