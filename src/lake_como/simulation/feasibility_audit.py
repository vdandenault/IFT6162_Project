"""Read-only forecast schedule diagnostics for the fixed protected hourly model."""
from itertools import product

import numpy as np

from .low_water import ProtectedOutletModel
from .model import WaterState
from .precision_mpc import service_requests
from .protected_mpc import evaluate_schedules


def first_exit(model, state, plan, inflows, targets, days):
    """Stop at the first predicted guard crossing; never evaluate beyond it."""
    for lead, (request, inflow, target, day) in enumerate(zip(plan, inflows, targets, days, strict=True)):
        for hour in range(24):
            before = model.lake.storage_to_level(state.storage)
            release = model.lake.actual_release(float(request), state.storage, int(day))
            following = state.storage + 3600*(inflow-release)
            level = model.lake.storage_to_level(following)
            side = ('upper' if level > model.lake.spec.upper_evaluation_m+1e-12
                    else 'lower' if level < model.lake.spec.lower_evaluation_m-1e-12 else None)
            if side:
                return dict(side=side, lead_day=lead+1, hour_in_day=hour+1,
                    level_before_m=float(before), attempted_level_m=float(level),
                    inflow_m3_s=float(inflow), release_m3_s=float(release))
            state, _ = model.advance(state, float(request), float(inflow), float(target), int(day))
    return None


def full_open_lower_bound(model, state, inflows):
    """Relax nominal commands: use all outlet capacity, subject to reserve cap.

    This is a state lower bound, not an executable control policy. The bound map
    h + dt/A*(q-min(capacity(h),reserve(h))) is order preserving when dt*L/A<=1.
    L bounds the slope of both release curves on the declared domain. Any actual
    release is no greater than this relaxed release at the same state. Induction
    then bounds every admissible trajectory from below, until the first exit.
    An upper exit certifies infeasibility for that forecast in this hourly model.
    A non-exit does not prove a nominally bounded schedule exists.
    """
    if type(model) is not ProtectedOutletModel or state.river_storage:
        raise ValueError('Fixed protected bypass model required')
    inflows = np.asarray(inflows, float)
    if inflows.ndim != 1 or not len(inflows) or not np.isfinite(inflows).all() or (inflows < 0).any():
        raise ValueError('Finite nonnegative daily inflows required')
    lake = model.lake
    if lake.beta < 1 or lake.C_r <= 0:
        raise ValueError('Monotone convex outlet curve required')
    capacity_slope = lake.C_r*lake.beta*(lake.spec.upper_evaluation_m-lake.alpha)**(lake.beta-1)
    reserve_slope = lake.surface/(lake.rule.reserve_hours*3600)
    slope_factor = 3600*max(capacity_slope, reserve_slope)/lake.surface
    if slope_factor > 1:
        raise ValueError('Hourly lower-bound map is not certified order preserving')
    storage = state.storage
    rows = []
    lake.check_domain(lake.storage_to_level(storage))
    for lead, inflow in enumerate(inflows):
        for hour in range(24):
            before = lake.storage_to_level(storage)
            release = min(lake.C_r*(before-lake.alpha)**lake.beta,
                          lake.rule.release_cap(before, lake.surface))
            following = storage+3600*(inflow-release)
            level = lake.storage_to_level(following)
            rows.append(dict(lead_day=lead+1, hour_in_day=hour+1, level_before_m=float(before),
                level_m=float(level), inflow_m3_s=float(inflow), relaxed_release_m3_s=float(release)))
            if level > lake.spec.upper_evaluation_m+1e-12:
                return dict(certified_upper_exit=True, first_exit_lead_day=lead+1,
                    first_exit_hour=hour+1, attempted_level_m=float(level),
                    order_preserving_slope_factor=float(slope_factor), rows=rows)
            if level < lake.spec.lower_evaluation_m-1e-12:
                raise AssertionError('Unexpected lower exit in the nonnegative-input relaxation')
            storage = following
    return dict(certified_upper_exit=False, first_exit_lead_day=None, first_exit_hour=None,
        attempted_level_m=None, order_preserving_slope_factor=float(slope_factor), rows=rows)


def augmented_schedules(model, targets, days, incumbent):
    """Fixed bounded audit: 17 constants, binary plans, nine-grid two-stage plans,
    service proposals and maximum-first/service-tail schedules. Not a new runner.
    """
    horizon = len(targets)
    grid = np.linspace(model.low, model.high, 9).astype(np.float32)
    plans = [np.asarray(incumbent, np.float32)]
    plans += [np.full(horizon, v, np.float32) for v in np.linspace(model.low, model.high, 17).astype(np.float32)]
    plans += [np.array(values, np.float32) for values in product([np.float32(model.low), np.float32(model.high)], repeat=horizon)]
    for cut in range(1, horizon):
        for first, second in product(grid, repeat=2):
            plans.append(np.r_[np.repeat(first, cut), np.repeat(second, horizon-cut)])
    for precise in [False, True]:
        service = service_requests(model, targets, days, precision_safe=precise)
        for cut in range(horizon+1):
            plan = service.copy()
            plan[:cut] = np.float32(model.high)
            plans.append(plan)
    return np.array(list({tuple(plan): plan for plan in plans}.values()), np.float32)


def audit_forecast(model, state, decision, days):
    flows = np.asarray(decision['forecast_inflows'])
    targets = np.asarray(decision['forecast_target'])
    incumbent = np.asarray(decision['selected_plan'], np.float32)
    schedules = augmented_schedules(model, targets, days, incumbent)
    results = evaluate_schedules(model, state, schedules, flows, targets, days)
    costs = results['scenario_costs'].mean(axis=1)
    finite = np.flatnonzero(np.isfinite(costs))
    witness = None
    if len(finite):
        index = int(finite[np.argmin(costs[finite])])
        witness = dict(plan=schedules[index].tolist(), expected_cost=float(costs[index]))
        # Feasibility witness independently checked against scalar physics.
        for path in flows:
            model.rollout(state, np.repeat(schedules[index], 24), np.repeat(path, 24),
                          np.repeat(targets, 24), np.repeat(days, 24))
    paths = []
    for index, path in enumerate(flows):
        bound = full_open_lower_bound(model, state, path)
        paths.append(dict(scenario=index, actual_schedule_exit=first_exit(model, state, incumbent, path, targets, days),
            relaxation={k: v for k, v in bound.items() if k != 'rows'}, bound_rows=bound['rows']))
    certified = any(path['relaxation']['certified_upper_exit'] for path in paths)
    if certified and witness is not None:
        raise AssertionError('Feasible witness contradicts the outlet-capacity bound')
    return dict(candidate_count=len(schedules), feasible_candidates=len(finite), witness=witness,
        certified_infeasible_forecast=certified, paths=paths,
        interpretation=('At least one scenario exceeds the domain even under relaxed full opening; no bounded schedule can satisfy all forecast paths.'
                        if certified else 'No hydraulic certificate; bounded search results alone do not establish global infeasibility.'))
