from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from reservoir_control.controllers.mpc.convex_response import solve_response


class InvalidFollowerInput(ValueError):
    """Malformed arrays, nonphysical routing, or invalid objective parameters."""


@dataclass(frozen=True)
class FollowerWeights:
    storage: float = 1.0
    release: float = 1.0
    change: float = 0.2
    terminal: float = 2.0


@dataclass(frozen=True)
class HardFollowerProblem:
    initial_storage_m3: float
    previous_release_m3_s: float
    net_inflow_m3_s: object
    ontario_executable_release_m3_s: object
    other_downstream_m3_s: object
    routing_matrix: object
    in_transit_contribution_m3_s: object
    downstream_capacity_m3_s: object
    storage_minimum_m3: object  # H+1, including the initial-state bound
    storage_maximum_m3: object
    release_minimum_m3_s: object  # H
    release_maximum_m3_s: object
    storage_target_m3: object  # H+1
    release_target_m3_s: object  # H
    duration_seconds: object  # H; may be unequal
    storage_scale_m3: float
    flow_scale_m3_s: float
    weights: FollowerWeights = FollowerWeights()


def make_problem(config, scenario, step, storage, previous, transit, forecast):
    """Build the follower inputs for one configured decision window."""
    horizon = len(forecast)
    window = slice(step, step + horizon)
    durations = np.full(horizon, config["clock"]["duration_seconds"])
    routing_matrix, arrivals = transit.forecast(durations)
    return HardFollowerProblem(
        initial_storage_m3=storage,
        previous_release_m3_s=previous,
        net_inflow_m3_s=forecast,
        ontario_executable_release_m3_s=config["known_ontario_executable_schedule_m3_s"][window],
        other_downstream_m3_s=config["known_other_downstream_forecast_m3_s"][window],
        routing_matrix=routing_matrix,
        in_transit_contribution_m3_s=arrivals,
        downstream_capacity_m3_s=scenario["downstream_capacity_m3_s"][window],
        storage_minimum_m3=np.full(horizon + 1, config["storage_minimum_m3"]),
        storage_maximum_m3=np.full(horizon + 1, config["storage_maximum_m3"]),
        release_minimum_m3_s=np.full(horizon, config["release_minimum_m3_s"]),
        release_maximum_m3_s=np.full(horizon, config["release_maximum_m3_s"]),
        storage_target_m3=np.full(horizon + 1, config["storage_target_m3"]),
        release_target_m3_s=np.full(horizon, config["release_target_m3_s"]),
        duration_seconds=durations,
        storage_scale_m3=config["storage_scale_m3"],
        flow_scale_m3_s=config["flow_scale_m3_s"],
        weights=FollowerWeights(**config["weights"]),
    )


@dataclass(frozen=True)
class FollowerTolerances:
    storage_m3: float = 0.01
    flow_m3_s: float = 1e-5
    normalized_kkt: float = 2e-4


@dataclass(frozen=True)
class HardFollowerResult:
    status: str
    message: str
    storage_m3: np.ndarray | None = None
    release_m3_s: np.ndarray | None = None
    downstream_flow_m3_s: np.ndarray | None = None
    first_release_m3_s: float | None = None
    objective: float | None = None
    residuals: dict = field(default_factory=dict)
    pending_planned_release_volume_m3: float | None = None
    raw_solver_success: bool = False
    elapsed_seconds: float = 0.0

    @property
    def success(self):
        return self.status in ("optimal", "optimal_kkt_checked")


def validate_problem(p):
    scalar = [
        p.initial_storage_m3,
        p.previous_release_m3_s,
        p.storage_scale_m3,
        p.flow_scale_m3_s,
        *vars(p.weights).values(),
    ]
    if (
        not np.isfinite(scalar).all()
        or min(p.storage_scale_m3, p.flow_scale_m3_s) <= 0
        or p.previous_release_m3_s < 0
        or p.weights.release <= 0
        or min(vars(p.weights).values()) < 0
    ):
        raise InvalidFollowerInput(
            "Finite inputs, positive scales, nonnegative weights and wr > 0 required"
        )
    inflow = np.asarray(p.net_inflow_m3_s, float)
    if inflow.ndim != 1 or not len(inflow):
        raise InvalidFollowerInput("A nonempty prediction horizon is required")
    h = len(inflow)
    arrays = {}
    names = (
        "net_inflow_m3_s",
        "ontario_executable_release_m3_s",
        "other_downstream_m3_s",
        "in_transit_contribution_m3_s",
        "downstream_capacity_m3_s",
        "release_minimum_m3_s",
        "release_maximum_m3_s",
        "release_target_m3_s",
        "duration_seconds",
        "storage_minimum_m3",
        "storage_maximum_m3",
        "storage_target_m3",
    )
    for name in names:
        value = np.asarray(getattr(p, name), float).copy()
        length = h + 1 if name.startswith("storage_") else h
        if value.shape != (length,) or not np.isfinite(value).all():
            raise InvalidFollowerInput(f"{name} must contain {length} finite values")
        arrays[name] = value
    a = np.asarray(p.routing_matrix, float).copy()
    if a.shape != (h, h) or not np.isfinite(a).all() or (a < 0).any() or np.any(np.triu(a, 1) != 0):
        raise InvalidFollowerInput("Routing must be a finite nonnegative causal H-by-H matrix")
    dt = arrays["duration_seconds"]
    if (dt <= 0).any():
        raise InvalidFollowerInput("Positive SI period durations required")
    if np.any(dt @ a > dt + 1e-9 * dt):
        raise InvalidFollowerInput(
            "Routing creates water: duration-weighted column sums exceed release volume"
        )
    for name in (
        "ontario_executable_release_m3_s",
        "other_downstream_m3_s",
        "in_transit_contribution_m3_s",
        "downstream_capacity_m3_s",
        "release_minimum_m3_s",
    ):
        if (arrays[name] < 0).any():
            raise InvalidFollowerInput(f"{name} must be nonnegative")
    if (
        (arrays["storage_minimum_m3"] < 0).any()
        or np.any(arrays["storage_minimum_m3"] > arrays["storage_maximum_m3"])
        or np.any(arrays["release_minimum_m3_s"] > arrays["release_maximum_m3_s"])
    ):
        raise InvalidFollowerInput("Ordered nonnegative storage/release bounds required")
    arrays["routing_matrix"] = a
    return arrays


def objective_value(p, storage, release):
    w = p.weights
    target_s, target_r = np.asarray(p.storage_target_m3), np.asarray(p.release_target_m3_s)
    previous = np.r_[p.previous_release_m3_s, release[:-1]]
    return float(
        w.storage * np.sum(((storage[1:] - target_s[1:]) / p.storage_scale_m3) ** 2)
        + w.release * np.sum(((release - target_r) / p.flow_scale_m3_s) ** 2)
        + w.change * np.sum(((release - previous) / p.flow_scale_m3_s) ** 2)
        + w.terminal * ((storage[-1] - target_s[-1]) / p.storage_scale_m3) ** 2
    )


def solve_hard_follower(p: HardFollowerProblem, tolerances=FollowerTolerances()):
    """Pure solve. Invalid input raises; infeasible/failure results contain no action.

    Hard constraints have no slacks. Pending volume counts new planned releases
    not delivered within this horizon; the caller must also preserve any initial
    transit scheduled beyond the supplied c horizon.
    """
    started = perf_counter()
    try:
        d = validate_problem(p)
    except InvalidFollowerInput:
        raise
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        raise InvalidFollowerInput(f"Malformed follower input: {error}") from error
    if (
        not np.isfinite(list(vars(tolerances).values())).all()
        or min(vars(tolerances).values()) <= 0
    ):
        raise InvalidFollowerInput("Positive finite acceptance tolerances required")

    def failed(status, message, residuals=None):
        return HardFollowerResult(
            status, message, residuals=residuals or {}, elapsed_seconds=perf_counter() - started
        )

    initial_violation = max(
        0.0,
        d["storage_minimum_m3"][0] - p.initial_storage_m3,
        p.initial_storage_m3 - d["storage_maximum_m3"][0],
    )
    if initial_violation > tolerances.storage_m3:
        return failed(
            "infeasible",
            "Fixed initial storage is outside its physical bounds",
            {"initial_storage_violation_m3": initial_violation},
        )
    dt, inflow = d["duration_seconds"], d["net_inflow_m3_s"]
    h = len(dt)
    identity = np.eye(h)
    w = p.weights
    f = (p.initial_storage_m3 + np.cumsum(dt * inflow)) / p.storage_scale_m3
    b = -np.tril(np.ones((h, h))) * dt[None, :] * p.flow_scale_m3_s / p.storage_scale_m3
    target = d["storage_target_m3"][1:] / p.storage_scale_m3
    rref = d["release_target_m3_s"] / p.flow_scale_m3_s
    diff = identity - np.eye(h, k=-1)
    previous = np.r_[p.previous_release_m3_s / p.flow_scale_m3_s, np.zeros(h - 1)]
    hessian = 2 * (
        w.storage * b.T @ b
        + w.release * identity
        + w.change * diff.T @ diff
        + w.terminal * np.outer(b[-1], b[-1])
    )
    linear = 2 * (
        w.storage * b.T @ (f - target)
        - w.release * rref
        - w.change * diff.T @ previous
        + w.terminal * b[-1] * (f[-1] - target[-1])
    )
    routing = d["routing_matrix"]
    constraints = np.vstack([b, -b, identity, -identity, routing])
    limits = np.r_[
        d["storage_maximum_m3"][1:] / p.storage_scale_m3 - f,
        f - d["storage_minimum_m3"][1:] / p.storage_scale_m3,
        d["release_maximum_m3_s"] / p.flow_scale_m3_s,
        -d["release_minimum_m3_s"] / p.flow_scale_m3_s,
        (
            d["downstream_capacity_m3_s"]
            - d["ontario_executable_release_m3_s"]
            - d["other_downstream_m3_s"]
            - d["in_transit_contribution_m3_s"]
        )
        / p.flow_scale_m3_s,
    ]
    if not all(np.isfinite(a).all() for a in (hessian, linear, constraints, limits)):
        raise InvalidFollowerInput("Input/scaling overflowed the QP representation")
    # Fixed downstream arrivals can produce zero QP rows. Assess these in SI
    # units first; sub-tolerance roundoff must not turn a feasible queue infeasible.
    fixed = ~np.any(constraints != 0.0, axis=1)
    fixed_tolerance = np.r_[
        np.full(2 * h, tolerances.storage_m3 / p.storage_scale_m3),
        np.full(3 * h, tolerances.flow_m3_s / p.flow_scale_m3_s),
    ]
    if np.any(limits[fixed] < -fixed_tolerance[fixed]):
        return failed("infeasible", "Fixed in-transit contribution exceeds downstream capacity")
    constraints, limits = constraints[~fixed], limits[~fixed]
    try:
        feasible = linprog(
            np.zeros(h),
            A_ub=constraints,
            b_ub=limits,
            bounds=[(None, None)] * h,
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
    except Exception as error:
        return failed("solver_failure", f"Feasibility LP failed: {type(error).__name__}: {error}")
    if feasible.status == 2:
        return failed("infeasible", "Hard constraints infeasible: " + feasible.message)
    if not feasible.success:
        return failed(
            "solver_failure", "Feasibility LP did not certify feasibility: " + feasible.message
        )
    try:
        fit = solve_response(hessian, linear, constraints, limits, feasible.x)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return failed("solver_failure", f"QP did not pass optimality checks: {error}")
    release = fit.x * p.flow_scale_m3_s
    storage = np.r_[p.initial_storage_m3, p.initial_storage_m3 + np.cumsum(dt * (inflow - release))]
    downstream = (
        d["ontario_executable_release_m3_s"]
        + d["other_downstream_m3_s"]
        + routing @ release
        + d["in_transit_contribution_m3_s"]
    )
    residuals = dict(
        initial_storage_error_m3=abs(storage[0] - p.initial_storage_m3),
        mass_balance_error_m3=float(np.max(np.abs(np.diff(storage) - dt * (inflow - release)))),
        storage_violation_m3=float(
            max(
                0.0,
                np.max(d["storage_minimum_m3"] - storage),
                np.max(storage - d["storage_maximum_m3"]),
            )
        ),
        release_violation_m3_s=float(
            max(
                0.0,
                np.max(d["release_minimum_m3_s"] - release),
                np.max(release - d["release_maximum_m3_s"]),
            )
        ),
        downstream_violation_m3_s=float(
            max(0.0, np.max(downstream - d["downstream_capacity_m3_s"]))
        ),
        normalized_primal=fit.primal_residual,
        normalized_stationarity=fit.stationarity_residual,
        normalized_complementarity=fit.complementarity_residual,
    )
    value = objective_value(p, storage, release)
    checks = [
        residuals[k] <= tolerances.storage_m3
        for k in ("initial_storage_error_m3", "mass_balance_error_m3", "storage_violation_m3")
    ]
    checks += [
        residuals[k] <= tolerances.flow_m3_s
        for k in ("release_violation_m3_s", "downstream_violation_m3_s")
    ]
    checks += [
        residuals[k] <= tolerances.normalized_kkt
        for k in ("normalized_primal", "normalized_stationarity", "normalized_complementarity")
    ]
    if (
        not all(checks)
        or not np.isfinite([*storage, *release, *downstream, value, *residuals.values()]).all()
    ):
        return failed(
            "solver_failure",
            "Solver candidate rejected by independent SI/KKT residual checks",
            residuals,
        )
    pending = float((dt - dt @ routing) @ release)
    return HardFollowerResult(
        "optimal" if fit.solver_success else "optimal_kkt_checked",
        "Strictly convex response accepted after SI and KKT checks",
        storage.copy(),
        release.copy(),
        downstream.copy(),
        float(release[0]),
        value,
        residuals,
        pending,
        fit.solver_success,
        perf_counter() - started,
    )
