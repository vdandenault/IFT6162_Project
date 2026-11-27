from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, nnls


@dataclass(frozen=True)
class ConvexResponse:
    x: np.ndarray
    objective: float
    primal_residual: float
    stationarity_residual: float
    complementarity_residual: float
    multipliers: np.ndarray
    active: np.ndarray
    solver_success: bool

    def sensitivity(self, hessian, linear_derivative, constraints, bounds_derivative=None):
        """dx/dtheta for fixed H/A and parameter-dependent g/b."""
        h = np.asarray(hessian, float)
        a = np.asarray(constraints, float)[self.active]
        rhs = np.asarray(linear_derivative, float)
        block = np.block([[h, a.T], [a, np.zeros((len(a), len(a)))]])
        db = (np.zeros((len(constraints), rhs.shape[1])) if bounds_derivative is None
              else np.asarray(bounds_derivative, float))
        target = np.vstack([-rhs, db[self.active]])
        return np.linalg.lstsq(block, target, rcond=1e-10)[0][:len(self.x)]


def solve_response(hessian, linear, constraints, bounds, initial):
    h, g, a, b, x0 = (np.asarray(v, float) for v in
                       (hessian, linear, constraints, bounds, initial))
    n = len(g)
    if (h.shape != (n, n) or a.ndim != 2 or a.shape[1] != n
            or b.shape != (len(a),) or x0.shape != (n,)
            or not all(np.isfinite(v).all() for v in (h, g, a, b, x0))
            or not np.allclose(h, h.T) or np.linalg.eigvalsh(h).min() <= 0):
        raise ValueError('Finite strictly convex QP with aligned arrays required')
    fit = minimize(lambda x: .5*x@h@x + g@x, x0,
                   jac=lambda x: h@x+g, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': lambda x: b-a@x,
                                'jac': lambda x: -a},
                   options={'ftol': 1e-11, 'maxiter': 300})
    slack = b-a@fit.x
    active = np.flatnonzero(slack < 2e-6)
    dual = np.zeros(len(a))
    if len(active):
        dual[active] = nnls(a[active].T, -(h@fit.x+g), maxiter=1000)[0]
    primal = float(max(0, -slack.min()))
    stationarity = float(np.max(np.abs(h@fit.x+g+a.T@dual)))
    comp = float(np.max(np.abs(dual*slack)))
    # For a strictly convex QP, independently checked KKT residuals certify the
    # solution even when SLSQP reports a line-search warning near an active bound.
    if max(primal, stationarity, comp) > 2e-4:
        raise RuntimeError(f'Follower QP failed: {fit.message}; KKT={primal, stationarity, comp}')
    return ConvexResponse(fit.x, float(fit.fun), primal, stationarity, comp, dual, active, bool(fit.success))
