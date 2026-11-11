"""Candidate safeguards and isolation of the experimental selection change."""
import ast
import inspect
import textwrap
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.mpc_retention import RetainingMPC, select_plan


class RetentionChecks(unittest.TestCase):
    def result(self, x, status=9, success=False):
        return OptimizeResult(
            x=np.array(x, dtype=float),
            success=success,
            status=status,
            fun=-9999.0,
            nit=35,
            nfev=100,
            message="Iteration limit reached",
        )

    def test_only_finite_bounded_cheaper_iteration_limit_iterate_is_retained(self):
        initial = np.array([0.8, 0.8])
        objective = lambda x: float(np.dot(x, x))
        cost = objective(initial)
        result = self.result([0.2, 0.3])
        plan, fallback, retained, candidate_cost = select_plan(
            result, initial, cost, objective, True
        )
        np.testing.assert_array_equal(plan, result.x)
        self.assertFalse(fallback)
        self.assertTrue(retained)
        self.assertAlmostEqual(
            candidate_cost, 0.13
        )  # result.fun is deliberately false.
        for rejected in [
            self.result([0.2, 0.3], status=8),
            self.result([np.nan, 0.3]),
            self.result([-0.1, 0.3]),
            self.result([1.0, 1.0]),
            self.result(initial),
            self.result([0.2]),
        ]:
            chosen, fallback, retained, _ = select_plan(
                rejected, initial, cost, objective, True
            )
            np.testing.assert_array_equal(chosen, initial)
            self.assertTrue(fallback)
            self.assertFalse(retained)
        _, fallback, retained, _ = select_plan(result, initial, cost, objective, False)
        self.assertTrue(fallback)
        self.assertFalse(retained)

    def test_nonfinite_recomputed_cost_is_rejected(self):
        initial = np.array([0.8])
        _, fallback, retained, _ = select_plan(
            self.result([0.2]), initial, 0.64, lambda x: float("nan"), True
        )
        self.assertTrue(fallback)
        self.assertFalse(retained)

    def test_solver_setup_is_identical_to_reference(self):
        def prefix(function):
            nodes = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
            # Both functions are identical through the SLSQP call.
            end = next(
                i
                for i, n in enumerate(nodes)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
            )
            return ast.dump(ast.Module(body=nodes[: end + 1], type_ignores=[]))

        self.assertEqual(
            prefix(DeterministicMPC.__call__), prefix(RetainingMPC.__call__)
        )

    def test_disabled_arm_matches_reference_and_retention_preserves_failure_flag(self):
        params = settings("train")
        config = MPCConfig(horizon_days=2)
        seasonal = DeterministicMPC.from_training(config).forecast
        old = DeterministicMPC(params, seasonal, config)
        control = RetainingMPC(params, seasonal, config)
        for day in [1, 2, 3, 1]:
            phase = 2 * np.pi * (day - 1) / 365
            obs = np.array([0.35, np.sin(phase), np.cos(phase), 0], dtype=np.float32)
            np.testing.assert_array_equal(old(obs), control(obs))
            self.assertEqual(
                old.diagnostics[-1]["planned_cost"],
                control.diagnostics[-1]["planned_cost"],
            )
        target = np.array([0.3, 0.7])
        experimental = RetainingMPC(
            params, seasonal, config, retain_iteration_limit=True
        )
        experimental.model.cost = lambda x, *args: float(np.sum((x - target) ** 2))
        with patch(
            "reservoir_control.systems.lake_como.mpc_retention.minimize", return_value=self.result(target)
        ):
            experimental(np.array([0.35, 0, 1, 0], dtype=np.float32))
        diagnostic = experimental.diagnostics[-1]
        self.assertTrue(diagnostic["retained_iteration_limit"])
        self.assertFalse(diagnostic["solver_success"])
        self.assertEqual(diagnostic["solver_status"], 9)
        self.assertFalse(diagnostic["used_fallback"])
        np.testing.assert_array_equal(experimental.previous_plan, target)


if __name__ == "__main__":
    unittest.main()
