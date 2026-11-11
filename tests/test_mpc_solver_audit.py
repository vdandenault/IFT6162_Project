"""Diagnostic selection and discarded-iterate accounting."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.mpc_solver_audit import (
    first_action_slice,
    inspect_result,
    select_slice_cases,
)


class SolverAuditChecks(unittest.TestCase):
    def test_nearest_control_ties_choose_earlier_and_avoid_fallback(self):
        frame = pd.DataFrame(
            dict(
                method=["persistence"] * 5,
                year=[1988] * 5,
                day=[1, 2, 3, 4, 5],
                solver_success=[True, True, False, True, True],
                used_fallback=[False, True, True, False, False],
            )
        )
        selected = select_slice_cases(frame)
        self.assertEqual(selected.control_day.iloc[0], 4)
        frame.loc[1, "used_fallback"] = False
        selected = select_slice_cases(frame)
        self.assertEqual(selected.control_day.iloc[0], 2)

    def test_discarded_feasible_iterate_can_improve_initial_without_success(self):
        objective = lambda x: float(np.sum(x * x))
        result = SimpleNamespace(
            x=np.array([0.2, 0.3]), jac=np.array([0.4, 0.6]), success=False
        )
        audit = inspect_result(result, objective, 1.0)
        self.assertTrue(audit["raw_within_bounds"])
        self.assertAlmostEqual(audit["raw_gain_over_initial"], 0.87)
        outside = inspect_result(SimpleNamespace(x=np.array([2.0])), objective, 1.0)
        self.assertFalse(outside["raw_within_bounds"])
        self.assertTrue(np.isnan(outside["raw_gain_over_initial"]))

    def test_slice_preserves_tail_and_includes_original_plan(self):
        class Model:
            low = 0.0
            high = 100.0

            def cost(self, plan, storage, inflows, demands, days):
                self_test.assertEqual(plan[1], 0.8)
                return (plan[0] - 0.5) ** 2

        self_test = self
        plan = np.array([0.1, 0.8])
        result = first_action_slice(
            Model(), plan, 0, [0, 0], [0, 0], np.array([1, 2]), 5
        )
        np.testing.assert_array_equal(plan, [0.1, 0.8])
        self.assertAlmostEqual(result["slice_request_m3_s"], 50.0)
        self.assertAlmostEqual(result["cost_gain"], 0.16)
        self.assertEqual(result["evaluated_points"], 6)


if __name__ == "__main__":
    unittest.main()
