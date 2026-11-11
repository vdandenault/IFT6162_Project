import unittest

import numpy as np

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.carryover_check import (
    BoundaryHistory, make_controller as original_controller, restore_branch, restore_memory, take_step,
)
from reservoir_control.systems.lake_como.simulation.precision_check import make_controller
from reservoir_control.systems.lake_como.simulation.precision_mpc import service_requests, select_schedule
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank


class PrecisionMPCChecks(unittest.TestCase):
    def setUp(self):
        self.training = settings('train')
        self.bank = CausalForecastBank(self.training['inflow'], self.training['demand'], before_year=1988)

    def test_exact_ties_prefer_lower_first_request_then_sum_then_order(self):
        schedules = np.array([[120., 120.], [80., 90.], [80., 80.], [80., 80.]])
        self.assertEqual(select_schedule([0., 0., 0., 0.], schedules), (0, 4))
        self.assertEqual(select_schedule([0., 0., 0., 0.], schedules, conservative=True), (2, 4))

    def test_no_primary_cost_sacrifice_even_for_tiny_differences(self):
        schedules = np.array([[120., 120.], [80., 80.]])
        self.assertEqual(select_schedule([0., 1e-12], schedules, conservative=True), (0, 1))
        self.assertEqual(select_schedule([np.inf, np.inf], schedules, conservative=True), (None, 0))
        self.assertEqual(select_schedule([np.inf, 0.], schedules, conservative=True), (1, 1))

    def test_precision_ceiling_serves_representable_targets_without_exceeding_bounds(self):
        controller = make_controller(self.training, self.bank, True, False)
        model = controller.model
        days = np.arange(1, 366)
        targets = np.linspace(0., 400., 365)
        requests = service_requests(model, targets, days, precision_safe=True)
        exact = targets + model.lake.min_env_flow[days-1]
        self.assertTrue((requests.astype(float) >= exact).all())
        self.assertTrue((requests <= model.high).all())
        # Unattainable targets must retain shortage, never exceed action bounds.
        requests = service_requests(model, np.full(365, 1000.), days, precision_safe=True)
        np.testing.assert_array_equal(requests, np.float32(model.high))

    def test_both_disabled_reproduce_original_closed_loop(self):
        history = BoundaryHistory(8760, 78.1, 41.4, 79., 8736, tuple([122.902496]*7))
        old = original_controller(self.training, self.bank)
        new = make_controller(self.training, self.bank, False, False)
        trajectories = []
        for controller in [old, new]:
            env, obs, info = restore_branch(self.bank, history, 84_986_537.)
            restore_memory(controller, history)
            trajectory = []
            for _ in range(8):
                obs, info, row, _, _ = take_step(env, controller, obs, info)
                trajectory.append(list(row.values()))
            trajectories.append(trajectory)
            env.close()
        np.testing.assert_array_equal(trajectories[0], trajectories[1])

    def test_all_arms_use_identical_causal_information_and_primary_costs(self):
        history = BoundaryHistory(8760, 78.1, 41.4, 79., 8736, tuple([122.902496]*7))
        env, obs, info = restore_branch(self.bank, history, 84_986_537.)
        controls = [make_controller(self.training, self.bank, p, t)
                    for p, t in [(False, False), (True, False), (False, True), (True, True)]]
        for controller in controls:
            restore_memory(controller, history)
            action = controller(obs, info)
            self.assertEqual(action.dtype, np.float32)
            self.assertEqual(controller.weights, controls[0].weights)
            self.assertTrue(env.action_space.contains(action))
            np.testing.assert_array_equal(controller.diagnostics[-1]['forecast_inflows'],
                                          controls[0].diagnostics[-1]['forecast_inflows'])
            np.testing.assert_array_equal(controller.diagnostics[-1]['forecast_target'],
                                          controls[0].diagnostics[-1]['forecast_target'])
        env.close()


if __name__ == '__main__':
    unittest.main()
