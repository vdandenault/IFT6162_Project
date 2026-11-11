"""Verify paired objectives and finite-grid selection independently of SLSQP."""
from copy import deepcopy
import unittest

import numpy as np

from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.mpc import MPCConfig, PlanningModel
from reservoir_control.systems.lake_como.mpc_objective_audit import compare_grid, score_plan


class MPCObjectiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = settings("train")
        cls.model = PlanningModel(cls.params, MPCConfig())

    def test_surrogate_matches_deployed_cost_and_discounting(self):
        model = self.model
        days = np.array([170, 171, 172])
        for level in [-0.46, 0.35, 1.26]:
            storage = model.lake.level_to_storage(level)
            requests = np.array([30.0, 150.0, 400.0], dtype=np.float32).astype(float)
            inflows = np.array([100.0, 250.0, 10.0])
            demands = np.array([200.0, 210.0, 220.0])
            pair = score_plan(model, requests, storage, inflows, demands, days)
            normalized = (requests - model.low) / (model.high - model.low)
            self.assertAlmostEqual(
                pair.surrogate,
                model.cost(normalized, storage, inflows, demands, days),
                places=10,
            )

    def test_binary_cost_matches_actual_environment_with_known_forcing(self):
        model = self.model
        env = make_env("validation", 1988)
        try:
            env.reset()
            sim = env.unwrapped
            sim.inflow = [0.0] * 365
            sim.demand = [100.0] * 365
            sim.level[0] = -0.449
            sim.storage[0] = sim.lake.level_to_storage(-0.449)
            initial = sim.storage[0]
            requests = np.array([20.0, 10.0], dtype=np.float32)
            pair = score_plan(
                model, requests, initial, [0.0, 0.0], [100.0, 100.0], np.array([1, 2])
            )
            rewards = []
            for request in requests:
                _, reward, _, _, _ = env.step(np.array([request], dtype=np.float32))
                rewards.append(reward)
            expected = (
                -np.dot(model.config.discount_factor ** np.arange(2), rewards) / 1000
            )
            self.assertAlmostEqual(pair.binary, expected, places=10)
            self.assertNotAlmostEqual(pair.surrogate, pair.binary, places=3)
        finally:
            env.close()

    def test_no_level_violations_give_identical_objectives_and_choices(self):
        model = self.model
        storage = model.lake.level_to_storage(0.35)
        row, choices, _ = compare_grid(
            model, storage, [0.0], [100.0], np.array([1]), 21
        )
        self.assertEqual(row["grid_evaluations"], 21)
        self.assertAlmostEqual(row["binary_regret"], 0.0, places=12)
        self.assertAlmostEqual(row["surrogate_regret"], 0.0, places=12)
        np.testing.assert_array_equal(
            choices["binary"]["requests"], choices["surrogate"]["requests"]
        )

    def test_tiny_violation_is_selected_by_surrogate_but_avoided_by_binary(self):
        params = deepcopy(self.params)
        params["lake_params"]["min_env_flow"] = np.zeros(365)
        model = PlanningModel(params, MPCConfig())
        row, choices, _ = compare_grid(
            model,
            model.lake.level_to_storage(-0.449),
            [0.0],
            [100.0],
            np.array([1]),
            401,
        )
        self.assertLess(choices["surrogate"]["score"].levels[0], model.low_level)
        self.assertGreaterEqual(choices["binary"]["score"].levels[0], model.low_level)
        self.assertGreater(row["binary_regret"], 100.0)
        self.assertGreater(row["surrogate_regret"], 0.0)
        self.assertTrue(row["material_binary_regret"])


if __name__ == "__main__":
    unittest.main()
