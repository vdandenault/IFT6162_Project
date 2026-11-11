"""The oracle intervention must change inflow alone and remain diagnostic-only."""
import unittest

import numpy as np

from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.mpc_hindsight import perfect_inflow_controller


class PerfectInflowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = settings("train")
        cls.config = MPCConfig(horizon_days=2)
        cls.reference = DeterministicMPC.from_training(cls.config)
        cls.validation = settings("validation")["inflow"]

    def test_only_inflow_changes_and_arrays_cannot_be_mutated(self):
        oracle = perfect_inflow_controller(
            1988, self.params, self.reference.forecast, self.config, self.validation
        )
        np.testing.assert_array_equal(
            oracle.forecast.inflow, np.array(self.validation[1988][:365])
        )
        np.testing.assert_array_equal(
            oracle.forecast.demand, self.reference.forecast.demand
        )
        self.assertEqual(oracle.config, self.config)
        self.assertEqual(
            oracle.forecast.demand_training_years,
            self.reference.forecast.training_years,
        )
        self.assertFalse(oracle.forecast.inflow.flags.writeable)
        self.assertFalse(oracle.forecast.demand.flags.writeable)
        self.assertFalse(
            np.array_equal(oracle.forecast.inflow, self.reference.forecast.inflow)
        )
        with self.assertRaisesRegex(ValueError, "restricted to validation"):
            perfect_inflow_controller(
                2000, self.params, self.reference.forecast, self.config, self.validation
            )

    def test_seasonal_inflow_injection_reproduces_original_controller(self):
        normal = DeterministicMPC(self.params, self.reference.forecast, self.config)
        injected = perfect_inflow_controller(
            1988,
            self.params,
            self.reference.forecast,
            self.config,
            {1988: self.reference.forecast.inflow},
        )
        for day in range(3):
            phase = 2 * np.pi * day / 365
            obs = np.array([0.35, np.sin(phase), np.cos(phase), 0], dtype=np.float32)
            np.testing.assert_array_equal(normal(obs), injected(obs))
            self.assertEqual(
                normal.diagnostics[-1]["planned_cost"],
                injected.diagnostics[-1]["planned_cost"],
            )

    def test_values_beyond_existing_horizon_do_not_change_first_action(self):
        first = np.array(self.validation[1988][:365])
        second = first.copy()
        second[2:] += 10000
        a = perfect_inflow_controller(
            1988, self.params, self.reference.forecast, self.config, {1988: first}
        )
        b = perfect_inflow_controller(
            1988, self.params, self.reference.forecast, self.config, {1988: second}
        )
        obs = np.array([0.35, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(a(obs), b(obs))

    def test_exact_inflow_prediction_on_real_transitions(self):
        oracle = perfect_inflow_controller(
            1988, self.params, self.reference.forecast, self.config, self.validation
        )
        env = make_env("validation", 1988)
        try:
            obs, info = env.reset()
            for day in range(3):
                action = oracle(obs, info)
                obs, _, _, _, info = env.step(action)
                diagnostic = oracle.diagnostics[-1]
                self.assertEqual(
                    diagnostic["forecast_inflow_m3_s"], self.validation[1988][day]
                )
                self.assertEqual(
                    diagnostic["forecast_demand_m3_s"],
                    self.reference.forecast.demand[day],
                )
                self.assertAlmostEqual(
                    diagnostic["predicted_level_m"], info["level"], places=6
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
