"""Horizon ablation isolation, information timing, and solver-budget checks."""
from dataclasses import asdict
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.mpc_horizon_study import make_policy


class HorizonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = settings("train")
        cls.config = MPCConfig.load()
        cls.seasonal = DeterministicMPC.from_training(cls.config).forecast

    def test_only_horizon_changes_and_slsqp_settings_are_shared(self):
        options = []
        for horizon in [1, 3, 7]:
            policy = make_policy(self.params, self.seasonal, self.config, horizon)
            expected = asdict(self.config)
            expected["horizon_days"] = horizon
            self.assertEqual(asdict(policy.controller.config), expected)
            self.assertTrue(policy.controller.retain_iteration_limit)

            def fake_minimize(objective, initial, **kwargs):
                self.assertEqual(len(initial), horizon)
                self.assertEqual(kwargs["method"], "SLSQP")
                self.assertEqual(kwargs["bounds"], [(0.0, 1.0)] * horizon)
                options.append(kwargs["options"])
                return OptimizeResult(
                    x=initial.copy(),
                    success=True,
                    status=0,
                    nit=1,
                    nfev=1,
                    message="test",
                )

            with patch("reservoir_control.systems.lake_como.mpc_retention.minimize", side_effect=fake_minimize):
                policy(
                    np.array([0.35, 0, 1, 0], dtype=np.float32),
                    {"storage": 1e8, "future_inflow": 99999},
                )
            np.testing.assert_array_equal(
                policy.forecasts[-1], self.seasonal.inflow[:horizon]
            )
        self.assertEqual(options[0], options[1])
        self.assertEqual(options[1], options[2])
        self.assertEqual(options[0]["maxiter"], 35)
        self.assertEqual(self.config.horizon_days, 7)
        with self.assertRaisesRegex(ValueError, "fixed"):
            make_policy(self.params, self.seasonal, self.config, 2)

    def test_all_horizons_shrink_to_one_on_last_day_without_future_profile(self):
        phase = 2 * np.pi * 364 / 365
        for horizon in [1, 3, 7]:
            policy = make_policy(self.params, self.seasonal, self.config, horizon)
            policy.previous_day = 363
            policy.previous_storage = 1e8
            policy(
                np.array([0.35, np.sin(phase), np.cos(phase), 0], dtype=np.float32),
                {"storage": 1e8 + 86400 * (123 - 50), "release": 50},
            )
            self.assertEqual(policy.diagnostics[-1]["horizon_days"], 1)
            np.testing.assert_allclose(policy.forecasts[-1], [123.0])


if __name__ == "__main__":
    unittest.main()
