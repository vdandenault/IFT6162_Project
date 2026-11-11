"""Causal forecast timing and training partition guards."""
import unittest
import numpy as np
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.mpc_forecast_study import HistoricalInflowPolicy, fit_anomaly_weight


class ForecastChecks(unittest.TestCase):
    def setUp(self):
        self.params = settings("train")
        self.config = MPCConfig(horizon_days=1)
        self.seasonal = DeterministicMPC.from_training(self.config).forecast

    def obs(self, day):
        phase = 2 * np.pi * (day - 1) / 365
        return np.array([0.35, np.sin(phase), np.cos(phase), 0], dtype=np.float32)

    def test_persistence_uses_completed_day_and_resets(self):
        p = HistoricalInflowPolicy(
            self.params, self.seasonal, self.config, "binary", "persistence", 0.8, 101
        )
        storage = 1e8
        p(self.obs(1), {"storage": storage, "future_inflow": 9999})
        self.assertEqual(
            p.diagnostics[-1]["forecast_inflow_m3_s"], self.seasonal.inflow[0]
        )
        p(
            self.obs(2),
            {
                "storage": storage + 86400 * (123 - 50),
                "release": 50,
                "future_inflow": 9999,
            },
        )
        self.assertAlmostEqual(p.diagnostics[-1]["forecast_inflow_m3_s"], 123)
        p(self.obs(1), {"storage": storage})
        self.assertIsNone(p.diagnostics[-1]["previous_inflow_estimate_m3_s"])
        with self.assertRaisesRegex(ValueError, "Consecutive"):
            p(self.obs(3), {"storage": storage, "release": 50})

    def test_fitted_weight_uses_training_partition_only(self):
        alpha = fit_anomaly_weight(self.params["inflow"], self.seasonal.inflow)
        self.assertTrue(0 <= alpha <= 1)
        wrong = dict(self.params["inflow"])
        wrong[1998] = wrong[1946]
        with self.assertRaisesRegex(ValueError, "training"):
            fit_anomaly_weight(wrong, self.seasonal.inflow)

    def test_seasonal_anomaly_formula(self):
        p = HistoricalInflowPolicy(
            self.params,
            self.seasonal,
            self.config,
            "surrogate",
            "seasonal_anomaly",
            0.8,
            101,
        )
        storage = 1e8
        p(self.obs(1), {"storage": storage})
        previous = self.seasonal.inflow[0] + 10
        p(self.obs(2), {"storage": storage + 86400 * previous, "release": 0})
        self.assertAlmostEqual(
            p.diagnostics[-1]["forecast_inflow_m3_s"], self.seasonal.inflow[1] + 8
        )


if __name__ == "__main__":
    unittest.main()
