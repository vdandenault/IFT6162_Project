"""Timing, multi-lead forecasts, and unchanged seasonal solver checks."""
import unittest
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.mpc_causal import CausalMPC, fit_horizon_weights, merge_severity
from reservoir_control.systems.lake_como.mpc_forecast_study import fit_anomaly_weight


class CausalMPCChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = settings("train")
        cls.config = MPCConfig.load()
        cls.seasonal = DeterministicMPC.from_training(cls.config).forecast
        cls.weights = fit_horizon_weights(cls.params["inflow"], cls.seasonal.inflow)

    def obs(self, day):
        phase = 2 * np.pi * (day - 1) / 365
        return np.array([0.35, np.sin(phase), np.cos(phase), 0], dtype=np.float32)

    def test_severity_merge_preserves_annual_extrema(self):
        annual = pd.DataFrame(
            dict(method=["seasonal"], year=[1988], max_level_m=[1.3], min_level_m=[0.0])
        )
        daily = pd.DataFrame(
            dict(method=["seasonal", "seasonal"], year=[1988, 1988], level_m=[1.3, 0.0])
        )
        merged = merge_severity(annual, daily)
        self.assertEqual(merged.max_level_m.iloc[0], 1.3)
        self.assertAlmostEqual(merged.flood_excess_m_days.iloc[0], 0.06)
        self.assertFalse(any(c.endswith("_x") or c.endswith("_y") for c in merged))

    def test_direct_lead_fit_and_partition(self):
        self.assertAlmostEqual(
            self.weights[0],
            fit_anomaly_weight(self.params["inflow"], self.seasonal.inflow),
        )
        self.assertFalse(self.weights.flags.writeable)
        anomalies = (
            np.array(
                [self.params["inflow"][y][:365] for y in sorted(self.params["inflow"])]
            )
            - self.seasonal.inflow
        )
        for lead, weight in enumerate(self.weights, 1):
            x = anomalies[:, :-lead].ravel()
            y = anomalies[:, lead:].ravel()
            self.assertAlmostEqual(weight, np.clip(np.dot(x, y) / np.dot(x, x), 0, 1))
        bad = dict(self.params["inflow"])
        bad[1998] = bad[1946]
        with self.assertRaisesRegex(ValueError, "training"):
            fit_horizon_weights(bad, self.seasonal.inflow)

    def test_persistence_horizon_and_year_reset(self):
        p = CausalMPC(
            self.params, self.seasonal, self.config, "persistence", self.weights
        )
        storage = 1e8
        p(self.obs(1), {"storage": storage, "future_inflow": 99999})
        np.testing.assert_array_equal(p.forecasts[-1], self.seasonal.inflow[:7])
        p(self.obs(2), {"storage": storage + 86400 * (123 - 50), "release": 50})
        np.testing.assert_allclose(p.forecasts[-1], np.full(7, 123))
        p(self.obs(1), {"storage": storage})
        np.testing.assert_array_equal(p.forecasts[-1], self.seasonal.inflow[:7])
        self.assertIsNone(p.diagnostics[-1]["previous_inflow_estimate_m3_s"])
        with self.assertRaisesRegex(ValueError, "Consecutive"):
            p(self.obs(3), {"storage": storage, "release": 50})

    def test_anomaly_forecast_uses_each_lead_and_shrinks_at_year_end(self):
        p = CausalMPC(
            self.params, self.seasonal, self.config, "seasonal_anomaly", self.weights
        )
        storage = 1e8
        p(self.obs(1), {"storage": storage})
        previous = self.seasonal.inflow[0] + 10
        p(self.obs(2), {"storage": storage + 86400 * previous, "release": 0})
        np.testing.assert_allclose(
            p.forecasts[-1], self.seasonal.inflow[1:8] + 10 * self.weights
        )
        # Inject only previous-day telemetry state, never current/future forcing.
        p.previous_day = 363
        p.previous_storage = storage
        p(self.obs(365), {"storage": storage, "release": 50})
        self.assertEqual(len(p.forecasts[-1]), 1)
        self.assertEqual(p.diagnostics[-1]["horizon_days"], 1)

    def test_seasonal_wrapper_matches_original_solver(self):
        original = DeterministicMPC(self.params, self.seasonal, self.config)
        wrapper = CausalMPC(
            self.params, self.seasonal, self.config, "seasonal", self.weights
        )
        for day in [1, 2, 3]:
            obs = self.obs(day)
            np.testing.assert_array_equal(original(obs), wrapper(obs, {"storage": 1e8}))
            self.assertEqual(
                original.diagnostics[-1]["planned_cost"],
                wrapper.diagnostics[-1]["planned_cost"],
            )
            self.assertEqual(
                original.diagnostics[-1]["used_fallback"],
                wrapper.diagnostics[-1]["used_fallback"],
            )


if __name__ == "__main__":
    unittest.main()
