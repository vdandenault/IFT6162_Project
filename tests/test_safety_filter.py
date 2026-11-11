"""Common scenario checks, intervention selection, recovery, and causal timing."""
import unittest
import numpy as np
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import MPCConfig, PlanningModel, SeasonalForecast
from reservoir_control.systems.lake_como.safety_filter import choose_request, FloodSafetyFilter, fit_error_limits


class SafetyFilterChecks(unittest.TestCase):
    def test_closest_safe_request_and_no_change_when_proposal_passes(self):
        requests = np.array([0.0, 100.0, 200.0])
        end = np.array([[1.3, 1.2, 1.1], [1.4, 1.22, 1.15]])
        peak = end.copy()
        index, feasible, *_ = choose_request(requests, 90.0, end, peak, 1.24)
        self.assertTrue(feasible)
        self.assertEqual(requests[index], 100)
        index, feasible, *_ = choose_request(requests, 200.0, end, peak, 1.24)
        self.assertEqual(requests[index], 200)

    def test_every_scenario_and_hourly_peak_must_pass(self):
        requests = np.array([0.0, 100.0])
        end = np.array([[1.1, 1.1], [1.1, 1.1]])
        peak = np.array([[1.1, 1.2], [1.3, 1.21]])
        index, feasible, *_ = choose_request(requests, 0.0, end, peak, 1.24)
        self.assertTrue(feasible)
        self.assertEqual(index, 1)

    def test_recovery_breaks_unavoidable_initial_peak_tie_with_endpoint(self):
        requests = np.array([0.0, 100.0, 200.0])
        end = np.array([[1.3, 1.2, 1.1]])
        peak = np.full((1, 3), 1.4)
        index, feasible, *_ = choose_request(requests, 0.0, end, peak, 1.24)
        self.assertFalse(feasible)
        self.assertEqual(index, 2)

    def test_filter_uses_only_past_telemetry_and_resets_history(self):
        params = settings("train")
        model = PlanningModel(params, MPCConfig.load())
        season = SeasonalForecast.fit(params["inflow"], params["demand"])
        a = FloodSafetyFilter(model, season.inflow[0], (-100, 200), 21)
        b = FloodSafetyFilter(model, season.inflow[0], (-100, 200), 21)
        obs = np.array([0.35, 0, 1, 0, season.inflow[0]], dtype=np.float32)
        np.testing.assert_array_equal(
            a([100.0], obs, {"storage": 1e8, "future_inflow": 0}),
            b([100.0], obs, {"storage": 1e8, "future_inflow": 99999}),
        )
        phase = 2 * np.pi / 365
        obs[1:3] = [np.sin(phase), np.cos(phase)]
        a([100.0], obs, {"storage": 1e8 + 86400 * (123 - 50), "release": 50})
        self.assertAlmostEqual(a.diagnostics[-1]["forecast_past_inflow_m3_s"], 123)
        obs[1:3] = [0, 1]
        a([100.0], obs, {"storage": 1e8})
        self.assertAlmostEqual(
            a.diagnostics[-1]["forecast_past_inflow_m3_s"], season.inflow[0]
        )
        with self.assertRaisesRegex(ValueError, "observation"):
            a([100.0], np.array([np.nan, 0, 1, 0]), {"storage": 1e8})

    def test_calibration_is_training_only_and_matches_empirical_quantiles(self):
        params = settings("train")
        season = SeasonalForecast.fit(params["inflow"], params["demand"])
        lower, upper = fit_error_limits(params["inflow"], season.inflow)
        self.assertLess(lower, 0)
        self.assertEqual(upper, 283.5)
        wrong = dict(params["inflow"])
        wrong[1988] = wrong[1946]
        with self.assertRaisesRegex(ValueError, "Training"):
            fit_error_limits(wrong, season.inflow)


if __name__ == "__main__":
    unittest.main()
