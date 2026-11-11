"""Scalar physics parity and conservative flood infeasibility bounds."""
import unittest
import numpy as np
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import MPCConfig, PlanningModel, SeasonalForecast
from reservoir_control.systems.lake_como.flood_feasibility import (
    grid_rollout,
    lower_bounds,
    training_error_quantile,
)


class FloodFeasibilityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = settings("train")
        cls.model = PlanningModel(cls.params, MPCConfig.load())
        cls.lake = cls.model.lake

    def test_grid_endpoints_and_hourly_peaks_match_scalar_branch_physics(self):
        lake = self.lake
        requests = np.linspace(0, self.model.high, 9).astype(np.float32).astype(float)
        levels = np.array([-0.5, -0.4, 0.35, 1.24, 1.25, 2.5])
        storage = np.array([lake.level_to_storage(h) for h in levels])
        inflow = np.array([0.0, 100.0, 2000.0, 450.0, 800.0, 2000.0])
        days = np.array([1, 90, 180, 240, 300, 365])
        end, peak = grid_rollout(lake, storage, inflow, days, requests)
        low, first = lower_bounds(lake, storage, inflow, days, self.model.high)
        for i, s in enumerate(storage):
            for j, request in enumerate(requests):
                scalar = s
                highest = lake.storage_to_level(s)
                first_level = None
                for hour in range(24):
                    scalar += 3600 * (
                        inflow[i] - lake.actual_release(request, scalar, int(days[i]))
                    )
                    level = lake.storage_to_level(scalar)
                    highest = max(highest, level)
                    if hour == 0:
                        first_level = level
                self.assertAlmostEqual(end[i, j], level, places=10)
                self.assertAlmostEqual(peak[i, j], highest, places=10)
                self.assertLessEqual(low[i], level + 1e-10)
                self.assertLessEqual(first[i], first_level + 1e-10)

    def test_already_flooded_start_cannot_be_hourly_safe_even_if_endpoint_recovers(
        self,
    ):
        s = np.array([self.lake.level_to_storage(1.26)])
        end, peak = grid_rollout(
            self.lake, s, np.array([0.0]), np.array([1]), np.array([self.model.high])
        )
        self.assertLess(end[0, 0], 1.24)
        self.assertGreater(peak[0, 0], 1.24)

    def test_first_hour_certificate_detects_extreme_inflow_from_safe_state(self):
        s = np.array([self.lake.level_to_storage(1.23)])
        low, first = lower_bounds(
            self.lake, s, np.array([10000.0]), np.array([1]), self.model.high
        )
        self.assertGreater(first[0], 1.24)
        end, peak = grid_rollout(
            self.lake,
            s,
            np.array([10000.0]),
            np.array([1]),
            np.linspace(0, self.model.high, 101),
        )
        self.assertTrue((peak > 1.24).all())
        self.assertLessEqual(low[0], end.min())

    def test_uncertainty_stress_calibration_rejects_validation_membership(self):
        seasonal = SeasonalForecast.fit(self.params["inflow"], self.params["demand"])
        bound = training_error_quantile(self.params["inflow"], seasonal.inflow)
        self.assertGreater(bound, 0)
        bad = dict(self.params["inflow"])
        bad[1988] = bad[1946]
        with self.assertRaisesRegex(ValueError, "training"):
            training_error_quantile(bad, seasonal.inflow)


if __name__ == "__main__":
    unittest.main()
