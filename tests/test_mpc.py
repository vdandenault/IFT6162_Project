"""MPC physics, information boundaries, optimizer behavior, and fallback checks."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig, PlanningModel, SeasonalForecast
from reservoir_control.systems.lake_como.reproduce import evaluate
from reservoir_control.systems.lake_como.splits import split_years


class MPCChecks(unittest.TestCase):
    def test_planning_transition_matches_actual_hourly_hydraulics(self):
        params = settings("train")
        model = PlanningModel(params, MPCConfig())
        env = make_env("validation", 1988)
        try:
            for level, request in [
                (-0.49, 100.0),
                (-0.4, 491.61),
                (0.35, 100.0),
                (1.3, 0.0),
            ]:
                _, info = env.reset()
                sim = env.unwrapped
                sim.storage[0] = sim.lake.level_to_storage(level)
                sim.level[0] = level
                levels, releases = model.predict(
                    sim.storage[0], [request], [sim.inflow[0]], [1]
                )
                _, _, _, _, actual = env.step(np.array([request], dtype=np.float32))
                self.assertAlmostEqual(levels[0], actual["level"], places=9)
                self.assertAlmostEqual(releases[0], actual["release"], places=9)
        finally:
            env.close()

    def test_forecasts_use_only_training_profiles_and_reject_extra_years(self):
        params = settings("train")
        forecast = SeasonalForecast.fit(params["inflow"], params["demand"])
        np.testing.assert_array_equal(
            forecast.inflow,
            np.array([params["inflow"][y][:365] for y in split_years("train")]).mean(
                axis=0
            ),
        )
        self.assertFalse(forecast.inflow.flags.writeable)
        with self.assertRaisesRegex(ValueError, "training years only"):
            SeasonalForecast.fit(
                dict(params["inflow"], **{"1988": [1.0] * 365}), params["demand"]
            )

    def test_equal_observations_ignore_hidden_future_and_previous_info(self):
        config = MPCConfig(horizon_days=2)
        a = DeterministicMPC.from_training(config)
        b = DeterministicMPC.from_training(config)
        obs = np.array([0.35, 0.0, 1.0, 0.0], dtype=np.float32)
        first = a(obs, {"inflow": [999999] * 365, "storage": -999999, "demand": 999999})
        second = b(obs, {"inflow": [0] * 365, "storage": 999999, "demand": 0})
        np.testing.assert_array_equal(first, second)
        self.assertFalse(hasattr(a, "env"))
        self.assertEqual(a.forecast.training_years, tuple(split_years("train")))

    def test_one_day_solution_agrees_with_dense_release_search(self):
        controller = DeterministicMPC.from_training(
            MPCConfig(horizon_days=1, max_iterations=80)
        )
        for level in [-0.44, 0.35, 1.23]:
            obs = np.array([level, 0.0, 1.0, 0.0], dtype=np.float32)
            controller(obs)
            model = controller.model
            storage = model.lake.level_to_storage(float(obs[0]))
            days = np.array([1])
            grid = np.linspace(0, 1, 1001)
            costs = [
                model.cost(
                    [x],
                    storage,
                    controller.forecast.inflow[:1],
                    controller.forecast.demand[:1],
                    days,
                )
                for x in grid
            ]
            self.assertLessEqual(
                controller.diagnostics[-1]["planned_cost"], min(costs) + 1e-4
            )

    def test_solver_failure_has_visible_bounded_fallback(self):
        controller = DeterministicMPC.from_training(MPCConfig(horizon_days=2))
        failed = OptimizeResult(
            success=False,
            x=np.array([np.nan, np.nan]),
            status=9,
            message="injected failure",
            nit=1,
            nfev=1,
        )
        with patch("reservoir_control.systems.lake_como.mpc.minimize", return_value=failed):
            action = controller(np.array([0.35, 0.0, 1.0, 0.0], dtype=np.float32))
        self.assertTrue(controller.diagnostics[-1]["used_fallback"])
        self.assertTrue(np.isfinite(action).all())
        self.assertGreaterEqual(action[0], controller.model.low)
        self.assertLessEqual(action[0], controller.model.high)

    def test_year_end_shrinks_horizon_and_reset_discards_old_plan(self):
        controller = DeterministicMPC.from_training(MPCConfig(horizon_days=7))
        phase = 2 * np.pi * 364 / 365
        controller(
            np.array([0.35, np.sin(phase), np.cos(phase), 1.0], dtype=np.float32)
        )
        self.assertEqual(controller.diagnostics[-1]["horizon_days"], 1)
        initial = np.array([0.35, 0, 1, 0], dtype=np.float32)
        reset = controller(initial)
        fresh = DeterministicMPC.from_training(MPCConfig(horizon_days=7))(initial)
        np.testing.assert_array_equal(reset, fresh)

    def test_evaluator_single_year_is_explicit_and_checked(self):
        frame, _ = evaluate(
            "ead",
            split="validation",
            protocol="chronological",
            year=1988,
            sequential_years=True,
        )
        self.assertEqual(frame.year.unique().tolist(), [1988])
        self.assertEqual(len(frame), 365)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            evaluate("ead", split="validation", protocol="chronological", year=2000)

    def test_horizon_anticipates_a_forecast_flood(self):
        params = settings("train")
        inflows = np.zeros(365)
        inflows[1] = 2000.0
        forecast = SeasonalForecast(
            inflows, np.full(365, 100.0), tuple(split_years("train"))
        )
        one = DeterministicMPC(params, forecast, MPCConfig(horizon_days=1))
        three = DeterministicMPC(params, forecast, MPCConfig(horizon_days=3))
        obs = np.array([0.8, 0, 1, 0], dtype=np.float32)
        self.assertGreater(three(obs)[0], one(obs)[0] + 20)

    def test_forecast_nonlevel_cost_matches_actual_reward(self):
        params = settings("train")
        model = PlanningModel(params, MPCConfig(horizon_days=1))
        env = make_env("validation", 1988)
        try:
            for request in [0.0, 100.0, 250.0]:
                _, info = env.reset()
                sim = env.unwrapped
                cost = model.cost(
                    [request / model.high],
                    info["storage"],
                    np.array(sim.inflow[:1]),
                    np.array(sim.demand[:1]),
                    np.array([1]),
                )
                _, reward, _, _, info = env.step(np.array([request], dtype=np.float32))
                self.assertGreater(info["level"], params["starving_level"])
                self.assertLess(info["level"], params["flood_level"])
                self.assertAlmostEqual(cost, -reward / 1000, places=8)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
