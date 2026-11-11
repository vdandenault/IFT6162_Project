import unittest

import numpy as np

from reservoir_control.systems.lake_como.continuous_environment import make_continuous_env
from reservoir_control.systems.lake_como.environment import make_env
from reservoir_control.systems.lake_como.hourly_telemetry import HourlyTelemetry, linear_exposure, make_hourly_env


class HourlyTelemetryChecks(unittest.TestCase):
    def test_interpolated_crossing_duration_and_integral(self):
        self.assertEqual(linear_exposure(0, 2, 1), (.5, .25))
        self.assertEqual(linear_exposure(2, 0, 1), (.5, .25))
        self.assertEqual(linear_exposure(2, 2, 1), (1., 1.))
        self.assertEqual(linear_exposure(0, 0, 1), (0., 0.))
        self.assertEqual(linear_exposure(0, 2, 1, above=False), (.5, .25))
        self.assertEqual(linear_exposure(1, 1, 1), (0., 0.))
        with self.assertRaises(ValueError):
            linear_exposure(0, 1, 0, hours=0)

    def test_two_year_parity_including_rollover_and_final_observations(self):
        original = make_continuous_env()
        logged = make_hourly_env()
        try:
            np.testing.assert_array_equal(original.reset()[0], logged.reset()[0])
            old_trace = None
            for step in range(730):
                action = np.array([20 + (step * 37) % 450], np.float32)
                obs, reward, term, trunc, info = original.step(action)
                actual, r, t, x, data = logged.step(action)
                np.testing.assert_array_equal(obs, actual)
                self.assertEqual((reward, term, trunc), (r, t, x))
                for key in info:
                    self.assertEqual(info[key], data[key])
                hourly = data['hourly']
                self.assertEqual(len(hourly), 24)
                self.assertEqual(hourly[0]['storage_before_m3'], info['storage_before'])
                self.assertEqual(hourly[-1]['storage_m3'], info['storage'])
                self.assertEqual(np.mean([h['release_m3_s'] for h in hourly]), info['release'])
                np.testing.assert_allclose([h['inflow_m3_s'] for h in hourly],
                    original.unwrapped.inflow_data[info['transition_year']][info['transition_day']-1])
                self.assertLess(data['physical_metrics']['max_hourly_balance_error_m3'], 1e-6)
                self.assertTrue(0 <= data['physical_metrics']['interpolated_flood_hours'] <= 24)
                if step == 364:
                    old_trace = [dict(h) for h in hourly]
                    held_reference = hourly
                if step == 365:
                    self.assertEqual(held_reference, old_trace)
                    self.assertEqual(hourly[0]['storage_before_m3'], old_trace[-1]['storage_m3'])
            logged.reset()
            self.assertEqual(logged.recorder.last_hourly, [])
        finally:
            original.close(); logged.close()

    def test_observation_contains_no_hourly_forcing_or_new_features(self):
        logged = make_hourly_env(years=(1946,))
        obs, info = logged.reset()
        self.assertEqual(obs.shape, (5,))
        self.assertNotIn('hourly', info)
        self.assertNotIn('physical_metrics', info)
        logged.close()

    def test_exposure_includes_initial_state_not_just_endpoints(self):
        env = HourlyTelemetry(make_env('train', 1946))
        env.reset()
        # Controlled high initial state: existing hydraulic equations drain it.
        core = env.unwrapped
        core.level[0] = 1.245
        core.storage[0] = core.lake.level_to_storage(1.245)
        _, _, _, _, info = env.step(np.array([491.61], np.float32))
        first = info['hourly'][0]
        self.assertGreater(first['interpolated_flood_hours'], 0)
        self.assertEqual(info['physical_metrics']['max_sampled_level_m'], 1.245)
        self.assertLess(info['level'], core.flood_level)
        self.assertEqual(info['weighted_reward']['overflow_reward'], 0)
        env.close()

    def test_reject_duplicate_instrumentation(self):
        env = make_hourly_env()
        with self.assertRaises(ValueError):
            HourlyTelemetry(env)
        env.close()


if __name__ == '__main__':
    unittest.main()
