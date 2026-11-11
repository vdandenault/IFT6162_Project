from pathlib import Path
import os
import calendar
import unittest
import numpy as np

from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.paths import UPSTREAM_ROOT


class ReservoirChecks(unittest.TestCase):
    def setUp(self):
        self.env = make_env('train', 1985)
        self.sim = self.env.unwrapped
        self.env.reset(seed=123)

    def tearDown(self):
        self.env.close()

    def test_adapter_matches_upstream_loader(self):
        from gym4real.envs.dam.utils import parameter_generator
        previous = Path.cwd()
        try:
            os.chdir(UPSTREAM_ROOT)
            upstream = parameter_generator('gym4real/envs/dam/world_train.yaml',
                                           'gym4real/envs/dam/lake.yaml')
        finally:
            os.chdir(previous)
        ours = settings('train', protocol='upstream')
        self.assertEqual(ours.keys(), upstream.keys())
        for key in ours:
            if key != 'lake_params':
                self.assertEqual(ours[key], upstream[key], key)
        for key in ours['lake_params']:
            np.testing.assert_equal(ours['lake_params'][key], upstream['lake_params'][key])

    def test_chronological_membership_and_alignment(self):
        groups = []
        for split, start, end in [('train', 1946, 1987), ('validation', 1988, 1997), ('test', 1998, 2010)]:
            profiles = settings(split)
            self.assertEqual(list(profiles['inflow']), list(range(start, end + 1)))
            self.assertEqual(profiles['inflow'].keys(), profiles['demand'].keys())
            groups.extend(profiles['inflow'])
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(set(groups), set(range(1946, 2011)))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            make_env('train', 2008)

    def test_leap_years_included_by_default_and_can_be_excluded(self):
        for split, full_count, filtered_count in [('train', 42, 32), ('validation', 10, 7), ('test', 13, 10)]:
            self.assertEqual(len(settings(split)['inflow']), full_count)
            profiles = settings(split, include_leap_years=False)
            self.assertEqual(len(profiles['inflow']), filtered_count)
            self.assertTrue(all(not calendar.isleap(y) for y in profiles['inflow']))
        self.assertEqual(len(settings('test')['inflow'][2008]), 366)
        env = make_env('test', 2008)
        try:
            env.reset(seed=123)
            self.assertEqual(env.unwrapped.curr_year_data, 2008)
            self.assertEqual(env.unwrapped.H, 365)
        finally:
            env.close()

    def test_mass_balance_and_hydraulic_limits(self):
        lake = self.sim.lake
        s = lake.level_to_storage(0.35)
        final, released = lake.integration(24, s, 100.0, 150.0, 1)
        self.assertAlmostEqual(final - s, 86400 * (150 - released), places=5)
        self.assertEqual(lake.actual_release(1e6, s, 1), lake.max_release(s, 1))
        self.assertEqual(lake.actual_release(-100, s, 1), lake.min_release(s, 1))
        self.assertEqual(lake.actual_release(100, lake.level_to_storage(-0.5), 1), 0)

    def test_annual_boundary_and_observation_timing(self):
        for day in range(365):
            obs, reward, terminated, truncated, info = self.env.step(np.array([100.0], dtype=np.float32))
            self.assertFalse(terminated)
            self.assertEqual(truncated, day == 364)
            expected = info['level']
            self.assertAlmostEqual(float(obs[0]), expected, places=5)
            self.assertAlmostEqual(reward, sum(info['weighted_reward'].values()))
            self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(self.sim.current_step, 365)
        self.assertNotAlmostEqual(float(obs[0]), self.sim.level[0], places=3)
        # A correct terminal observation does not imply continuous multiyear physics.
        reset_obs, reset_info = self.env.reset(seed=123)
        self.assertAlmostEqual(reset_info['level'], 0.35)
        self.assertAlmostEqual(float(reset_obs[0]), 0.35, places=5)

    def test_terminal_fix_preserves_dynamics_and_rewards(self):
        original = make_env('train', 1985, correct_terminal_level=False)
        try:
            original.reset(seed=123)
            for day in range(365):
                action = np.array([100.0], dtype=np.float32)
                fixed_obs, fixed_reward, fixed_done, fixed_trunc, fixed_info = self.env.step(action)
                raw_obs, raw_reward, raw_done, raw_trunc, raw_info = original.step(action)
                self.assertEqual(fixed_reward, raw_reward)
                self.assertEqual((fixed_done, fixed_trunc), (raw_done, raw_trunc))
                self.assertEqual(fixed_info['storage'], raw_info['storage'])
                if day < 364:
                    np.testing.assert_array_equal(fixed_obs, raw_obs)
                else:
                    np.testing.assert_array_equal(fixed_obs[1:], raw_obs[1:])
                    self.assertNotEqual(fixed_obs[0], raw_obs[0])
        finally:
            original.close()

    def test_demand_observation_uses_elapsed_day(self):
        obs, info = self.env.reset(seed=123)
        self.assertEqual(float(obs[-1]), 0)
        obs, _, _, _, info = self.env.step(np.array([100.0], dtype=np.float32))
        self.assertAlmostEqual(float(obs[-1]), 0.2 * self.sim.demand[0], places=4)


if __name__ == '__main__':
    unittest.main()
