from pathlib import Path
import os
import unittest
import numpy as np

from lake_como.environment import make_env, settings
from lake_como.paths import UPSTREAM_ROOT


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
        ours = settings('train')
        self.assertEqual(ours.keys(), upstream.keys())
        for key in ours:
            if key != 'lake_params':
                self.assertEqual(ours[key], upstream[key], key)
        for key in ours['lake_params']:
            np.testing.assert_equal(ours['lake_params'][key], upstream['lake_params'][key])

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
            # Upstream terminal observation wraps to initial level; record, don't fix silently.
            expected = self.sim.level[0] if truncated else info['level']
            self.assertAlmostEqual(float(obs[0]), expected, places=5)
            self.assertAlmostEqual(reward, sum(info['weighted_reward'].values()))
            self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(self.sim.current_step, 365)

    def test_demand_observation_uses_elapsed_day(self):
        obs, info = self.env.reset(seed=123)
        self.assertEqual(float(obs[-1]), 0)
        obs, _, _, _, info = self.env.step(np.array([100.0], dtype=np.float32))
        self.assertAlmostEqual(float(obs[-1]), 0.2 * self.sim.demand[0], places=4)


if __name__ == '__main__':
    unittest.main()
