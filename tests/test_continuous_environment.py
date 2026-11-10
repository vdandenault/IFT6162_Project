"""Physical parity, sequence boundaries, causal observations and split isolation."""
import unittest
from copy import deepcopy

import numpy as np
from gymnasium.utils.env_checker import check_env

from lake_como.continuous_environment import ContinuousDamEnv, make_continuous_env
from lake_como.environment import make_env, settings


class ContinuousEnvironmentChecks(unittest.TestCase):
    def test_single_year_matches_benchmark_for_every_transition(self):
        current = make_env('train', 1946)
        extended = make_continuous_env(years=(1946,), past_inflow=False)
        try:
            np.testing.assert_array_equal(current.reset()[0], extended.reset()[0])
            for day in range(365):
                request = np.array([20 + (day * 37) % 400], np.float32)
                a, ra, ta, xa, ia = current.step(request)
                b, rb, tb, xb, ib = extended.step(request)
                np.testing.assert_array_equal(a, b)
                self.assertEqual((ra, ta, xa), (rb, tb, xb))
                for key in ['storage', 'release', 'level', 'exponential_average_demand', 'weighted_reward']:
                    self.assertEqual(ia[key], ib[key])
                self.assertLess(abs(ib['balance_error_m3']), 1e-6)
        finally:
            current.close(); extended.close()

    def test_year_boundary_preserves_state_memory_and_correct_next_forcing(self):
        env = make_continuous_env()
        with self.assertRaises(RuntimeError):
            env.step(np.array([100], np.float32))
        initial, _ = env.reset()
        for day in range(365):
            before, _, terminated, truncated, info = env.step(np.array([100], np.float32))
            self.assertFalse(terminated or truncated)
        self.assertTrue(info['year_boundary'])
        self.assertEqual((info['transition_year'], info['transition_day']), (1946, 365))
        self.assertEqual((info['next_year'], info['next_day']), (1947, 1))
        self.assertEqual(before[0], np.float32(info['level']))
        self.assertNotEqual(before[3], 0)  # demand memory survives
        self.assertEqual(before[-1], np.float32(env.unwrapped.inflow_data[1946][364]))
        self.assertEqual(env.unwrapped.release[0], info['release'])
        following, _, _, _, next_info = env.step(np.array([100], np.float32))
        self.assertEqual(next_info['storage_before'], info['storage'])
        self.assertEqual((next_info['transition_year'], next_info['transition_day']), (1947, 1))
        expected_ema = .8 * info['exponential_average_demand'] + .2 * env.unwrapped.demand_data[1947][0]
        self.assertEqual(next_info['exponential_average_demand'], expected_ema)
        self.assertEqual(following[-1], np.float32(env.unwrapped.inflow_data[1947][0]))
        for _ in range(364):
            final, _, terminated, truncated, end = env.step(np.array([100], np.float32))
        self.assertFalse(terminated); self.assertTrue(truncated)
        self.assertEqual(end['elapsed_steps'], 730)
        self.assertIsNone(end['next_year'])
        self.assertEqual(final[0], np.float32(end['level']))
        with self.assertRaises(RuntimeError):
            env.step(np.array([100], np.float32))
        np.testing.assert_array_equal(initial, env.reset()[0])
        env.close()

    def test_future_year_inputs_are_not_in_current_observations(self):
        params = settings('train'); changed = deepcopy(params)
        changed['inflow'][1947][0] += 500
        changed['demand'][1947][0] += 200
        a = ContinuousDamEnv(params, (1946, 1947)); b = ContinuousDamEnv(changed, (1946, 1947))
        np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
        for _ in range(365):
            left = a.step(np.array([100], np.float32)); right = b.step(np.array([100], np.float32))
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1], right[1])
        self.assertNotEqual(a.step(np.array([100], np.float32))[1], b.step(np.array([100], np.float32))[1])
        a.close(); b.close()

    def test_split_sequence_and_action_validation(self):
        for years in [(), (1946, 1948), (1947, 1946), (1946, 1946), (1987, 1988)]:
            with self.assertRaises(ValueError):
                make_continuous_env('train', years)
        env = make_continuous_env()
        env.reset()
        for action in [np.array([np.nan]), np.array([-1]), np.array([500]), np.array([100, 100])]:
            with self.assertRaises(ValueError):
                env.step(action)
        env.close()

    def test_replay_links_years_but_stores_record_end_before_reset(self):
        from lake_como.sac import ReplayBuffer
        env = make_continuous_env()
        replay = ReplayBuffer(730, 5, 1)
        observation, _ = env.reset()
        request = np.array([100.], np.float32)
        normalized = 2 * request / env.action_space.high - 1
        for _ in range(730):
            following, reward, term, trunc, _ = env.step(request)
            replay.add(observation, normalized, reward, following, term, trunc)
            observation = following
        self.assertEqual(replay.data['ends'][364, 0], 0)
        np.testing.assert_array_equal(replay.data['next_observations'][364], replay.data['observations'][365])
        self.assertEqual(replay.data['ends'][:730].sum(), 1)
        self.assertEqual(replay.data['ends'][729, 0], 1)
        np.testing.assert_array_equal(replay.data['next_observations'][729], observation)
        reset, _ = env.reset()
        self.assertNotEqual(float(reset[0]), float(replay.data['next_observations'][729, 0]))
        env.close()

    def test_gymnasium_contract(self):
        env = make_continuous_env(years=(1946,), past_inflow=False)
        check_env(env, skip_render_check=True)
        env.close()


if __name__ == '__main__':
    unittest.main()
