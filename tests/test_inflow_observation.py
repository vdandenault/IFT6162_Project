"""Past-inflow timing, reset isolation, and unchanged simulator behavior."""
import unittest
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from lake_como.environment import make_env
from lake_como.inflow_observation import PastInflowObservation
from lake_como.ppo_inflow_study import first_day_fallback


class TelemetryOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 500, shape=(1,), dtype=np.float32)
    lake = SimpleNamespace(evaporation=False)

    @property
    def inflow(self):
        raise AssertionError("Future/forcing profile accessed")

    def reset(self, **kwargs):
        self.storage = 1e8
        return np.zeros(4, dtype=np.float32), {"storage": self.storage}

    def step(self, action):
        self.storage += 86400 * (123 - 50)
        return (
            np.ones(4, dtype=np.float32),
            -10,
            False,
            False,
            {"storage": self.storage, "release": 50},
        )


class InflowObservationChecks(unittest.TestCase):
    def test_only_completed_telemetry_is_used_and_reset_discards_history(self):
        env = PastInflowObservation(TelemetryOnlyEnv(), 67.0)
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            env.step(np.array([100], dtype=np.float32))
        obs, _ = env.reset()
        self.assertEqual(obs[-1], 67)
        obs, reward, *_ = env.step(np.array([100], dtype=np.float32))
        self.assertEqual(obs[-1], 123)
        self.assertEqual(reward, -10)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        obs, _ = env.reset()
        self.assertEqual(obs[-1], 67)

    def test_rewards_dynamics_first_four_features_and_terminal_are_unchanged(self):
        base = make_env("validation", 1989)
        wrapped = PastInflowObservation(
            make_env("validation", 1989), first_day_fallback()
        )
        try:
            a, ia = base.reset()
            b, ib = wrapped.reset()
            np.testing.assert_array_equal(a, b[:4])
            for day in range(365):
                action = np.array([100.0], dtype=np.float32)
                a, ra, ta, xa, ia = base.step(action)
                b, rb, tb, xb, ib = wrapped.step(action)
                np.testing.assert_array_equal(a, b[:4])
                self.assertEqual(ra, rb)
                self.assertEqual(ta, tb)
                self.assertEqual(xa, xb)
                self.assertEqual(ia["storage"], ib["storage"])
                self.assertAlmostEqual(
                    float(b[-1]), base.unwrapped.inflow[day], places=4
                )
            self.assertTrue(xb)
            self.assertAlmostEqual(float(b[0]), ib["level"], places=6)
            reset, _ = wrapped.reset()
            self.assertAlmostEqual(float(reset[-1]), first_day_fallback(), places=4)
        finally:
            base.close()
            wrapped.close()

    def test_evaporation_and_invalid_fallback_are_rejected(self):
        env = TelemetryOnlyEnv()
        with self.assertRaisesRegex(ValueError, "fallback"):
            PastInflowObservation(env, float("nan"))
        env.lake = SimpleNamespace(evaporation=True)
        with self.assertRaisesRegex(ValueError, "evaporation"):
            PastInflowObservation(env, 67.0)


if __name__ == "__main__":
    unittest.main()
