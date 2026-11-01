"""FQI action ordering, Bellman masks, nonmyopic behavior and causal observations."""
import importlib.util
import unittest
import numpy as np
from lake_como.fqi import state_actions, greedy_values, bellman_targets, FQIController


class FQIChecks(unittest.TestCase):
    def test_state_action_order_and_chunked_greedy_maximum(self):
        states = np.zeros((3, 5), dtype=np.float32)
        states[:, 0] = [1, 2, 3]
        actions = np.array([0.0, 0.5, 1.0])
        features = state_actions(states, actions)
        np.testing.assert_array_equal(features[:, 0], [1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(
            features[:, -1], [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]
        )

        class Model:
            def predict(self, x):
                return x[:, 0] + x[:, -1]

        np.testing.assert_array_equal(
            greedy_values(Model(), states, actions, batch_size=2), [2, 3, 4]
        )

    def test_truncation_mask_stops_year_boundary_bootstrap(self):
        result = bellman_targets([-1.0, -2.0], [False, True], [10.0, 100.0], 0.9)
        np.testing.assert_array_equal(result, [8.0, -2.0])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            bellman_targets([1.0], [True, False], [1.0], 0.9)

    def test_policy_observes_completed_inflow_and_resets(self):
        class Model:
            def predict(self, x):
                self.last = x.copy()
                return x[:, -1]

        model = Model()
        p = FQIController(model, [0, 0.5, 1.0], 491.61, 67.0)
        obs = np.array([0.35, 0, 1, 0], dtype=np.float32)
        action = p(obs, {"storage": 1e8, "future_inflow": 99999})
        self.assertEqual(model.last[0, 4], 67.0)
        self.assertAlmostEqual(float(action[0]), 491.61, places=4)
        phase = 2 * np.pi / 365
        obs[1:3] = [np.sin(phase), np.cos(phase)]
        p(obs, {"storage": 1e8 + 86400 * (123 - 50), "release": 50})
        self.assertEqual(model.last[0, 4], 123.0)
        obs[1:3] = [0, 1]
        p(obs, {"storage": 1e8})
        self.assertEqual(model.last[0, 4], 67.0)

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn"),
        "Run the pinned scikit-learn overlay for the FQI integration check",
    )
    def test_fitted_backup_prefers_delayed_reward_in_small_mdp(self):
        from sklearn.ensemble import ExtraTreesRegressor

        states = np.zeros((4, 5), dtype=np.float32)
        states[2:, 0] = 1
        actions = np.array([0, 1, 0, 1], dtype=np.float32)
        x = np.column_stack([states, actions])
        following = states.copy()
        following[0, 0] = 1
        rewards = np.array([0.0, 1.0, 5.0, 5.0])
        done = np.array([False, True, True, True])
        first = ExtraTreesRegressor(
            n_estimators=8, min_samples_leaf=1, random_state=1
        ).fit(x, rewards)
        values = greedy_values(first, following, [0, 1])
        second = ExtraTreesRegressor(
            n_estimators=8, min_samples_leaf=1, random_state=1
        ).fit(x, bellman_targets(rewards, done, values, 0.9))
        predictions = second.predict(state_actions(np.zeros((1, 5)), [0, 1]))
        self.assertGreater(predictions[0], predictions[1])
        np.testing.assert_allclose(predictions, [4.5, 1.0])


if __name__ == "__main__":
    unittest.main()
