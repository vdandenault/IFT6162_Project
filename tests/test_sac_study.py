"""Failure gates and parity between SAC training observations and evaluation adapter."""
import unittest
import numpy as np
import torch

from reservoir_control.systems.lake_como.environment import make_env
from reservoir_control.systems.lake_como.inflow_observation import PastInflowObservation
from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import SACConfig, checked_optimizer_step
from reservoir_control.systems.lake_como.sac_study import SACController


class SACStudyChecks(unittest.TestCase):
    def test_nonfinite_gradient_is_rejected_before_optimizer_mutation(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.Adam([parameter], lr=.01)
        with self.assertRaises(FloatingPointError):
            checked_optimizer_step(parameter.sqrt(), optimizer, [parameter])
        self.assertEqual(parameter.item(), 0.)
        self.assertEqual(len(optimizer.state), 0)

    def test_nonfinite_loss_is_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.Adam([parameter], lr=.01)
        with self.assertRaises(FloatingPointError):
            checked_optimizer_step(parameter * float('nan'), optimizer, [parameter])
        self.assertEqual(parameter.item(), 1.)

    def test_controller_matches_wrapped_observations_for_an_entire_training_year(self):
        fallback = 67.27619047619051
        agent = SACAgent(SACConfig(hidden_sizes=(8, 8)),
                         observation_scale=[1, 1, 1, 100, 100], seed=3)
        controller = SACController(agent, fallback)
        env = PastInflowObservation(make_env('train', 1946), fallback)
        try:
            observation, info = env.reset(seed=123)
            for day in range(365):
                expected = agent.physical_action(torch.from_numpy(agent.act(observation, deterministic=True))).numpy()
                # Hidden forcing is deliberately present in info but must have no effect.
                info['future_inflow'] = 99999
                actual = controller(observation[:4], info)
                np.testing.assert_array_equal(actual, expected)
                observation, _, terminated, truncated, info = env.step(actual)
                self.assertFalse(terminated)
                self.assertEqual(truncated, day == 364)
            observation, info = env.reset()
            expected = agent.physical_action(torch.from_numpy(agent.act(observation, deterministic=True))).numpy()
            np.testing.assert_array_equal(controller(observation[:4], info), expected)
        finally:
            env.close()

    def test_seeded_reset_replays_training_trajectory(self):
        env = PastInflowObservation(make_env('train'), 67.27619047619051)
        trajectories = []
        try:
            for _ in range(2):
                obs, _ = env.reset(seed=123)
                year = env.unwrapped.curr_year_data
                rows = [obs]
                for _ in range(10):
                    obs, reward, _, _, _ = env.step(np.array([100.], np.float32))
                    rows.append(np.r_[obs, reward])
                trajectories.append((year, rows))
            self.assertEqual(trajectories[0][0], trajectories[1][0])
            for first, second in zip(trajectories[0][1], trajectories[1][1]):
                np.testing.assert_array_equal(first, second)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
