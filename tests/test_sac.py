"""SAC mathematical/gradient checks, boundary handling and learner persistence."""
import math
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import SACConfig, ReplayBuffer, SquashedGaussianActor, polyak_update, soft_bellman_target
from reservoir_control.systems.lake_como.sac_loop import train_steps


class SACChecks(unittest.TestCase):
    def agent(self, **kwargs):
        return SACAgent(SACConfig(observation_dim=2, hidden_sizes=(8, 8)), seed=12, **kwargs)

    def batch(self):
        replay = ReplayBuffer(8, 2, 1, seed=2)
        for i in range(8):
            replay.add([i / 8, 1], [0.2], -10 - i, [i / 8 + .1, 1], False, i == 7)
        return replay.sample(8)

    def test_soft_target_numbers_boundary_mask_and_detachment(self):
        q = torch.tensor([[10.], [10.], [10.]], requires_grad=True)
        target = soft_bellman_target(torch.tensor([[-1000.], [-1000.], [-1000.]]),
                                    torch.tensor([[0.], [1.], [1.]]), q,
                                    torch.full((3, 1), -2.), .2, .9, .001)
        torch.testing.assert_close(target, torch.tensor([[8.36], [-1.], [-1.]]))
        self.assertFalse(target.requires_grad)

    def test_replay_wraparound_copy_and_both_end_types(self):
        replay = ReplayBuffer(2, 2, 1)
        observation = np.array([1., 2.])
        replay.add(observation, [0], -3, [3, 4], True, False)
        observation[:] = 99
        np.testing.assert_array_equal(replay.data["observations"][0], [1, 2])
        replay.add([2, 3], [.2], -4, [4, 5], False, True)
        replay.add([3, 4], [.3], -5, [5, 6], False, False)
        self.assertEqual(len(replay), 2)
        np.testing.assert_array_equal(replay.data["ends"][:, 0], [0, 1])
        with self.assertRaises(ValueError):
            replay.add([0, 1], [491.61], 0, [0, 1], False, False)
        with self.assertRaises(ValueError):
            replay.sample(3)

    def test_squashed_density_matches_change_of_variables(self):
        actor = SquashedGaussianActor(SACConfig(observation_dim=2, hidden_sizes=(8,)))
        for p in actor.parameters():
            nn.init.zeros_(p)
        action, logp = actor(torch.zeros(64, 2), generator=torch.Generator().manual_seed(3))
        z = action.atanh()
        expected = (-.5 * (z.square() + math.log(2 * math.pi))
                    - torch.log1p(-action.square())).sum(-1, keepdim=True)
        torch.testing.assert_close(logp, expected, atol=3e-5, rtol=3e-5)
        self.assertTrue((action.abs() <= 1).all())
        # Extreme means must retain finite log probabilities and useful gradients.
        actor.net[-1].bias.data[0] = 100
        _, saturated_logp = actor(torch.zeros(2, 2))
        self.assertTrue(torch.isfinite(saturated_logp).all())
        saturated_logp.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_actor_gradient_flows_through_frozen_critic(self):
        agent = self.agent()
        with torch.no_grad():
            for p in agent.actor.parameters():
                p.zero_()
        agent.q1.requires_grad_(False)
        agent.q1.net = nn.Linear(3, 1)
        with torch.no_grad():
            agent.q1.net.weight.copy_(torch.tensor([[0., 0., 1.]]))
            agent.q1.net.bias.zero_()
        agent.q1.requires_grad_(False)
        observation = torch.zeros(4, 2)
        action, _ = agent.actor(observation, deterministic=True)
        # Use an analytic Q=a to check direction, rather than random critic slopes.
        loss = -agent.q1(observation, action).mean()
        loss.backward()
        self.assertLess(agent.actor.net[-1].bias.grad[0].item(), 0)
        self.assertTrue(all(p.grad is None for p in agent.q1.parameters()))

    def test_updates_change_correct_parameters_and_polyak_targets(self):
        agent = self.agent()
        actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        old_target = [p.clone() for p in agent.target_q1.parameters()]
        alpha_before = agent.alpha.item()
        metrics = agent.update(self.batch())
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertNotEqual(alpha_before, agent.alpha.item())
        for old, target, online in zip(old_target, agent.target_q1.parameters(), agent.q1.parameters()):
            torch.testing.assert_close(target, (1 - agent.config.tau) * old + agent.config.tau * online)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in agent.target_q1.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.q1.parameters()))

    def test_polyak_known_values(self):
        online, target = nn.Linear(1, 1), nn.Linear(1, 1)
        for p in online.parameters():
            p.data.fill_(4)
        for p in target.parameters():
            p.data.fill_(2)
        polyak_update(online, target, .25)
        for p in target.parameters():
            torch.testing.assert_close(p, torch.full_like(p, 2.5))

    def test_alpha_adjustment_direction(self):
        # Too little entropy => alpha increases; above target => decreases.
        for logp, expected_sign in [(2., 1), (-2., -1)]:
            log_alpha = torch.tensor(-1., requires_grad=True)
            loss = -(log_alpha * (torch.tensor(logp) - 1)).mean()
            loss.backward()
            delta = -.01 * log_alpha.grad.item()
            self.assertGreater(delta * expected_sign, 0)

    def test_normalization_physical_mapping_and_checkpoint_next_update(self):
        agent = self.agent(observation_mean=[1, 2], observation_scale=[2, 4])
        torch.testing.assert_close(agent.normalize(torch.tensor([3., 6.])), torch.ones(2))
        np.testing.assert_allclose(agent.physical_action(torch.tensor([[-1.], [0.], [1.]])).numpy(),
                                   [[0], [245.805], [491.61]], rtol=1e-6)
        batch = self.batch()
        agent.update(batch)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "learner.pt"
            agent.save(path)
            loaded = SACAgent.load(path)
            np.testing.assert_array_equal(agent.act([1, 2], deterministic=True),
                                          loaded.act([1, 2], deterministic=True))
            np.testing.assert_array_equal(agent.act([1, 2]), loaded.act([1, 2]))
            self.assertEqual(agent.update(batch), loaded.update(batch))
            for key, tensor in agent.state_dict().items():
                torch.testing.assert_close(tensor, loaded.state_dict()[key], rtol=0, atol=0)

    def test_critic_can_fit_known_terminal_values(self):
        agent = self.agent()
        observations = torch.zeros(16, 2)
        actions = torch.linspace(-1, 1, 16).unsqueeze(1)
        truth = -actions.square()
        before = F.mse_loss(agent.q1(observations, actions), truth).item()
        optimizer = torch.optim.Adam(agent.q1.parameters(), lr=.01)
        for _ in range(150):
            loss = F.mse_loss(agent.q1(observations, actions), truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), before * .1)

    def test_loop_stores_final_observation_before_reset(self):
        class TwoDayEnv(gym.Env):
            observation_space = gym.spaces.Box(-100, 100, (2,), dtype=np.float32)
            action_space = gym.spaces.Box(0, np.float32(491.61), (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.day = 0
                return np.zeros(2, np.float32), {}

            def step(self, request):
                assert self.action_space.contains(request)
                self.day += 1
                return np.full(2, self.day, np.float32), -10., False, self.day == 2, {}

        agent = self.agent()
        replay = ReplayBuffer(8, 2, 1)
        result = train_steps(TwoDayEnv(), agent, replay, steps=5, warmup_steps=2, batch_size=2)
        np.testing.assert_array_equal(replay.data["ends"][:5, 0], [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(replay.data["next_observations"][1], [2, 2])
        self.assertEqual(result["episodes"], [{"return": -20., "length": 2}] * 2)
        self.assertEqual(result["updates"], 3)


if __name__ == "__main__":
    unittest.main()
