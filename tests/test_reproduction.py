"""Checks for matching source reward accounting and PPO checkpoint inference."""
import importlib.util
import unittest
import numpy as np

from reservoir_control.systems.lake_como.baselines import rollout
from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.reproduce import evaluate, CheckpointPolicy, CHECKPOINT


class ReproductionChecks(unittest.TestCase):
    def test_original_years_and_objective_accounting(self):
        frame, _ = evaluate('ead')
        self.assertEqual(len(frame), 4745)
        self.assertEqual(frame.year.drop_duplicates().tolist(), list(settings('test', protocol='upstream')['inflow']))
        objective = sum(frame[f'weighted_{k}'] for k in ('overflow_reward','starving_reward','daily_deficit_reward'))
        np.testing.assert_allclose(frame.objective_reward, objective)
        np.testing.assert_allclose(frame.full_reward, objective + frame.weighted_wasted_water_reward + frame.weighted_clipping_reward)
        self.assertLess(frame.balance_error_m3.abs().max(), 1e-4)
        reference, _ = rollout('ead', 1950, split='test', protocol='upstream')
        subset = frame[frame.year == 1950]
        np.testing.assert_allclose(subset.full_reward, reference.reward)
        np.testing.assert_allclose(subset.level_m, reference.level_after_m)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('skrl'),
                         'Run with uv run --group reproduction to check the PPO checkpoint')
    def test_frozen_mean_action_matches_skrl_agent(self):
        import torch
        from copy import deepcopy
        from gymnasium.wrappers import RescaleAction
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        from gym4real.algorithms.dam.ppo_skrl import Value
        env = RescaleAction(make_env('test', protocol='upstream'), -1., 1.)
        try:
            controller = CheckpointPolicy(env.observation_space, env.action_space, CHECKPOINT)
            state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
            value = Value(env.observation_space, env.action_space, device='cpu', net_arch=(16,16))
            value.load_state_dict(state['value'])
            cfg = deepcopy(PPO_DEFAULT_CONFIG)
            cfg.update(state_preprocessor=RunningStandardScaler,
                       state_preprocessor_kwargs={'size':env.observation_space,'device':'cpu'},
                       mixed_precision=False)
            cfg['experiment'].update(write_interval=0, checkpoint_interval=0)
            agent = PPO(models={'policy':controller.policy,'value':value}, memory=None,
                        cfg=cfg, observation_space=env.observation_space,
                        action_space=env.action_space, device='cpu')
            agent._state_preprocessor.load_state_dict(state['state_preprocessor'])
            agent.set_mode('eval')
            before = controller.scaler.current_count.clone()
            for obs in [np.array([.35,0,1,0],dtype=np.float32),
                        np.array([1.2,.7,-.7,180],dtype=np.float32),
                        np.array([-.4,-.8,.6,70],dtype=np.float32)]:
                expected = agent.act(torch.as_tensor(obs).reshape(1,-1),0,365)[-1]['mean_actions']
                np.testing.assert_allclose(controller(obs), expected.detach().numpy().reshape(-1),atol=1e-6)
            self.assertTrue(torch.equal(controller.scaler.current_count, before))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
