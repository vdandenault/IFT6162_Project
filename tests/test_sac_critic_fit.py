"""Frozen-actor fit mathematics and separation of historical training-year partitions."""
import unittest
import numpy as np
import torch

from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import SACConfig
from reservoir_control.systems.lake_como.sac_critic_fit import (critic_pair_losses,fixed_targets,training_partitions,
                                     freeze_noncritics,frozen_state,assert_frozen)
from reservoir_control.systems.lake_como.splits import split_years


class CriticFitChecks(unittest.TestCase):
    def test_pair_loss_gradient_favors_the_known_cheaper_action(self):
        q1=torch.tensor([[1.,0.]],requires_grad=True)
        q2=torch.tensor([[1.,0.]],requires_grad=True)
        targets=torch.tensor([[0.,.5]])
        loss,td,pair=critic_pair_losses(q1,q2,targets,torch.tensor([.5]),10.)
        self.assertAlmostEqual(pair.item(),4.5)
        loss.backward()
        self.assertGreater(q1.grad[0,0],0)
        self.assertLess(q1.grad[0,1],0)
        self.assertTrue(torch.isfinite(q1.grad).all())

    def test_perfect_pair_targets_have_no_auxiliary_loss(self):
        q1=torch.tensor([[1.,1.5],[2.,2.25]])
        _,_,pair=critic_pair_losses(q1,q1,q1,torch.tensor([.5,.25]),10.)
        self.assertEqual(pair.item(),0.)
        with self.assertRaises(ValueError):
            critic_pair_losses(q1,q1,q1,torch.ones(2,1),10.)

    def test_common_continuation_cancels_and_terminal_target_is_immediate_reward(self):
        agent=SACAgent(SACConfig(observation_dim=2,hidden_sizes=(8,)))
        freeze_noncritics(agent)
        snapshot=frozen_state(agent);rng=agent.generator.get_state().clone()
        rewards=np.array([[-10.,-5.],[-20.,-10.]],np.float64)
        targets=fixed_targets(agent,np.zeros((2,2),np.float32),rewards,
                              np.array([0.,1.],np.float32),samples=4,seed=90)
        torch.testing.assert_close(targets[:,1]-targets[:,0],torch.tensor([.005,.01]),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(targets[1],torch.tensor([-.02,-.01]))
        self.assertFalse(targets.requires_grad)
        assert_frozen(agent,snapshot,rng)

    def test_partition_uses_only_disjoint_training_years(self):
        fit,check=training_partitions()
        self.assertEqual(len(fit),21)
        self.assertEqual(len(check),21)
        self.assertFalse(set(fit)&set(check))
        self.assertEqual(set(fit)|set(check),set(split_years('train')))
        self.assertLess(max(fit+check),1988)


if __name__=='__main__':
    unittest.main()
