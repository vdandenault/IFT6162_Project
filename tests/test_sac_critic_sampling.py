"""Matched budgets/sampling, split isolation, and shared counterfactual targets."""
import unittest
import numpy as np
import pandas as pd
import torch
from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import SACConfig
from reservoir_control.systems.lake_como.sac_critic_sampling import (SamplingConfig,matched_minibatches,
                                          merge_counterfactuals,common_targets)


class SamplingChecks(unittest.TestCase):
    def test_matched_batches_and_focus_use_only_fitting_members(self):
        config=SamplingConfig(budgets=(2,10),batch_size=8,focused_pairs_per_batch=2)
        pairs=np.array([[0,4],[1,5]])
        batch=matched_minibatches(np.arange(6),pairs,config,seed=3)
        self.assertEqual(batch['uniform'].shape,(10,8))
        self.assertEqual(batch['focused'].shape,(10,8))
        np.testing.assert_array_equal(batch['uniform'][:,:4],batch['focused'][:,:4])
        self.assertTrue(np.isin(batch['focused'],np.arange(6)).all())
        self.assertTrue(np.isin(batch['focused'][:,4:].reshape(-1,2),pairs).all())
        for pair in batch['focused'][:,4:].reshape(-1,2):
            self.assertTrue(any(np.array_equal(pair,p) for p in pairs))
        again=matched_minibatches(np.arange(6),pairs,config,seed=3)
        np.testing.assert_array_equal(batch['uniform'],again['uniform'])
        np.testing.assert_array_equal(batch['focused'],again['focused'])
        with self.assertRaises(ValueError):
            matched_minibatches(np.arange(4),pairs,config,seed=3)

    def fixture(self):
        base=dict(observations=np.array([[0.,0.],[1.,1.]],np.float32),
                  next_observations=np.array([[.1,.2],[.3,.4]],np.float32),
                  requests=np.array([200.,100.],np.float32),rewards=np.array([-10.,-20.]),
                  ends=np.array([0.,1.],np.float32),years=np.array([1946,1947]),days=np.array([90,180]))
        pairs=dict(observations=base['observations'][:1],next_observations=base['next_observations'][:1],
                   requests=np.array([[200.,150.]],np.float32),rewards=np.array([[-10.,-5.]]),
                   ends=base['ends'][:1],years=base['years'][:1])
        rows=pd.DataFrame(dict(year=[1946],day=[90]))
        return base,pairs,rows

    def test_pool_appends_only_cheaper_request_and_preserves_actor_indices(self):
        base,pairs,rows=self.fixture()
        pool,indices=merge_counterfactuals(base,pairs,rows)
        self.assertEqual(len(pool['years']),3)
        np.testing.assert_array_equal(indices,[[0,2]])
        np.testing.assert_array_equal(pool['requests'],[200.,100.,150.])
        np.testing.assert_array_equal(pool['next_observations'][0],pool['next_observations'][2])
        pairs['rewards'][0,0]=-9.
        with self.assertRaises(ValueError):
            merge_counterfactuals(base,pairs,rows)

    def test_shared_next_value_cancels_and_terminal_bootstrap_is_zero(self):
        base,pairs,rows=self.fixture()
        pool,indices=merge_counterfactuals(base,pairs,rows)
        agent=SACAgent(SACConfig(observation_dim=2,hidden_sizes=(8,)))
        rng=agent.generator.get_state().clone()
        targets=common_targets(agent,pool,indices,samples=4,seed=8)
        self.assertAlmostEqual(float(targets[2]-targets[0]),.005,places=6)
        self.assertAlmostEqual(float(targets[1]),-.020,places=6)
        self.assertFalse(targets.requires_grad)
        self.assertTrue(torch.equal(rng,agent.generator.get_state()))


if __name__=='__main__':
    unittest.main()
