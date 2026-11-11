"""Check isolation and direction of the entropy regularizer."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.entropy_study import COEFFICIENT, decision_from_means
from reservoir_control.systems.lake_como.stabilize import agent_config


class EntropyChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('skrl'),'Requires reproduction group')
    def test_only_entropy_coefficient_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            control=agent_config('capped_scheduler',Path(directory))
            candidate=agent_config('capped_scheduler',Path(directory),entropy_loss_scale=COEFFICIENT)
            self.assertEqual(control.pop('entropy_loss_scale'),0)
            self.assertEqual(candidate.pop('entropy_loss_scale'),.01)
            self.assertEqual(control,candidate)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('skrl'),'Requires reproduction group')
    def test_entropy_loss_encourages_larger_exploration_spread(self):
        import torch
        from gymnasium.spaces import Box
        from gym4real.algorithms.dam.ppo_skrl import Policy
        policy=Policy(Box(-np.inf,np.inf,shape=(4,),dtype=np.float32),
                      Box(-1,1,shape=(1,),dtype=np.float32),device='cpu',net_arch=(16,16),initial_log_std=-.5)
        policy.act({'states':torch.zeros((1,4))},role='policy')
        entropy_loss=-COEFFICIENT*policy.get_entropy(role='policy').mean()
        gradient=torch.autograd.grad(entropy_loss,policy.log_std_parameter)[0]
        self.assertLess(gradient.item(),0)  # Gradient descent increases log standard deviation.
        self.assertAlmostEqual(gradient.item(),-.01,places=6)

    def test_selection_rejects_unstable_candidate_with_better_mean(self):
        rows=[]
        for arm,values in [('zero_entropy',[-20]*5),('entropy_bonus',[-1,-1,-1,-1,-70])]:
            rows.extend(dict(exploration=arm,training_seed=seed,full_return=value)
                        for seed,value in zip(range(123,128),values))
        frame=pd.DataFrame(rows).set_index(['exploration','training_seed'])
        self.assertEqual(decision_from_means(frame)['selected'],'zero_entropy')
        frame.loc['entropy_bonus','full_return']=[-10]*5
        self.assertEqual(decision_from_means(frame)['selected'],'entropy_bonus')


if __name__=='__main__':unittest.main()
