"""Independent references, exact 10% allocation, and preservation of controls."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from reservoir_control.systems.lake_como.sac import SACAgent
from reservoir_control.controllers.sac.agent import SACConfig
from reservoir_control.systems.lake_como.sac_critic_reference import (ReferenceConfig,gentle_batches,reference_errors,
                                           verify_critic_reproduction)
from reservoir_control.systems.lake_como.sac_critic_sampling import matched_minibatches,SamplingConfig


class ReferenceChecks(unittest.TestCase):
    def test_exact_ten_percent_focus_and_unchanged_uniform_control(self):
        config=ReferenceConfig(budgets=(5,10))
        fit=np.arange(20);pairs=np.array([[0,10],[1,11],[2,12]])
        batches,counts=gentle_batches(fit,pairs,config,seed=8)
        previous=matched_minibatches(fit,pairs,SamplingConfig(budgets=(5,10)),seed=8)
        np.testing.assert_array_equal(batches['uniform'],previous['uniform'])
        self.assertEqual(counts[:5].sum(),64)
        self.assertEqual(counts.sum(),128)
        for i,count in enumerate(counts):
            np.testing.assert_array_equal(batches['focus10'][i,:128-count],previous['uniform'][i,:128-count])
            np.testing.assert_array_equal(batches['focus10'][i,-count:],previous['focused'][i,64:64+count])
        self.assertTrue(np.isin(batches['focus10'],fit).all())
        with self.assertRaises(ValueError):
            gentle_batches(fit,pairs,ReferenceConfig(budgets=(1,10)),seed=8)

    def test_conditional_mc_error_and_sampling_noise_correction(self):
        values=np.array([[1.,3.],[3.,5.]])
        metrics=reference_errors(np.array([3.,3.]),values)
        self.assertEqual(metrics['bias'],0.)
        self.assertEqual(metrics['mae'],1.)
        self.assertEqual(metrics['mse'],1.)
        self.assertEqual(metrics['rmse'],1.)
        self.assertEqual(metrics['mean_mc_standard_error'],1.)
        self.assertEqual(metrics['noise_adjusted_mse'],0.)
        with self.assertRaises(ValueError):
            reference_errors(np.zeros(2),np.zeros((2,1)))
        with self.assertRaises(ValueError):
            reference_errors(np.zeros(1),values)

    def test_uniform_reproduction_rejects_changed_critic_weights(self):
        agent=SACAgent(SACConfig(observation_dim=2,hidden_sizes=(8,)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'critics.pt'
            torch.save(dict(q1=agent.q1.state_dict(),q2=agent.q2.state_dict()),path)
            verify_critic_reproduction(agent,path)
            with torch.no_grad():
                next(agent.q1.parameters()).add_(.01)
            with self.assertRaises(AssertionError):
                verify_critic_reproduction(agent,path)


if __name__=='__main__':
    unittest.main()
