import unittest
import numpy as np
import torch
from lake_como.sac import SACAgent,SACConfig
from lake_como.sac_iterative_td import td_target,policy_state

class IterativeTDChecks(unittest.TestCase):
    def test_terminal_mask_and_shared_noise(self):
        agent=SACAgent(SACConfig(observation_dim=2,hidden_sizes=(8,)))
        data=dict(next_observations=np.zeros((2,2),np.float32),rewards=np.array([-10.,-20.]),ends=np.array([0,1],np.float32))
        original=policy_state(agent);rng=agent.generator.get_state().clone()
        first=td_target(agent,data,np.array([0,1]),torch.Generator().manual_seed(3))
        second=td_target(agent,data,np.array([0,1]),torch.Generator().manual_seed(3))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertAlmostEqual(first[1].item(),-.020,places=6)
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.equal(rng,agent.generator.get_state()))
        self.assertTrue(all(torch.equal(v,agent.state_dict()[k]) for k,v in original.items()))
        self.assertFalse(any(k.startswith('target_q') for k in original))

if __name__=='__main__': unittest.main()
