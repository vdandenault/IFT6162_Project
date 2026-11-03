import tempfile
from pathlib import Path
import unittest
import gymnasium as gym
import numpy as np
import torch
from lake_como.sac import SACConfig
from lake_como.residual_mpc import combine_request,MPCProposalEnv,CombinedCritic,ProposalSACAgent


class ResidualChecks(unittest.TestCase):
    def test_bound_and_saturation_semantics(self):
        self.assertEqual(combine_request(100.,0.),np.float32(100.))
        self.assertEqual(combine_request(490.,50.),np.float32(491.61))
        self.assertEqual(combine_request(10.,-50.),0.)
        with self.assertRaises(ValueError): combine_request(float('nan'),0.)

    def test_critic_uses_combined_action_and_clips_gradients(self):
        cfg=SACConfig(observation_dim=6,hidden_sizes=(8,))
        critic=CombinedCritic(cfg,'residual',50.,500.)
        obs=torch.zeros(3,6);obs[:,-1]=torch.tensor([0.,.9,-.9])
        a=torch.tensor([[.5],[1.],[-1.]],requires_grad=True)
        combined=critic.combined_action(obs,a)
        torch.testing.assert_close(combined,torch.tensor([[.1],[1.],[-1.]]))
        combined.sum().backward()
        torch.testing.assert_close(a.grad,torch.tensor([[.2],[0.],[0.]]))

    def test_terminal_does_not_call_planner(self):
        calls=[]
        class Planner:
            diagnostics=[]
            def __call__(self,obs,info):
                calls.append(1);self.diagnostics=[dict(solver_success=True)];return np.array([100.],np.float32)
        class Env(gym.Env):
            observation_space=gym.spaces.Box(-np.inf,np.inf,(5,),np.float32)
            action_space=gym.spaces.Box(0,500.,(1,),np.float32)
            def reset(self,seed=None,options=None): return np.zeros(5,np.float32),{}
            def step(self,a): self.last=a;return np.ones(5,np.float32),-1.,False,True,{}
        underlying=Env();env=MPCProposalEnv(underlying,Planner)
        obs,_=env.reset();self.assertEqual(obs[-1],100.)
        following,_,_,end,info=env.step(np.zeros(1,np.float32))
        np.testing.assert_array_equal(underlying.last,[100.]);self.assertTrue(end)
        self.assertEqual(following[-1],0.);self.assertEqual(len(calls),1)
        self.assertEqual(info['effective_correction_m3_s'],0.)

    def test_checkpoint_round_trip_and_zero_mean_residual(self):
        cfg=SACConfig(observation_dim=6,hidden_sizes=(8,8))
        agent=ProposalSACAgent(cfg,observation_mean=[0,0,0,0,0,250],observation_scale=[1,1,1,1,1,250],high=500.)
        obs=np.array([.35,0,1,100,100,200],np.float32)
        np.testing.assert_array_equal(agent.act(obs,deterministic=True),[0.])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'policy.pt';agent.save(path);loaded=ProposalSACAgent.load(path)
            np.testing.assert_array_equal(agent.act(obs),loaded.act(obs))
            for name,value in agent.state_dict().items(): torch.testing.assert_close(value,loaded.state_dict()[name],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
