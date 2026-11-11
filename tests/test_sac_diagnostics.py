"""Diagnostic accounting, selection scope, and entropy/boundary conventions."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch

from reservoir_control.systems.lake_como.sac_diagnostics import (water_allocations, selected_training_years,
                                      rollout_branch, same_transition)
from reservoir_control.systems.lake_como.splits import split_years


class DiagnosticChecks(unittest.TestCase):
    def test_water_accounting_including_release_below_environmental_flow(self):
        frame=pd.DataFrame(dict(day=[1,1,1],release_m3_s=[5.,15.,30.],
                                demand_m3_s=[10.,10.,10.],request_m3_s=[6.,16.,31.],
                                deficit_m3=np.array([10.,5.,0.])*86400))
        result=water_allocations(frame,np.full(365,10.))
        np.testing.assert_array_equal(result.useful_m3,np.array([0.,5.,10.])*86400)
        np.testing.assert_array_equal(result.surplus_after_mef_m3,np.array([0.,0.,10.])*86400)
        np.testing.assert_array_equal(result.environmental_allocation_m3,np.array([5.,10.,10.])*86400)

    def test_year_selection_does_not_accept_validation_membership(self):
        profiles={year:[float(year)]*365 for year in split_years('train')}
        selected=selected_training_years(profiles)
        self.assertEqual(selected,dict(dry=1946,median=1966,wet=1987))
        with self.assertRaises(ValueError):
            selected_training_years({1988:[1.]*365})

    def test_branch_excludes_first_entropy_and_stops_at_truncation(self):
        class Env:
            day=0
            def step(self,action):
                self.day+=1
                return np.array([self.day],np.float32),-10.,False,self.day==2,{}
            def close(self):
                pass
        class Agent:
            config=SimpleNamespace(gamma=.5,reward_scale=.1)
            alpha=torch.tensor(.2)
            def normalize(self,obs):
                return obs
            def actor(self,obs,generator=None):
                return torch.zeros(1,1),torch.full((1,1),-2.)
            def physical_action(self,action):
                return action
        env=Env()
        result=rollout_branch(env,Agent(),100.,noise_seed=7)
        # r0*.1 + gamma*(r1*.1 - alpha*logp1) = -1 + .5*(-1+.4).
        self.assertAlmostEqual(result['soft_return'],-1.3,places=6)
        self.assertEqual(result['discounted_original_return'],-15.)
        self.assertEqual(result['continuation_steps'],2)
        self.assertEqual(env.day,0)  # branching must not advance the source snapshot
        env.day=1
        terminal=rollout_branch(env,Agent(),100.,noise_seed=7)
        self.assertEqual(terminal['continuation_steps'],1)
        self.assertEqual(terminal['soft_return'],-1.)

    def test_transition_identity_requires_exact_storage_and_observation(self):
        first=(np.array([1.,2.]),-1.,dict(storage=3.,release=4.),False)
        better=(np.array([1.,2.]),0.,dict(storage=3.,release=4.),False)
        self.assertTrue(same_transition(first,better))
        changed=(np.array([1.,2.]),0.,dict(storage=3.000001,release=4.),False)
        self.assertFalse(same_transition(first,changed))


if __name__=='__main__':
    unittest.main()
