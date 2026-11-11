"""Ensure precision is isolated from optimization settings and selection is guarded."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.precision_study import decision_from_means
from reservoir_control.systems.lake_como.stabilize import agent_config


class PrecisionChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('skrl'),'Requires reproduction group')
    def test_configuration_differs_only_in_mixed_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            control=agent_config('capped_scheduler',Path(directory))
            fp32=agent_config('capped_scheduler',Path(directory),mixed_precision=False)
            self.assertTrue(control.pop('mixed_precision'))
            self.assertFalse(fp32.pop('mixed_precision'))
            self.assertEqual(control,fp32)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('skrl'),'Requires reproduction group')
    def test_autocast_switch_changes_forward_precision(self):
        import torch
        from gym4real.algorithms.dam.ppo_skrl import Policy
        from gymnasium.spaces import Box
        policy=Policy(Box(-np.inf,np.inf,shape=(4,),dtype=np.float32),
                      Box(-1,1,shape=(1,),dtype=np.float32),device='cpu',net_arch=(16,16))
        x={'states':torch.zeros((1,4))}
        with torch.autocast('cpu',enabled=True): mixed=policy.compute(x)[0]
        with torch.autocast('cpu',enabled=False): full=policy.compute(x)[0]
        self.assertEqual(mixed.dtype,torch.bfloat16)
        self.assertEqual(full.dtype,torch.float32)

    def test_selection_rejects_lower_variance_with_worse_mean(self):
        rows=[]
        for arm,values in [('mixed_precision',[-10,-20,-30,-40,-50]),('fp32',[-60]*5)]:
            rows.extend(dict(precision=arm,training_seed=seed,full_return=value)
                        for seed,value in zip(range(123,128),values))
        means=pd.DataFrame(rows).set_index(['precision','training_seed'])
        self.assertEqual(decision_from_means(means)['selected'],'mixed_precision')
        means.loc['fp32','full_return']=[-10]*5
        self.assertEqual(decision_from_means(means)['selected'],'fp32')


if __name__=='__main__':unittest.main()
