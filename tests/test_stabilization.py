"""Verify the controlled ablation and rejection of stable but poor policies."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
import pandas as pd
from lake_como.stabilize import agent_config, select_candidate, VARIANTS, COMMON


class StabilizationChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('skrl'),'Requires reproduction dependency group')
    def test_only_learning_rate_scheduler_changes(self):
        with tempfile.TemporaryDirectory() as d:
            configs=[agent_config(v,Path(d)) for v in VARIANTS]
            self.assertEqual(configs[0]['learning_rate_scheduler_kwargs']['max_lr'],1e-2)
            self.assertEqual(configs[1]['learning_rate_scheduler_kwargs']['max_lr'],1e-4)
            self.assertIsNone(configs[2]['learning_rate_scheduler'])
            for cfg in configs:
                self.assertEqual(cfg['learning_rate'],8e-6)
                cfg.pop('learning_rate_scheduler');cfg.pop('learning_rate_scheduler_kwargs')
            self.assertEqual(configs[0],configs[1])
            self.assertEqual(configs[1],configs[2])

    def test_reject_stable_but_worse_and_select_qualifying_candidate(self):
        rows=[]
        values={'original_scheduler':[-10,-20,-30,-40,-50],
                'capped_scheduler':[-15,-20,-25,-30,-35],
                'fixed_lr':[-60]*5}
        for variant, scores in values.items():
            rows.extend({'variant':variant,'training_seed':seed,'full_return':score}
                        for seed,score in zip(range(123,128),scores))
        frame=pd.DataFrame(rows).set_index(['variant','training_seed'])
        result=select_candidate(frame)
        self.assertEqual(result['selected'],'capped_scheduler')
        self.assertEqual(result['qualifying_candidates'],['capped_scheduler'])
        frame.loc['capped_scheduler','full_return']=[-80]*5
        self.assertEqual(select_candidate(frame)['selected'],'original_scheduler')

    def test_annual_horizon_held_fixed(self):
        self.assertFalse(COMMON['time_limit_bootstrap'])
        self.assertTrue(COMMON['correct_terminal_level'])
        self.assertEqual(COMMON['steps'],200000)
        self.assertEqual(COMMON['protocol'],'chronological')


if __name__=='__main__':unittest.main()
