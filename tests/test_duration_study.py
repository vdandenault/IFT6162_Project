"""Protect duration pairing and detect optimizer-state mismatches at the prefix."""
from copy import deepcopy
import importlib.util
import tempfile
from pathlib import Path
import unittest
import pandas as pd
from reservoir_control.systems.lake_como.duration_study import trees_equal, compare_prefix, decision_from_means, decide_million


class DurationChecks(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Requires reproduction group')
    def test_comparison_includes_optimizer_and_tensor_dtype(self):
        import torch
        a={'policy':{'weight':torch.ones(2)},'optimizer':{'state':{0:{'exp_avg':torch.zeros(2)}},'param_groups':[{'lr':1e-4}]}}
        b=deepcopy(a)
        self.assertTrue(trees_equal(a,b))
        b['optimizer']['state'][0]['exp_avg'][0]=1
        self.assertFalse(trees_equal(a,b))
        self.assertFalse(trees_equal(torch.ones(2),torch.ones(2,dtype=torch.float64)))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Requires reproduction group')
    def test_prefix_mismatch_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            a=Path(temporary)/'a.pt';b=Path(temporary)/'b.pt'
            torch.save({'policy':{'weight':torch.ones(2)},'optimizer':{'lr':1e-4}},a)
            torch.save({'policy':{'weight':torch.ones(2)},'optimizer':{'lr':1e-4}},b)
            self.assertTrue(all(compare_prefix(a,b).values()))
            torch.save({'policy':{'weight':torch.ones(2)},'optimizer':{'lr':1e-3}},b)
            with self.assertRaisesRegex(ValueError,'prefix mismatch'):compare_prefix(a,b)

    def test_longer_run_requires_mean_variance_and_weakest_seed_guardrails(self):
        rows=[]
        for budget,values in [(200000,[-20]*5),(400000,[-1,-1,-1,-1,-70])]:
            rows.extend(dict(budget=budget,training_seed=seed,full_return=value)
                        for seed,value in zip(range(123,128),values))
        frame=pd.DataFrame(rows).set_index(['budget','training_seed'])
        self.assertEqual(decision_from_means(frame)['selected_steps'],200000)
        frame.loc[400000,'full_return']=[-10]*5
        self.assertEqual(decision_from_means(frame)['selected_steps'],400000)

    def test_600k_comparison_uses_400k_reference(self):
        rows = [dict(budget=budget, training_seed=seed, full_return=value)
                for budget, value in [(200000, -1), (400000, -20), (600000, -10)]
                for seed in range(123,128)]
        means = pd.DataFrame(rows).set_index(['budget','training_seed'])
        decision = decision_from_means(means, baseline=400000, candidate=600000)
        self.assertEqual(decision['selected_steps'],600000)
        self.assertIn('versus 400k',decision['rule'])
        self.assertFalse(decision['test_evaluated'])

    def test_million_endpoints_compared_independently_against_600k(self):
        rows = [dict(budget=budget,training_seed=seed,full_return=value)
                for budget,values in [(600000,[-20]*5),(800000,[-10]*5),
                                      (1000000,[-1,-1,-1,-1,-25])]
                for seed,value in zip(range(123,128),values)]
        means=pd.DataFrame(rows).set_index(['budget','training_seed'])
        result=decide_million(means)
        self.assertTrue(result['comparisons']['800000']['longer_qualifies'])
        self.assertFalse(result['comparisons']['1000000']['longer_qualifies'])
        self.assertEqual(result['selected_steps'],800000)
        means.loc[1000000,'full_return']=[-5]*5
        self.assertEqual(decide_million(means)['selected_steps'],1000000)
        means.loc[1000000,'full_return']=[-10]*5
        self.assertEqual(decide_million(means)['selected_steps'],800000)
        means.loc[800000,'full_return']=[-30]*5
        means.loc[1000000,'full_return']=[-30]*5
        self.assertEqual(decide_million(means)['selected_steps'],600000)


if __name__=='__main__':unittest.main()
