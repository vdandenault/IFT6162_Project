"""Target-only scope, shared minibatches, and rollout reference alignment."""
import unittest
import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.sac_critic_targets import (TargetConfig,check_fit_scope,target_batches,
                                         assert_identical_non_targets,ordered_returns)
from reservoir_control.systems.lake_como.sac_critic_fit import training_partitions


class TargetChecks(unittest.TestCase):
    def test_only_fitting_years_enter_minibatches(self):
        fit,check=training_partitions()
        years=np.array([fit[0],fit[1]])
        config=TargetConfig(budgets=(2,10),batch_size=8)
        first=target_batches(years,config,seed=123)
        second=target_batches(years,config,seed=123)
        self.assertEqual(first.shape,(10,8))
        np.testing.assert_array_equal(first,second)
        self.assertTrue(np.isin(years[first],fit).all())
        for bad in [np.array([]),np.array([check[0]]),np.array([1988]),np.array([1998])]:
            with self.assertRaises(ValueError):
                check_fit_scope(bad)

    def test_only_target_values_may_differ(self):
        common=dict(observations=np.zeros((2,5)),requests=np.array([100.,200.]),
                    years=np.array([1946,1948]),days=np.array([45,135]))
        left={**common,'targets':np.array([1.,2.])}
        right={**common,'targets':np.array([3.,4.])}
        assert_identical_non_targets(left,right)
        right['requests']=np.array([101.,200.])
        with self.assertRaises(AssertionError):
            assert_identical_non_targets(left,right)

    def test_return_matrix_matches_records_not_dataframe_order(self):
        records=[dict(seed=123,year=1946,day=45),dict(seed=123,year=1948,day=135)]
        rows=[]
        for i,record in enumerate(records):
            for replica in reversed(range(8)):
                rows.append(dict(**record,replicate=replica,soft_return=i*100+replica))
        result=ordered_returns(records,pd.DataFrame(rows).iloc[::-1])
        np.testing.assert_array_equal(result,[np.arange(8),100+np.arange(8)])
        with self.assertRaises(ValueError):
            ordered_returns(records,pd.DataFrame(rows[:-1]))


if __name__=='__main__':
    unittest.main()
