import unittest
import numpy as np
from lake_como.sac_action_diverse import diverse_batches,gap_metrics,old
from lake_como.sac_critic_fit import training_partitions

class ActionDiverseChecks(unittest.TestCase):
    def test_same_states_and_fitting_scope(self):
        years=np.repeat(training_partitions()[0],4)
        single,multi=diverse_batches(years,old.TargetConfig(),123)
        np.testing.assert_array_equal(single//3,multi//3)
        self.assertTrue(np.all(single%3==0))
        self.assertEqual(set(np.unique(multi%3)),{0,1,2})
        with self.assertRaises(ValueError):diverse_batches(np.array(training_partitions()[1]),old.TargetConfig(),123)
    def test_paired_gap_and_duplicate_exclusion(self):
        returns=np.repeat(np.array([[10.,11.,8.],[5.,5.,6.]])[:,:,None],8,axis=2).reshape(6,8)
        result=gap_metrics(np.array([100.,101.,98.,200.,200.,201.]),returns,[50,0,100,0,0,50])
        self.assertEqual(result['gap_pairs'],3)
        self.assertEqual(result['informative_pairs'],3)
        self.assertEqual(result['sign_correct'],3)
        self.assertEqual(result['gap_mae'],0)
    def test_noisy_gap_not_called_informative(self):
        returns=np.zeros((3,8));returns[1]=[-10,10]*4;returns[2]=.00001
        result=gap_metrics([0,1,1],returns,[50,0,100])
        self.assertEqual(result['informative_pairs'],0)

if __name__=='__main__':unittest.main()
