import unittest
import numpy as np
from reservoir_control.systems.lake_como.simulation.weighted_probability_audit import distribution_metrics
from reservoir_control.systems.lake_como.simulation.weighted_forecast import weighted_crps


class ProbabilityAuditChecks(unittest.TestCase):
    def test_sorted_crps_matches_pairwise_definition(self):
        x=np.array([8.,0.,3.,12.]);p=np.array([.1,.2,.4,.3])
        for truth in [-1.,2.,7.,20.]:
            result=distribution_metrics(x,p,truth,5.)
            self.assertAlmostEqual(result['volume_crps'],weighted_crps(x,truth,p))

    def test_quantile_coverage_and_event_probability_are_distinct(self):
        result=distribution_metrics([0.,1.,2.,3.,10.],[.25,.25,.25,.2,.05],4.,5.)
        self.assertAlmostEqual(result['wet_probability'],.05)
        self.assertFalse(result['wet_truth'])
        self.assertFalse(result['q50_covered']);self.assertFalse(result['q90_covered'])
        self.assertFalse(result['q95_covered'])
        self.assertAlmostEqual(result['brier'],.05**2)

    def test_invalid_probabilities_cannot_be_silently_renormalized(self):
        with self.assertRaises(ValueError):distribution_metrics([0.,1.],[.1,.1],.5,1.)
        with self.assertRaises(ValueError):distribution_metrics([0.,1.],[1.,0.],.5,1.)


if __name__=='__main__':unittest.main()
