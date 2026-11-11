import tempfile
from pathlib import Path
import unittest
import numpy as np
from reservoir_control.systems.lake_como.simulation.calibration import fit_balance_area,balance_scores,read_pangaea,require_operational_calibration

class CalibrationEvidenceChecks(unittest.TestCase):
    def test_robust_fit_recovers_balance_identity_without_outlier_tuning(self):
        delta=np.array([.005,.01,-.01,.02,-.005,.04,0])
        flow=145152000*delta/86400;flow[-2]+=100
        self.assertAlmostEqual(fit_balance_area(delta,flow),145152000)
        self.assertEqual(balance_scores(delta[:4],flow[:4],145152000)['flow_mae_m3_s'],0)
        with self.assertRaises(ValueError):fit_balance_area(np.zeros(3),np.ones(3))

    def test_parser_rejects_test_years_before_numeric_scoring(self):
        with self.assertRaises(ValueError):read_pangaea('not-read.tsv',[1998])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.tsv'
            path.write_text('/* Flow rate, daily mean */\nDate/Time\tQ [m**3/s]\n1946-01-01\t100\n')
            with self.assertRaises(ValueError):read_pangaea(path,[1946])

    def test_partial_identification_cannot_be_promoted(self):
        with self.assertRaises(ValueError):require_operational_calibration({'fully_calibrated':True,'evidence':{}})
        evidence={k:True for k in ['confirmed_flow_interval','independent_outlet_measurements','gate_ramp_identified','river_routing_identified','downstream_reference_verified','irrigation_demand_verified']}
        self.assertTrue(require_operational_calibration({'evidence':evidence}))
