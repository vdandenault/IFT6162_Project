"""Regression checks for oracle scope and first-step feasibility certificates."""
from dataclasses import replace
import unittest

import numpy as np

from reservoir_control.systems.loslr.causal import CausalOntarioSession
from reservoir_control.systems.loslr.data import load_inputs
from reservoir_control.systems.loslr.model import OntarioModel
from reservoir_control.systems.loslr.mpc_diagnostics import OracleInflowAdapter, oracle_decision, release_scan
from reservoir_control.systems.loslr.native import NativeReference, default_checkout
from reservoir_control.systems.loslr.planning import MPCSettings, OntarioPlanningAdapter


@unittest.skipUnless((default_checkout() / '.git').exists(), 'Pinned Ontario source required')
class OntarioDiagnosticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = NativeReference()
        cls.frame = load_inputs(cls.reference)
        cls.model = OntarioModel(cls.reference)

    def test_oracle_changes_inflows_and_preserves_causal_hydraulic_information(self):
        session = CausalOntarioSession(self.model, self.frame)
        causal, time = session.information()
        oracle = oracle_decision(self.frame, session.index, causal)
        adapter = OracleInflowAdapter(self.model, MPCSettings(), self.frame)
        stages = tuple(adapter.forecast_stages(oracle, time))
        for lead, (decision, _) in enumerate(stages):
            actual = self.frame.iloc[session.index + lead]
            self.assertEqual(decision.forecast.nts_m3_s[0], actual.ontNTS)
            self.assertEqual(decision.forecast.slon_m3_s, actual.stlouisontOut)
            self.assertEqual(decision.ice, causal.ice)
            self.assertEqual(decision.pointe_claire_roughness, causal.pointe_claire_roughness)
            self.assertEqual(decision.forecast.annual_nts_m3_s, causal.forecast.annual_nts_m3_s)
        self.assertEqual(session.index, 96)

    def test_relaxed_cap_conflict_excludes_all_scanned_executed_releases(self):
        session = CausalOntarioSession(self.model, self.frame)
        decision, time = session.information()
        wet_state = replace(session.state, level_m=76.5)
        adapter = OntarioPlanningAdapter(self.model)
        diagnostic, grid = release_scan(adapter, wet_state, decision, time, grid_step=100.)
        self.assertTrue(diagnostic['relaxed_caps_conflict'])
        self.assertFalse(grid.feasible.any())
        self.assertGreater(diagnostic['necessary_release_min_m3_s'],
                           diagnostic['necessary_release_max_m3_s'])

    def test_invalid_objective_weights_rejected(self):
        for weight in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                OntarioPlanningAdapter(self.model, MPCSettings(tracking_weight=weight))
