import unittest

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.simulation.low_water import ProtectedOutletModel
from lake_como.simulation.model import WaterState
from lake_como.simulation.feasibility_audit import full_open_lower_bound, audit_forecast
from lake_como.simulation.flood_diagnosis import (
    source, event_bounds, branch_specs, forecast_bank, restore, contrasts,
)


class FloodDiagnosisChecks(unittest.TestCase):
    def test_full_capacity_relaxation_lower_bounds_nominal_hourly_trajectories(self):
        model = ProtectedOutletModel(settings('train'))
        state = WaterState(model.lake.level_to_storage(.6))
        flows = np.array([100., 150., 90., 200., 80., 150., 100.])
        bound = full_open_lower_bound(model, state, flows)
        self.assertFalse(bound['certified_upper_exit'])
        self.assertLess(bound['order_preserving_slope_factor'], 1.)
        lower = np.array([r['level_m'] for r in bound['rows']])
        for seed in range(4):
            requests = np.random.default_rng(seed).uniform(0., model.high, 7)
            _, rows = model.rollout(state, np.repeat(requests, 24), np.repeat(flows, 24),
                                    np.repeat(100., 168), np.repeat(np.arange(1,8), 24))
            self.assertTrue((lower <= np.array([r['level_m'] for r in rows])+1e-12).all())

    def test_upper_exit_certificate_and_search_witness_cannot_disagree(self):
        model = ProtectedOutletModel(settings('train'))
        state = WaterState(model.lake.level_to_storage(.5))
        decision = dict(forecast_inflows=[[10000.]*7], forecast_target=[100.]*7,
                        selected_plan=[float(np.float32(model.high))]*7)
        audit = audit_forecast(model, state, decision, np.arange(1,8))
        self.assertTrue(audit['certified_infeasible_forecast'])
        self.assertEqual(audit['feasible_candidates'], 0)
        self.assertEqual(audit['paths'][0]['actual_schedule_exit']['side'], 'upper')
        bound = audit['paths'][0]['bound_rows']
        self.assertLessEqual(bound[-1]['level_before_m'], model.lake.spec.upper_evaluation_m)
        self.assertGreater(bound[-1]['level_m'], model.lake.spec.upper_evaluation_m)
        # We stop at this first crossing, rather than extending the rating curve.
        self.assertLess(len(bound), 168)

    def test_feasible_witness_is_distinct_from_a_relaxation_non_exit(self):
        model = ProtectedOutletModel(settings('train'))
        state = WaterState(model.lake.level_to_storage(.35))
        decision = dict(forecast_inflows=[[100.]*7], forecast_target=[100.]*7,
                        selected_plan=[float(np.float32(model.high))]*7)
        audit = audit_forecast(model, state, decision, np.arange(1,8))
        self.assertFalse(audit['certified_infeasible_forecast'])
        self.assertGreater(audit['feasible_candidates'], 0)
        self.assertIsNotNone(audit['witness'])
        with self.assertRaises(ValueError): full_open_lower_bound(model, state, [-1.])

    def test_storage_swap_preserves_clock_and_exogenous_history(self):
        _, bank = forecast_bank()
        common = dict(hour=(365+253)*24, target_ema=150.123456789, last_inflow=120.,
                      state=dict(storage=100000000., command=100., river_storage=0.))
        other = dict(common, state=dict(storage=120000000., command=200., river_storage=0.))
        a, obs_a, info_a = restore(1993, bank, common, common)
        b, obs_b, info_b = restore(1993, bank, common, other)
        np.testing.assert_array_equal(obs_a[1:], obs_b[1:])
        self.assertNotEqual(obs_a[0], obs_b[0])
        self.assertEqual(info_a, dict(info_b, storage=info_a['storage']))
        self.assertEqual(a.demand_ema, 150.123456789)
        self.assertEqual(info_a['remaining_hours'], (730-365-253)*24)
        a.close(); b.close()
        with self.assertRaises(ValueError): restore(1998, bank, common, common)
        with self.assertRaises(ValueError): source(1998, 901, 'original')

    def test_anchor_grid_is_fixed_by_flood_union_not_best_branch_outcome(self):
        frames = [pd.DataFrame(dict(year=[1993]*3, model_day=[268,269,270], flood_excess_m_hours=[1.,1.,1.]))]
        event = event_bounds(1993, frames)
        self.assertEqual(event['anchors'], [dict(lead=14,start_day=254),dict(lead=7,start_day=261)])
        self.assertEqual(event['end_day'],277)
        cases = branch_specs(dict(events=[event]))
        self.assertEqual(len(cases),50)
        self.assertEqual(sum(c['memory']=='combined' for c in cases),10)

    def test_memory_effect_is_separate_in_both_exact_contrasts(self):
        result = contrasts(dict(oo=100.,oc=110.,co=130.,cc=145.,own=147.))
        self.assertEqual(result['memory_effect'],2.)
        self.assertEqual(result['observed_factual_delta'],47.)
        self.assertEqual(result['storage_effect_original_controller'],30.)
        self.assertEqual(result['controller_effect_at_combined_stock'],15.)


if __name__ == '__main__':
    unittest.main()
