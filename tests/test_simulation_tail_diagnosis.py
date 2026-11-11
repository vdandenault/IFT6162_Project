import unittest

import numpy as np

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.history_forecast import HistoryForecastBank
from reservoir_control.systems.lake_como.simulation.tail_diagnosis import pool_details,nested_draws,support_record,fallback_contrasts,forecast_infeasible
from reservoir_control.systems.lake_como.simulation.low_water import ProtectedOutletModel
from reservoir_control.systems.lake_como.simulation.model import WaterState
from reservoir_control.systems.lake_como.simulation.feasibility_audit import full_open_lower_bound


class TailDiagnosisChecks(unittest.TestCase):
    def test_frozen_five_paths_are_reconstructed_and_extra_sets_are_nested(self):
        training=settings('train')
        bank=HistoryForecastBank(training['inflow'],training['demand'],before_year=1976,method='history_scaled_stratified')
        past=np.array([80.,90.,110.,120.,140.,170.,200.]);bank.set_history(past)
        details=pool_details(bank,120,past)
        for seed in [901,905]:
            sets=nested_draws(details['conditioned'],seed)
            frozen=bank.forecast('conditional',120,200.,seed=seed)[0]
            np.testing.assert_array_equal(details['conditioned'][sets[5]],frozen)
            np.testing.assert_array_equal(sets[15][:5],sets[5])
            np.testing.assert_array_equal(sets[25][:15],sets[15])
            self.assertTrue(set(sets[25]).issubset(set(range(128))))

    def test_upper_miss_categories_separate_support_conditioning_and_sampling(self):
        paths=np.repeat(np.arange(128.)[:,None],7,axis=1)
        details=dict(conditioned=paths,seasonal=np.repeat(np.arange(256.)[:,None],7,axis=1),
                     raw_conditioned=paths,uncapped_conditioned=paths,ratios=np.ones(128))
        scale=7*86400/1e6
        self.assertEqual(support_record(details,300*scale,0)['category'],'outside_seasonal_support')
        self.assertEqual(support_record(details,200*scale,0)['category'],'excluded_by_history_pool')
        self.assertEqual(support_record(details,127*scale,0)['category'],'missed_by_five_sampling')
        self.assertEqual(support_record(details,0.,0)['category'],'covered_by_five')
        row=support_record(details,127*scale,0)
        self.assertAlmostEqual(row['conditional_five_upper_coverage_probability'],1/25)
        self.assertLessEqual(row['max_5_million_m3'],row['max_15_million_m3'])
        self.assertLessEqual(row['max_15_million_m3'],row['max_25_million_m3'])

    def test_counterfactual_orders_both_reconcile_without_unique_percentages(self):
        result=fallback_contrasts(dict(old_state_old_forecast=[0,1,0],old_state_new_forecast=[1,0,0],
                                       new_state_old_forecast=[0,1,1],new_state_new_forecast=[1,1,1]))
        self.assertEqual(result['observed_extra_fallbacks'],2)
        self.assertEqual(result['forecast_effect_at_old_states']+result['state_effect_under_new_forecast'],2)
        self.assertEqual(result['state_effect_under_old_forecast']+result['forecast_effect_at_new_states'],2)
        self.assertEqual(result['forecast_new_warnings_at_old_states'],1)
        self.assertEqual(result['forecast_removed_warnings_at_old_states'],1)
        with self.assertRaises(ValueError):fallback_contrasts(dict(old_state_old_forecast=[0],old_state_new_forecast=[],
            new_state_old_forecast=[0],new_state_new_forecast=[1]))

    def test_fast_non_exit_bound_matches_full_hourly_certificate(self):
        model=ProtectedOutletModel(settings('train'))
        for level in [-.4,.35,1.2,2.4]:
            for flow in [0.,100.,700.,10000.]:
                paths=np.full((1,7),flow)
                rounded=float(np.float32(level))
                full=full_open_lower_bound(model,WaterState(model.lake.level_to_storage(rounded)),paths[0])
                self.assertEqual(forecast_infeasible(model,level,paths),full['certified_upper_exit'])
        with self.assertRaises(ValueError):forecast_infeasible(model,.35,[[-1.]*7])


if __name__=='__main__':unittest.main()
