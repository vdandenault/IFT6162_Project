"""Synthetic fixtures validate scenario accounting, not Baskatong calibration."""
import unittest
import numpy as np
import pandas as pd
from reservoir_control.study_config import OTTAWA_EVIDENCE, load_settings
from reservoir_control.systems.ottawa.baskatong_scenario import (
    LinearStorage, align_centers, effective_inputs, prescribe, simulate,
    evaluate_case, negative_diagnostics, validate,
)


class ScenarioChecks(unittest.TestCase):
    def fixture(self):
        c = load_settings(OTTAWA_EVIDENCE, "baskatong_release_scenario")
        c['observed_inputs'].update(first_date='2021-01-01', last_interval_start_date='2021-01-04',
                                     following_context_date='2021-01-05')
        c['experimental_bounds_assumed'].update(level_minimum_m=0., level_maximum_m=30.,
                                               release_minimum_m3_s=0., release_maximum_m3_s=20.)
        c['prescribed_release_experiment'].update(retention_start_inclusive='2021-01-01',
            retention_end_exclusive='2021-01-03', reduction_fraction=.2,
            repayment_start_inclusive='2021-01-03', repayment_end_exclusive='2021-01-05',
            maximum_additional_release_m3_s=2.)
        dates = pd.date_range('2021-01-01', '2021-01-05').strftime('%Y-%m-%d')
        h = pd.DataFrame({'DATE': dates, 'LEVEL': [10.]*5, 'LEVEL_SYMBOL_EN': [None]*5})
        q = pd.DataFrame({'DATE': dates, 'DISCHARGE': [10.]*5, 'DISCHARGE_SYMBOL_EN': [None]*5})
        return c, h, q, LinearStorage(1e6, 0.)

    def test_mapping_inverts_and_storage_origin_is_explicit(self):
        mapping = LinearStorage(2e6, 200.)
        np.testing.assert_array_equal(mapping.level(mapping.storage([201., 205.])), [201., 205.])
        self.assertEqual(mapping.storage(200.), 0.)
        with self.assertRaises(ValueError): LinearStorage(0., 200.)
        with self.assertRaises(ValueError): LinearStorage(np.inf, 200.)

    def test_center_alignment_adjacent_means_and_flag_union(self):
        c, h, q, _ = self.fixture()
        q['DISCHARGE'] = [2., 4., 6., 8., 10.]
        q.loc[1, 'DISCHARGE_SYMBOL_EN'] = 'Partial Day'
        rows = align_centers(h, q, c)
        np.testing.assert_array_equal(rows.proxy_interval_release_m3_s, [3., 5., 7., 9.])
        self.assertEqual(rows.source_flagged.tolist(), [True, True, False, False])
        self.assertEqual(rows.model_interval_start.iloc[0], '2021-01-01T12:00:00')
        self.assertEqual(rows.model_interval_end.iloc[-1], '2021-01-05T12:00:00')
        self.assertTrue(rows.duration_seconds.eq(86400.).all())

    def test_missing_final_context_remains_unavailable(self):
        c, h, q, model = self.fixture()
        rows = align_centers(h.iloc[:-1], q.iloc[:-1], c)
        self.assertFalse(rows.input_available.iloc[-1])
        self.assertTrue(np.isnan(rows.proxy_interval_release_m3_s.iloc[-1]))
        inputs = effective_inputs(rows, model)
        replay, alternative, summary = evaluate_case(inputs, model, prescribe(rows, c), c)
        self.assertTrue(np.isnan(replay.level_end_m.iloc[-1]))
        self.assertFalse(summary['alternative_complete'])
        self.assertFalse(summary['release_totals_match'])

    def test_water_conservation_same_effective_input_and_volume_accounting(self):
        c, h, q, model = self.fixture()
        inputs = effective_inputs(align_centers(h, q, c), model)
        original = inputs.effective_net_input_m3_s.copy()
        ledger = prescribe(inputs, c)
        replay, alternative, summary = evaluate_case(inputs, model, ledger, c)
        np.testing.assert_array_equal(inputs.effective_net_input_m3_s, original)
        np.testing.assert_array_equal(replay.effective_net_input_m3_s, alternative.effective_net_input_m3_s)
        np.testing.assert_allclose(replay.level_end_m, 10., atol=1e-12)
        np.testing.assert_allclose(alternative.storage_end_m3-replay.storage_end_m3,
                                   ledger.outstanding_retained_m3, atol=1e-7)
        self.assertTrue(summary['release_totals_match'])
        self.assertAlmostEqual(summary['retained_volume_m3'], 345600.)
        self.assertAlmostEqual(summary['repaid_volume_m3'], 345600.)
        self.assertAlmostEqual(summary['final_level_difference_m'], 0.)
        self.assertLess(summary['max_mass_balance_error_m3'], 1e-7)

    def test_unrepaid_volume_is_not_claimed_equal(self):
        c, h, q, model = self.fixture()
        c['prescribed_release_experiment']['maximum_additional_release_m3_s'] = .5
        inputs = effective_inputs(align_centers(h, q, c), model)
        _, _, summary = evaluate_case(inputs, model, prescribe(inputs, c), c)
        self.assertAlmostEqual(summary['remaining_unreleased_volume_m3'], 259200.)
        self.assertFalse(summary['release_totals_match'])
        self.assertAlmostEqual(summary['final_level_difference_m'], 259200./model.effective_area_m2)

    def test_level_bounds_are_reported_without_clipping(self):
        c, h, q, model = self.fixture()
        c['experimental_bounds_assumed']['level_maximum_m'] = 10.1
        inputs = effective_inputs(align_centers(h, q, c), model)
        _, alt, summary = evaluate_case(inputs, model, prescribe(inputs, c), c)
        self.assertGreater(alt.level_end_m.max(), 10.1)
        self.assertFalse(summary['alternative_feasible_under_assumptions'])
        self.assertTrue(summary['release_totals_match'])
        self.assertGreater(summary['alternative_upper_excess_peak_m'], 0.)

    def test_negative_inputs_and_flag_exclusion_remain_visible(self):
        c, h, q, model = self.fixture()
        h['LEVEL'] = [10., 9., 8., 7., 6.]
        q.loc[1, 'DISCHARGE_SYMBOL_EN'] = 'Partial Day'
        inputs = effective_inputs(align_centers(h, q, c), model)
        self.assertTrue(inputs.effective_net_input_m3_s.lt(0).all())
        self.assertEqual(negative_diagnostics(inputs, False)['negative_input_intervals'], 4)
        self.assertEqual(negative_diagnostics(inputs, True)['negative_input_intervals'], 2)
        replay = simulate(inputs, model, inputs.proxy_interval_release_m3_s, c)
        np.testing.assert_allclose(replay.level_end_m, h.LEVEL.to_numpy()[1:])
        self.assertTrue(replay.state_depends_on_flagged_input.all())

    def test_outlet_bounds_respected_without_repairing_proxy(self):
        c, h, q, model = self.fixture()
        c['experimental_bounds_assumed']['release_maximum_m3_s'] = 9.
        inputs = effective_inputs(align_centers(h, q, c), model)
        ledger = prescribe(inputs, c)
        self.assertTrue(ledger.experiment_release_m3_s.isna().all())
        self.assertTrue(inputs.proxy_interval_release_m3_s.eq(10.).all())
        _, _, summary = evaluate_case(inputs, model, ledger, c)
        self.assertFalse(summary['alternative_feasible_under_assumptions'])
        self.assertFalse(summary['release_totals_match'])

    def test_no_reset_across_missing_intermediate_input(self):
        c, h, q, model = self.fixture()
        q.loc[1, 'DISCHARGE'] = np.nan
        inputs = effective_inputs(align_centers(h, q, c), model)
        replay = simulate(inputs, model, inputs.proxy_interval_release_m3_s, c)
        self.assertTrue(replay.level_end_m.isna().all())
        self.assertTrue(inputs.input_available.iloc[-1])

    def test_area_changes_level_effect_and_input_interpretation(self):
        c, h, q, model = self.fixture()
        h['LEVEL'] = [10., 10.1, 10.2, 10.3, 10.4]
        intervals = align_centers(h, q, c); ledger = prescribe(intervals, c)
        a, b = effective_inputs(intervals, model), effective_inputs(intervals, LinearStorage(2e6, 0.))
        self.assertFalse(np.array_equal(a.effective_net_input_m3_s, b.effective_net_input_m3_s))
        _, _, s1 = evaluate_case(a, model, ledger, c)
        _, _, s2 = evaluate_case(b, LinearStorage(2e6, 0.), ledger, c)
        self.assertAlmostEqual(s1['max_level_increase_m'], 2*s2['max_level_increase_m'])
        self.assertEqual(s1['retained_volume_m3'], s2['retained_volume_m3'])

    def test_invalid_scenario_config_is_rejected(self):
        c, _, _, _ = self.fixture()
        validate(c)
        c['prescribed_release_experiment']['reduction_fraction'] = 1.1
        with self.assertRaises(ValueError): validate(c)


if __name__ == '__main__': unittest.main()
