import unittest

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank
from reservoir_control.systems.lake_como.simulation.precision_validation import (
    SEEDS, METRICS, specs, make_controller, annual_records, paired_annual, scope_summary,
)


def annual_fixture():
    rows = []
    for seed in [901, 902]:
        for year in [1994, 1995]:
            for arm in ['original', 'combined']:
                # Opposite seed effects must stay paired, never cherry-picked.
                cost = 100. + (10. if seed == 902 else -20.) * (arm == 'combined')
                rows.append(dict(seed=seed, block_start=1994, year=year,
                    position_in_pair=year-1993, arm=arm, complete=True,
                    **{metric: cost if metric == 'annual_cost' else 0. for metric in METRICS}))
    return pd.DataFrame(rows)


class PrecisionValidationChecks(unittest.TestCase):
    def test_fixed_complete_validation_grid_and_forecast_seeds(self):
        cases = specs()
        self.assertEqual(len(cases), 50)
        self.assertEqual(len({s['name'] for s in cases}), 50)
        self.assertEqual(set(y for s in cases for y in s['years']), set(range(1988, 1998)))
        self.assertEqual(SEEDS, (901, 902, 903, 904, 905))
        self.assertEqual(sum(s['years'] == [1994, 1995] for s in cases), 10)

    def test_seed_changes_only_scenario_sampling_not_target_fit_or_costs(self):
        training = settings('train')
        bank = CausalForecastBank(training['inflow'], training['demand'], before_year=1988)
        original = make_controller(training, bank, 'original', 901)
        combined = make_controller(training, bank, 'combined', 902)
        self.assertEqual(original.weights, combined.weights)
        np.testing.assert_array_equal(original.target_forecaster.slopes, combined.target_forecaster.slopes)
        a = bank.forecast('conditional', 120, 150., seed=901+24*119)
        b = bank.forecast('conditional', 120, 150., seed=902+24*119)
        np.testing.assert_array_equal(a[1], b[1])
        self.assertFalse(np.array_equal(a[0], b[0]))
        self.assertEqual(bank.years, tuple(range(1946, 1988)))
        with self.assertRaises(ValueError): make_controller(training, bank, 'combined', 42)

    def test_pairing_keeps_seed_and_year_and_does_not_select_best_seed(self):
        paired = paired_annual(annual_fixture().sample(frac=1., random_state=3))
        a = scope_summary(paired, [1994, 1995], [901, 902])
        self.assertTrue(a['all_required_complete'])
        self.assertEqual(a['mean_delta_annual_cost'], -5.)
        self.assertEqual((a['wins'], a['losses']), (2, 2))
        self.assertEqual(scope_summary(paired, [1994, 1995], [902])['mean_delta_annual_cost'], 10.)

    def test_incomplete_or_missing_member_suppresses_primary_average(self):
        annual = annual_fixture()
        annual.loc[(annual.seed == 902) & (annual.year == 1995) & annual.arm.eq('combined'),
                   ['complete', 'annual_cost']] = [False, np.nan]
        paired = paired_annual(annual)
        result = scope_summary(paired, [1994, 1995], [901, 902])
        self.assertFalse(result['all_required_complete'])
        self.assertEqual(result['complete_pairs'], 3)
        self.assertIsNone(result['mean_delta_annual_cost'])
        missing = scope_summary(paired.iloc[:2], [1994, 1995], [901, 902])
        self.assertFalse(missing['all_required_complete'])
        self.assertIsNone(missing['mean_original_annual_cost'])

    def test_empty_failed_episode_preserves_missing_years(self):
        from reservoir_control.systems.lake_como.simulation.carryover_check import FIELDS
        frame = pd.DataFrame(columns=FIELDS)
        rows = annual_records(frame, dict(years=[1994, 1995], seed=901, arm='combined', name='failed'),
                              [80., 120.], 145900000.)
        self.assertEqual([r['observed_days'] for r in rows], [0, 0])
        self.assertTrue(all(r['annual_cost'] is None and r['end_storage_million_m3'] is None for r in rows))


if __name__ == '__main__':
    unittest.main()
