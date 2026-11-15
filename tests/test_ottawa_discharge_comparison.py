"""Synthetic comparison fixtures only, not replacement historical observations."""
import unittest
import numpy as np
from reservoir_control.systems.ottawa.discharge_comparison import align, statistics


class DischargeComparisonChecks(unittest.TestCase):
    def payload(self, station, values, flags):
        return {'features': [{'properties': dict(STATION_NUMBER=station,
            DATE=f'1970-01-{day:02d}', DISCHARGE=q, DISCHARGE_SYMBOL_EN=flag,
            DISCHARGE_SYMBOL_FR=None)} for day, (q, flag) in enumerate(zip(values, flags), 1)]}

    def fixture(self):
        return {'02LG007': self.payload('02LG007', [10., 20., None], [None, None, 'Estimated']),
                '02LG006': self.payload('02LG006', [7., 20., 5.], [None, 'Partial Day', None])}

    def test_signed_difference_flags_and_missing_values_preserved(self):
        rows = align(self.fixture())
        self.assertEqual(len(rows), 365)
        self.assertEqual(rows.q_02LG007_m3_s.iloc[0], 10.)
        self.assertEqual(rows.q_02LG006_m3_s.iloc[0], 7.)
        self.assertEqual(rows.difference_007_minus_006_m3_s.iloc[0], 3.)
        self.assertEqual(rows.difference_007_minus_006_m3_s.iloc[1], 0.)
        self.assertTrue(np.isnan(rows.difference_007_minus_006_m3_s.iloc[2]))
        self.assertEqual(rows.symbol_02LG006_en.iloc[1], 'Partial Day')
        self.assertTrue(rows.any_source_flag.iloc[1])
        self.assertTrue(np.isnan(rows.q_02LG006_m3_s.iloc[3]))

    def test_flagged_and_unflagged_statistics_do_not_hide_disagreement(self):
        rows = align(self.fixture())
        all_data = statistics(rows, 'all')
        self.assertEqual(all_data['paired_dates'], 2)
        self.assertEqual(all_data['missing_pairs'], 363)
        self.assertEqual(all_data['mean_signed_difference_m3_s'], 1.5)
        self.assertEqual(all_data['equal_dates'], 1)
        flagged = statistics(rows[rows.any_source_flag], 'flagged')
        self.assertEqual(flagged['paired_dates'], 1)
        self.assertEqual(flagged['missing_pairs'], 1)
        self.assertEqual(flagged['mean_absolute_difference_m3_s'], 0.)
        self.assertEqual(statistics(rows[~rows.any_source_flag], 'unflagged')['max_absolute_difference_m3_s'], 3.)

    def test_empty_subset_and_duplicate_source_dates(self):
        rows = align(self.fixture())
        empty = statistics(rows.iloc[:0], 'empty')
        self.assertEqual(empty['paired_dates'], 0)
        self.assertNotIn('mean_absolute_difference_m3_s', empty)
        payloads = self.fixture()
        payloads['02LG007']['features'].append(payloads['02LG007']['features'][0])
        with self.assertRaises(ValueError): align(payloads)


if __name__ == '__main__': unittest.main()
