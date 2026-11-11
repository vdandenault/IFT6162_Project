import unittest
import numpy as np
from reservoir_control.systems.lake_como.calendar_audit import describe_year


class CalendarAuditChecks(unittest.TestCase):
    def test_leap_last_sample_is_unused_not_february_29(self):
        info = describe_year(1948, np.arange(366.))
        self.assertEqual(info['unused_source_columns'], [365])
        self.assertEqual(info['assumed_last_consumed_date'], '1948-12-30')
        self.assertEqual(info['assumed_omitted_date'], '1948-12-31')
        self.assertIn('If', info['date_assumption'])

    def test_ordinary_year_has_no_unused_valid_sample(self):
        info = describe_year(1947, np.r_[np.arange(365.), np.nan])
        self.assertEqual(info['source_samples'], 365)
        self.assertEqual(info['unused_source_columns'], [])
        self.assertEqual(info['assumed_last_consumed_date'], '1947-12-31')

    def test_reject_internal_gaps_instead_of_shifting_data(self):
        values = np.arange(366.); values[30] = np.nan
        with self.assertRaises(ValueError):
            describe_year(1948, values)
        with self.assertRaises(ValueError):
            describe_year(1948, np.arange(365.))


if __name__ == '__main__':
    unittest.main()
