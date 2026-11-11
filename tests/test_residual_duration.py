import unittest
import pandas as pd
from reservoir_control.systems.lake_como.residual_duration import extended_config,check_prefix
from reservoir_control.systems.lake_como.residual_study import CONFIG

class DurationChecks(unittest.TestCase):
    def test_only_steps_change_without_mutating_original(self):
        result=extended_config(CONFIG)
        self.assertEqual(CONFIG['steps'],3650)
        self.assertEqual(result['steps'],36500)
        self.assertEqual(result['replay_capacity'],3650)
        self.assertEqual([k for k in CONFIG if CONFIG[k]!=result[k]],['steps'])
        result['seeds'].append(999)
        self.assertEqual(CONFIG['seeds'],[123,124,125])
    def test_prefix_is_exact_not_merely_similar(self):
        short=pd.DataFrame(dict(step=[365,730],full_return=[-10.,-20.]))
        long=pd.DataFrame(dict(step=[365,730,1095],full_return=[-10.,-20.,-30.]))
        check_prefix(short,long)
        long.loc[0,'full_return']=-10.000001
        with self.assertRaises(AssertionError): check_prefix(short,long)
        with self.assertRaises(ValueError): check_prefix(short,short.iloc[:1])

if __name__=='__main__': unittest.main()
