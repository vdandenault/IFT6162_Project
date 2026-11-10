import unittest
import numpy as np
import pandas as pd
from lake_como.simulation.weighted_validation import specs,pair_annual,scope_summary,METRICS
from lake_como.simulation.precision_validation import annual_records
from lake_como.simulation.carryover_check import FIELDS


def fixture():
    rows=[]
    for seed in [901,902]:
        for year in [1994,1995]:
            for arm in ['legacy','weighted']:
                cost=100.+(10. if seed==902 else -20.)*(arm=='weighted')
                rows.append(dict(seed=seed,block_start=1994,year=year,position_in_pair=year-1993,
                    arm=arm,regime='wet' if year==1994 else 'dry',complete=True,
                    **{k:cost if k=='annual_cost' else 0. for k in METRICS}))
    return pd.DataFrame(rows)


class WeightedValidationChecks(unittest.TestCase):
    def test_declared_grid_uses_all_validation_pairs_and_no_test_years(self):
        cases=specs('mix20_tail05')
        self.assertEqual(len(cases),50);self.assertEqual(len({r['name'] for r in cases}),50)
        self.assertEqual(set(y for r in cases for y in r['years']),set(range(1988,1998)))
        self.assertEqual(set(r['seed'] for r in cases),{901,902,903,904,905})
        self.assertTrue(all(r['variant']==('legacy' if r['arm']=='legacy' else 'mix20_tail05') for r in cases))
        with self.assertRaises(ValueError):specs('unfrozen')

    def test_pairing_is_by_seed_and_year_without_cherry_picking(self):
        paired=pair_annual(fixture().sample(frac=1,random_state=5))
        result=scope_summary(paired,[1994,1995],[901,902])
        self.assertTrue(result['all_required_complete']);self.assertEqual(result['mean_delta_annual_cost'],-5.)
        self.assertEqual((result['cost_wins'],result['cost_losses']),(2,2))
        self.assertEqual(scope_summary(paired,[1995],[901,902])['mean_delta_annual_cost'],-5.)
        self.assertEqual(set(paired.regime),{'wet','dry'})

    def test_missing_or_incomplete_pair_suppresses_scope_average(self):
        annual=fixture()
        annual.loc[annual.arm.eq('weighted')&annual.seed.eq(902)&annual.year.eq(1995),['complete','annual_cost']]=[False,np.nan]
        paired=pair_annual(annual);summary=scope_summary(paired,[1994,1995],[901,902])
        self.assertFalse(summary['all_required_complete']);self.assertEqual(summary['complete_pairs'],3)
        self.assertIsNone(summary['mean_delta_annual_cost'])
        missing=scope_summary(paired.iloc[:2],[1994,1995],[901,902])
        self.assertFalse(missing['all_required_complete']);self.assertIsNone(missing['mean_weighted_annual_cost'])
        self.assertFalse(scope_summary(paired,[],[901,902])['all_required_complete'])

    def test_absent_years_do_not_receive_zero_cost_or_stock(self):
        records=annual_records(pd.DataFrame(columns=FIELDS),dict(years=[1994,1995],arm='weighted',name='failed',seed=901),
                              [80.,120.],145900000.)
        self.assertEqual([r['observed_days'] for r in records],[0,0])
        self.assertTrue(all(r['annual_cost'] is None and r['end_storage_million_m3'] is None for r in records))


if __name__=='__main__':unittest.main()
