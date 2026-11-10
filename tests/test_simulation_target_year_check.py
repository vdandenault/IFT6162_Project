import unittest

import numpy as np
import pandas as pd

from lake_como.simulation.target_year_check import (
    validation_pairs,annual_metrics,paired_scope,DAILY_FIELDS,
)


def record(year,days,cost=1.):
    return pd.DataFrame([dict(year=year,model_day=d,reward=-cost,
        level_before_m=.35,minimum_level_m=.2,maximum_level_m=.5,inflow_m3_s=100.,
        unmet_target_m3=cost*1e6,environmental_shortage_m3=0.,
        flood_excess_m_hours=0.,low_water_excess_m_hours=0.,balance_error_m3=0.)
        for d in range(1,days+1)])


class BroaderYearChecks(unittest.TestCase):
    def test_complete_validation_only_no_test_years_or_cherry_picked_subset(self):
        self.assertEqual(validation_pairs(),[(1988,1989),(1990,1991),(1992,1993),(1994,1995),(1996,1997)])
        for years in [(1998,1999),tuple(range(1990,1998)),tuple(range(1988,2000))]:
            with self.assertRaises(ValueError):validation_pairs(years)

    def test_partial_year_cannot_look_like_cheap_full_year(self):
        frame=pd.concat([record(1990,365),record(1991,10)],ignore_index=True)
        rows=annual_metrics(frame,(1990,1991),'candidate',(80,120))
        self.assertTrue(rows[0]['complete']);self.assertEqual(rows[0]['annual_cost'],365)
        self.assertFalse(rows[1]['complete']);self.assertIsNone(rows[1]['annual_cost'])
        self.assertEqual(rows[1]['observed_prefix_cost'],10)
        baseline=annual_metrics(pd.concat([record(1990,365,2),record(1991,365,2)]),
                                (1990,1991),'baseline',(80,120))
        comparison=paired_scope(pd.DataFrame(rows+baseline),'baseline','candidate',(1990,1991))
        self.assertEqual(comparison['complete_pairs'],1)
        self.assertFalse(comparison['all_pairs_complete'])
        self.assertIsNone(comparison['mean_annual_cost_delta'])
        self.assertEqual(comparison['complete_blocks'],0)

    def test_missing_second_year_stays_missing_and_duplicate_days_rejected(self):
        rows=annual_metrics(record(1990,365),(1990,1991),'candidate',(80,120))
        self.assertEqual(rows[1]['observed_days'],0)
        self.assertIsNone(rows[1]['annual_cost']);self.assertIsNone(rows[1]['observed_prefix_cost'])
        empty=annual_metrics(pd.DataFrame(columns=DAILY_FIELDS),(1990,1991),'candidate',(80,120))
        self.assertTrue(all(row['annual_cost'] is None for row in empty))
        with self.assertRaises(ValueError):annual_metrics(pd.concat([record(1990,365),record(1990,1)]),
                                                         (1990,1991),'candidate',(80,120))


if __name__=='__main__':unittest.main()
