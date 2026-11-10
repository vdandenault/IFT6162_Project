from dataclasses import replace
import unittest

import numpy as np

from lake_como.environment import settings
from lake_como.simulation.carryover_check import (
    BoundaryHistory,restore_branch,restore_memory,make_controller,decompose,
)
from lake_como.simulation.protected_mpc import CausalForecastBank
from lake_como.simulation.low_water import ProtectedOutletModel


class CarryoverExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.training=settings('train')
        self.bank=CausalForecastBank(self.training['inflow'],self.training['demand'],before_year=1988)
        self.model=ProtectedOutletModel(self.training)
        self.history=BoundaryHistory(8760,100.,120.,100.,8736,tuple([150.]*7))

    def test_swap_changes_only_level_and_preserves_second_year_clock(self):
        low,lo,li=restore_branch(self.bank,self.history,self.model.lake.level_to_storage(-.463))
        high,hi,hinfo=restore_branch(self.bank,self.history,self.model.lake.level_to_storage(.082))
        np.testing.assert_array_equal(lo[1:],hi[1:])
        self.assertNotEqual(lo[0],hi[0]);self.assertEqual(lo[-1],.5)
        self.assertEqual(li['elapsed_hours'],8760);self.assertEqual(li['remaining_hours'],8760)
        self.assertEqual(li['next_year'],1995);self.assertEqual(li['next_day'],1)
        self.assertEqual(low.demand_ema,self.history.target_ema)
        np.testing.assert_array_equal(low.forcing[1995],high.forcing[1995])
        low.close();high.close()

    def test_warm_start_control_changes_memory_not_forecast(self):
        env,obs,info=restore_branch(self.bank,self.history,self.model.lake.level_to_storage(-.463))
        warm=make_controller(self.training,self.bank);cold=make_controller(self.training,self.bank)
        restore_memory(warm,self.history);restore_memory(cold,self.history,cold=True)
        warm(obs,info);cold(obs,info)
        a=warm.diagnostics[-1];b=cold.diagnostics[-1]
        self.assertTrue(a['shifted_previous_plan']);self.assertFalse(b['shifted_previous_plan'])
        np.testing.assert_array_equal(a['forecast_inflows'],b['forecast_inflows'])
        np.testing.assert_array_equal(a['forecast_target'],b['forecast_target'])
        env.close()

    def test_original_clock_and_domain_are_required(self):
        for bad in [replace(self.history,hour=0),replace(self.history,previous_hour=None),
                    replace(self.history,previous_plan=tuple([float('nan')]*7))]:
            with self.assertRaises(ValueError):restore_branch(self.bank,bad,0.)
        with self.assertRaises(ValueError):
            restore_branch(self.bank,self.history,self.model.lake.level_to_storage(2.7))

    def test_interaction_does_not_get_hidden_by_unique_percentage_claim(self):
        result=decompose(dict(mpc_low=100.,mpc_high=70.,feedback_low=95.,feedback_high=60.))
        self.assertEqual(result['observed_gap'],40)
        self.assertEqual(result['storage_benefit_mpc'],30)
        self.assertEqual(result['storage_benefit_feedback'],35)
        self.assertEqual(result['interaction'],-5)
        self.assertEqual(result['symmetric_storage_component']+result['symmetric_controller_component'],40)
        with self.assertRaises(ValueError):decompose(dict(mpc_low=100.))


if __name__=='__main__':unittest.main()
