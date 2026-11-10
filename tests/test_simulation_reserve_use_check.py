import unittest

import numpy as np
import pandas as pd

from lake_como.environment import settings
from lake_como.simulation.carryover_check import BoundaryHistory, restore_branch, restore_memory
from lake_como.simulation.protected_mpc import CausalForecastBank
from lake_como.simulation.reserve_use_check import PerfectTargetRecord, make_controller, inventory_trace
from lake_como.simulation.reserve_tie_audit import service_rounding
from lake_como.simulation.low_water import ProtectedOutletModel
from lake_como.simulation.model import WaterState
from lake_como.simulation.protected_mpc import evaluate_schedules


class ReserveDiagnosticChecks(unittest.TestCase):
    def test_privileged_record_requires_original_1995_clock(self):
        record=PerfectTargetRecord(np.arange(365.))
        np.testing.assert_array_equal(record.horizon(365,7),np.arange(7.))
        np.testing.assert_array_equal(record.horizon(729,1),[364.])
        for clock,length in [(0,7),(364,7),(730,1),(729,2),(365,8)]:
            with self.assertRaises(ValueError):record.horizon(clock,length)
        with self.assertRaises(ValueError):PerfectTargetRecord(np.zeros(365),year=1998)
        with self.assertRaises(ValueError):record.values[0]=1.

    def test_target_intervention_preserves_inflow_forecast_and_realized_forcing(self):
        training=settings('train')
        bank=CausalForecastBank(training['inflow'],training['demand'],before_year=1988)
        history=BoundaryHistory(8760,100.,120.,100.,8736,tuple([150.]*7))
        env,obs,info=restore_branch(bank,history,80_000_000.)
        forcing={year:values.copy() for year,values in env.forcing.items()}
        record=PerfectTargetRecord(np.arange(365.)+20.)
        causal=make_controller(training,bank,'causal',None)
        oracle=make_controller(training,bank,'perfect_target',record)
        for controller in [causal,oracle]:
            restore_memory(controller,history);controller(obs,info)
        a=causal.diagnostics[-1];b=oracle.diagnostics[-1]
        np.testing.assert_array_equal(a['forecast_inflows'],b['forecast_inflows'])
        np.testing.assert_array_equal(a['forecast_target'],b['forecast_metadata']['causal_target_forecast'])
        np.testing.assert_array_equal(b['forecast_target'],np.arange(7.)+20.)
        self.assertTrue(b['oracle_targets'])
        self.assertEqual(a['search_mode'],b['search_mode'])
        self.assertEqual(causal.weights,oracle.weights)
        for year,values in forcing.items():np.testing.assert_array_equal(values,env.forcing[year])
        env.close()

    def test_inventory_keeps_negative_daily_deltas(self):
        # Same inflow: high starts with 30 units more, releases 10 more,
        # then releases 5 less. Signed net release is 5, not 15.
        low=pd.DataFrame(dict(elapsed_hour=[8760,8784],model_day=[1,2],
            release_m3_s=np.array([10.,10.])/86400,delivered_target_m3=[5.,5.],
            remaining_river_m3=[5.,5.],level_before_m=[100.,90.],level_m=[90.,80.]))
        high=pd.DataFrame(dict(elapsed_hour=[8760,8784],model_day=[1,2],
            release_m3_s=np.array([20.,5.])/86400,delivered_target_m3=[7.,3.],
            remaining_river_m3=[13.,2.],level_before_m=[130.,110.],level_m=[110.,105.]))
        result=inventory_trace(low,high,1.)
        np.testing.assert_allclose(result.extra_remaining_river_m3,[8.,-3.])
        self.assertAlmostEqual(result.cumulative_extra_release_m3.iloc[-1],5.)
        self.assertAlmostEqual(result.extra_storage_after_m3.iloc[-1],25.)

    def test_rounding_diagnostic_removes_undershoot_without_cost_change(self):
        model=ProtectedOutletModel(settings('train'))
        days=np.arange(1,8);targets=np.repeat(74.7,7)
        nearest,upward=service_rounding(model,targets,days)
        self.assertTrue((upward.astype(float)>=targets+model.lake.min_env_flow[days-1]).all())
        self.assertLess(float(np.max(upward-nearest)),1e-4)
        state=WaterState(model.lake.level_to_storage(.08),79.,0.)
        plans=np.array([np.repeat(np.float32(122.902496),7),upward])
        result=evaluate_schedules(model,state,plans,np.repeat(60.,7)[None,:],targets,days)
        np.testing.assert_array_equal(result['scenario_costs'],0.)
        self.assertGreater(result['daily_levels'][1,0,-1],result['daily_levels'][0,0,-1])


if __name__=='__main__':unittest.main()
