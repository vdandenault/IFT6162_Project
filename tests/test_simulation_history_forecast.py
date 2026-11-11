from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.history_forecast import HistoryForecastBank,HistoryMPC,METHODS
from reservoir_control.systems.lake_como.simulation.forecast_study import select,crps
from reservoir_control.systems.lake_como.simulation.protected_mpc import CausalForecastBank,ProtectedMPCConfig
from reservoir_control.systems.lake_como.simulation.precision_validation import make_controller
from reservoir_control.systems.lake_como.simulation.low_water import make_protected_outlet_env
from reservoir_control.systems.lake_como.simulation.carryover_check import take_step


class HistoryForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.training=settings('train')

    def test_holdout_records_do_not_affect_fit_or_paths(self):
        training=self.training
        changed=deepcopy(training['inflow'])
        for year in range(1976,1988):changed[year]=[99999.]*365
        a=HistoryForecastBank(training['inflow'],training['demand'],before_year=1976,method='history_stratified')
        b=HistoryForecastBank(changed,training['demand'],before_year=1976,method='history_stratified')
        past=np.array([100.,110.,120.,130.,150.,170.,200.])
        for bank in [a,b]:bank.set_history(past)
        x=a.forecast('conditional',120,200.,seed=901);y=b.forecast('conditional',120,200.,seed=901)
        np.testing.assert_array_equal(x[0],y[0]);self.assertEqual(a.years,tuple(range(1946,1976)))
        self.assertTrue(all(s+6<len(a.flat) for s in x[2]['source_start_indices']))

    def test_horizon_prefix_and_five_scenario_count_stay_fixed(self):
        bank=HistoryForecastBank(self.training['inflow'],self.training['demand'],before_year=1976,method='history_scaled_stratified')
        bank.set_history([80.,90.,100.,110.,120.,130.,140.])
        a=bank.forecast('conditional',100,140.,horizon=7,seed=902)
        b=bank.forecast('conditional',100,140.,horizon=1,seed=902)
        np.testing.assert_array_equal(a[0][:,:1],b[0]);self.assertEqual(a[0].shape,(5,7))
        self.assertTrue(np.isfinite(a[0]).all() and (a[0]>=0).all())
        with self.assertRaises(ValueError):bank.forecast('conditional',100,141.)

    def test_baseline_forecasts_are_exactly_unchanged(self):
        a=CausalForecastBank(self.training['inflow'],self.training['demand'],before_year=1976)
        b=HistoryForecastBank(self.training['inflow'],self.training['demand'],before_year=1976,method='baseline')
        b.set_history([100.]*7)
        for seed in [901,905]:
            x=a.forecast('conditional',240,100.,seed=seed);y=b.forecast('conditional',240,100.,seed=seed)
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])

    def test_selection_is_gated_not_just_lowest_error(self):
        summary=pd.DataFrame([
            dict(method='baseline',volume_crps=10.,wet_upper_coverage=.5,first_mae=100.),
            dict(method='history',volume_crps=9.,wet_upper_coverage=.6,first_mae=105.),
            dict(method='history_stratified',volume_crps=7.,wet_upper_coverage=.4,first_mae=95.),
            dict(method='history_scaled_stratified',volume_crps=8.,wet_upper_coverage=.7,first_mae=120.)])
        self.assertEqual(select(summary)['selected_method'],'history')
        summary.loc[summary.method.eq('history'),'volume_crps']=11.
        self.assertEqual(select(summary)['selected_method'],'baseline')
        self.assertEqual(crps([1.,1.],1.),0.);self.assertAlmostEqual(crps([0.,2.],1.),.5)

    def test_online_history_uses_only_returned_observations_and_resets(self):
        physical={k:self.training[k] for k in ['lake_params','action','flood_level','starving_level']}
        bank=HistoryForecastBank(self.training['inflow'],self.training['demand'],before_year=1988,method='baseline')
        new=HistoryMPC(physical,bank,forecast='conditional',config=ProtectedMPCConfig(seed=901),
                       ema_coefficient=self.training['exponential_average_coeff'])
        old=make_controller(self.training,bank,'combined',901)
        records=[]
        for c in [old,new]:
            env=make_protected_outlet_env('validation',(1992,1993));env.fallback=bank.initial_last_inflow
            obs,info=env.reset(seed=901);rows=[]
            for _ in range(9):
                obs,info,row,_,_=take_step(env,c,obs,info);rows.append(list(row.values()))
            records.append(rows);env.close()
        np.testing.assert_array_equal(records[0],records[1])
        self.assertEqual(len(new.observed_flows),7)
        np.testing.assert_array_equal(new.observed_flows,np.asarray(settings('validation')['inflow'][1992][1:8],np.float32))
        with self.assertRaises(ValueError):new(np.zeros(8,np.float32),dict(elapsed_hours=24*20))
        new.reset();self.assertEqual(new.observed_flows,[])
        with self.assertRaises(ValueError):new.prime_history([100.]*6,24*10)
        new.prime_history([100.]*7,24*10)
        with self.assertRaises(ValueError):new(np.zeros(8,np.float32),dict(elapsed_hours=24*10))


if __name__=='__main__':unittest.main()
