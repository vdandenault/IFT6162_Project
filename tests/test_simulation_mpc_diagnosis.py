from copy import deepcopy
import unittest

import numpy as np

from lake_como.environment import settings
from lake_como.simulation.low_water import make_protected_outlet_env
from lake_como.simulation.mpc_diagnosis import TargetForecaster,DiagnosticMPC,SearchExpansion
from lake_como.simulation.protected_mpc import CausalForecastBank,ProtectedMPC


class MPCDiagnosisChecks(unittest.TestCase):
    def setUp(self):
        self.params=settings('train')
        self.bank=CausalForecastBank(self.params['inflow'],self.params['demand'],before_year=1986)

    def test_restricted_seasonal_is_exact_frozen_controller(self):
        env=make_protected_outlet_env(years=(1986,1987));obs,info=env.reset()
        original=ProtectedMPC(self.params,self.bank,forecast='conditional')
        diagnosed=DiagnosticMPC(self.params,self.bank,forecast='conditional')
        for _ in range(12):
            a=original(obs,info);b=diagnosed(obs,info)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(original.diagnostics[-1]['expected_cost'],diagnosed.diagnostics[-1]['expected_cost'])
            obs,_,_,_,info=env.step(a)
        env.close()

    def test_fit_excludes_future_and_reset_has_no_artificial_target_drop(self):
        fitted=TargetForecaster(self.bank,.8)
        changed=deepcopy(self.params)
        for year in (1986,1987):changed['demand'][year]=[9000.]*len(changed['demand'][year])
        other=TargetForecaster(CausalForecastBank(changed['inflow'],changed['demand'],before_year=1986),.8)
        np.testing.assert_array_equal(fitted.slopes,other.slopes)
        forecast,residual=fitted.predict(1,0.,0)
        np.testing.assert_array_equal(forecast,self.bank.target_mean[:7]);self.assertEqual(residual,0)
        # After one observed day, account for the 0.2 mass of zero-initialized EMA.
        forecast,residual=fitted.predict(2,.2*(self.bank.target_mean[0]+100),1)
        self.assertAlmostEqual(residual,100)
        np.testing.assert_allclose(forecast,np.maximum(0,self.bank.target_mean[1:8]+fitted.slopes*100))

    def test_expansion_retains_incumbent_and_does_not_change_inflow_forecast(self):
        env=make_protected_outlet_env(years=(1986,));obs,info=env.reset()
        obs[0]=.9;obs[3]=120;info.update(elapsed_hours=100*24,remaining_hours=265*24)
        for target_mode in ('seasonal','ema_residual'):
            a=DiagnosticMPC(self.params,self.bank,forecast='conditional',target_mode=target_mode)
            b=DiagnosticMPC(self.params,self.bank,forecast='conditional',target_mode=target_mode,search='expanded')
            a(obs,info);action=b(obs,info)
            self.assertTrue(env.action_space.contains(action))
            da=a.diagnostics[-1];db=b.diagnostics[-1]
            np.testing.assert_array_equal(da['forecast_inflows'],db['forecast_inflows'])
            np.testing.assert_array_equal(da['forecast_target'],db['forecast_target'])
            self.assertEqual(da['expected_cost'],db['restricted_seed_cost'])
            self.assertLessEqual(db['expected_cost'],da['expected_cost']+1e-8)
            self.assertGreater(db['expanded_candidate_evaluations'],0)
            self.assertLess(db['first_day_parity_error_m'],1e-8)
        env.close()

    def test_target_forecast_uses_only_supplied_past_ema_and_clock(self):
        forecaster=TargetForecaster(self.bank,.8)
        baseline,_=forecaster.predict(101,100.,100)
        changed,_=forecaster.predict(101,200.,100)
        self.assertFalse(np.array_equal(baseline,changed))
        with self.assertRaises(ValueError):forecaster.predict(100,100.,100)
        with self.assertRaises(ValueError):SearchExpansion(sweeps=0)


if __name__=='__main__':unittest.main()
