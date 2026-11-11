from copy import deepcopy
import unittest

import numpy as np

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.low_water import ProtectedOutletModel,make_protected_outlet_env
from reservoir_control.systems.lake_como.simulation.model import WaterState,PhysicalWeights,physical_cost
from reservoir_control.systems.lake_como.simulation.outlet import OutletDomainError
from reservoir_control.systems.lake_como.simulation.protected_mpc import (
    CausalForecastBank,ProtectedMPC,ProtectedMPCConfig,evaluate_schedules,
)


class ProtectedPlanningChecks(unittest.TestCase):
    def test_forecast_excludes_replay_and_future_years_and_shared_first_day(self):
        params=settings('train');bank=CausalForecastBank(params['inflow'],params['demand'],before_year=1986)
        self.assertEqual(bank.years[-1],1985)
        changed=deepcopy(params)
        for year in (1986,1987):
            changed['inflow'][year]=[9000.]*len(changed['inflow'][year])
            changed['demand'][year]=[8000.]*len(changed['demand'][year])
        other=CausalForecastBank(changed['inflow'],changed['demand'],before_year=1986)
        for mode in ('persistence','seasonal','conditional'):
            a,targets,metadata=bank.forecast(mode,200,123,horizon=7,seed=14)
            b,d,_=other.forecast(mode,200,123,horizon=7,seed=14)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(targets,d)
            short,_,_=bank.forecast(mode,200,123,horizon=1,seed=14)
            np.testing.assert_array_equal(short[:,0],a[:,0])
            self.assertLessEqual(max(metadata['training_years']),1985)
        with self.assertRaises(ValueError):CausalForecastBank(params['inflow'],params['demand'],before_year=1946)

    def test_batch_kernel_matches_scalar_model_full_horizon_and_cost(self):
        model=ProtectedOutletModel(settings('train'))
        schedules=np.array([[0,100,250],[100,200,0],[491.609985,400,100]],float)
        paths=np.array([[20,50,100],[500,600,100]],float)
        targets=np.array([50,100,150]);days=np.array([120,121,122])
        for level in [-.49,-.45,.35,1.249,1.5]:
            state=WaterState(model.lake.level_to_storage(level))
            result=evaluate_schedules(model,state,schedules,paths,targets,days)
            for c,requests in enumerate(schedules):
                for s,inflow in enumerate(paths):
                    final,rows=model.rollout(state,np.repeat(requests,24),np.repeat(inflow,24),
                        np.repeat(targets,24),np.repeat(days,24))
                    self.assertTrue(result['valid'][c,s])
                    self.assertAlmostEqual(physical_cost(rows,PhysicalWeights()),
                                           result['scenario_costs'][c,s],places=7)
                    np.testing.assert_allclose(result['daily_levels'][c,s],
                        [rows[i]['level_m'] for i in (23,47,71)],rtol=0,atol=1e-9)

    def test_predicted_domain_exit_is_infeasible_and_does_not_continue_curve(self):
        model=ProtectedOutletModel(settings('train'))
        state=WaterState(model.lake.level_to_storage(2.63))
        result=evaluate_schedules(model,state,[[0],[400]],[[3000]],[100],[200])
        self.assertFalse(result['valid'].any());self.assertTrue(np.isinf(result['scenario_costs']).all())
        self.assertTrue((result['daily_levels']<=2.64).all())
        with self.assertRaises(OutletDomainError):model.rollout(state,[0]*24,[3000]*24,[100]*24,[200]*24)

    def test_first_day_parity_and_horizon_crosses_year_without_reset(self):
        params=settings('train');bank=CausalForecastBank(params['inflow'],params['demand'],before_year=1986)
        env=make_protected_outlet_env(years=(1986,1987));obs,info=env.reset(seed=1)
        info.update(elapsed_hours=364*24,remaining_hours=366*24,next_day=365)
        controller=ProtectedMPC(params,bank,forecast='conditional')
        action=controller(obs,info)
        self.assertTrue(env.action_space.contains(action))
        diag=controller.diagnostics[-1]
        self.assertLess(diag['first_day_parity_error_m'],1e-8)
        info.update(elapsed_hours=365*24,remaining_hours=365*24,next_day=1)
        controller(obs,info)
        self.assertTrue(controller.diagnostics[-1]['shifted_previous_plan'])
        self.assertEqual(controller.diagnostics[-1]['horizon_days'],7)
        env.close()

    def test_no_feasible_schedule_uses_labeled_uncertified_recovery(self):
        class ExtremeBank:
            def forecast(self,*args,horizon,**kwargs):
                return np.full((1,horizon),3000.),np.full(horizon,100.),{}
        params=settings('train');env=make_protected_outlet_env(years=(1986,));obs,info=env.reset()
        obs[0]=2.63
        controller=ProtectedMPC(params,ExtremeBank())
        action=controller(obs,info)
        self.assertTrue(env.action_space.contains(action))
        self.assertTrue(controller.diagnostics[-1]['domain_recovery'])
        self.assertIsNone(controller.diagnostics[-1]['expected_cost'])
        self.assertEqual(controller.diagnostics[-1]['feasible_candidates'],0)
        env.close()


if __name__=='__main__':unittest.main()
