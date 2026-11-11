import unittest
import numpy as np
import pandas as pd
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.weighted_forecast import (
    WeightedForecastBank,mixture_distribution,probability_strata,weighted_crps,expected_costs,CONFIGS,
)
from reservoir_control.systems.lake_como.simulation.weighted_study import make_controller,choose
from reservoir_control.systems.lake_como.simulation.forecast_study import controller as old_controller
from reservoir_control.systems.lake_como.simulation.tail_diagnosis import pool_details
from reservoir_control.systems.lake_como.simulation.low_water import make_protected_outlet_env
from reservoir_control.systems.lake_como.simulation.carryover_check import take_step
from reservoir_control.systems.lake_como.simulation.flood_diagnosis import restore


class WeightedForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.training=settings('train')

    def test_mixture_preserves_seasonal_mass_and_zero_control(self):
        b=WeightedForecastBank(self.training['inflow'],self.training['demand'],before_year=1976,variant='mix10_tail01')
        past=np.array([80.,90.,110.,130.,150.,170.,200.])
        conditioned=pool_details(b,120,past)['starts']
        paths,p,starts=mixture_distribution(b,120,past,.1)
        self.assertGreater(len(p),128);self.assertTrue((p>0).all());self.assertAlmostEqual(p.sum(),1.)
        outside=~np.isin(starts,conditioned)
        self.assertAlmostEqual(p[outside].sum(),.1*outside.sum()/len(p))
        _,zero,zero_starts=mixture_distribution(b,120,past,0.)
        self.assertEqual(len(zero),128);self.assertEqual(set(zero_starts),set(conditioned))

    def test_tail_mass_is_not_twenty_percent_and_atoms_can_span_bands(self):
        x=np.ones((1,7))*200.
        paths,p,starts=probability_strata(x,[1.],[42],.01,901)
        self.assertEqual(paths.shape,(5,7));self.assertAlmostEqual(p[-1],.01)
        self.assertAlmostEqual(p.sum(),1.);np.testing.assert_array_equal(starts,np.repeat(42,5))
        self.assertAlmostEqual(paths[:,0]@p,200.)
        self.assertAlmostEqual(weighted_crps([0.,2.],1.,[.5,.5]),.5)

    def test_weighted_selection_changes_ranking_but_never_hides_invalid_paths(self):
        matrix=np.array([[0.,100.],[50.,0.]])
        np.testing.assert_allclose(expected_costs(matrix,[.99,.01]),[1.,49.5])
        self.assertEqual(np.argmin(expected_costs(matrix,[.5,.5])),1)
        self.assertEqual(np.argmin(expected_costs(matrix,[.99,.01])),0)
        self.assertTrue(np.isinf(expected_costs([[0.,np.inf]],[.99,.01])[0]))
        with self.assertRaises(ValueError):expected_costs(matrix,[1.,0.])
        with self.assertRaises(ValueError):expected_costs(matrix,[.2,.2])

    def test_horizon_and_observed_history_contract(self):
        b=WeightedForecastBank(self.training['inflow'],self.training['demand'],before_year=1976,variant='mix05_tail05')
        b.set_history([80.,90.,110.,130.,150.,170.,200.])
        a=b.forecast('conditional',120,200.,horizon=7,seed=901)
        c=b.forecast('conditional',120,200.,horizon=1,seed=901)
        np.testing.assert_array_equal(a[0][:,:1],c[0]);self.assertAlmostEqual(a[2]['scenario_probabilities'][-1],.05)
        self.assertEqual(a[2]['training_years'],list(range(1946,1976)))
        with self.assertRaises(ValueError):b.forecast('conditional',120,201.)
        with self.assertRaises(ValueError):b.forecast('conditional',120,200.,count=6)

    def test_legacy_policy_closed_loop_matches_frozen_history_mpc(self):
        old,old_bank=old_controller(self.training,'history_scaled_stratified',901)
        new,new_bank=make_controller(self.training,'legacy',901)
        trajectories=[]
        for c,bank in [(old,old_bank),(new,new_bank)]:
            env=make_protected_outlet_env('validation',(1992,1993));env.fallback=bank.initial_last_inflow
            obs,info=env.reset(seed=901);rows=[]
            for _ in range(10):
                obs,info,row,_,_=take_step(env,c,obs,info);rows.append(list(row.values()))
            trajectories.append(rows);env.close()
        np.testing.assert_array_equal(trajectories[0],trajectories[1])

    def test_uniform_ablation_keeps_paths_and_reward_weights_identical(self):
        history=dict(hour=(365+253)*24,target_ema=120.,last_inflow=200.,
                     previous_hour=(365+252)*24,previous_plan=[150.]*7,
                     state=dict(storage=100000000.,command=100.,river_storage=0.))
        controls=[make_controller(self.training,'mix10_tail01',901,mode)[0] for mode in ['weighted','uniform_ablation']]
        for c in controls:
            env,obs,info=restore(1993,c.history_bank,history,history)
            c.previous_hour=history['previous_hour'];c.previous_plan=np.array(history['previous_plan'],np.float32)
            c.prime_history([80.,90.,110.,130.,150.,170.,200.],history['hour'])
            action=c(obs,info);self.assertTrue(env.action_space.contains(action));env.close()
        a,b=[c.diagnostics[-1] for c in controls]
        np.testing.assert_array_equal(a['forecast_inflows'],b['forecast_inflows'])
        np.testing.assert_array_equal(a['forecast_target'],b['forecast_target'])
        self.assertAlmostEqual(a['applied_probabilities'][-1],.01)
        self.assertEqual(b['applied_probabilities'],[.2]*5)
        self.assertEqual(controls[0].weights,controls[1].weights)

    def test_selection_can_retain_legacy_or_choose_zero_mixture_ablation(self):
        rows=[dict(variant=name,volume_crps=30.,wet_upper_coverage=.8,first_mae=100.) for name in CONFIGS]
        f=pd.DataFrame(rows);f.loc[f.variant.eq('legacy'),'volume_crps']=20.
        self.assertEqual(choose(f)['selected_variant'],'legacy')
        f.loc[f.variant.eq('mix00_tail01'),'volume_crps']=19.
        self.assertEqual(choose(f)['selected_variant'],'mix00_tail01')
        f.loc[f.variant.eq('mix00_tail01'),'wet_upper_coverage']=.7
        self.assertEqual(choose(f)['selected_variant'],'legacy')


if __name__=='__main__':unittest.main()
