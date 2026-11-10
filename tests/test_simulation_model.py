import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from lake_como.environment import settings,make_env
from lake_como.simulation import make_operational_env,Operations,PhysicalWeights
from lake_como.simulation.model import OperationalModel,WaterState

class OperationalPhysicsChecks(unittest.TestCase):
    def test_full_year_benchmark_parity(self):
        new=make_operational_env(years=(1946,),objective='benchmark');old=make_env('train',1946)
        new.reset();old.reset()
        for day in range(365):
            action=np.array([20+(day*37)%400],np.float32)
            _,reward,term,trunc,info=new.step(action);_,expected,_,legacy_end,legacy=old.step(action)
            self.assertEqual(info['storage'],legacy['storage'])
            self.assertAlmostEqual(reward,expected,places=9)
            self.assertEqual(term or trunc,legacy_end)
        self.assertTrue(term);self.assertFalse(trunc)
        new.close();old.close()

    def test_subdaily_additivity_and_physical_state_match(self):
        ops=Operations(ramp_m3_s_per_hour=5,river_residence_hours=6)
        daily=make_operational_env(years=(1946,),operations=ops)
        sub=make_operational_env(years=(1946,),decision_hours=6,operations=ops)
        daily.reset();sub.reset();a=np.array([100],np.float32)
        for _ in range(3):
            obs,reward,*_=daily.step(a);total=0
            for _ in range(4):last,r,*_=sub.step(a);total+=r
            self.assertAlmostEqual(reward,total,places=12)
            self.assertEqual(daily.state,sub.state);np.testing.assert_array_equal(obs,last)
        daily.close();sub.close()

    def test_routing_conserves_water_and_environmental_flow_not_consumed_twice(self):
        params=settings('train');model=OperationalModel(params,Operations(river_residence_hours=6))
        state=WaterState(model.lake.level_to_storage(.35))
        for _ in range(48):
            following,row=model.advance(state,100,100,40,1)
            self.assertLess(abs(row['river_balance_error_m3']),1e-6)
            self.assertAlmostEqual(state.river_storage+row['release_m3_s']*3600,
                following.river_storage+row['diversion_m3']+row['river_out_m3'],places=6)
            self.assertGreaterEqual(row['river_out_m3'],0)
            state=following
        self.assertAlmostEqual(row['diversion_m3']/3600,40,places=8)

    def test_ramp_limits_command_but_not_forced_release(self):
        model=OperationalModel(settings('train'),Operations(ramp_m3_s_per_hour=1))
        state=WaterState(model.lake.level_to_storage(2.))
        following,row=model.advance(state,100,100,100,1)
        self.assertEqual(following.command,1);self.assertGreater(row['release_m3_s'],1)

    def test_future_input_causality_and_evaporation_units(self):
        a=make_operational_env(years=(1946,));b=make_operational_env(years=(1946,))
        b.forcing[1946][1]+=100
        np.testing.assert_array_equal(a.reset()[0],b.reset()[0])
        np.testing.assert_array_equal(a.step(np.array([100],np.float32))[0],b.step(np.array([100],np.float32))[0])
        e=make_operational_env(years=(1946,),evaporation_mm_day={1946:np.ones(365)})
        e.reset();rows=e.step(np.array([100],np.float32))[-1]['hourly']
        self.assertAlmostEqual(sum(r['evaporation_m3'] for r in rows),145900.)
        for r in rows:self.assertLess(abs(r['lake_balance_error_m3']),1e-6)

    def test_continuing_record_truncates_without_time_to_go(self):
        env=make_operational_env(years=(1946,),terminal_mode='continuing_record');obs,_=env.reset()
        self.assertEqual(obs[-1],1)
        for _ in range(365):obs,_,term,trunc,info=env.step(np.array([100],np.float32))
        self.assertFalse(term);self.assertTrue(trunc);self.assertEqual(obs[-1],1)
        self.assertEqual(obs[0],np.float32(env.model.lake.storage_to_level(env.state.storage)))
        with self.assertRaises(RuntimeError):env.step(np.array([100],np.float32))

    def test_gym_contract_and_invalid_settings(self):
        check_env(make_operational_env(years=(1946,),decision_hours=6),skip_render_check=True)
        for kwargs in [dict(decision_hours=5),dict(decision_hours=6,objective='benchmark'),dict(initial_level_range=(-1,.5))]:
            with self.assertRaises(ValueError):make_operational_env(years=(1946,),**kwargs)
        with self.assertRaises(ValueError):make_operational_env('train',(1987,1988))
        with self.assertRaises(ValueError):Operations(ramp_m3_s_per_hour=-1)
