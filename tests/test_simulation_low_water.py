import unittest
from dataclasses import replace

import numpy as np
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.model import WaterState
from reservoir_control.systems.lake_como.simulation.outlet import OutletSpec, make_outlet_env
from reservoir_control.systems.lake_como.simulation.low_water import (
    LowWaterRule, ProtectedOutletLake, ProtectedOutletModel, make_protected_outlet_env,
)


class LowWaterChecks(unittest.TestCase):
    def test_operating_rule_does_not_change_hydraulic_capacity(self):
        params=settings('train')['lake_params']
        lake=ProtectedOutletLake(params,OutletSpec(),LowWaterRule())
        storage=lake.level_to_storage(-.5)
        self.assertGreater(lake.max_release(storage,1),100)
        self.assertEqual(lake.actual_release(100,storage,1),0)
        self.assertGreater(LowWaterRule().release_cap(-.49,lake.surface),0)

    def test_floor_invariant_and_conservation_under_nonnegative_inflow(self):
        model=ProtectedOutletModel(settings('train'))
        for h in [-.5,-.4999,-.49,-.45,0.,1.2]:
            for inflow in [0.,1.,100.]:
                state=WaterState(model.lake.level_to_storage(h))
                for _ in range(48):
                    state,row=model.advance(state,model.high,inflow,100,1)
                    self.assertGreaterEqual(row['level_m'],-.5-1e-12)
                    self.assertLess(abs(row['lake_balance_error_m3']),1e-6)
                    self.assertLessEqual(row['release_m3_s'],row['outlet_capacity_m3_s'])
                    self.assertLessEqual(row['release_m3_s'],row['low_water_release_cap_m3_s'])

    def test_priority_and_environmental_shortfall_are_reported(self):
        model=ProtectedOutletModel(settings('train'))
        state=WaterState(model.lake.level_to_storage(-.49999))
        _,row=model.advance(state,100,0,100,1)
        self.assertTrue(row['low_water_intervened'])
        self.assertGreater(row['operating_minimum_shortfall_m3'],0)
        self.assertGreater(row['environmental_shortage_m3'],0)
        self.assertGreater(row['unmet_demand_m3'],0)

    def test_no_future_inflow_in_release_and_no_negative_storage_clamp(self):
        model=ProtectedOutletModel(settings('train'))
        state=WaterState(model.lake.level_to_storage(-.6))
        outcomes=[model.advance(state,100,inflow,100,1) for inflow in [0.,100.]]
        self.assertEqual(outcomes[0][1]['release_m3_s'],outcomes[1][1]['release_m3_s'])
        self.assertEqual(outcomes[0][1]['release_m3_s'],0)
        self.assertAlmostEqual(outcomes[0][1]['level_m'],-.6)
        self.assertLess(outcomes[0][0].storage,0)
        shifted=ProtectedOutletModel(settings('train'),replace(OutletSpec(),storage_reference_m=0))
        other=WaterState(shifted.lake.level_to_storage(-.49))
        original=WaterState(model.lake.level_to_storage(-.49))
        self.assertEqual(model.advance(original,100,0,100,1)[1]['release_m3_s'],
                         shifted.advance(other,100,0,100,1)[1]['release_m3_s'])

    def test_disabled_rule_has_exact_unprotected_parity(self):
        a=make_outlet_env(years=(1946,));b=make_protected_outlet_env(years=(1946,),rule=None)
        np.testing.assert_array_equal(a.reset()[0],b.reset()[0])
        for _ in range(40):
            old=a.step(np.array([100],np.float32));new=b.step(np.array([100],np.float32))
            np.testing.assert_array_equal(old[0],new[0]);self.assertEqual(old[1:4],new[1:4])
        a.close();b.close()

    def test_invalid_protection_assumptions_rejected(self):
        for kwargs in [dict(reserve_hours=0),dict(floor_m=float('nan'))]:
            with self.assertRaises(ValueError):LowWaterRule(**kwargs)
        model=ProtectedOutletModel(settings('train'))
        with self.assertRaises(ValueError):model.advance(WaterState(0),100,100,100,1,1.)
        with self.assertRaises(ValueError):
            make_protected_outlet_env(evaporation_mm_day={1946:np.ones(365)})
        with self.assertRaises(ValueError):
            model.lake.integrate_interval(0,100,0,1,duration_seconds=172800,substeps=1)


if __name__=='__main__':unittest.main()
