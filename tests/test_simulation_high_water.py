from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.simulation.high_water import ReplayAnchor, flood_replay, capacity_volume_bound
from reservoir_control.systems.lake_como.simulation.low_water import ProtectedOutletModel, ProtectedOutletLake, make_protected_outlet_env
from reservoir_control.systems.lake_como.simulation.model import WaterState


class FloodDiagnosticChecks(unittest.TestCase):
    def test_hourly_replay_matches_real_environment_and_branch_state(self):
        params=settings('train')
        for policy in ['constant_100','causal_feedback','maximum_nominal']:
            trace,snapshots,failure=flood_replay(params,(1946,),policy=policy,end_hour=20*24)
            self.assertIsNone(failure)
            env=make_protected_outlet_env(years=(1946,));obs,_=env.reset()
            original=[]
            for _ in range(20):
                request=100 if policy=='constant_100' else (
                    env.model.high if policy=='maximum_nominal' else np.clip(float(obs[3])+20+150*(float(obs[0])-.35),0,env.model.high))
                obs,_,_,_,info=env.step(np.array([request],np.float32));original.extend(info['hourly'])
            np.testing.assert_array_equal(trace.level_m,[r['level_m'] for r in original])
            np.testing.assert_array_equal(trace.release_m3_s,[r['release_m3_s'] for r in original])
            resumed,_,failure=flood_replay(params,(1946,),policy=policy,anchor=snapshots[10*24],end_hour=20*24)
            np.testing.assert_array_equal(resumed.level_m,trace.level_m.iloc[10*24:])
            self.assertIsNone(failure);env.close()

    def test_exit_is_trial_only_and_curve_never_evaluated_beyond_domain(self):
        params=deepcopy(settings('train'));params['inflow'][1946][0]=3000
        model=ProtectedOutletModel(params)
        anchor=ReplayAnchor(0,WaterState(model.lake.level_to_storage(2.63)),0.)
        original=ProtectedOutletLake.max_release
        evaluated=[]
        def guarded(lake,storage,day):
            level=lake.storage_to_level(storage);evaluated.append(level)
            self.assertLessEqual(level,2.64)
            return original(lake,storage,day)
        with patch.object(ProtectedOutletLake,'max_release',guarded):
            trace,_,failure=flood_replay(params,(1946,),anchor=anchor,end_hour=24)
        self.assertEqual(len(trace),1);self.assertFalse(trace.accepted.iloc[0])
        self.assertEqual(failure['boundary'],'upper');self.assertGreater(failure['trial_level_m'],2.64)
        self.assertTrue(evaluated)
        self.assertTrue(capacity_volume_bound(params,(1946,),anchor,1)['certifies_upper_exit'])

    def test_integration_refinement_and_volume_balance(self):
        params=deepcopy(settings('train'));params['inflow'][1946][0]=1000
        model=ProtectedOutletModel(params)
        anchor=ReplayAnchor(0,WaterState(model.lake.level_to_storage(1.5)),0.)
        reference,_,failure=flood_replay(params,(1946,),anchor=anchor,end_hour=24,substeps_per_hour=64)
        self.assertIsNone(failure)
        errors=[]
        for count in [1,2,4,8]:
            trace,_,failure=flood_replay(params,(1946,),anchor=anchor,end_hour=24,substeps_per_hour=count)
            self.assertIsNone(failure)
            self.assertLess(trace.balance_error_m3.abs().max(),1e-6)
            errors.append(abs(trace.level_m.iloc[-1]-reference.level_m.iloc[-1]))
        self.assertTrue(all(b<a for a,b in zip(errors,errors[1:])))

    def test_maximum_nominal_reaches_available_capacity_and_forced_opening(self):
        model=ProtectedOutletModel(settings('train'));lake=model.lake
        for level in np.linspace(-.705,2.64,201):
            storage=lake.level_to_storage(level)
            expected=min(lake.max_release(storage,1),lake.rule.release_cap(level,lake.surface))
            self.assertAlmostEqual(lake.actual_release(float(np.float32(model.high)),storage,1),expected)
        storage=lake.level_to_storage(2.)
        self.assertEqual(lake.actual_release(0,storage,1),lake.actual_release(model.high,storage,1))
        with self.assertRaises(ValueError):flood_replay(settings('train'),(1998,))


if __name__=='__main__':unittest.main()
