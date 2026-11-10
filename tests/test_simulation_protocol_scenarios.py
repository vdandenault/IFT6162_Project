import unittest
import numpy as np
from lake_como.environment import settings
from lake_como.simulation.protocol import TaskProtocol,TrainingEpisodes,replay_terminal
from lake_como.simulation.scenarios import HistoricalScenarios,within_day_shape,stress_profiles
from lake_como.splits import split_years

class OperationalProtocolChecks(unittest.TestCase):
    def test_seeded_train_windows_spinup_and_discount(self):
        protocol=TaskProtocol(burn_in_days=2,decision_hours=6,initial_level_range=(0,.5))
        a=TrainingEpisodes(protocol);b=TrainingEpisodes(protocol)
        x,info=a.reset(seed=901);y,j=b.reset(seed=901)
        np.testing.assert_array_equal(x,y);self.assertEqual(info['episode_years'],j['episode_years'])
        self.assertTrue(set(info['episode_years']).issubset(split_years('train')))
        self.assertEqual(info['elapsed_hours'],48);self.assertEqual(a.burn_in_transitions,8)
        self.assertEqual(a.agent_transitions,0);self.assertAlmostEqual(protocol.gamma_per_decision**4,.995)
        a.step(np.array([100],np.float32));self.assertEqual(a.agent_transitions,1)
        self.assertFalse(replay_terminal(False,True));self.assertTrue(replay_terminal(True,False))
        a.close();b.close()

    def test_scenarios_are_train_only_contiguous_and_reproducible(self):
        params=settings('train');bank=HistoricalScenarios(params['inflow'],params['demand'])
        paths,meta=bank.paths(365,100,horizon=7,count=8,seed=4)
        np.testing.assert_array_equal(paths,bank.paths(365,100,horizon=7,count=8,seed=4)[0])
        for path,start in zip(paths,meta['source_start_indices']):np.testing.assert_array_equal(path,bank.flat[start:start+7])
        self.assertTrue((paths>=0).all())
        more,details=bank.paths(365,100,horizon=7,count=32,seed=4)
        np.testing.assert_array_equal(paths,more[:8])
        self.assertEqual(meta['conditioning_pool_size'],details['conditioning_pool_size'])
        altered=dict(params['inflow']);altered[1998]=[0]*365
        with self.assertRaises(ValueError):HistoricalScenarios(altered,params['demand'])

    def test_synthetic_intraday_shapes_preserve_volume(self):
        daily=np.array([0.,100.,300.])
        for kind in ['flat','front_loaded','late_loaded']:
            hourly=within_day_shape(daily,kind);np.testing.assert_allclose(hourly.mean(axis=1),daily)
        params=settings('train');new,hourly=stress_profiles(params,(1946,),'drought_half')
        self.assertNotEqual(new['inflow'][1946][0],params['inflow'][1946][0])
        np.testing.assert_allclose(hourly[1946].mean(axis=1),np.array(params['inflow'][1946][:365])*.5)
