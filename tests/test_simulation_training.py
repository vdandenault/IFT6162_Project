import unittest
import numpy as np
from lake_como.simulation.protocol import TaskProtocol,TrainingEpisodes
from lake_como.simulation.training import train_operational
from lake_como.simulation import make_operational_env,PhysicalWeights

class OperationalLearningChecks(unittest.TestCase):
    def test_record_reset_and_bootstrap_masks_are_separate(self):
        for mode,mask in [('finite_horizon',True),('continuing_record',False)]:
            env=TrainingEpisodes(TaskProtocol(episode_years=1,burn_in_days=0,terminal_mode=mode))
            agent,trace,report=train_operational(env,steps=365,warmup=400,seed=901)
            self.assertEqual(len(report['episodes']),1)
            self.assertEqual(report['episodes'][0]['length'],365)
            self.assertEqual(report['updates'],0)
            self.assertEqual(trace[-1]['replay_end'],mask)
            self.assertEqual(trace[-1]['terminated'] or trace[-1]['truncated'],True)
            self.assertEqual(sum(row['replay_end'] for row in trace),int(mask))
            self.assertEqual(env.episodes,1)  # No unused reset/spin-up at end of budget.
            env.close()

    def test_continuing_cutoff_cannot_add_a_finite_terminal_penalty(self):
        with self.assertRaises(ValueError):
            make_operational_env(years=(1946,),terminal_mode='continuing_record',weights=PhysicalWeights(terminal_reserve_per_million_m3=1))

    def test_terminal_reserve_cost_is_explicit_and_only_at_end(self):
        a=make_operational_env(years=(1946,),weights=PhysicalWeights(terminal_reserve_per_million_m3=2))
        b=make_operational_env(years=(1946,),weights=PhysicalWeights())
        a.reset();b.reset()
        for step in range(365):
            action=np.array([300],np.float32)
            _,r,term,_,info=a.step(action);_,s,*_=b.step(action)
            if not term:self.assertEqual(r,s);self.assertEqual(info['terminal_reserve_cost'],0)
        target=a.model.lake.level_to_storage(.35)
        self.assertAlmostEqual(s-r,2*max(target-a.state.storage,0)/1e6)
        a.close();b.close()
