"""Check late-window summaries without treating log points as independent trials."""
import unittest
import pandas as pd
from reservoir_control.systems.lake_como.learning_curves import window_summary,TAGS,SEEDS


class LearningCurveChecks(unittest.TestCase):
    def traces(self):
        rows=[]
        points=[(100000,1000),(112500,1),(125000,3),(137500,5),(150000,7),
                (162500,9),(175000,11),(187500,13),(200000,15),(200001,9999)]
        for i,seed in enumerate(SEEDS):
            for metric in TAGS:
                rows.extend(dict(training_seed=seed,metric=metric,step=step,value=value+i)
                            for step,value in points)
        return pd.DataFrame(rows)

    def test_windows_are_open_left_closed_right_and_changes_are_descriptive(self):
        windows,changes=window_summary(self.traces())
        first=windows[(windows.training_seed==123)&(windows.metric=='episode_reward')]
        self.assertEqual(first['mean'].tolist(),[2,6,10,14])
        self.assertEqual(first.logged_points.tolist(),[2]*4)
        self.assertEqual(changes.episode_reward_previous_50k.tolist(),[4,5,6,7,8])
        self.assertEqual(changes.episode_reward_last_50k.tolist(),[12,13,14,15,16])
        self.assertEqual(changes.episode_reward_change.tolist(),[8]*5)

    def test_missing_late_window_rejected(self):
        f=self.traces();f=f[~((f.training_seed==123)&(f.metric=='episode_reward')&(f.step>175000))]
        with self.assertRaisesRegex(ValueError,'Missing'):window_summary(f)


if __name__=='__main__':unittest.main()
