import unittest

import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.scoring import ScoreSettings, period_cost, score_trajectories


def trajectory(controller="mpc", year=2017):
    return pd.DataFrame(
        dict(
            controller=controller,
            year=year,
            quarter_month=np.arange(1, 49),
            ontario_end_m=np.full(48, 75.0),
            pointe_claire_m=np.full(48, 21.0),
            release_m3_s=np.full(48, 7000.0),
        )
    )


class OntarioScoringChecks(unittest.TestCase):
    def test_scalar_and_trajectory_scores_share_the_same_cost(self):
        rows = trajectory()
        rows.loc[0, ["ontario_end_m", "pointe_claire_m"]] = [76.0, 22.3]
        settings = ScoreSettings(ontario_weight=2.0, pointe_claire_weight=3.0)
        expected = 2 * 0.5 + 3 * 0.2
        self.assertAlmostEqual(float(period_cost(76.0, 22.3, settings)), expected)
        result = score_trajectories(rows, settings).iloc[0]
        self.assertAlmostEqual(result.cost, expected)
        self.assertAlmostEqual(result.score, -expected)
        self.assertAlmostEqual(result.ontario_excess_m_periods, 0.5)
        self.assertAlmostEqual(result.pointe_claire_excess_m_periods, 0.2)

    def test_controller_name_cannot_affect_score(self):
        frame = pd.concat([trajectory("mpc"), trajectory("ppo")])
        scored = score_trajectories(frame)
        self.assertEqual(scored.score.tolist(), [0.0, 0.0])

    def test_incomplete_duplicate_and_unmatched_years_are_rejected(self):
        rows = trajectory()
        invalid = [
            rows.iloc[:-1],
            pd.concat([rows, rows.iloc[[0]]]),
            pd.concat([rows, trajectory("ppo", 2019)]),
        ]
        for frame in invalid:
            with self.assertRaises(ValueError):
                score_trajectories(frame)
        with self.assertRaises(ValueError):
            period_cost(float("nan"), 22.0)
        with self.assertRaises(ValueError):
            ScoreSettings(ontario_scale_m=0.0)


if __name__ == "__main__":
    unittest.main()
