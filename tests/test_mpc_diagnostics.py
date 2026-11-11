"""Protect diagnostic comparisons from alignment, cost, and replay mistakes."""
import unittest

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig
from reservoir_control.systems.lake_como.mpc_diagnostics import (
    boundary_audit,
    exhaustive_grid,
    level_penalties,
    longest_episode,
    replay_snapshots,
)


class MPCDiagnosticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller = DeterministicMPC.from_training(MPCConfig(horizon_days=1))

    def test_identical_endpoints_isolate_penalty_shape(self):
        binary, surrogate = level_penalties(
            np.array([1.24, 1.25, -0.45, -0.46]), self.controller.model
        )
        np.testing.assert_allclose(binary, [0, 1000, 0, 250])
        np.testing.assert_allclose(surrogate, [0, 40, 0, 10], atol=1e-10)

    def test_boundary_comparison_joins_by_day_and_rejects_missing_data(self):
        frame = pd.DataFrame(
            dict(
                year=[1988, 1988],
                day=[1, 2],
                level_m=[1.25, -0.46],
                inflow_m3_s=[10.0, 20.0],
                weighted_overflow_reward=[-1000.0, 0.0],
                weighted_starving_reward=[0.0, -250.0],
            )
        )
        diagnostics = pd.DataFrame(
            dict(
                year=[1988, 1988],
                day=[2, 1],
                predicted_level_m=[-0.44, 1.23],
                forecast_inflow_m3_s=[20.0, 10.0],
                forecast_demand_m3_s=[1.0, 1.0],
                solver_success=[True, True],
                used_fallback=[False, False],
                planned_cost=[0.0, 0.0],
            )
        )
        audit, summary = boundary_audit(frame, diagnostics, self.controller.model)
        np.testing.assert_allclose(audit.predicted_level_m, [1.23, -0.44])
        self.assertEqual(summary.forecast_missed_days.tolist(), [1, 1])
        with self.assertRaisesRegex(ValueError, "identical days"):
            boundary_audit(frame, diagnostics.iloc[:1], self.controller.model)

    def test_cartesian_grid_finds_known_minimum(self):
        optimum = np.array([0.25, 0.75])
        plan, cost = exhaustive_grid(lambda x: float(((x - optimum) ** 2).sum()), 2, 5)
        np.testing.assert_array_equal(plan, optimum)
        self.assertEqual(cost, 0.0)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            exhaustive_grid(lambda x: np.nan, 1, 3)

    def test_episode_selection_uses_consecutive_days_and_earliest_tie(self):
        frame = pd.DataFrame(dict(day=[1, 2, 4, 5, 7, 8, 9]))
        episode = longest_episode(frame, pd.Series(True, index=frame.index))
        self.assertEqual(episode.day.tolist(), [7, 8, 9])
        episode = longest_episode(frame, frame.day < 7)
        self.assertEqual(episode.day.tolist(), [1, 2])

    def test_historical_replay_rejects_changed_actions(self):
        config = MPCConfig(horizon_days=1)
        controller = DeterministicMPC.from_training(config)
        action = controller(np.array([0.35, 0, 1, 0], dtype=np.float32))
        frame = pd.DataFrame(
            [
                dict(
                    year=1988,
                    day=1,
                    level_before_m=0.35,
                    request_m3_s=float(action[0]),
                    planned_cost=controller.diagnostics[-1]["planned_cost"],
                )
            ]
        )
        cases = pd.DataFrame([dict(case="control", year=1988, day=1)])
        snapshots, checked = replay_snapshots(frame, cases, config)
        self.assertEqual(checked, 1)
        self.assertIn((1988, 1), snapshots)
        frame["request_m3_s"] += 1
        with self.assertRaises(AssertionError):
            replay_snapshots(frame, cases, config)


if __name__ == "__main__":
    unittest.main()
