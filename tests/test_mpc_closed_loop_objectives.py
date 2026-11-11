"""Validate the common grid implementation against canonical scalar physics."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DeterministicMPC, MPCConfig, PlanningModel
from reservoir_control.systems.lake_como.mpc_hindsight import perfect_inflow_controller
from reservoir_control.systems.lake_como.mpc_objective_audit import compare_grid, score_plan
from reservoir_control.systems.lake_como.mpc_objective_closed_loop import (
    OneDayGridPolicy,
    score_actions,
    level_severity,
    forecast_for_mode,
    search_signature,
)


class ClosedLoopObjectiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = settings("train")
        cls.config = MPCConfig(horizon_days=1)
        cls.model = PlanningModel(cls.params, cls.config)
        cls.reference = DeterministicMPC.from_training(cls.config)
        cls.inflows = settings("validation")["inflow"]

    def test_batched_scores_match_scalar_hydraulics_and_costs(self):
        model = self.model
        requests = np.linspace(0, model.high, 21).astype(np.float32).astype(float)
        for level in [
            -0.5,
            -0.4999,
            -0.45,
            -0.40001,
            -0.4,
            0.35,
            1.24,
            1.25,
            1.25001,
            2.0,
        ]:
            for day, inflow in [(1, 0.0), (180, 100.0), (365, 2000.0)]:
                storage = model.lake.level_to_storage(level)
                batch = score_actions(model, requests, storage, inflow, 100.0, day)
                for index, request in enumerate(requests):
                    scalar = score_plan(
                        model, [request], storage, [inflow], [100.0], np.array([day])
                    )
                    np.testing.assert_allclose(
                        batch.costs[index],
                        [scalar.surrogate, scalar.binary],
                        rtol=1e-9,
                        atol=1e-9,
                    )
                    self.assertAlmostEqual(
                        batch.levels[index], scalar.levels[0], places=9
                    )
                    self.assertAlmostEqual(
                        batch.releases[index], scalar.releases[0], places=7
                    )

    def test_hydraulic_switch_uses_scalar_rechecks(self):
        requests = np.linspace(0, self.model.high, 21).astype(np.float32).astype(float)
        batch = score_actions(
            self.model,
            requests,
            self.model.lake.level_to_storage(1.25),
            100.0,
            100.0,
            1,
        )
        self.assertEqual(batch.scalar_rechecks, len(requests))

    def test_common_grid_winners_match_exhaustive_scalar_reference(self):
        model = self.model
        requests = np.linspace(0, model.high, 101).astype(np.float32).astype(float)
        for level in [-0.449, 0.35, 1.245]:
            storage = model.lake.level_to_storage(level)
            batch = score_actions(model, requests, storage, 50.0, 100.0, 1)
            _, chosen, _ = compare_grid(
                model, storage, [50.0], [100.0], np.array([1]), 101
            )
            for objective, column in [("surrogate", 0), ("binary", 1)]:
                winner = batch.costs[:, column].argmin()
                self.assertEqual(requests[winner], chosen[objective]["requests"][0])

    def test_policy_keeps_oracle_demand_and_one_day_scope(self):
        oracle = perfect_inflow_controller(
            1988, self.params, self.reference.forecast, self.config, self.inflows
        )
        a = OneDayGridPolicy(
            self.params, oracle.forecast, self.config, "surrogate", 101
        )
        b = OneDayGridPolicy(self.params, oracle.forecast, self.config, "binary", 101)
        obs = np.array([0.35, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(a(obs), b(obs))
        self.assertEqual(
            a.diagnostics[-1]["forecast_demand_m3_s"], self.reference.forecast.demand[0]
        )
        self.assertEqual(
            b.diagnostics[-1]["forecast_inflow_m3_s"], self.inflows[1988][0]
        )
        with self.assertRaisesRegex(ValueError, "one-day"):
            OneDayGridPolicy(
                self.params, oracle.forecast, MPCConfig(horizon_days=7), "binary"
            )

    def test_seasonal_forecast_never_reads_held_out_inflows(self):
        class ForbiddenInflows:
            def __getitem__(self, key):
                raise AssertionError("Held-out inflows accessed")

        forecast = forecast_for_mode(
            "seasonal", 1988, self.params, self.reference,
            self.config, ForbiddenInflows(),
        )
        self.assertIs(forecast, self.reference.forecast)
        self.assertFalse(forecast.inflow.flags.writeable)
        obs = np.array([0.35, 0, 1, 0], dtype=np.float32)
        for objective in ["binary", "surrogate"]:
            a = OneDayGridPolicy(self.params, forecast, self.config, objective, 101)
            b = OneDayGridPolicy(self.params, forecast, self.config, objective, 101)
            np.testing.assert_array_equal(
                a(obs, {"future_inflow": 0}), b(obs, {"future_inflow": 2000})
            )
            self.assertEqual(a.diagnostics[-1]["forecast_inflow_m3_s"], forecast.inflow[0])
        with self.assertRaisesRegex(ValueError, "validation-only"):
            forecast_for_mode("seasonal", 1998, self.params, self.reference, self.config)

    def test_search_signature_detects_computational_changes(self):
        from reservoir_control.systems.lake_como import mpc_objective_closed_loop as module
        source = Path(module.__file__).read_text()
        self.assertEqual(search_signature(source), search_signature(source + "\n# comment\n"))
        changed = source.replace("obs.shape != (4,)", "obs.shape != (5,)")
        self.assertNotEqual(search_signature(source), search_signature(changed))

    def test_fewer_flood_days_can_have_higher_exceedance_integral(self):
        daily = pd.DataFrame(
            dict(
                objective=["surrogate", "surrogate", "binary", "binary"],
                year=[1988] * 4,
                level_m=[1.241, 1.241, 1.3, 1.0],
            )
        )
        result = level_severity(daily)
        self.assertAlmostEqual(
            result.loc[("surrogate", 1988), "flood_excess_m_days"], 0.002
        )
        self.assertAlmostEqual(
            result.loc[("binary", 1988), "flood_excess_m_days"], 0.06
        )


if __name__ == "__main__":
    unittest.main()
