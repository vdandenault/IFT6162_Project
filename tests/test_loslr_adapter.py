from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from reservoir_control.adapters.planning import PlanningAdapter
from reservoir_control.study_config import ONTARIO_CONFIG
from reservoir_control.systems.loslr.adapter import OntarioSession, decision_time, initial_state
from reservoir_control.systems.loslr.baseline import (
    adapter_trace,
    archived_glrrm_comparison,
    compare_traces,
    run,
    study_recipe,
)
from reservoir_control.systems.loslr.data import baseline_parameters, forcing_from_row, load_inputs
from reservoir_control.systems.loslr.model import OntarioModel, Plan2014
from reservoir_control.systems.loslr.native import NativeReference, default_checkout


@unittest.skipUnless(
    (default_checkout() / ".git").exists(), "Pinned external LOSLR checkout not installed"
)
class LoslrParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = NativeReference()
        cls.frame = load_inputs(cls.reference)

    def test_causal_release_is_invariant_to_current_and_future_forcing(self):
        from reservoir_control.systems.loslr.causal import CausalOntarioSession

        model = OntarioModel(self.reference)
        original = CausalOntarioSession(model, self.frame)
        modified = self.frame.copy()
        # Supplied forecasts must have no effect, including during warmup.
        forecast_columns = ["forNTS", "slonFlow_QM1", *(f"ontNTS_QM{k}" for k in range(1, 5))]
        modified.loc[48:, forecast_columns] += 9000
        modified.loc[original.index :, ["ontNTS", "stlouisontOut"]] += 2500
        modified.loc[original.index :, ["ptclaireR", "longsaultR"]] *= 1.2
        modified.loc[original.index :, "iceInd"] = 0
        altered = CausalOntarioSession(model, modified)

        decision, time = original.information()
        self.assertEqual(altered.information(), (decision, time))
        self.assertEqual(original.state.level_m, altered.state.level_m)
        action = Plan2014(self.reference).action(original.state, decision.forecast, time)
        first, second = original.step(action), altered.step(action)
        self.assertEqual(first.info["native_limits"], second.info["native_limits"])
        self.assertEqual(
            first.metrics["executed_release_m3_s"], second.metrics["executed_release_m3_s"]
        )
        self.assertNotEqual(first.state.level_m, second.state.level_m)
        self.assertNotEqual(
            first.metrics["pointe_claire_level_m"], second.metrics["pointe_claire_level_m"]
        )

    def test_mpc_projection_matches_execution_at_forecast_forcing(self):
        from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
        from reservoir_control.systems.loslr.causal import CausalOntarioSession
        from reservoir_control.systems.loslr.planning import OntarioPlanningAdapter

        model = OntarioModel(self.reference)
        session = CausalOntarioSession(model, self.frame)
        decision, time = session.information()
        controller = RecedingHorizonSearch(OntarioPlanningAdapter(model))
        plan = controller.plan(session.state, decision, time)
        self.assertTrue(plan.feasible)
        self.assertEqual(len(plan.steps), 4)
        forcing = forcing_from_row(self.frame.iloc[session.index])
        metadata = dict(forcing.native_metadata, ptclaireR=decision.pointe_claire_roughness)
        forecast_forcing = replace(
            forcing,
            nts_m3_s=decision.forecast.nts_m3_s[0],
            slon_m3_s=decision.forecast.slon_m3_s,
            ice=decision.ice,
            native_metadata=metadata,
        )
        realized = model.transition(
            session.state, plan.actions[0], forecast_forcing, time, decision=decision
        )
        self.assertEqual(plan.steps[0].release_m3_s, realized.metrics["executed_release_m3_s"])
        self.assertEqual(plan.steps[0].pointe_claire_m, realized.metrics["pointe_claire_level_m"])
        self.assertEqual(plan.steps[0].state.level_m, realized.state.level_m)

    def test_mpc_reports_fallback_across_year_boundary(self):
        from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
        from reservoir_control.systems.loslr.causal import CausalOntarioSession
        from reservoir_control.systems.loslr.planning import MPCSettings, OntarioPlanningAdapter

        model = OntarioModel(self.reference)
        session = CausalOntarioSession(model, self.frame)
        decision, time = session.information()
        time = replace(time, period=47)
        settings = MPCSettings(ontario_max_m=73.0)
        plan = RecedingHorizonSearch(OntarioPlanningAdapter(model, settings)).plan(
            session.state, decision, time
        )
        self.assertFalse(plan.feasible)
        self.assertEqual(len(plan.actions), 4)
        self.assertGreater(plan.max_excess_m, 0)
        self.assertGreaterEqual(plan.total_excess_m, plan.max_excess_m)

    def test_native_import_does_not_pollute_paths_or_absolute_module_names(self):
        path = sys.path[:]
        existing = {
            name: module
            for name, module in sys.modules.items()
            if name == "functions" or name.startswith("functions.")
        }
        NativeReference()
        self.assertEqual(sys.path, path)
        after = {
            name: module
            for name, module in sys.modules.items()
            if name == "functions" or name.startswith("functions.")
        }
        self.assertEqual(after, existing)

    def test_pin_verification_rejects_modified_loaded_source(self):
        original = Path.read_bytes
        target = self.reference.root / "functions/utils.py"

        def changed(path):
            value = original(path)
            return value + b"\n# modified\n" if path == target else value

        with patch.object(Path, "read_bytes", changed):
            with self.assertRaisesRegex(ValueError, "differs from pinned source"):
                NativeReference()

    def test_dirty_archived_glrrm_reference_rejected_before_comparison(self):
        relative = "output/data/baseline/Plan2014_GLRRM/simulation/historic/1900_2020/sim.csv"
        original = Path.read_bytes
        target = self.reference.root / relative

        def changed(path):
            value = original(path)
            return value + b"\n# modified\n" if path == target else value

        with patch.object(Path, "read_bytes", changed):
            with self.assertRaisesRegex(ValueError, "differs from pinned source"):
                archived_glrrm_comparison(self.reference, None)

    def test_schema_translation_and_mixed_unit_execution(self):
        params = baseline_parameters(self.reference)
        self.assertEqual(params["C1*"], 260)
        self.assertEqual(params["dryFlowAdjustment"], -20)
        state = initial_state(self.frame)
        self.assertEqual(state.previous_release_m3_s, 5950)
        row = self.frame.iloc[48]
        forcing = forcing_from_row(row)
        time = decision_time(row, 0)
        action = Plan2014(self.reference).action(state, forcing.forecast, time)
        result = OntarioModel(self.reference).transition(state, action, forcing, time)
        native = result.metrics["executed_release_m3_s"] / 10
        expected = self.reference.utils.round_d((forcing.nts_m3_s / 10 - native) / 2970, 6)
        self.assertEqual(result.state.level_m, state.level_m + expected)
        self.assertAlmostEqual(
            result.metrics["lake_st_louis_flow_m3_s"], native * 10 + forcing.slon_m3_s
        )

    def test_full_period_native_parity_including_year_carryover_and_events(self):
        n = self.reference
        native = (
            n.simulation.simulation(
                self.frame.copy(deep=True),
                {},
                n.release.getReleaseFunctionInputs,
                n.release.releaseFunction,
                n.september.septemberRule,
                n.limits.getPlanLimitsInputs,
                n.limits.planLimits,
                n.routing.getStLawrenceRoutingInputs,
                n.routing.stLawrenceRouting,
                baseline_parameters(n),
            )
            .iloc[48:]
            .reset_index(drop=True)
        )
        adapted = adapter_trace(n, self.frame)
        comparison = compare_traces(native, adapted)
        self.assertTrue(comparison["complete"], comparison)
        self.assertEqual(comparison["decisions"], 121 * 48)
        self.assertLessEqual(max(comparison["max_absolute_errors_native_units"].values()), 1e-10)
        self.assertEqual(adapted.loc[adapted.Year.eq(2017)].shape[0], 48)
        self.assertEqual(adapted.loc[adapted.Year.eq(2019)].shape[0], 48)

    def test_planning_is_isolated_from_live_replay_and_terminal_state_preserved(self):
        session = OntarioSession(OntarioModel(self.reference), self.frame.iloc[:96])
        original = session.state
        forecast, time = session.information()
        action = Plan2014(self.reference).action(original, forecast, time)
        planner = PlanningAdapter(session.model)
        predicted = planner.transition(
            original, action, forcing_from_row(self.frame.iloc[48]), time
        )
        self.assertEqual(session.index, 48)
        self.assertEqual(session.state, original)
        actual = session.step(action)
        self.assertEqual(predicted.metrics, actual.metrics)
        policy = Plan2014(self.reference)
        while not session.done:
            forecast, time = session.information()
            final = session.step(policy.action(session.state, forecast, time))
        self.assertEqual(session.state, final.state)
        with self.assertRaises(RuntimeError):
            session.step(action)
        self.assertEqual(session.reset().level_m, 74.55)

    def test_unsupported_duration_rejected(self):
        row = self.frame.iloc[48]
        state = initial_state(self.frame)
        with self.assertRaisesRegex(ValueError, "duration"):
            OntarioModel(self.reference).transition(
                state,
                6000,
                forcing_from_row(row),
                replace(decision_time(row, 0), duration_seconds=3600),
            )

    def test_native_import_failure_leaves_global_paths_unchanged(self):
        path = sys.path[:]
        source = b'import sys\nsys.path.append("unexpected-native-path")\nraise RuntimeError("native failed")'
        with patch.object(self.reference, "verify_file", return_value=source):
            with self.assertRaisesRegex(RuntimeError, "native failed"):
                self.reference.load("functions/limits/Bv7_glamUpdates.py")
        self.assertEqual(sys.path, path)

    def test_recipe_rejects_unimplemented_variant_and_invalid_data_range(self):
        recipe = study_recipe(ONTARIO_CONFIG)
        recipe["limits"] = "onlyPhysicalLimits"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "Unsupported reference recipe limits"):
                study_recipe(path)
        for year in (1899, 2021):
            with self.assertRaises(ValueError):
                load_inputs(self.reference, year)

    def test_nonempty_result_directory_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                run(path)
            self.assertEqual((path / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
