"""Migration gates: isolated planning, optional imports and exact Como parity."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from reservoir_control.core import DecisionTime, EpisodeStart, EpisodeStep, Transition
from reservoir_control.adapters.planning import PlanningAdapter


class SharedContractChecks(unittest.TestCase):
    def test_common_imports_do_not_load_simulator_or_gym(self):
        code = ('import sys; import reservoir_control.core; import reservoir_control.adapters; '
                'assert not any(k.startswith(("gym4real", "reservoir_control.systems.lake_como", "gymnasium", "torch")) '
                'for k in sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_shared_learning_implementations_do_not_load_a_system(self):
        code = (
            'import sys; '
            'import reservoir_control.controllers.ppo.agent; '
            'import reservoir_control.controllers.sac.agent; '
            'import reservoir_control.controllers.fqi.agent; '
            'import reservoir_control.controllers.mpc.receding_horizon; '
            'assert not any(k.startswith(("reservoir_control.systems", "gym4real", "gymnasium")) '
            'for k in sys.modules)'
        )
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_legacy_package_is_removed(self):
        import importlib.util
        self.assertIsNone(importlib.util.find_spec('lake_como'))

    def test_frozen_source_verification_reports_the_historical_restore_path(self):
        from reservoir_control.evaluation.provenance import verify_sources
        with self.assertRaisesRegex(ValueError, 'pre-migration'):
            verify_sources({'/missing/src/lake_como/simulation/model.py': 'original-digest'})

    def test_time_supports_different_system_calendars(self):
        DecisionTime(0, 2020, 366, 86400.)
        DecisionTime(1, 2020, 48, 657450.)
        for duration in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                DecisionTime(0, 2020, 1, duration)

    def test_planning_isolates_mutating_model_and_forecasts(self):
        class MutatingModel:
            def transition(self, state, action, forcing, time):
                state['storage'] += action[0] + forcing.pop('inflow')
                action[0] = -999
                return Transition(state, {'storage': state['storage']})
        adapter = PlanningAdapter(MutatingModel())
        initial = {'storage': 10.}
        actions = [[2.], [3.]]
        forcing = [{'inflow': 4.}, {'inflow': 5.}]
        times = [DecisionTime(i, 2020, i + 1, 1.) for i in range(2)]
        rows = adapter.rollout(initial, actions, forcing, times)
        self.assertEqual([r.state['storage'] for r in rows], [16., 24.])
        self.assertEqual(initial, {'storage': 10.})
        self.assertEqual(actions, [[2.], [3.]])
        self.assertEqual(forcing, [{'inflow': 4.}, {'inflow': 5.}])
        self.assertEqual(rows[0].state['storage'], 16.)
        with self.assertRaises(ValueError):
            adapter.rollout(initial, actions, forcing[:1], times)
        with self.assertRaises(ValueError):
            adapter.rollout(initial, actions, forcing, [times[0], DecisionTime(3, 2020, 4, 1.)])

    def test_episode_runner_does_not_invent_terminal_flags_or_call_other_controller(self):
        from reservoir_control.evaluation import run_episode
        from reservoir_control.evaluation.rollout import IncompleteEpisodeError
        class Session:
            def reset(self, *, seed=None, options=None):
                self.count = 0
                return EpisodeStart([0.])
            def step(self, action):
                self.count += 1
                return EpisodeStep([float(self.count)], -float(action), self.count == 2, False)
        session = Session()
        result = run_episode(session, lambda obs, info: obs[0] + 1., max_steps=3)
        self.assertEqual(result.total_reward, -3.)
        self.assertEqual([row.action for row in result.steps], [1., 2.])
        self.assertTrue(result.steps[-1].result.terminated)
        with self.assertRaises(IncompleteEpisodeError):
            run_episode(session, lambda obs, info: 1., max_steps=1)

    def test_gym_boundary_preserves_terminal_observation_and_requires_reset(self):
        import gymnasium as gym
        from reservoir_control.adapters.gymnasium import GymnasiumAdapter
        class Session:
            def reset(self, *, seed=None, options=None):
                return EpisodeStart(np.array([1.], np.float32))
            def step(self, action):
                return EpisodeStep(np.array([9.], np.float32), -2., False, True, {'actual': 9.})
            def close(self):
                pass
        env = GymnasiumAdapter(Session(), action_space=gym.spaces.Box(0., 1., (1,)),
                               observation_space=gym.spaces.Box(0., 10., (1,)))
        with self.assertRaises(RuntimeError): env.step(np.array([0.]))
        env.reset(seed=3)
        obs, reward, term, trunc, info = env.step(np.array([0.]))
        self.assertEqual((obs[0], reward, term, trunc, info['actual']), (9., -2., False, True, 9.))
        with self.assertRaises(RuntimeError): env.step(np.array([0.]))
        env.reset(seed=3)
        env.close()


class ComoAdapterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reservoir_control.systems.lake_como.environment import settings
        cls.params = settings('train')

    def test_two_year_gym_trajectory_matches_native_and_frozen_boundary_fixture(self):
        from reservoir_control.systems.lake_como.simulation.environment import OperationalEnv
        from reservoir_control.systems.lake_como.adapters import ComoGymnasiumAdapter
        native = OperationalEnv(self.params, (1946, 1947))
        adapter = ComoGymnasiumAdapter(self.params, (1946, 1947))
        fixture = json.loads((Path(__file__).parent / 'fixtures/como_adapter_boundary.json').read_text())
        native_obs, native_info = native.reset(seed=123)
        obs, info = adapter.reset(seed=123)
        np.testing.assert_array_equal(obs, native_obs)
        self.assertEqual(info, native_info)
        for index in range(730):
            action = np.array([20 + (index * 37) % 400], np.float32)
            expected = native.step(action)
            actual = adapter.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])
            if str(index) in fixture['samples']:
                saved = fixture['samples'][str(index)]
                np.testing.assert_array_equal(actual[0], saved['observation'])
                self.assertEqual(actual[1], saved['reward'])
                self.assertEqual(actual[-1]['storage'], saved['storage_m3'])
                self.assertEqual(actual[-1]['year_boundary'], saved['year_boundary'])
            if index == 364:
                self.assertFalse(actual[2] or actual[3])
                self.assertEqual(actual[-1]['next_year'], 1947)
            if index == 365:
                self.assertEqual(actual[-1]['hourly'][0]['storage_before_m3'], fixture['samples']['364']['storage_m3'])
        self.assertEqual(actual[2:4], (True, False))
        self.assertEqual(actual[0][0], np.float32(native.model.lake.storage_to_level(adapter.physical_state.storage)))
        with self.assertRaises(RuntimeError): adapter.step(action)
        native.close(); adapter.close()

    def test_hourly_planning_matches_physics_across_year_boundary(self):
        from reservoir_control.systems.lake_como.simulation.model import OperationalModel, WaterState
        from reservoir_control.systems.lake_como.adapters import ComoForcing, ComoPlanningAdapter
        model = OperationalModel(self.params)
        initial = WaterState(model.lake.level_to_storage(.35))
        adapter = ComoPlanningAdapter(self.params)
        actions = [100.] * 48
        forcings = [ComoForcing(float(self.params['inflow'][year][day-1]),
                    float(self.params['demand'][year][day-1]), day)
                    for year, day in [(1946, 365)] * 24 + [(1947, 1)] * 24]
        times = [DecisionTime(i, 1946 if i < 24 else 1947, f.model_day, 3600.)
                 for i, f in enumerate(forcings)]
        planned = adapter.rollout(initial, actions, forcings, times)
        state = initial
        for result, action, forcing in zip(planned, actions, forcings):
            state, row = model.advance(state, action, forcing.inflow_m3_s,
                                      forcing.demand_m3_s, forcing.model_day)
            self.assertEqual(result.state, state)
            self.assertEqual(result.metrics, row)
        self.assertEqual(initial.storage, model.lake.level_to_storage(.35))
        self.assertEqual(adapter.rollout(initial, actions, forcings, times), planned)
        with self.assertRaises(ValueError):
            adapter.transition(initial, 100., forcings[0], DecisionTime(0, 1946, 365, 86400.))

    def test_continuing_record_uses_truncation_and_actual_terminal_level(self):
        from reservoir_control.systems.lake_como.adapters import ComoGymnasiumAdapter
        env = ComoGymnasiumAdapter(self.params, (1946,), terminal_mode='continuing_record')
        env.reset(seed=123)
        for _ in range(365): obs, _, term, trunc, info = env.step(np.array([100.], np.float32))
        self.assertEqual((term, trunc), (False, True))
        self.assertEqual(obs[-1], 1.)
        self.assertEqual(obs[0], np.float32(env.session.env.model.lake.storage_to_level(info['storage'])))
        env.close()


if __name__ == '__main__':
    unittest.main()
