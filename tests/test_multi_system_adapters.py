import subprocess
import sys
import unittest

import numpy as np

from reservoir_control.adapters.planning import PlanningAdapter
from reservoir_control.core import DecisionTime, EpisodeStart, EpisodeStep, Transition


class SharedContractChecks(unittest.TestCase):
    def test_common_imports_do_not_load_simulator_or_gym(self):
        code = (
            "import sys; import reservoir_control.core; import reservoir_control.adapters; "
            'assert not any(k.startswith(("gymnasium", "torch")) '
            "for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_shared_learning_implementations_do_not_load_a_system(self):
        code = (
            "import sys; "
            "import reservoir_control.controllers.ppo.agent; "
            "import reservoir_control.controllers.mpc.receding_horizon; "
            "import reservoir_control.controllers.mpc.bilevel; "
            "import reservoir_control.controllers.mpc.convex_response; "
            'assert not any(k.startswith(("reservoir_control.systems", "gymnasium")) '
            "for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_frozen_source_verification_rejects_missing_sources(self):
        from reservoir_control.evaluation.provenance import verify_sources

        with self.assertRaisesRegex(ValueError, "Frozen source is missing"):
            verify_sources({"/missing/source.py": "original-digest"})

    def test_time_supports_different_system_calendars(self):
        DecisionTime(0, 2020, 366, 86400.0)
        DecisionTime(1, 2020, 48, 657450.0)
        for duration in [0.0, -1.0, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                DecisionTime(0, 2020, 1, duration)

    def test_planning_isolates_mutating_model_and_forecasts(self):
        class MutatingModel:
            def transition(self, state, action, forcing, time):
                state["storage"] += action[0] + forcing.pop("inflow")
                action[0] = -999
                return Transition(state, {"storage": state["storage"]})

        adapter = PlanningAdapter(MutatingModel())
        initial = {"storage": 10.0}
        actions = [[2.0], [3.0]]
        forcing = [{"inflow": 4.0}, {"inflow": 5.0}]
        times = [DecisionTime(i, 2020, i + 1, 1.0) for i in range(2)]
        rows = adapter.rollout(initial, actions, forcing, times)
        self.assertEqual([r.state["storage"] for r in rows], [16.0, 24.0])
        self.assertEqual(initial, {"storage": 10.0})
        self.assertEqual(actions, [[2.0], [3.0]])
        self.assertEqual(forcing, [{"inflow": 4.0}, {"inflow": 5.0}])
        self.assertEqual(rows[0].state["storage"], 16.0)
        with self.assertRaises(ValueError):
            adapter.rollout(initial, actions, forcing[:1], times)
        with self.assertRaises(ValueError):
            adapter.rollout(initial, actions, forcing, [times[0], DecisionTime(3, 2020, 4, 1.0)])

    def test_episode_runner_does_not_invent_terminal_flags_or_call_other_controller(self):
        from reservoir_control.evaluation import run_episode
        from reservoir_control.evaluation.rollout import IncompleteEpisodeError

        class Session:
            def reset(self, *, seed=None, options=None):
                self.count = 0
                return EpisodeStart([0.0])

            def step(self, action):
                self.count += 1
                return EpisodeStep([float(self.count)], -float(action), self.count == 2, False)

        session = Session()
        result = run_episode(session, lambda obs, info: obs[0] + 1.0, max_steps=3)
        self.assertEqual(result.total_reward, -3.0)
        self.assertEqual([row.action for row in result.steps], [1.0, 2.0])
        self.assertTrue(result.steps[-1].result.terminated)
        with self.assertRaises(IncompleteEpisodeError):
            run_episode(session, lambda obs, info: 1.0, max_steps=1)

    def test_gym_boundary_preserves_terminal_observation_and_requires_reset(self):
        import gymnasium as gym

        from reservoir_control.adapters.gymnasium import GymnasiumAdapter

        class Session:
            def reset(self, *, seed=None, options=None):
                return EpisodeStart(np.array([1.0], np.float32))

            def step(self, action):
                return EpisodeStep(np.array([9.0], np.float32), -2.0, False, True, {"actual": 9.0})

            def close(self):
                pass

        env = GymnasiumAdapter(
            Session(),
            action_space=gym.spaces.Box(0.0, 1.0, (1,)),
            observation_space=gym.spaces.Box(0.0, 10.0, (1,)),
        )
        with self.assertRaises(RuntimeError):
            env.step(np.array([0.0]))
        env.reset(seed=3)
        obs, reward, term, trunc, info = env.step(np.array([0.0]))
        self.assertEqual(
            (obs[0], reward, term, trunc, info["actual"]), (9.0, -2.0, False, True, 9.0)
        )
        with self.assertRaises(RuntimeError):
            env.step(np.array([0.0]))
        env.reset(seed=3)
        env.close()
