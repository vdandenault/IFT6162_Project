"""Coupled model conservation, causal timing, QP optimality and PPO boundaries."""
from dataclasses import replace
import unittest
import numpy as np
from reservoir_control.controllers.mpc.convex_response import solve_response
from reservoir_control.systems.ottawa.follower import OttawaFollower
from reservoir_control.systems.loslr.coupled import (
    load_protocol, f_limit_target, operating_cost, start_states, CoupledSession, DURATION,
)
from reservoir_control.systems.loslr.model import OntarioModel
from reservoir_control.systems.loslr.native import NativeReference
from reservoir_control.systems.loslr.data import load_inputs


class ResponseTests(unittest.TestCase):
    def test_f_tiers_and_shared_objective(self):
        for x, expected in ((75.29, 22.1), (75.3, 22.2), (75.37, 22.33),
                            (75.5, 22.4), (75.6, 22.48)):
            self.assertEqual(f_limit_target(x), expected)
        c = load_protocol()
        cost, terms = operating_cost(75.5, 75.7, 22.6, 8000, 7000, c)
        self.assertAlmostEqual(cost, .2**2+.2**2+.01)
        self.assertAlmostEqual(cost, sum(terms.values()))

    def test_qp_active_constraint_and_sensitivity(self):
        # min (x-2)^2 with x <= 1; derivative dx/db = 1.
        r = solve_response([[2.]], [-4.], [[1.]], [1.], [0.])
        self.assertAlmostEqual(r.x[0], 1.)
        self.assertLess(r.stationarity_residual, 1e-7)
        derivative = r.sensitivity([[2.]], [[0.]], [[1.]], [[1.]])
        self.assertAlmostEqual(derivative[0, 0], 1.)
        with self.assertRaises(ValueError): solve_response([[-1.]], [0.], [[1.]], [1.], [0.])
        with self.assertRaises(RuntimeError):
            solve_response([[2.]], [0.], [[1.], [-1.]], [0., -1.], [0.])

    def test_response_conservation_and_announcement_dependence(self):
        c = load_protocol(); follower = OttawaFollower(c)
        args = (3e9, np.full(4, 1500.), np.full(4, 2000.))
        low = follower.solve(*args, np.full(4, 6500.), np.full(4, 10000.), DURATION)
        high = follower.solve(*args, np.full(4, 9000.), np.full(4, 10000.), DURATION)
        self.assertLess(high.release_m3_s.sum(), low.release_m3_s.sum())
        expected = 3e9+np.cumsum((1500-high.release_m3_s)*DURATION)
        np.testing.assert_allclose(high.storage_m3, expected, atol=1e-5)
        self.assertTrue((high.storage_m3 >= -1e-5).all())
        self.assertLess(high.response.stationarity_residual, 2e-4)
        # Independently perturb announcement and compare with local active-set KKT.
        perturb = np.full(4, 9000.); perturb[0] += .1
        changed = follower.solve(*args, perturb, np.full(4, 10000.), DURATION)
        np.testing.assert_allclose((changed.release_m3_s-high.release_m3_s)/.1,
                                   high.announcement_derivative[:, 0], atol=.003)


class CoupledIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_protocol(); cls.model = OntarioModel(NativeReference())
        cls.frame = load_inputs(cls.model.reference, last_year=1902)
        cls.states = start_states(cls.model, cls.frame, [1901])

    def test_realized_future_does_not_change_decision_or_response(self):
        a = CoupledSession(self.model, self.frame, self.states[1901], self.c)
        changed = self.frame.copy()
        index = self.states[1901][1]
        changed.loc[index:, 'stlouisontOut'] += 1000
        changed.loc[index:, 'ontNTS'] += 1000
        b = CoupledSession(self.model, changed, self.states[1901], self.c)
        for sim in (a, b):
            decision, time = sim.information()
            sim.test_plan = sim.adapter.plan(sim.session.state, sim.storage_m3, decision, time,
                                             anticipating=True)
        self.assertEqual(a.test_plan.requests, b.test_plan.requests)
        np.testing.assert_array_equal(a.test_plan.follower.release_m3_s,
                                      b.test_plan.follower.release_m3_s)
        row = a.step(a.test_plan.requests)
        self.assertLess(abs(row['ottawa_balance_error_m3']), 1e-5)
        self.assertEqual(row['reward'], -row['cost'])
        self.assertAlmostEqual(row['release_m3_s'], a.test_plan.predicted_releases[0])

    def test_gym_complete_annual_boundary_and_reward(self):
        from reservoir_control.systems.loslr.gym_env import OntarioOttawaEnv
        env = OntarioOttawaEnv(self.model, self.frame, self.states, self.c, [1901])
        obs, _ = env.reset(seed=123)
        self.assertEqual(obs.shape, (64,))
        np.testing.assert_allclose(obs[13:61],
            np.asarray(env.simulation.session.state.mean_level_history_m)-74.8, atol=1e-6)
        for i in range(48):
            obs, reward, term, trunc, info = env.step(np.array([0.], np.float32))
            self.assertEqual(term, i == 47)
            self.assertFalse(trunc)
            self.assertEqual(reward, -info['cost'])
        self.assertAlmostEqual(float(obs[0]), info['ontario_end_m']-74.8, places=5)
        with self.assertRaises(RuntimeError): env.step(np.array([0.]))

    def test_final_data_year_preserves_terminal_observation(self):
        from reservoir_control.systems.loslr.gym_env import OntarioOttawaEnv
        states = start_states(self.model, self.frame, [1902])
        env = OntarioOttawaEnv(self.model, self.frame, states, self.c, [1902])
        env.reset(seed=123)
        for _ in range(48):
            obs, _, term, _, info = env.step(np.array([0.], np.float32))
        self.assertTrue(term)
        self.assertTrue(np.isfinite(obs).all())
        self.assertAlmostEqual(float(obs[0]), info['ontario_end_m']-74.8, places=5)


if __name__ == '__main__': unittest.main()
