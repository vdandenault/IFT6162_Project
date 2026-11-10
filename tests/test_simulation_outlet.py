import unittest
from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env
from lake_como.environment import settings
from lake_como.simulation import make_operational_env
from lake_como.simulation.outlet import (
    OutletSpec, OutletLake, OutletDomainError, make_outlet_env,
)


class OutletChecks(unittest.TestCase):
    def setUp(self):
        self.params = settings('train')['lake_params']

    def test_reference_is_bookkeeping_not_no_flow_level(self):
        for reference in (-.5, 0., 2.):
            lake = OutletLake(self.params, OutletSpec(storage_reference_m=reference))
            storage = lake.level_to_storage(-.6)
            self.assertLess(storage, 0)
            self.assertAlmostEqual(lake.storage_to_level(storage), -.6)
            self.assertAlmostEqual(lake.actual_release(100., storage, 1), 100.)
        legacy = OutletLake(self.params, OutletSpec(capacity='legacy'))
        self.assertEqual(legacy.actual_release(100, legacy.level_to_storage(-.6), 1), 0)

    def test_only_capacity_changes_and_operating_policy_is_retained(self):
        candidate = OutletLake(self.params)
        legacy = OutletLake(self.params, OutletSpec(capacity='legacy'))
        for level in [-.705, -.6, -.5, -.45, -.4, 0., 1.25, 1.5, 2.64]:
            storage = candidate.level_to_storage(level)
            self.assertEqual(candidate.min_release(storage, 1), legacy.min_release(storage, 1))
            self.assertGreaterEqual(candidate.max_release(storage, 1), 0)
            if level > -.4:
                self.assertEqual(candidate.max_release(storage, 1), legacy.max_release(storage, 1))
        left = candidate.max_release(candidate.level_to_storage(-.50000001), 1)
        right = candidate.max_release(candidate.level_to_storage(-.49999999), 1)
        self.assertLess(abs(left-right), .00001)
        high = candidate.level_to_storage(2.)
        self.assertEqual(candidate.actual_release(0, high, 1), candidate.max_release(high, 1))

    def test_balance_and_integration_convergence(self):
        lake = OutletLake(self.params)
        storage = lake.level_to_storage(-.6)
        errors = []
        reference = lake.integrate_interval(storage, 400, 20, 1, substeps=1536)
        for steps in [24, 48, 96, 192]:
            result = lake.integrate_interval(storage, 400, 20, 1, substeps=steps)
            self.assertLess(abs(result['balance_error_m3']), 1e-5)
            errors.append(abs(result['storage_m3']-reference['storage_m3']))
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))

    def test_domain_is_explicit_and_no_extrapolation(self):
        lake = OutletLake(self.params)
        for level in [-.706, 2.641]:
            with self.assertRaises(OutletDomainError):
                lake.max_release(lake.level_to_storage(level), 1)
        for changes in [dict(area_m2=-1), dict(lower_evaluation_m=-2.5), dict(capacity='unknown')]:
            with self.assertRaises(ValueError):
                replace(OutletSpec(), **changes)
        with self.assertRaises(ValueError):
            lake.integrate_interval(0, -1, 1, 1)

    def test_feasible_minimum_and_capacity_differs_from_nominal_action_bound(self):
        lake = OutletLake(self.params)
        for level in np.linspace(-.705, 2.64, 81):
            storage = lake.level_to_storage(level)
            for day in range(1,366):
                self.assertLessEqual(lake.min_release(storage,day),lake.max_release(storage,day))
        high = lake.level_to_storage(2.)
        self.assertGreater(lake.max_release(high,1), settings('train')['action']['high'])

    def test_finite_sac_updates_on_actual_candidate_transitions(self):
        # Learner plumbing check on a short prescribed-action dataset. This is
        # separate from the free-action study that hits its domain in warmup.
        import torch
        from lake_como.sac import SACAgent, SACConfig, ReplayBuffer
        torch.set_num_threads(1)
        env = make_outlet_env(years=(1946,))
        agent = SACAgent(SACConfig(observation_dim=8),observation_mean=np.zeros(8),
            observation_scale=[1,1,1,100,100,250,1e7,1],
            action_low=env.action_space.low,action_high=env.action_space.high,seed=901)
        replay = ReplayBuffer(64,8,1,seed=901)
        obs,_ = env.reset(seed=901)
        physical = np.array([100],np.float32)
        normalized = 2*(physical-env.action_space.low)/(env.action_space.high-env.action_space.low)-1
        for _ in range(16):
            following,reward,term,trunc,_ = env.step(physical)
            replay.add(obs,normalized,reward,following,term,trunc)
            obs = following
        for _ in range(4):
            diagnostics = agent.update(replay.sample(16))
            self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
            self.assertGreater(diagnostics['critic_gradient_norm'],0)
            self.assertGreater(diagnostics['actor_gradient_norm'],0)
        self.assertEqual(agent.updates,4)
        env.close()

    def test_legacy_opt_in_env_has_exact_benchmark_parity(self):
        original = make_operational_env(years=(1946,), objective='benchmark')
        wrapped = make_outlet_env(years=(1946,), outlet=OutletSpec(capacity='legacy'), objective='benchmark')
        np.testing.assert_array_equal(original.reset()[0], wrapped.reset()[0])
        for day in range(365):
            action = np.array([20+(day*37)%400], np.float32)
            old = original.step(action)
            new = wrapped.step(action)
            np.testing.assert_array_equal(old[0], new[0])
            self.assertEqual(old[1:4], new[1:4])
            self.assertEqual(old[-1]['storage'], new[-1]['storage'])
        original.close(); wrapped.close()

    def test_candidate_gym_contract_and_correct_finite_boundary(self):
        env = make_outlet_env(years=(1946,), objective='benchmark')
        check_env(env, skip_render_check=True)
        env.reset()
        env.hour = env.total_hours-24
        obs, reward, term, trunc, info = env.step(np.array([100], np.float32))
        self.assertTrue(term); self.assertFalse(trunc)
        self.assertEqual(obs[-1], 0)
        self.assertEqual(obs[0], np.float32(env.model.lake.storage_to_level(env.state.storage)))
        self.assertTrue(np.isfinite(reward)); self.assertFalse(info['outlet_extrapolated'])
        with self.assertRaises(RuntimeError): env.step(np.array([100], np.float32))
        env.close()


if __name__ == '__main__':
    unittest.main()
