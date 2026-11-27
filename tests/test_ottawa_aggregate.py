import unittest

from reservoir_control.core.contracts import DecisionTime
from reservoir_control.systems.ottawa.model import (
    AggregateStorageModel,
    OttawaForcing,
    OttawaParameters,
    OttawaState,
)


class OttawaAggregateTests(unittest.TestCase):
    """Synthetic quantities verify accounting; none are Ottawa parameter estimates."""

    def setUp(self):
        self.model = AggregateStorageModel(OttawaParameters(0, 10_000_000, 0, 1000))
        self.time = DecisionTime(0, 2017, 1, 86400)

    def test_balance_and_no_state_mutation(self):
        state = OttawaState(5_000_000)
        result = self.model.transition(state, 80, OttawaForcing(100), self.time)
        self.assertEqual(result.state.storage_m3, 6_728_000)
        self.assertEqual(result.metrics["balance_error_m3"], 0)
        self.assertEqual(result.metrics["release_m3"], 6_912_000)
        self.assertEqual(state.storage_m3, 5_000_000)
        self.assertFalse(result.info["calibrated"])

    def test_subdivision_conserves_volume(self):
        forcing = OttawaForcing(100)
        start = OttawaState(5_000_000)
        full = self.model.transition(start, 80, forcing, self.time)
        half_time = DecisionTime(0, 2017, 1, 43200)
        first = self.model.transition(start, 80, forcing, half_time)
        second = self.model.transition(first.state, 80, forcing, half_time)
        self.assertEqual(full.state, second.state)

    def test_storage_limits_constrain_release_and_no_implicit_clipping(self):
        forcing = OttawaForcing(0)
        start = OttawaState(86400)
        self.assertEqual(self.model.feasible_release_bounds(start, forcing, self.time), (0, 1))
        self.assertEqual(self.model.transition(start, 1, forcing, self.time).state.storage_m3, 0)
        with self.assertRaisesRegex(ValueError, "next-storage"):
            self.model.transition(start, 2, forcing, self.time)

    def test_unavoidable_overflow_is_reported_without_deleting_water(self):
        start = OttawaState(10_000_000)
        forcing = OttawaForcing(1001)
        self.assertIsNone(self.model.feasible_release_bounds(start, forcing, self.time))
        with self.assertRaisesRegex(ValueError, "spill/emergency"):
            self.model.transition(start, 1000, forcing, self.time)

    def test_negative_net_inflow_can_represent_explicit_losses(self):
        result = self.model.transition(OttawaState(5_000_000), 0, OttawaForcing(-1), self.time)
        self.assertEqual(result.state.storage_m3, 4_913_600)

    def test_invalid_parameters_and_inputs(self):
        for args in [(0, 0, 0, 1), (-1, 10, 0, 1), (0, 10, 2, 1), (0, 10, 0, float("inf"))]:
            with self.assertRaises(ValueError):
                OttawaParameters(*args)
        for constructor in [OttawaState, OttawaForcing]:
            with self.assertRaises(ValueError):
                constructor(float("nan"))
        with self.assertRaisesRegex(ValueError, "initial storage"):
            self.model.transition(OttawaState(-1), 0, OttawaForcing(0), self.time)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.transition(OttawaState(1), float("nan"), OttawaForcing(0), self.time)


if __name__ == "__main__":
    unittest.main()
