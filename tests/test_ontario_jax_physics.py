import numpy as np
import pytest

jax = pytest.importorskip("jax")

from bilevel_reservoirs.simulators.physics import (
    CONVERSION,
    DATUM,
    FLOW_SCALE,
    PREPROJECT_SLOPE,
    checked_lake_step,
    dlake_step_dh,
    dlake_step_du,
    lake_step,
)


def test_lake_step_conserves_water_and_has_consistent_gradients():
    level, adjustment, supply = 74.55, 40.0, 7000.0
    following = checked_lake_step(level, adjustment, supply)
    head = (level + following) / 2 - DATUM
    total_release = FLOW_SCALE * (PREPROJECT_SLOPE * head**1.5 + adjustment)
    assert FLOW_SCALE * CONVERSION * (following - level) == pytest.approx(
        supply - total_release, abs=1e-6
    )

    delta = 1e-3
    grad_u = (
        checked_lake_step(level, adjustment + delta, supply)
        - checked_lake_step(level, adjustment - delta, supply)
    ) / (2 * delta)
    grad_h = (
        checked_lake_step(level + delta, adjustment, supply)
        - checked_lake_step(level - delta, adjustment, supply)
    ) / (2 * delta)
    with jax.experimental.enable_x64():
        assert float(dlake_step_du(level, adjustment, supply)) == pytest.approx(grad_u, abs=1e-8)
        assert float(dlake_step_dh(level, adjustment, supply)) == pytest.approx(grad_h, abs=1e-8)
        levels = jax.numpy.array([level, level + 0.5])
        batch = jax.vmap(lake_step, in_axes=(0, None, None))(levels, adjustment, supply)
    np.testing.assert_allclose(
        batch, [checked_lake_step(float(h), adjustment, supply) for h in levels], atol=1e-10, rtol=0
    )
