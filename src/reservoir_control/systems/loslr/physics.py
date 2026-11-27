"""Differentiable Lake Ontario transition for adjustment-based planning.

Levels are in metres, net total supply is in m³/s, and the adjustment is
in LOSLR native units (one unit equals 10 m³/s).
"""

from math import isfinite

import jax
import jax.numpy as jnp
from jax import lax

DATUM = 69.474 + 0.0014 * (2010 - 1985)
FLOW_SCALE = 10
CONVERSION = 2970
PREPROJECT_SLOPE = 55.5823


@jax.jit
def lake_step(h_t, u_t, Q_NTS):
    """Return the next lake level, or NaN when no hydraulic-head root exists."""
    area = FLOW_SCALE * PREPROJECT_SLOPE
    volume_factor = FLOW_SCALE * CONVERSION
    a = 2 * volume_factor / area
    b = (Q_NTS - FLOW_SCALE * u_t + 2 * volume_factor * (h_t - DATUM)) / area
    argument = 27 * b / (2 * a**3) - 1

    def trig(arg):
        return (a / 3) * (2 * jnp.cos(jnp.arccos(jnp.clip(arg, -1.0, 1.0)) / 3) - 1)

    def hyperbolic(arg):
        return (a / 3) * (2 * jnp.cosh(jnp.arccosh(jnp.maximum(arg, 1.0)) / 3) - 1)

    root = lax.cond(argument <= 1.0, trig, hyperbolic, argument)
    return jnp.where(argument < -1.0, jnp.nan, 2 * (DATUM + root**2) - h_t)


dlake_step_du = jax.jit(jax.grad(lake_step, argnums=1))
dlake_step_dh = jax.jit(jax.grad(lake_step, argnums=0))


def checked_lake_step(h_t: float, u_t: float, Q_NTS: float) -> float:
    """Run the JAX kernel in double precision at the scalar simulator boundary."""
    with jax.experimental.enable_x64():
        level = float(lake_step(h_t, u_t, Q_NTS))
    if not isfinite(level):
        raise ValueError("No finite nonnegative hydraulic-head root")
    return level
