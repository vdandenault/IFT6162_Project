"""Run a validation-only, perfect-inflow hindsight diagnostic.

uv run --group reproduction python scripts/analysis/perfect_inflow_mpc.py
This explicitly provides privileged future inflows only in the separate
experiment. The deployable MPC factory and its settings are untouched.
"""
from reservoir_control.systems.lake_como.mpc_hindsight import main


if __name__ == "__main__":
    main()
