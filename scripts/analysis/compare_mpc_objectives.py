"""Compare level-penalty objectives on identical hindsight action grids.

uv run --group reproduction python scripts/analysis/compare_mpc_objectives.py
This scores shared finite candidate sets at saved validation states. It does
not change deployed MPC, run a local optimizer, or evaluate new annual policies.
"""
from reservoir_control.systems.lake_como.mpc_objective_audit import main


if __name__ == "__main__":
    main()
