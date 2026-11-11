"""Run the common one-day grid, closed-loop objective diagnostic.

uv run --group reproduction python scripts/analysis/closed_loop_mpc_objectives.py
Add --inflow-source seasonal for training-only non-oracle inflows. Both arms
use the identical exhaustive grid. This does not
change the deployed seven-day MPC or access chronological test years.
"""
from reservoir_control.systems.lake_como.mpc_objective_closed_loop import main


if __name__ == "__main__":
    main()
