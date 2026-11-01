"""Audit saved MPC boundary failures and short-horizon optimization.

Run: uv run --group reproduction python scripts/analysis/diagnose_mpc.py
This replays recorded states and scores hypothetical plans with the original
training-only forecasts. It does not change the controller, run new closed-loop
episodes, or supply actual future inflows to the optimizer.
"""
from lake_como.mpc_diagnostics import main


if __name__ == "__main__":
    main()
