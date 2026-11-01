"""Validation forecast ablation with original seven-day MPC cost and solver.

Run: uv run --group reproduction python scripts/analysis/compare_causal_mpc.py
"""
from lake_como.mpc_causal import main

if __name__ == "__main__":
    main()
