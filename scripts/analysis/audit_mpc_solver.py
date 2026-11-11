"""Replay saved causal MPC decisions and audit discarded iterates and action slices."""
from reservoir_control.systems.lake_como.mpc_solver_audit import main

if __name__ == "__main__":
    main()
