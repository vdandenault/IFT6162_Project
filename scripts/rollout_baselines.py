"""Compatibility entry point; prefer lake-como-rollout."""
from reservoir_control.systems.lake_como.baselines import main

if __name__ == "__main__":
    raise SystemExit(main())
