"""Compare training-only climatology and causal past-inflow forecasts.

Run: uv run --group reproduction python scripts/analysis/compare_mpc_forecasts.py
Outputs remain ignored under results/mpc/forecast_study_v1/.
"""
from reservoir_control.systems.lake_como.mpc_forecast_study import main

if __name__ == "__main__":
    main()
