"""Inspect the first fixed MPC prototype; reads saved validation results only.

Run: uv run --group reproduction python scripts/analysis/mpc_validation.py
Generate inputs with lake-como-compare --controllers ead max mpc ppo
--output results/comparison/mpc_validation_v1.
"""
import json

import pandas as pd
from lake_como.paths import PROJECT_ROOT


def main():
    root = PROJECT_ROOT / "results/comparison/mpc_validation_v1"
    metadata = json.loads((root / "run.json").read_text())
    assert metadata["complete"] and metadata["split"] == "validation"
    assert not metadata["test_evaluated"]
    annual = pd.read_csv(root / "annual.csv")
    diagnostics = pd.read_csv(root / "mpc_diagnostics.csv")
    summary = pd.read_csv(root / "summary.csv")
    assert len(diagnostics) == 3650
    assert diagnostics.groupby("year").size().eq(365).all()
    assert annual.groupby("controller").year.nunique().eq(10).all()
    print(
        summary[
            [
                "family",
                "mean_full_return",
                "mean_objective_return",
                "mean_flood_days",
                "mean_low_water_days",
                "mean_deficit_million_m3",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )
    print(
        diagnostics.groupby("year")
        .agg(
            fallback_days=("used_fallback", "sum"),
            solver_failures=("solver_success", lambda s: int((~s).sum())),
            inflow_mae_m3_s=("inflow_error_m3_s", lambda s: s.abs().mean()),
        )
        .round(2)
        .to_string()
    )
    print(pd.read_csv(root / "timings.csv").round(2).to_string(index=False))
    print("Plot:", root / "annual_comparison.png")
    print("Plot:", root / "paired_ppo_vs_rules.png")
    # A valid implementation is not yet a credible competitive engineering baseline.
    # This initial surrogate/climatology/horizon combination performs poorly.
    # Separate forecast, horizon/terminal, surrogate, and optimization effects
    # before attributing the gap to the MPC method itself.


if __name__ == "__main__":
    main()
