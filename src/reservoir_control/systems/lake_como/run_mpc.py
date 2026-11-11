"""Run a deterministic MPC validation smoke test; no fitting on held-out inputs."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from reservoir_control.systems.lake_como.compare import annual_metrics
from reservoir_control.systems.lake_como.environment import settings
from reservoir_control.systems.lake_como.mpc import DEFAULT_CONFIG, DeterministicMPC, MPCConfig, save_diagnostics
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT, DAM_CONFIG, DAM_DATA
from reservoir_control.systems.lake_como.reproduce import evaluate
from reservoir_control.systems.lake_como.splits import split_years


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="validation"
    )
    parser.add_argument(
        "--year", type=int, help="Default: first year of selected split"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    year = args.year if args.year is not None else split_years(args.split)[0]
    if year not in split_years(args.split):
        parser.error("Year is outside the selected chronological split")
    output = args.output or PROJECT_ROOT / f"results/mpc/{args.split}_{year}_v1"
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; use a new --output")
    config = MPCConfig.load(args.config)
    started = time.perf_counter()
    controller = DeterministicMPC.from_training(config)
    setup_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    frame, seconds = evaluate(
        "mpc",
        split=args.split,
        year=year,
        protocol="chronological",
        correct_terminal_level=True,
        sequential_years=True,
        measure_decisions=True,
        controller_factory=lambda env: controller,
    )
    annual = annual_metrics(frame)
    if frame.balance_error_m3.abs().max() >= 1e-4:
        raise RuntimeError("Water balance failed")
    frame.to_csv(output / "trajectories.csv", index=False)
    annual.to_csv(output / "annual.csv", index=False)
    diagnostics = save_diagnostics(controller, frame, output)
    pd.DataFrame(
        dict(
            day=np.arange(1, 366),
            inflow_m3_s=controller.forecast.inflow,
            demand_m3_s=controller.forecast.demand,
        )
    ).to_csv(output / "seasonal_forecast.csv", index=False)
    files = [
        Path(__file__),
        PROJECT_ROOT / "src/reservoir_control/systems/lake_como/mpc.py",
        PROJECT_ROOT / "src/reservoir_control/systems/lake_como/reproduce.py",
        PROJECT_ROOT / "src/reservoir_control/systems/lake_como/environment.py",
        PROJECT_ROOT / "configs/splits.json",
        PROJECT_ROOT / "uv.lock",
        args.config.resolve(),
        *sorted(DAM_CONFIG.glob("*.py")),
        *sorted(DAM_CONFIG.glob("*.yaml")),
        *sorted(DAM_DATA.rglob("*.csv")),
    ]
    metadata = dict(
        complete=True,
        split=args.split,
        year=year,
        test_evaluated=args.split == "test",
        config=asdict(config),
        forecast="Training-only 365-day inflow/demand climatology; no online future inputs",
        training_years=list(controller.forecast.training_years),
        setup_seconds=setup_seconds,
        evaluation_seconds=seconds,
        solver_failure_days=int((~diagnostics.solver_success).sum()),
        fallback_days=int(diagnostics.used_fallback.sum()),
        decision_median_ms=float(frame.decision_ns.median() / 1e6),
        decision_p95_ms=float(frame.decision_ns.quantile(0.95) / 1e6),
        inflow_mae_m3_s=float(diagnostics.inflow_error_m3_s.abs().mean()),
        terminal="No terminal penalty by default; horizon shrinks to annual boundary, annual reset retained",
        objective="Squared level-violation surrogate; other penalties follow forecast DamEnv terms; true reward used for scoring",
        planning="Exact upstream 24-substep hydraulics; forecast error and observation float32 rounding remain",
        versions={p: version(p) for p in ["numpy", "scipy", "gymnasium"]},
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    axes[0].plot(frame.day, frame.level_m, label="Realized simulated level")
    axes[0].plot(
        frame.day,
        diagnostics.predicted_level_m,
        alpha=0.7,
        label="One-day forecast level",
    )
    axes[0].axhline(controller.model.flood_level, color="firebrick", ls="--")
    axes[0].axhline(controller.model.low_level, color="gray", ls="--")
    axes[0].set(ylabel="Level (m)")
    axes[0].legend(fontsize=9)
    axes[1].plot(frame.day, frame.request_m3_s, label="Requested release")
    axes[1].plot(
        frame.day,
        frame.release_m3_s,
        label="Hydraulically feasible actual release",
        alpha=0.7,
    )
    axes[1].set(ylabel="Release (m³/s)")
    axes[1].legend(fontsize=9)
    axes[2].plot(frame.day, frame.inflow_m3_s, label="Realized historical inflow")
    axes[2].plot(
        frame.day,
        diagnostics.forecast_inflow_m3_s,
        label="Training-only seasonal forecast",
    )
    axes[2].set(ylabel="Inflow (m³/s)", xlabel="Day")
    axes[2].legend(fontsize=9)
    fig.suptitle(
        f"Deterministic MPC prototype · {year} · {config.horizon_days}-day horizon"
    )
    fig.savefig(output / "mpc_rollout.png", dpi=170)
    plt.close(fig)
    print(annual.round(2).to_string(index=False))
    print(
        "Fallback days:",
        metadata["fallback_days"],
        "; decision median ms:",
        round(metadata["decision_median_ms"], 2),
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
