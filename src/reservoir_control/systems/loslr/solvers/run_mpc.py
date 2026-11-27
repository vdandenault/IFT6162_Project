from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

from reservoir_control.systems.loslr.operator import AdjustFlowConfig, OntarioConfig
from reservoir_control.systems.loslr.solvers.mpc_beam import MPCBeamSearchCfg
from reservoir_control.systems.loslr.solvers.mpc_mppi import MPCMPPICfg
from reservoir_control.systems.loslr.solvers.mpc_shooting import MPCShootingCfg

SOLVERS = {"shooting": MPCShootingCfg, "beam": MPCBeamSearchCfg, "mppi": MPCMPPICfg}


def main():
    parser = ArgumentParser(
        description="Run the merged Lake Ontario simulator with shooting, beam or MPPI MPC."
    )
    parser.add_argument("--solver", choices=SOLVERS, default="shooting")
    parser.add_argument("--lookahead", type=int, choices=range(1, 5), default=4,
                        help="Prediction window in quarter-months (default: 4)")
    parser.add_argument("--steps", type=int, default=12,
                        help="Number of simulated quarter-months (default: 12)")
    parser.add_argument("--start-step", type=int, default=48,
                        help="Zero-based row in the historical input (default: 48)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.steps < 1 or args.start_step < 48:
        parser.error("--steps must be positive and --start-step must be at least 48")
    if args.output.exists():
        parser.error("--output must be a fresh directory")

    config = SOLVERS[args.solver]().training(
        data_path=args.data_dir,
        start_step=args.start_step,
        prediction_horizon=args.lookahead,
    )
    config.environment(horizon=args.steps, env_config={"mechanism_id": None})
    config.agents(
        OntarioConfig(
            id="ont",
            policy_id="ont",
            mechanisms=AdjustFlowConfig(
                id="flow",
                acts_on=("ont", "flow"),
                action_space=Box(0.0, np.inf, shape=(1,), dtype=np.float64),
            ),
        )
    )
    result = config.build_optimizer().train()
    args.output.mkdir(parents=True, exist_ok=False)
    result["data"].to_csv(args.output / "trajectory.csv", index=False)
    summary = {
        "solver": args.solver,
        "lookahead_quarter_months": args.lookahead,
        "executed_steps": len(result["data"]),
        "start_step": args.start_step,
        "input": str(config.input_path.resolve()),
        "running_cost": result["running_cost"].tolist(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {summary['executed_steps']} Ontario periods to {args.output}")


if __name__ == "__main__":
    main()
