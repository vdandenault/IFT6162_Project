# Lake Como reservoir control

IFT6162 project: compare a literature-based MPC controller with PPO for daily
Lake Como releases using Gym4ReaL's DamEnv. Demand, flood, and low-water control
are the core scope; hydropower is optional.

## Start here

- [Project plan](PROJECT_PLAN.md): model, scope, current status, and remaining work.
- [PPO findings](docs/PPO.md): selected settings and consolidated experimental evidence.
- [Evaluation](docs/EVALUATION.md): run the comparison and interpret metrics.
- [Simulator development](docs/SIMULATOR.md): continuous/subdaily research gym, physical checks, scenario uncertainty, routing and explicit training protocols.
- [Literature review](docs/LITERATURE_REVIEW.md): references and MPC/FQI decisions.

For a visual introduction, run [the environment walkthrough](scripts/analysis/environment_and_data.py).
For current results, run [the controller comparison](scripts/analysis/controller_comparison.py).
The historical analysis scripts retain detailed experimental evidence; they are optional
reading, linked from the PPO findings.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone
with the pinned simulator fork:

```bash
git clone --recurse-submodules https://github.com/vdandenault/IFT6162_Project.git
cd IFT6162_Project
uv sync --locked --group reproduction
```

For an existing checkout, run `git submodule update --init --recursive` first.
uv creates `.venv` using Python 3.12 and installs the project and Gym4ReaL workspace
in editable mode. Scripts import the installed packages directly; no `sys.path` edits or shell
activation are needed. `uv.lock` pins dependencies. Use `uv add` for changes
and review both the project file and lockfile.

## Common commands

```bash
uv run --group reproduction lake-como-audit
uv run --group reproduction lake-como-rollout --split validation --years 1990
uv run --group reproduction lake-como-compare
uv run --group reproduction python -m unittest discover -s tests -v
```

The comparison defaults to Random, Mean, Max, EAD, and all five selected 1M PPO
checkpoints on 1988–1997 validation years. It loads local checkpoints; it performs
no training or test evaluation. [Evaluation options](docs/EVALUATION.md) cover
controller subsets, other splits, and explicit checkpoint paths. Existing nonempty
comparison output directories are rejected; choose a new `--output` for another run.
PPO reproduction/training commands are listed together in [PPO findings](docs/PPO.md#reproduce-or-inspect-details).

## Start the MPC prototype

```bash
uv run --group reproduction lake-como-mpc
```

This runs the first validation year with the seven-day deterministic controller in
`configs/mpc.json`, saving an annual rollout PNG and solver diagnostics. Use a new
`--output` if the default run already exists. [Controller design](docs/LITERATURE_REVIEW.md#implemented-deterministic-mpc-prototype)
and [evaluation](docs/EVALUATION.md#mpc-prototype) explain the forecast, surrogate
objective, limitations, and full validation command. The prototype runs correctly
but performs poorly; it needs investigation before serving as the engineering benchmark.
Run `uv run --group reproduction python scripts/analysis/diagnose_mpc.py` for
[the boundary and optimizer audit](docs/EVALUATION.md#boundary-and-optimizer-diagnosis).
The separate [perfect-inflow hindsight diagnostic](docs/EVALUATION.md#perfect-inflow-hindsight-diagnostic)
runs with `uv run --group reproduction python scripts/analysis/perfect_inflow_mpc.py`;
it intentionally supplies privileged inflow information without changing deployable MPC.
[The paired objective audit](docs/EVALUATION.md#binary-versus-surrogate-objective-on-common-grids)
uses `uv run --group reproduction python scripts/analysis/compare_mpc_objectives.py`
to compare penalty choices on common hindsight grids without new annual rollouts.
The [closed-loop objective comparison](docs/EVALUATION.md#closed-loop-objective-comparison-with-common-search)
runs via `uv run --group reproduction python scripts/analysis/closed_loop_mpc_objectives.py`;
it uses a separate common one-day search, not deployed seven-day MPC. Add
`--inflow-source seasonal` for the training-only forecast comparison, including
flood/low-water severity and the saved hindsight reference.

The causal [forecast comparison](docs/EVALUATION.md#causal-past-inflow-forecast-comparison)
runs with `uv run --group reproduction python scripts/analysis/compare_mpc_forecasts.py`.
It compares climatology, persistence, and a training-fitted seasonal anomaly forecast
under identical one-day searches, using only completed-day inflow estimates.

To repeat the [seven-day causal MPC forecast ablation](docs/EVALUATION.md#causal-forecasts-in-seven-day-mpc):
`uv run --group reproduction python scripts/analysis/compare_causal_mpc.py`.
This preserves the original MPC costs and solver.

The [solver-failure audit](docs/EVALUATION.md#audit-of-additional-causal-mpc-solver-failures)
runs with `uv run --group reproduction python scripts/analysis/audit_mpc_solver.py`
and replays saved decisions without changing the controller.

The paired [iteration-limit retention experiment](docs/EVALUATION.md#cost-checked-retention-of-iteration-limit-iterates)
runs with `uv run --group reproduction python scripts/analysis/compare_mpc_retention.py`.
It preserves solver failure flags and keeps the original controller unchanged.

The [matched MPC horizon study](docs/EVALUATION.md#matched-mpc-horizon-experiment)
runs with `uv run --group reproduction python scripts/analysis/compare_mpc_horizons.py`.
It compares 1, 3, and 7 days under the same persistence forecast and optimizer.

The [PPO past-inflow observation study](docs/PPO.md#past-inflow-observation-ablation)
runs with `uv run --group reproduction python scripts/analysis/compare_ppo_inflow.py`.
It trains five 1M-step policies and compares against verified four-feature controls.

The [wet/dry decision analysis](docs/PPO.md#what-changes-in-wet-and-dry-years)
runs with `uv run --group reproduction python scripts/analysis/analyze_ppo_inflow.py`
on frozen PPO trajectories, with years selected only by total inflow.

The [flood feasibility audit](docs/EVALUATION.md#flood-safety-feasibility-audit)
runs with `uv run --group reproduction python scripts/analysis/audit_flood_feasibility.py`.
It checks saved states without changing any controller or claiming a safety guarantee.

The [common safety-filter comparison](docs/EVALUATION.md#common-finite-scenario-safety-filter)
runs with `uv run --group reproduction python scripts/analysis/compare_safety_filter.py`.
Frozen reference hashes and the plain/filtered comparison tracks are recorded in
`configs/experiments/control_protocol_v1.json`; test years remain reserved.

The [first FQI pilot](docs/EVALUATION.md#first-tree-based-fqi-pilot) uses
`configs/fqi.json` and runs without changing the historical lock:

```bash
uv run --group reproduction --with scikit-learn==1.5.2 \
  --with numpy==2.2.4 --with scipy==1.15.3 \
  python scripts/analysis/run_fqi_pilot.py
```

## Readable analysis scripts

```bash
uv run --group reproduction python scripts/analysis/environment_and_data.py
uv run --group reproduction python scripts/analysis/controller_comparison.py
```

The walkthrough runs short training-year diagnostics and saves PNGs under
`results/analysis/environment_and_data/`. The other analysis scripts read saved
experiment results, print tables/checks, and print paths to existing PNG plots.
They do not launch training or new policy evaluations. If results are missing,
run the corresponding experiment command listed in [PPO findings](docs/PPO.md).
Scripts retain the explanations as comments and run top-to-bottom through `main()`.
Generated outputs stay ignored; original notebook files were archived locally in
`reports/archive/notebooks_before_scripts/`.

## Configuration and local artifacts

- `configs/splits.json`: train 1946–1987, validation 1988–1997, test 1998–2010.
- `configs/ppo.json`: current selected PPO settings and default checkpoint locations.
- `configs/mpc.json`: initial deterministic MPC horizon, solver, and surrogate settings.
- `configs/experiments/ppo_studies.json`: exact historical experiment plans, grouped by study.

Use validation for design decisions; reserve test years for the final comparison.
Leap years currently retain 365-step episodes, annual storage resets to level
0.35 m, and terminal level observations are corrected. See the plan for limitations.
`lake-como-rollout --no-include-leap-years` supports sensitivity checks;
`--protocol upstream` selects the separate original benchmark membership.

Generated `results/`, `reports/`, `background/`, and local PDF papers are ignored.
Only PNG graphics are needed. Our [Gym4ReaL fork](https://github.com/vdandenault/gym4ReaL)
is pinned at `external/gym4real/`; keep simulator changes in that fork and project
experiments here. Its existing packaging change is separate from simulator physics.
Review and commit fork changes separately before updating the parent submodule pin.

## Team

- Nadine Mohamed
- Daniel Martineau
- Vincent Dandenault
