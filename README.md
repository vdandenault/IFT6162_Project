# Reservoir control: Lake Como and Lake Ontario–Ottawa

IFT6162 project: compare Plan 2014, MPC and Stackelberg MPC for Lake Ontario–
St. Lawrence releases with an aggregated Ottawa reservoir operator. PPO is an
additional comparison. Lake Como remains a supported research and regression case.

All application code lives under `src/reservoir_control/`. Shared PPO, SAC and
FQI implementations live in `controllers/`. Each system owns its physics,
observations, operational policies and study runners under `systems/`.
Planning and Gymnasium adapters connect those systems to the controllers.
The MPC search lives in `controllers/mpc/receding_horizon.py`. Ontario's planning
adapter lives in `systems/loslr/planning.py` and supplies its calendar, release
rules, candidate actions, stage constraints and objective.

```text
reservoir_control/
  controllers/       mpc/, ppo/, sac/, fqi/
  core/              time, state and information contracts
  adapters/          planning and Gymnasium interfaces
  evaluation/        rollout and source verification
  systems/
    lake_como/       Como model, operational policies and studies
    loslr/           Ontario–St. Lawrence model and studies
    ottawa/          aggregate Ottawa storage model
```

Lake Como console commands retain their `lake-como-` names. Python module commands
use the new package, for example
`python -m reservoir_control.systems.lake_como.simulation.mapped_ppo_duration`.

## Current status

The package migration is complete. The Ontario native reference and causal
rule-curve replays have run, and the first four-quarter-month MPC comparison
covers 1995, 2017 and 2019. MPC matches the rule curve in two cases and makes a
small upstream/downstream tradeoff in 2017. Follow-up diagnostics found conflicting
one-step forecast caps at 32 of 37 fallback states. Removing the tracking penalties
preserved every tested release, and perfect inflows reduced downstream peaks while
increasing downstream total exceedance. The Ottawa aggregate balance is
implemented. Follower optimization, Stackelberg MPC and Ontario PPO are upcoming.
See [RESULTS.md](RESULTS.md) for measured outcomes and [PROJECT_PLAN.md](PROJECT_PLAN.md)
for the next implementation phases.

## Start here

- [Project plan](PROJECT_PLAN.md): current scope, completed work and next phases.
- [Results](RESULTS.md): Ontario pilot and Como reference outcomes.
- [Architecture and model details](docs/MULTI_SYSTEM_PLAN.md): Adapter pattern and system specifications.
- [Ottawa model specification](docs/OTTAWA_MODEL_SPEC.md): available evidence, aggregate storage and coupling assumptions.
- [PPO findings](docs/PPO.md): selected settings and consolidated experimental evidence.
- [Evaluation](docs/EVALUATION.md): run the comparison and interpret metrics.
- [Simulator development](docs/SIMULATOR.md): continuous/subdaily research gym, physical checks, scenario uncertainty, routing and explicit training protocols.
- [Literature review](docs/LITERATURE_REVIEW.md): references and MPC/FQI decisions.

## Dataset locations

| Data | Location |
|---|---|
| Como benchmark inflow, demand, levels and environmental flow | `external/gym4real/gym4real/data/dam/` |
| Como downloaded calibration sources | `results/simulator/calibration_research/sources/` |
| Ontario historical supplies and supplied forecasts | `external/loslr_regulation_optimization/input/historic/1900_2020/` |
| Generated trajectories, checkpoints and summaries | `results/` |

The Ontario reference reads `12month_sqAR.txt`. Its SLON series represents the
non-Ontario contribution used in routing. A separate Ottawa reservoir dataset
and controlled/uncontrolled decomposition still need to be established.

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

## Historical Como studies

Frozen manifests retain the original source paths and hashes. The pre-migration
source, workspace dependency and environment files are archived locally under
`results/migration/como_before_multisystem_v1/`. To inspect or reproduce that
source version, restore it into an empty directory:

```bash
python scripts/migration/preserve_como.py --restore-to /tmp/lake-como-reference
```

Generated results and checkpoints remain in their existing directories. Studies
with absolute manifest paths require the original checkout location or explicit
path mapping in a copied study. Current source verification reports a historical
source-tree mismatch when an old manifest references the removed package.

## Ontario baseline and architecture figures

The native runner reproduces the pinned upstream Python implementation. The
causal runner uses past-only inputs for the Python rule curve and Bv7 limits
and reports differences against the archived GLRRM baseline. See the
[migration status](docs/MULTI_SYSTEM_PLAN.md#implementation-status).

The upstream checkout is currently local and ignored, pending the licensing audit;
it is not a registered submodule. It is already present in this workspace. For a
fresh checkout, these commands fetch only the required upstream directories:

```bash
git clone --filter=blob:none --no-checkout --single-branch --branch dps \
  https://github.com/ksemmendinger/loslr_regulation_optimization.git \
  external/loslr_regulation_optimization
git -C external/loslr_regulation_optimization fetch --depth 1 origin \
  8b8de9e89487552f8590f4d42d87484e5edc6872
git -C external/loslr_regulation_optimization sparse-checkout set \
  functions objectiveFunctions input/historic/1900_2020 \
  output/data/baseline/Plan2014_GLRRM
git -C external/loslr_regulation_optimization checkout --detach \
  8b8de9e89487552f8590f4d42d87484e5edc6872
```

```bash
# Existing output directories are protected; use a fresh path for another replay.
uv run python -m reservoir_control.systems.loslr.baseline \
  --output results/loslr/native_baseline_review
uv run python -m reservoir_control.systems.loslr.causal \
  --output results/loslr/causal_baseline_review
uv run python -m reservoir_control.systems.loslr.mpc \
  --output results/loslr/mpc_review --years 1995 2017 2019
uv run python -m reservoir_control.systems.loslr.mpc_diagnostics \
  --output results/loslr/mpc_diagnostics_review
uv run python scripts/analysis/multisystem_overview.py
```

The figures are PNGs under ignored `background/images/`: physical coupling,
independent MPC/PPO adapter paths, and the 2017/2019 native baseline trajectories
(when the `native_baseline_v1` trace is available). No new training is launched.

## Score Ontario controllers

Apply the same declared flood-exposure score to every controller trajectory:

```bash
uv run python -m reservoir_control.systems.loslr.scoring \
  --output results/loslr/controller_scores_review
```

The default input is the saved MPC/rule-curve pilot. Repeat `--input path/to/run.csv`
to include additional controller outputs. Each CSV needs a `controller`, `arm` or
`policy` column plus `year`, `quarter_month`, `ontario_end_m`, `pointe_claire_m` and
`release_m3_s`. Comparisons require the same complete 48-period years. Use matched
initial states and information when generating controller runs.

`configs/studies/loslr/scoring.json` declares thresholds, weights and level scales.
The initial score is the negative sum of Ontario and Pointe-Claire exceedance,
with equal weights and 1 m scales. Higher scores are better. The output keeps
annual scores, both physical exceedance components, peaks, counts and terminal
Ontario level. The same period-cost function can supply an Ontario PPO reward.

## Ontario visual walkthrough

Open [the Ontario playback](background/ontario_simulator.html) locally to explore
1995, 2017 and 2019. Select the causal rule curve, MPC or the labelled perfect-inflow
diagnostic. Step through quarter-months to see Ontario storage levels, executed
release, SLON contribution and Pointe-Claire stage. The charts overlay the
controller's four-period forecasts on the recorded outcomes.

The page includes [the Ontario MPC diagram](background/images/ontario_mpc_controller.png)
and [the planned Ontario PPO diagram](background/images/ontario_ppo_controller.png).
The PPO application still needs its Ontario observations, reward and training
protocol. The main controller programme remains rule curve, MPC, Stackelberg MPC
and PPO. Shared FQI and SAC implementations are available for optional later work.

Rebuild the local HTML and both PNGs from saved experiment traces with:

```bash
uv run python scripts/analysis/ontario_visualization.py
```

For freshly generated review runs, pass their paths explicitly:

```bash
uv run python scripts/analysis/ontario_visualization.py \
  --comparison results/loslr/mpc_review/comparison.csv \
  --diagnostics results/loslr/mpc_diagnostics_review
```

The generated HTML and PNGs live under ignored `background/`. The builder and
HTML template live under `scripts/analysis/`.

## Validation and Como references

```bash
uv run --group reproduction --with pytest python -m pytest tests -q
```

The improved Como simulator uses continuous two-year episodes and has its own
matched MPC/PPO studies. The earlier DamEnv benchmark uses annual episodes.
[Simulator development](docs/SIMULATOR.md) documents the improved model and
[current results](RESULTS.md#lake-como) records its comparison.

<details>
<summary>Earlier Como annual benchmark commands and analysis scripts</summary>

### Como annual benchmark commands

```bash
uv run --group reproduction lake-como-audit
uv run --group reproduction lake-como-rollout --split validation --years 1990
uv run --group reproduction lake-como-compare
uv run --group reproduction --with pytest python -m pytest tests -q
```

The comparison defaults to Random, Mean, Max, EAD, and all five selected 1M PPO
checkpoints on 1988–1997 validation years. It loads local checkpoints; it performs
no training or test evaluation. [Evaluation options](docs/EVALUATION.md) cover
controller subsets, other splits, and explicit checkpoint paths. Existing nonempty
comparison output directories are rejected; choose a new `--output` for another run.
PPO reproduction/training commands are listed together in [PPO findings](docs/PPO.md#reproduce-or-inspect-details).

### Como MPC prototype

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

### Como analysis scripts

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

</details>

## Configuration and local artifacts

- `configs/studies/loslr/native_baseline.json` specifies the native Ontario recipe.
- `src/reservoir_control/systems/loslr/planning.py` contains the current Ontario MPC settings, study caps
  and objective. The runner accepts selected years and a fresh output directory.
- `configs/splits.json`, `configs/ppo.json` and `configs/mpc.json` define the earlier
  Como annual benchmark and its selected checkpoints.
- `configs/experiments/` retains recorded Como study protocols and plans.

Ontario data uses 48 quarter-months per year and the pinned balance conversion.
The current causal replay scores 1901–2020 after initialization and warmup. MPC
case studies use common year-start states and preserve within-year storage.
Como's annual and improved two-year protocols are described in their study docs.

Generated `results/`, `reports/`, `background/`, PDFs and packaging metadata are
ignored. Explanatory graphics use PNG. The Gym4ReaL fork is pinned under
`external/gym4real/`. The Ontario upstream checkout is currently local and ignored.

## Team

- Nadine Mohamed
- Daniel Martineau
- Vincent Dandenault
