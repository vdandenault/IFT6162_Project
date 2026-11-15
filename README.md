# Reservoir control: Lake Como and Lake Ontario–Ottawa

Standalone Ottawa hard-follower demos and reproduction commands are documented in
[the Ottawa study guide](configs/studies/ottawa/README.md). This synthetic Step 2
controller is separate from the earlier soft follower and Ontario experiments.

IFT6162 project: compare Plan 2014, MPC and Stackelberg MPC for Lake Ontario–
St. Lawrence releases with an aggregated Ottawa reservoir operator. PPO is an
additional comparison. Lake Como remains a supported research and regression case.

All application code lives under `src/reservoir_control/`. Shared PPO, SAC and
FQI implementations live in `controllers/`. Each system owns its physics,
observations, operational policies and study runners under `systems/`.
Planning and Gymnasium adapters connect those systems to the controllers.
MPC beam search lives in `controllers/mpc/receding_horizon.py`; finite nested
leader search and convex response solving live beside it. Ontario planning
adapters (`systems/loslr/planning.py` and `coupled.py`) supply calendars, release
rules, candidate actions, constraints and objectives. The Gymnasium adapter
(`systems/loslr/gym_env.py`) connects the same coupled physics to the shared PPO
agent. MPC and PPO remain separate controller implementations.

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

The Ontario native reference and causal
rule-curve replays have run, and the first four-quarter-month MPC comparison
covers 1995, 2017 and 2019. MPC matches the rule curve in two cases and makes a
small upstream/downstream tradeoff in 2017. Follow-up diagnostics found conflicting
one-step forecast caps at 32 of 37 fallback states. Removing the tracking penalties
preserved every tested release, and perfect inflows reduced downstream peaks while
increasing downstream total exceedance. A new coupled research pilot also completed:
a convex Ottawa follower, nested Stackelberg MPC and five short Ontario PPO training runs. The coupled study uses
state-dependent F-limit targets and declared synthetic Ottawa parameters. It is
separate from the earlier fixed-cap benchmark.
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
with the pinned simulator submodules:

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

Frozen manifests retain the original source paths and hashes. Historical studies
require the source revision and environment recorded by their manifests; the
current source tree is checked strictly against those hashes. Generated results
and checkpoints remain in their existing directories.

## Baskatong: approximate release-response scenarios

```bash
uv run python -m reservoir_control.systems.ottawa.baskatong_scenario \
  --pilot results/ottawa/baskatong_1970_observed_v1 \
  --output results/ottawa/baskatong_scenario_review
```

This separate offline model demonstrates release → storage → level under three
explicit effective-area assumptions and both discharge interpretations. It uses
assumed daily-center timing and **effective net input**, not measured natural
inflow. Primary-proxy release retention raises modeled level by approximately
2.20 / 1.10 / 0.55 m across the scales; the smallest exceeds the assumed upper
experiment bound. All retained volume is later repaid in this prescribed run.
These are scenario results, not calibration or an optimized controller. See
[the scenario guide](configs/studies/ottawa/README.md). Strict reconstruction,
observations and Ontario controllers are unchanged.

## Baskatong: actual 1970 observations

```bash
uv run python -m reservoir_control.systems.ottawa.historical_pilot \
  --output results/ottawa/baskatong_1970_review
```

The [one-year pilot guide](configs/studies/ottawa/README.md) describes acquisition,
offline replay and source definitions. ECCC records supply 365 paired dates of
Baskatong daily mean level and downstream Mercier discharge; 28 discharges carry
Partial Day flags. The outputs show observed seasonal behavior and normalized
water-level position. They do not establish storage or net inflow: levels are
means in an arbitrary datum, clock/outlet completeness remains unverified, and
no storage curve or usable K is assumed. Ontario controllers remain unchanged.

For the two-station discharge overlay, differences, flags and outlet-definition
review, use `uv run python -m reservoir_control.systems.ottawa.discharge_comparison
--output results/ottawa/baskatong_discharge_comparison_review` (one line).
See [the data guide](configs/studies/ottawa/README.md) for the full command.

## Baskatong historical data pilot

The [measurement and ingestion guide](configs/studies/ottawa/README.md) documents
Baskatong's boundary, source coverage, permissions and daily input schema.
Reproduce the source assessment without downloading restricted observations:

```bash
uv run python -m reservoir_control.systems.ottawa.reconstruction \
  --output results/ottawa/baskatong_assessment_review
```

This currently produces an **assessment**, not a historical reconstruction: no
suitable aligned, licensed 2016–2019 level/total-outlet pair or storage curve was
obtained. The module can process permitted records with explicit timing/datum
contracts; normalized, curve-based and assumed-volume scenarios remain separate.
It does not modify the synthetic Ottawa model or run a controller.

## Coupled Ontario–Ottawa comparison

```bash
uv run --group reproduction python -m reservoir_control.systems.loslr.coupled_study \
  --output results/loslr/coupled_review
```

This runs the rule curve, external-flow MPC, nested bilevel MPC and five PPO seeds
on 1995, 2017 and 2019. The default PPO budget is 4,096 decisions per seed, a
pipeline pilot. A different budget requires a fresh output folder, for example
`--training-decisions 20000`; the actual budget is frozen before training. Use
`--skip-ppo` for a labelled controller-only diagnostic.

The [study protocol](configs/studies/loslr/README.md) explains sourced F-limit
thresholds, research objectives, causal information, the announcement-response
game, splits and parameter assumptions. Plans, source hashes, annual trajectories,
physical metrics, training curves and final checkpoints are saved under the output
folder. Historical SLON is split synthetically; this is not operationally calibrated
Ottawa control. Native Bv7 limits apply to every controller. MPC additionally
searches for forecast stage-cap feasibility and explicitly reports recovery
fallbacks. PPO receives the same stage exposure in its cost; its policy does not
provide an additional hard stage-safety guarantee. No final test years are run.

## Ontario baseline and architecture figures

The native runner reproduces the pinned upstream Python implementation. The
causal runner uses past-only inputs for the Python rule curve and Bv7 limits
and reports differences against the archived GLRRM baseline. See [RESULTS.md](RESULTS.md) for the current benchmark findings.

The upstream simulator is registered as a submodule at
`external/loslr_regulation_optimization/`, pinned to inspected commit
`8b8de9e89487552f8590f4d42d87484e5edc6872`. Initialize it on an existing clone:

```bash
git submodule update --init --recursive
```

Cloning this project with `--recurse-submodules` initializes both simulators.
The Ontario submodule points directly to the authors' repository; local changes
and parent-project commits do not modify that upstream repository.

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
uv run python scripts/visualization/multisystem_overview.py
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
uv run python scripts/visualization/ontario_visualization.py
```

For freshly generated review runs, pass their paths explicitly:

```bash
uv run python scripts/visualization/ontario_visualization.py \
  --comparison results/loslr/mpc_review/comparison.csv \
  --diagnostics results/loslr/mpc_diagnostics_review
```

The generated HTML and PNGs live under ignored `background/`. The builder and
HTML template live under `scripts/visualization/`.

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
Run `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_diagnostics` for
[the boundary and optimizer audit](docs/EVALUATION.md#boundary-and-optimizer-diagnosis).
The separate [perfect-inflow hindsight diagnostic](docs/EVALUATION.md#perfect-inflow-hindsight-diagnostic)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_hindsight`;
it intentionally supplies privileged inflow information without changing deployable MPC.
[The paired objective audit](docs/EVALUATION.md#binary-versus-surrogate-objective-on-common-grids)
uses `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_objective_audit`
to compare penalty choices on common hindsight grids without new annual rollouts.
The [closed-loop objective comparison](docs/EVALUATION.md#closed-loop-objective-comparison-with-common-search)
runs via `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_objective_closed_loop`;
it uses a separate common one-day search, not deployed seven-day MPC. Add
`--inflow-source seasonal` for the training-only forecast comparison, including
flood/low-water severity and the saved hindsight reference.

The causal [forecast comparison](docs/EVALUATION.md#causal-past-inflow-forecast-comparison)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_forecast_study`.
It compares climatology, persistence, and a training-fitted seasonal anomaly forecast
under identical one-day searches, using only completed-day inflow estimates.

To repeat the [seven-day causal MPC forecast ablation](docs/EVALUATION.md#causal-forecasts-in-seven-day-mpc):
`uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_causal`.
This preserves the original MPC costs and solver.

The [solver-failure audit](docs/EVALUATION.md#audit-of-additional-causal-mpc-solver-failures)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_solver_audit`
and replays saved decisions without changing the controller.

The paired [iteration-limit retention experiment](docs/EVALUATION.md#cost-checked-retention-of-iteration-limit-iterates)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_retention_study`.
It preserves solver failure flags and keeps the original controller unchanged.

The [matched MPC horizon study](docs/EVALUATION.md#matched-mpc-horizon-experiment)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.mpc_horizon_study`.
It compares 1, 3, and 7 days under the same persistence forecast and optimizer.

The [PPO past-inflow observation study](docs/PPO.md#past-inflow-observation-ablation)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.ppo_inflow_study`.
It trains five 1M-step policies and compares against verified four-feature controls.

The [wet/dry decision analysis](docs/PPO.md#what-changes-in-wet-and-dry-years)
runs with `uv run --group reproduction python scripts/lake_como/analyze_ppo_inflow.py`
on frozen PPO trajectories, with years selected only by total inflow.

The [flood feasibility audit](docs/EVALUATION.md#flood-safety-feasibility-audit)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.flood_feasibility`.
It checks saved states without changing any controller or claiming a safety guarantee.

The [common safety-filter comparison](docs/EVALUATION.md#common-finite-scenario-safety-filter)
runs with `uv run --group reproduction python -m reservoir_control.systems.lake_como.safety_study`.
Frozen reference hashes and the plain/filtered comparison tracks are recorded in
`configs/experiments/control_protocol_v1.json`; test years remain reserved.

The [first FQI pilot](docs/EVALUATION.md#first-tree-based-fqi-pilot) uses
`configs/fqi.json` and runs without changing the historical lock:

```bash
uv run --group reproduction --with scikit-learn==1.5.2 \
  --with numpy==2.2.4 --with scipy==1.15.3 \
  python -m reservoir_control.systems.lake_como.fqi_study
```

### Como analysis scripts

See [the scripts guide](scripts/README.md) for the remaining utilities and their purpose.

```bash
uv run --group reproduction python scripts/lake_como/environment_and_data.py
uv run --group reproduction python scripts/lake_como/controller_comparison.py
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
`external/gym4real/`; the Ontario simulator is pinned as a submodule under
`external/loslr_regulation_optimization/`.

## Team

- Nadine Mohamed
- Daniel Martineau
- Vincent Dandenault
