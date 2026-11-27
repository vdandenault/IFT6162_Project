# Reservoir control: Lake Como and Lake Ontario–Ottawa

Standalone Ottawa hard-follower demos and reproduction commands are documented in
[the Ottawa study guide](configs/studies/ottawa/README.md). This synthetic Step 2
controller is separate from the earlier soft follower and Ontario experiments.

IFT6162 project: compare Plan 2014, MPC and Stackelberg MPC for Lake Ontario–
St. Lawrence releases with an aggregated Ottawa reservoir operator. PPO is an
additional comparison. Lake Como remains a supported research and regression case.

The existing studies live under `src/reservoir_control/`. Nadine’s Ontario
adapter and solvers live under `src/bilevel_reservoirs/`. Shared PPO, SAC and
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
| Ontario adapter parity input and fixtures | `data/input/historic/1961_2020/` and `data/baseline/` |
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
uv creates `.venv` with Python 3.12 and installs both project packages and the
Gym4ReaL workspace. The `reproduction` group retains the pinned numerical and
RL dependencies used by the existing studies.

Use a separate environment for the Ontario adapter. MetaMARL and the saved
Ontario reference fixture require a different set of dependency versions.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv sync --locked --group ontario
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario pytest tests/integration tests/test_ontario_jax_physics.py
```

The `jax` group also enables the standalone physics kernel alongside the
reproduction stack.

```bash
uv run --locked --group reproduction --group jax pytest tests/test_ontario_jax_physics.py
```

## Run the standalone MPC simulators

From the repo root, run either command with a fresh output directory.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.run_mpc \
  --solver shooting --lookahead 4 --steps 12 --output results/loslr/shooting_h4

uv run --locked --group reproduction \
  python -m reservoir_control.systems.ottawa.hard_follower_study \
  --forecast perfect --lookahead 5 --output results/ottawa/follower_h5
```

`--lookahead` sets the number of quarter-month periods MPC plans over at each
decision. Ontario supports 1–4 periods because the inputs provide four forecast
leads. `--steps` sets the Ontario rollout length separately. Use `--solver beam`
or `--solver mppi` to switch Ontario solvers. Ottawa supports 1–12 periods in the
bundled study, with `--forecast noisy` for its noisy scenarios. If omitted, the
Ontario look-ahead and the bundled Ottawa configuration both use four periods.
Use the Ottawa [matched-forecast study](configs/studies/ottawa/README.md#matched-forecast-horizon-sensitivity)
when comparing horizons under the same noisy forecasts.

## Ontario integration and JAX

Run the fixed Ontario schedule through the Ottawa H5 follower using the
delayed synthetic scenario:

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.fixed_schedule_replay \
  --trajectory results/loslr/shooting_h4/trajectory.csv \
  --output results/loslr/fixed_ontario_ottawa_h5_rerun
```

The replay uses perfect Ottawa inflow forecasts and a synthetic downstream
background. It writes a joint trace, Ottawa planning inputs and a summary.
A commitment rejected by Ontario's limits stops both actions for that period.

Add `--repair-offline` to find a feasible test schedule. This starts a fresh
replay after each change to the first rejected commitment. It saves
`fixed_schedule.csv` and the edits in `summary.json`. Verify the result by
passing the saved CSV as `--trajectory` in a fresh run with a new output
directory. The schedule search uses perfect Ottawa forecasts and serves as
an offline feasibility benchmark.

Compare the Ontario planners against the same Ottawa follower and fixed
fifth-period commitment:

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.compare_ottawa_response \
  --trajectory results/loslr/shooting_h4/trajectory.csv \
  --fixed-schedule results/loslr/feasible_fixed_schedule_h5/fixed_schedule.csv \
  --output results/loslr/ottawa_response_comparison_rerun
```

The default grid varies the four Ontario commitments independently around the
verified fixed schedule. Both planners use the same grid and native Ontario
stage cost. The external arm assumes a 900 m³/s Ottawa release and routes it
with the actual transit queue. The anticipating arm solves the Ottawa follower
for each candidate. Both call the same follower at execution. The study uses
perfect synthetic Ottawa inflow forecasts. The summary compares costs over the
common executed prefix and the candidate log records rejected commitments.

Pass `--candidates limits` to generate additional candidates by applying the
native limits to forecast releases. Both arms use the same generation rule and
keep the same fifth commitment. The period-3 diagnostic is available as
`python -m bilevel_reservoirs.solvers.diagnose_ottawa_forecast --help`.
Its substitutions use replay inputs offline to measure their effect.

Combine `--candidates limits --align-f` to cap each candidate's first four
Ottawa leads with Ontario's forecast F ceiling. Both arms execute the selected
schedule with the capped Ottawa follower. The external arm still ranks against
assumed arrivals. `--align-from 3` reproduces the earlier policy states and
switches on alignment at the original rejection. `--roughness-forecast
trailing_mean` uses the mean of up to four observations available at issue time
for leads 2–4. Lead 1 uses the current roughness.

Add `--screen-external` to include a third arm that ranks candidates using the
external forecast, then tests them in order with the Ottawa follower. The
summary reports a separate matched cost comparison between this arm and the
anticipating arm, along with the number of Ottawa candidate solves.

Run the paired stress test with the same inputs:

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.stress_ottawa_response \
  --trajectory results/loslr/shooting_h4/trajectory.csv \
  --fixed-schedule results/loslr/feasible_fixed_schedule_h5/fixed_schedule.csv \
  --output results/loslr/ottawa_noisy_stress_rerun
```

The stress test saves the shared forecast banks, executed traces, matched
cost components and offline feasibility witnesses. Storage advances with the
realized inflow. The witness search uses true future inputs only after a run
stops, and checks each witness against the native Ontario limits.

Pass `--recovery` to compare the screened baseline with two variants. The
first tests additional schedules on the original forecast after the primary
search fails. The second also retries with the last observed Ottawa inflow
held constant over the window. The next decision uses the original forecast
bank again. Both variants retain the native execution checks, and the runner
records and stops on realized storage violations.

Combine `--recovery --storage-guard` to test one- and two-period storage checks
at the persistence retry. These reserve room for an extra one forecast-noise
standard deviation of inflow in each checked period, carrying the extra
volume forward through the trial schedule. The next decision replans using
its ordinary constraints and original forecast.

Use `--holdout` to compare the frozen one-period +700 m³/s check with the
unguarded retry under seeds 101–106 and a perfect-forecast control. Add
`--historical-year 2017` or `--historical-year 2019` in place of the trajectory
and fixed-schedule inputs to run April–June on the pinned Ontario data. These
windows start from a causal Plan 2014 warmup. The Ontario schedule anchor uses
the previous release and the declared synthetic flow cap. Ottawa inflows,
routing and downstream capacities remain synthetic in all windows.

Add `--replan-guard` with `--holdout` to compare applying that same storage
check on the first ordinary decision after a retry and on every ordinary
decision. Both variants still check the retry itself. The primary and expanded
candidate searches use the same bound at a guarded decision.

Reproduce the 2019 seed-101 stop and test a targeted recovery using the archived
Ontario inputs and Ottawa forecasts. The recovery searches lower Ontario
commitments for the period when the first Ottawa release arrives downstream.
The diagnostic writes the frozen feasibility checks, candidate audit, and replay
trace to a fresh directory.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.diagnose_replan_stop \
  --archive results/loslr/ottawa_replan_guard_2019 \
  --output results/loslr/ottawa_replan_stop_2019_reproduction
```

Compare the opt-in targeted search against the archived guard on the 36 noisy
cases across the 1961, 2017 and 2019 Ontario windows. The runner uses the saved
forecast banks and checks trajectory agreement up to the first recovery.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_delay_headroom \
  --archive-root results/loslr \
  --output results/loslr/ottawa_delay_headroom_holdout_reproduction
```

Diagnose the two remaining period-6 stops using the saved forecast banks,
guarded persistence forecasts, and realized inflows for the offline checks.
The second command compares the after-retry trigger with an opt-in trigger
that also tries the targeted candidates when the ordinary or persistence
search is exhausted. Both arms share the same forecasts for seeds 107–112.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.diagnose_remaining_stops \
  --output results/loslr/ottawa_remaining_stops_reproduction

UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.validate_delay_headroom_trigger \
  --output results/loslr/ottawa_broader_headroom_reproduction
```

Compare external-ranked MPC with finite-search Stackelberg MPC on the 36
saved forecast-noise cases. Both arms use the same candidate generator,
Ottawa feasibility solve, native limit checks, and recovery rules. The
Stackelberg arm evaluates Ontario's objective using Ottawa's routed best
response for each eligible schedule. The report includes the check against
the archived external-ranked trajectories and the full-window paired outcomes.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_response_ranking \
  --baseline results/loslr/ottawa_broader_headroom_validation_v1 \
  --output results/loslr/ottawa_response_ranking_reproduction
```

Compare the current baseline with two local candidate expansions. The generic
and Ottawa-guided arms may each try 12 new Ontario schedules after finding a
feasible incumbent. Both use the same ranking and feasibility checks. Run
the saved spring cases first, then the fresh noise seeds 113–118.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_candidate_proposals \
  --sample saved --output results/loslr/ottawa_candidate_proposals_saved_reproduction

UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_candidate_proposals \
  --sample fresh --output results/loslr/ottawa_candidate_proposals_fresh_reproduction
```

Reproduce the three seed-115 storage overflows and compare the after-retry,
always-on, and tight-headroom guards on the archived 36-case matrix. The
runner saves the period-6 feasibility witnesses and period-5-only replays.
The offline witnesses use realized inflows. The replay controllers use the
original archived forecasts.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_storage_guard \
  --output results/loslr/ottawa_storage_guard_seed115_reproduction
```

Validate the frozen +700 m³/s guard against the declared early-peak,
sustained-wet and dry Ottawa inflow shapes. The two arms share the same
forecast bank for each year, inflow shape and seed. The runner saves
storage outcomes and continuation checks for stopped runs.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.validate_storage_guard \
  --output results/loslr/ottawa_storage_guard_hydrology_reproduction
```

Diagnose the three extra sharp-peak stops from that study. The runner
reproduces the archived stops and checks the same Ontario candidates with
and without the storage reserve. It uses realized Ottawa inflows only for
offline first-lead, tail, and full-window substitutions.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.diagnose_sharp_peak \
  --output results/loslr/ottawa_sharp_peak_diagnostic_reproduction
```

Test a causal recession forecast under the same always-on +700 m³/s
guard. The recession arm keeps lead one and the saved noise draws while
letting above-recent inflow decay in later leads. Both arms retain the
same persistence fallback. The first command uses the archived
comparison and the second runs both arms on fresh noise seeds.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_ottawa_recession \
  --sample archived --output results/loslr/ottawa_recession_archived_reproduction

UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_ottawa_recession \
  --sample fresh --output results/loslr/ottawa_recession_fresh_reproduction
```

Compare the same two forecasts with an isolated-jump rule and a new Ottawa
inflow case that stays high after the sharp rise. The study checks that
the two sharp-rise cases issue identical forecasts through the first
decision after the jump. Both forecasts with recession preserve lead one.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_ottawa_recession \
  --sample isolated-jump --output results/loslr/ottawa_isolated_jump_reproduction
```

Check common Ontario–Ottawa first actions offline on the archived sharp-rise
pairs and compare the causal two-path controller with isolated-jump
recession on fresh pulse heights, durations, and noise seeds. Both online
arms use the always-on +700 m³/s guard.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_scenario_recourse \
  --output results/loslr/ottawa_two_path_recourse_reproduction
```

Isolate the effect of candidate coverage and the storage reserve at the
12 saved two-path stops. The runner uses the issued forecasts and reports
shared actions for the current and wider search, each with and without
the extra reserve.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.diagnose_two_path_stops \
  --output results/loslr/ottawa_two_path_factorial_reproduction
```

Compare the wider search in closed loop with isolated-jump MPC and the
original two-path MPC. Replay the archived 36 cases, then test 36 fresh
cases with different Ottawa peaks, high-flow durations, and noise seeds.
All three arms retain the +700 m³/s reserve.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_scenario_recourse \
  --sample wider-archived --output results/loslr/ottawa_two_path_wider_archived_reproduction

UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_scenario_recourse \
  --sample wider-fresh --output results/loslr/ottawa_two_path_wider_fresh_reproduction
```

Check sustained-rise recoverability at every executed prefix of the fresh
wider-search cases. The offline witnesses preserve known transit and use
realized inflows through the endpoint. A separate stopped-decision probe
keeps the +700 m³/s reserve and swaps the issued Ottawa inputs for the
realized plateau at that decision only.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.diagnose_plateau_recoverability \
  --output results/loslr/ottawa_plateau_recoverability_reproduction
```

## Paired hard-follower comparison

Run the nine-state diagnostic and the optional reserve-fallback sensitivity.
The sensitivity relaxes only the forecast reserve when the guarded two-path
search fails. The hard execution constraints still apply.

```bash
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.diagnose_final_constraints \
  --output results/loslr/ottawa_final_constraints_reproduction

UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_scenario_recourse \
  --sample wider-fresh --reserve-fallback \
  --output results/loslr/ottawa_reserve_fallback_reproduction
```

The paired study uses Plan 2014 requests, external-ranked MPC, Stackelberg
MPC, and PPO residual requests. They share the same hard Ottawa follower,
native limit checks, causal inputs, and operating-cost score. PPO fits on
1901–1980 only and the runner evaluates the saved final checkpoints.
Keep the synthetic stress windows separate from the annual comparisons.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-ontario
study_root=results/coupled_final_rerun
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
  --mode smoke --output "$study_root/smoke"
for seed in 123 124 125 126 127; do
  uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
    --mode train --seed "$seed" --output "$study_root/final_coupled_training_v1/seed_$seed"
done
for sample in development reserved stress; do
  uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
    --mode "$sample" --training-root "$study_root/final_coupled_training_v1" \
    --output "$study_root/final_coupled_${sample}_v1"
done
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.report_final_coupled \
  --root "$study_root" --output "$study_root/report" \
  --playback background/ontario_coupled_final.html
```

### Shared recovery follow-up

Use the frozen final PPO checkpoints to compare the common recovery choices.
The audit records which options would reopen each stopped decision while
preserving the original executed path. The closed-loop runs allow recovery
and keep the realized storage and flow limits.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-ontario
train_root=results/loslr/final_coupled_training_v1
for recovery in audit reserve profiles proactive_profiles peak_protected; do
  uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
    --mode development --event-only --recovery "$recovery" --training-root "$train_root" \
    --output "results/loslr/final_recovery_${recovery}_events_reproduction"
done
for recovery in original profiles peak_protected; do
  uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
    --mode stress --stress-design configs/studies/ottawa/shared_recovery_final_check_v1.json \
    --recovery "$recovery" --training-root "$train_root" \
    --output "results/loslr/final_recovery_${recovery}_confirmation_reproduction"
done
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.report_shared_recovery \
  --output results/loslr/final_shared_recovery_report_reproduction
```

For the PPO execution-contract ablation, train the same five seeds with
`--mode train --recovery profiles` and pass their output root to evaluation.
The saved report also uses the first stress holdout defined in
`configs/studies/ottawa/shared_recovery_holdout_v1.json`.

Run the Ottawa storage and routing sensitivity using the archived peak-protected
event and confirmation runs under `results/loslr/`. The output includes a
side-by-side 2017 and 2019 figure and a local interactive replay.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_ottawa_ranking_sensitivity \
  --output results/loslr/ottawa_ranking_sensitivity_rerun \
  --playback background/ontario_peak_protected_rerun.html
```

Audit anticipation at downstream-sensitive decisions and assemble the final
comparison from the archived peak-protected event, reserved and confirmation
runs. Use a fresh output directory.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m bilevel_reservoirs.solvers.study_downstream_anticipation \
  --output results/loslr/downstream_anticipation_rerun
```

Check official Ottawa station coverage and probe the shared execution
filter on archived PPO trajectories. The coverage audit downloads HYDAT
data from the public ECCC GeoMet endpoint. The PPO audit uses local
peak-protected archives and replays ten flood-year episodes with their
saved requests.

```sh
export UV_PROJECT_ENVIRONMENT=.venv-ontario
uv run --locked --group ontario python -m reservoir_control.systems.ottawa.flow_source_audit \
  --output results/ottawa/flow_source_audit_rerun
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.diagnose_ppo_filter \
  --output results/loslr/ppo_filter_audit_rerun
```

Fit the daily Carillon forecast benchmark and enumerate the exact PPO
release menu on development trajectories using the frozen checkpoints.

```sh
export UV_PROJECT_ENVIRONMENT=.venv-ontario
uv run --locked --group ontario python -m reservoir_control.systems.ottawa.forecast_benchmark \
  --source results/ottawa/flow_source_audit_v1/historical_daily.csv \
  --output results/ottawa/carillon_forecast_benchmark_rerun
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.diagnose_ppo_menu \
  --output results/loslr/ppo_development_menu_rerun
```

Benchmark quarter-month SLON forecasts with the latest completed value
available or delayed by one period. The scores report high-flow periods
separately and leave Ontario MPC inputs unchanged.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.loslr.slon_forecast \
  --output results/loslr/slon_forecasts_rerun
```

Replay the same standalone Ontario MPC with persistence and seasonal-history
SLON forecasts under both availability settings. The runner verifies its
forecast bank against the offline benchmark and uses historical aggregate
SLON for realized routing.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.loslr.slon_mpc_replay \
  --benchmark results/loslr/slon_forecasts_verified_v1 \
  --output results/loslr/slon_mpc_replay_rerun
```

Diagnose changed exposures by swapping forecasts at the same inherited
states. The runner checks the original decisions against the archive and
also probes lead one separately from leads two through four.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.loslr.slon_state_diagnostic \
  --archive results/loslr/slon_mpc_replay_verified_v1 \
  --output results/loslr/slon_state_diagnostic_rerun
```

Run the follow-up screens with the same archived forecast bank. The
smaller-margin run automatically evaluates a frozen winner on the reserved
years if it passes the development gate.

```sh
export UV_PROJECT_ENVIRONMENT=.venv-ontario
uv run --locked --group ontario python -m reservoir_control.systems.loslr.slon_upper_bound_study \
  --output results/loslr/slon_upper_bound_rerun
uv run --locked --group ontario python -m reservoir_control.systems.loslr.slon_upper_bound_study \
  --follow-up --output results/loslr/slon_upper_bound_followup_rerun
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.study_ppo_modal_release \
  --output results/loslr/ppo_modal_release_rerun
uv run --locked --group ontario python -m bilevel_reservoirs.solvers.study_ppo_duration \
  --workers 4 --output results/loslr/ppo_duration_peak_rerun
```

Check observed downstream stage response and the exposure trade-off using the
saved HYDAT station audit and reserved SLON replay.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.ottawa.routing_validation \
  --output results/ottawa/routing_validation_rerun
```

Run the earlier-release hindsight search and its gated causal comparison
using the archived Ontario replay.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.loslr.release_timing_study \
  --output results/loslr/release_timing_rerun
```

Test early high-SLON onset alerts from observed Britannia flow. Omit
`--raw` to retrieve the same date ranges from ECCC GeoMet in a fresh run.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.loslr.slon_warning \
  --raw results/loslr/britannia_slon_warning_v1/raw \
  --output results/loslr/britannia_slon_warning_rerun
```

Test early Pointe-Claire headroom alerts at the original issue-time
Ontario releases. This scores alerts without changing the controller.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.loslr.headroom_alert \
  --output results/loslr/pointe_claire_headroom_alert_rerun
```

Build the observed downstream headroom benchmark from Water Survey of Canada
LaSalle discharge and Pointe-Claire stage, USGS Cornwall discharge, and the
Canadian Hydrographic Service datum cross-check.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.ottawa.downstream_observations \
  --output results/ottawa/downstream_gauges_v1
```

For offline reproduction, pass `--raw results/ottawa/downstream_gauges_v1/raw`
and a fresh `--output`. The report links to the data terms and records the
coverage and flow boundaries. Keep the CHS responses in local results.

Check daily routing and near-threshold stage errors using the frozen
downstream gauge benchmark plus HYDAT Britannia and Rivière des Prairies
discharge. Omit `--raw` to fetch the upstream gauge responses again.

```sh
UV_PROJECT_ENVIRONMENT=.venv-ontario uv run --locked --group ontario \
  python -m reservoir_control.systems.ottawa.reach_validation \
  --downstream results/ottawa/downstream_gauges_verified_v1 \
  --raw results/ottawa/reach_inputs_v1/raw \
  --output results/ottawa/reach_validation_replay
```

Use `bilevel_reservoirs.simulators.physics.lake_step` as the JAX transition for
adjustment-based planning. It accepts lake level in metres, a release adjustment
in units of 10 m³/s, and net total supply in m³/s. It is usable with `jax.jit`,
`jax.vmap` and `jax.grad` independently of MetaMARL. The simulator uses double
precision at its scalar boundary. Set JAX x64 explicitly for numerical parity
when calling the kernel or its derivatives directly.

`LoslrEnv` exposes two action contracts. In `flow` mode, the rule curve and
existing MPC solvers supply an already constrained total discharge, and the
simulator uses the reference water balance and rounding. In `adjustment` mode,
the simulator decodes the adjustment with the JAX lake transition, applies
the native limits to total flow, and reconciles the level using the applied
flow. The full rollout still uses Python and NumPy for routing, performance
indicators and operating limits. The JAX gradients apply to the lake kernel.

For the Ottawa integration, pass the executable **total** Ontario discharge
in m³/s to the Ottawa follower. Convert the native flow units at that boundary.
The next integration test should verify that the flow used by Ottawa matches
the flow Ontario actually executes, including when an operational limit binds.
The adjustment adapter currently reads the selected `ontNTS_QM*` column.
Check the realized-supply and forecast inputs explicitly when connecting it to
the coupled rollout.

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
