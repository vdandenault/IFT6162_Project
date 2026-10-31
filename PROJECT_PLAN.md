# Lake Como reservoir control: project plan

## Research question and current status

Can a learned daily release policy improve the trade-off between downstream water
supply and flood/low-water risk on held-out Lake Como years?

The required comparison is **literature-based deterministic MPC versus PPO** in
DamEnv, with matched decision-time information, physical constraints, and evaluation
metrics. PPO and the shared evaluator are ready; **the next task is specifying MPC**.
Simple rules are sanity checks. Actual operator use of the proposed MPC remains
unverified, so describe it as a literature-based engineering controller.

Keep the core to two substantial controllers. FQI is the preferred optional third
method after the core works; scenario MPC and stochastic DP are alternatives,
not additional mandatory implementations. Hydropower requires a justified turbine,
head, efficiency, and possibly price model. Do not interpret surplus release or
stored water as measured electricity generation.

## The simulated control problem

Each day, the controller observes lake level, seasonal sine/cosine, and exponentially
averaged past demand, then requests a release in m³/s. No inflow forecast is included
by default. The simulator holds the day's historical inflow through 24 hourly
water-balance updates, clipping releases to level-dependent hydraulic limits.
An action of 100 m³/s over one day corresponds to 8.64 million m³.

```text
next_storage = storage + seconds × (inflow - actual_release - evaporation)
level = storage / surface_area + minimum_level
```

Surface area is 145.9 km². Storage is relative to the model's minimum-level datum,
not total lake volume. Evaporation is disabled. Simulated levels follow our actions;
observed historical levels do not overwrite the trajectory.

The nominal action range is 0–491.61 m³/s. Hydraulic level boundaries are -0.5 and
1.25 m; reward penalties begin below -0.45 m or above 1.24 m. Hydraulic release
limits do not guarantee flood avoidance under extreme inflows. The reward penalizes
unmet demand, flooding, low water, surplus release, and action clipping. Demand
penalties vary seasonally; report physical metrics alongside weighted return.

An episode has 365 decisions and 8,760 hourly updates. Reset restores level 0.35 m
and demand average zero; training randomly selects a year, not an initial level.
Stop rollouts on `terminated or truncated`. A wrapper corrects the final observed
level. Annual resets still prevent storage continuity between years; default PPO
uses `time_limit_bootstrap=False`. Any different terminal treatment needs an explicit
comparison protocol. MPC does not automatically repair resets or observations.

## Data and evaluation protocol

Daily Lake Como inflow, downstream demand, and observed levels cover 1946–2010.
Files live in `external/gym4real/gym4real/data/dam/`: `inflow/`, `demand/`, `level/`,
`MEF.csv` (environmental flow), and `metadata_dam.json` (provenance/license metadata).
The audit records hashes and unresolved provenance/license discrepancies locally.

[configs/splits.json](configs/splits.json) defines train 1946–1987 (42 years),
validation 1988–1997 (10), and test 1998–2010 (13). Leap years retain the upstream
365-step approximation: the final sample is unused and seasonal alignment is
approximate. Excluding leap years remains available for sensitivity checks.
The original paper's 52/13 membership is a separate reproduction protocol.

Select settings on validation, retain every learner seed, and reserve chronological
test years for the final comparison. Future historical inputs are simulator forcing,
not policy information. Any perfect-inflow controller must be labeled as an oracle.

The current PPO baseline is 1M steps with a capped learning-rate scheduler, mixed
precision, zero entropy bonus, and seeds 123–127. See [PPO findings](docs/PPO.md)
and [current settings](configs/ppo.json). Its improved reward does not imply every
physical outcome improved. [Evaluation](docs/EVALUATION.md) documents the common
runner, metrics, pairing, provenance, runtime, and validation results.

## Remaining work

### Phase 1 — environment and data: complete

- [x] Pin the simulator fork and uv environment; audit years, units, checksums, and licenses.
- [x] Run the environment/rule walkthrough; check water balance, clipping, timing, and observation boundaries.

### Phase 2 — benchmark and evaluation: MPC specification remains

- [x] Implement chronological membership and retain separate upstream reproduction.
- [x] Reproduce rule/PPO benchmark qualitatively and investigate five-seed sensitivity.
- [x] Complete controlled PPO ablations and stop duration tuning at 1M.
- [x] Evaluate every year of a selected split; save paired metrics, reward components, provenance, terminal storage, and decision runtime.
- [x] Compare four rules and five PPO seeds on all validation years; produce tables and paired plots without adding objectives.
- [ ] Extract the deterministic MPC specification from the [focused literature review](docs/LITERATURE_REVIEW.md).

### Phase 3 — engineering versus learning

- [ ] Specify observations, forecast timing, planning dynamics, solver, horizon, objective, constraints, and terminal treatment before implementing MPC.
- [ ] Fit a training-only seasonal inflow forecast, then consider persistence/history; require matched information for the main MPC/PPO comparison. Retrain a separately labeled PPO if inputs change.
- [ ] Implement and validate deterministic MPC, applying only its first action before replanning. Proposed 7/14/30-day horizons are new daily-DamEnv choices, selected on validation.
- [ ] Compare full annual trajectories, physical metrics, both reward scores, terminal storage, training interactions, and decision time. Inspect failures and wet/dry years using a predeclared selection rule.
- [ ] Run one focused ablation (horizon or forecast availability), then freeze the protocol before final test evaluation.
- [ ] If useful and feasible, add one optional method: tree-based FQI (preferred), scenario MPC, or a reduced-state SDP with grid-refinement checks.

Cestari et al. (2023) uses hourly actions, a 24-hour horizon, and a 90-day simulation
starting January 1, 2000; reported planning uses 380 scenarios and 50 Monte Carlo
inflow realizations. Its costs differ from DamEnv. Our MPC will be an adaptation,
not a numerical reproduction. Disclose any planning-model or reward surrogate,
validate resulting actions in unchanged DamEnv physics, and do not transfer scenario
guarantees without checking assumptions. Deterministic/scenario comparisons need
the same forecast generator and horizon. An approximate oracle is not automatically
a performance bound.

### Phase 4 — final analysis and presentation

- [ ] Evaluate the frozen controllers on held-out test years and report paired trade-offs, seed variability, and limitations.
- [ ] Add reward-weight or hydropower extensions only if time and defensible parameters support them.
- [ ] Prepare the report/poster and interactive results website; keep code, figures, and claims reproducible.

## Reading path and artifacts

Read [PPO findings](docs/PPO.md), [evaluation](docs/EVALUATION.md), and
[literature review](docs/LITERATURE_REVIEW.md) as needed. Start with
[environment walkthrough script](scripts/analysis/environment_and_data.py) for the system or
[controller comparison script](scripts/analysis/controller_comparison.py) for current comparisons;
intermediate analysis scripts are optional detailed evidence.

`configs/ppo.json` is the current selected baseline; `configs/experiments/ppo_studies.json`
consolidates exact historical study plans. Keep experiment code in `src/lake_como/`
and simulator changes in the fork at `external/gym4real/`. Generated `results/`,
`reports/`, `background/`, and local PDFs remain ignored. Nothing is committed
without review and approval.
