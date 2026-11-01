# Lake Como reservoir control: project plan

## Research question and current status

Can a learned daily release policy improve the trade-off between downstream water
supply and flood/low-water risk on held-out Lake Como years?

The required comparison is **literature-based deterministic MPC versus PPO** in
DamEnv, with matched decision-time information, physical constraints, and evaluation
metrics. PPO and the shared evaluator are ready. A deterministic MPC prototype is implemented;
**the next task is diagnosing and validating it as an engineering benchmark**.
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

### Phase 2 — benchmark and evaluation: complete

- [x] Implement chronological membership and retain separate upstream reproduction.
- [x] Reproduce rule/PPO benchmark qualitatively and investigate five-seed sensitivity.
- [x] Complete controlled PPO ablations and stop duration tuning at 1M.
- [x] Evaluate every year of a selected split; save paired metrics, reward components, provenance, terminal storage, and decision runtime.
- [x] Compare four rules and five PPO seeds on all validation years; produce tables and paired plots without adding objectives.
- [x] Extract the deterministic MPC specification from the [focused literature review](docs/LITERATURE_REVIEW.md#implemented-deterministic-mpc-prototype).

### Phase 3 — engineering versus learning

- [x] Specify the initial MPC observations, training-only climatology, hourly planning dynamics, local solver, seven-day horizon, surrogate objective, constraints, and zero terminal value.
- [x] Fit training-only seasonal inflow/demand climatology with no held-out future inputs.
- [x] Compare seasonal, persistence, and training-fitted seasonal-anomaly forecasts under matched one-day search and both costs. Persistence improves return in all validation years; prediction-error ranking differs from control ranking, and unmet demand increases. See [forecast study](docs/EVALUATION.md#causal-past-inflow-forecast-comparison).
- [x] Extend causal forecasts to seven-day MPC with horizon-specific training fits and fixed costs/solver: both improve every validation year. Persistence returns -28,931 versus seasonal -50,274, but raises unmet demand and solver failures. Preserve severity/supply trade-offs and the extra-history label; see [seven-day study](docs/EVALUATION.md#causal-forecasts-in-seven-day-mpc).
- [x] Audit additional solver failures via exact replay of all 10,950 decisions: all hit the iteration cap; 148/149 discarded persistence iterates have lower predicted cost within request bounds. Fixed-tail first-action slices improve 141/149 failed states. See [solver audit](docs/EVALUATION.md#audit-of-additional-causal-mpc-solver-failures).
- [x] Test cost-checked iteration-limit retention under persistence with fixed forecast/cost/budget: return improves in all validation years (-26,361 versus -28,931), low-water severity falls 39%, unmet demand rises 0.16%, and all 131 retained iterates remain flagged as failures. Experimental only; see [retention study](docs/EVALUATION.md#cost-checked-retention-of-iteration-limit-iterates).
- [ ] Resolve the supply-versus-level-risk criterion before promoting a baseline or tuning safety margins. Match PPO information before a final comparison.
- [x] Implement deterministic MPC and test physics, information timing, optimization, fallbacks, and resets; apply only the first action before replanning.
- [x] Audit boundary failures and short-horizon optimization without changing the deployed controller: exact historical replay, selected grid/refinement checks, and seven-day cost slices. Forecast, objective, and solver issues remain distinct; see [diagnosis](docs/EVALUATION.md#boundary-and-optimizer-diagnosis).
- [x] Run a clearly labeled perfect-inflow counterfactual with fixed demand forecast, surrogate, solver, and horizon: validation return improves from -50,274 to -28,962 across all ten years, but violations and optimization issues remain.
- [x] Isolate objective preferences on common hindsight grids: the surrogate has material binary-cost regret in 19/36 selected short-horizon problems; 15 change the first release. See [objective audit](docs/EVALUATION.md#binary-versus-surrogate-objective-on-common-grids).
- [x] Compare objectives in closed loop under common one-day exhaustive hindsight search: binary improves validation reward in all ten years (-15,976 versus -50,066), but unmet demand and flood-exceedance severity increase slightly. This is not the deployed seven-day MPC.
- [x] Test training-only seasonal inflows under the same one-day search, retaining severity: binary versus surrogate returns are -50,463 versus -50,716; the large hindsight gain disappears. See [forecast comparison](docs/EVALUATION.md#non-oracle-forecasts-under-the-same-search).
- [ ] Establish the engineering-risk criterion using supply, event counts, and exceedance severity; investigate forecast uncertainty and safety margins before promoting a controller or extending the horizon. Preserve the initial baseline.
- [ ] Validate engineering-baseline quality: the initial seven-day prototype performs poorly. Separate forecast, surrogate-objective, horizon/terminal, and optimizer effects; retain the first result and avoid claiming PPO superiority from this prototype alone.
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

### Phase 4 — understand and extend the control problem

Start from [DamEnv's existing equations](https://github.com/Daveonwave/gym4ReaL/blob/main/gym4real/envs/dam/env.py)
(use our pinned local source for experiments) and investigate **information and safety**
before expanding the algorithm list. Preserve the current PPO/MPC results as the
reference; these are new experiments, not retroactive benchmark changes.

- [ ] Explain the hourly storage balance, hydraulic release limits, daily observation/action timing, and all reward terms in a short mathematical walkthrough. Distinguish physical release limits from soft lake-level penalties, and document the current no-evaporation and annual-reset assumptions.
- [x] Compare MPC horizons **1, 3, and 7 days** with persistence, surrogate, retention, initialization recipe, and solver cap fixed: validation returns -36,015 / -28,077 / -26,361. Seven days beats one day in all years and three days in nine; shorter horizons use less CPU and leave less demand unmet. Keep seven days as the current best mean-reward setting, not a universal optimum. See [matched horizon study](docs/EVALUATION.md#matched-mpc-horizon-experiment).
- [ ] Add **last completed-day inflow** to PPO observations through a project wrapper, with explicit January 1 fallback/reset and causality checks. Do not expose the current day's realized inflow before its transition. Give PPO the same past-inflow information as causal MPC; retain the four-feature PPO as the ablation control.
- [ ] Retrain both PPO observation variants with matched training budget and all five seeds, fit normalization on training only, and compare full validation trajectories. Keep reward, dynamics, calendar, and split fixed. Optionally test a training-fitted forecast/history representation only after this single-feature ablation.
- [ ] Report whether added information changes decisions and physical outcomes, not just reward or forecast error. Identify wet/dry failure regimes using a declared selection rule; adding a feature alone is an ablation, while the contribution is the controlled engineering analysis. Confirm project expectations with the instructor; do not assume this extension alone earns a particular grade.
- [ ] Explore a **safety-first formulation** separately: define the flood threshold, daily versus hourly enforcement, admissible inflow uncertainty, and priority over irrigation. Check whether feasible releases can actually satisfy the condition from each state. A larger penalty is not a hard constraint.
- [ ] Prototype constrained MPC or a common safety filter only after the feasibility audit. Record infeasibility explicitly and define a minimum-violation fallback; do not claim unconditional flood prevention under unbounded/unmodeled inflows. Evaluate the supply cost, event severity, violations, and intervention frequency. Apply the same safety requirement/information to learning and engineering controllers for comparisons; keep original-reward results separately labeled.
- [ ] If time permits **one additional controller, prioritize tree-based FQI**: fit Q-functions on training-only simulated transitions, declare the release grid and behavior-data coverage, retain multi-step/terminal handling, and compare unseen-year decisions under matched information/objective. Investigate the paper's exact setup before claiming reproduction.
- [ ] Use stochastic DP only as an optional reduced-state, coarse-grid reference or FQI sanity check. Fit inflow transition probabilities from training years, account for season, and check grid refinement. Do not make both SDP and FQI prerequisites: PPO plus a credible MPC already covers the main comparison, and extra methods must answer a concrete question.

### Phase 5 — final analysis and presentation

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
