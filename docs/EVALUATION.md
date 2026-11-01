# Evaluation: commands, metrics, and results

Run the shared chronological comparison:

```bash
uv run --group reproduction lake-como-compare
```

The default evaluates Random, Mean, Max, EAD, and all five selected PPO checkpoints
from [configs/ppo.json](../configs/ppo.json) on 1988–1997 validation years. It does
no training or chronological test evaluation. See [controller comparison script](../scripts/analysis/controller_comparison.py)
for the table, reward components, terminal storage, decision timings, and plots.

## Results and their limits

The completed comparison covers 90 annual episodes. PPO has the highest full
return averaged across seeds in every validation year, and its scores match the
saved 1M study. EAD beats one PPO seed in one year; PPO wins the other 49 seed/year
comparisons against EAD. These are paired outcomes, not 50 independent trials.

| Controller | Full return | Flood days/year | Low-water days/year | Unmet demand (million m³/year) |
|---|---:|---:|---:|---:|
| Random | -48,353 | 12.30 | 45.20 | 936.16 |
| Mean | -40,469 | 9.90 | 63.80 | 562.77 |
| Max | -53,370 | 4.20 | 64.30 | 520.27 |
| EAD | -33,858 | 23.80 | 11.20 | 356.14 |
| PPO | -17,232 | 4.40 | 5.76 | 514.91 |

PPO improves the weighted objective, not every physical metric. EAD leaves less
unmet demand; Max has slightly fewer flood days but many more low-water days.
The PPO setting was tuned on these validation years. This is a development
comparison against simple rules, not evidence of superiority over MPC or operators.

## Run another comparison

```bash
uv run --group reproduction lake-como-compare --split validation \
  --controllers ead max ppo --output results/comparison/validation_subset
```

`--split` accepts `train`, `validation`, or `test`; reserve test for the final frozen
comparison. `--controllers` also accepts `constant` (100 m³/s). Repeat
`--ppo-checkpoint SEED=PATH` for explicit checkpoint choices. Adjacent training
`run.json` must match the seed, chronological training membership, protocol, and
checkpoint hash. Nonempty output directories are rejected; choose a new `--output`.

Every controller visits the same years in order, with initial level 0.35 m and
demand average zero. The runner verifies identical forcing and resets, complete
365-day episodes, reward accounting, and water balance. Leap years retain the
365-step approximation; the terminal observation is corrected, but storage does
not carry between years. Random has one action RNG seeded at 123 across the split,
not multiple random-rule seeds. PPO uses frozen normalization, deterministic mean
actions, CPU FP32, and one PyTorch thread.

## Outputs and metric definitions

Results remain ignored under `results/comparison/validation_v1/`:

- `trajectories.csv` / `annual.csv`: daily and annual results, all weighted reward terms, terminal storage, and decision timing; each PPO seed is a separate controller.
- `controller_means.csv`, `family_years.csv`, `summary.csv`: controller means and equal-weight PPO averages. Seed SD uses five seed means; year SD uses ten yearly seed averages. Rules have no learner-seed SD.
- `paired_annual.csv` / `paired_family_annual.csv`: candidate-minus-reference differences joined by year, before/after PPO seed averaging. Higher reward and lower physical penalties are better.
- `annual_comparison.png` / `paired_ppo_vs_rules.png`: year-by-year PNG plots. Shading is the observed PPO seed range, not a confidence interval.
- `timings.csv` / `run.json`: runtimes, checkpoint/input/source hashes, versions, split, reset/calendar/inference settings, and reward coefficients.

Full return includes five penalties; the three-term objective excludes waste and
clipping. Flood/low-water days use daily endpoints, not hourly exceedance counts.
Unmet demand is unweighted physical volume. Terminal storage is relative to the
model's minimum-level datum; no new terminal penalty is imposed.

CPU decision medians were about 36–37 µs for PPO and below 2 µs for deterministic
rules. Timings exclude simulation, action rescaling in `env.step`, loading, CSV,
and per-year PPO warmup. Python/timer overhead and local noise are material.

A future controller can reuse `evaluate(name, ..., controller_factory=factory,
sequential_years=True, measure_decisions=True)`. The factory receives the environment
and returns `(observation, previous_info) -> physical_release_request`. Respect
decision-time information; future historical inputs must not leak into the policy.
Adding the MPC CLI adapter belongs to implementing MPC, our next task.

## MPC prototype

Run one validation-year smoke test (defaults to 1988), or add MPC to the shared
all-year comparison:

```bash
uv run --group reproduction lake-como-mpc
uv run --group reproduction lake-como-compare --controllers ead max mpc ppo \
  --output results/comparison/mpc_validation_v1
uv run --group reproduction python scripts/analysis/mpc_validation.py
```

Existing outputs are preserved; choose a new `--output` when rerunning. The smoke
command accepts `--year`, `--split`, and `--config`. The shared runner accepts
`--mpc-config`; its original default rule/PPO comparison is unchanged.

The first fixed seven-day prototype scored mean full validation return -50,274,
with 30 flood days, 36.9 low-water days, and 530.68 million m³ unmet demand per
year. It is substantially worse than PPO and EAD on reward; this does not establish
an inherent limitation of MPC. Seasonal forecasts, the surrogate objective,
short horizon/no terminal value, and local optimization all require investigation.
No settings were changed in response to these results; the prototype is not yet
our final engineering benchmark.

`mpc_diagnostics.csv` records every optimizer status, iteration count, fallback,
forecast, and one-step predicted level. Twenty decisions hit the iteration limit;
151 used a fallback (including successful solves with worse initial-plan cost).
Median decision time was about 4.1 ms and p95 about 26.8 ms locally. Forecast inflow
MAE ranged from about 51 to 102 m³/s across years. The smoke PNG contrasts realized
levels/releases/inflow with the predictions; all plots/results remain ignored.

### Boundary and optimizer diagnosis

```bash
uv run --group reproduction python scripts/analysis/diagnose_mpc.py
```

This audits saved states without new closed-loop episodes or controller changes.
It checks six fixed cases, 1–3-day exhaustive action grids with refinement, and
first-action slices of the original seven-day plans. Historical replay verified
719 decisions. Outputs, three PNGs, and `report.md` are local under ignored
`results/mpc/diagnosis_v1/`; select a new `--output` to rerun.

The forecast missed 359/369 low-water days and 173/300 flood days. A selected
persistent-flood plan predicts four flood endpoints, charging 6.04 surrogate level
cost versus 4,000 binary cost on identical forecasts (undiscounted sums). A fine
grid beats one of 18 selected cold short-horizon solutions materially; first-action
slices improve two of six original seven-day plans materially. These checks expose
forecast, objective, and optimizer issues; they neither prove global optimization
nor attribute the annual performance gap to a single cause. The perfect-inflow
counterfactual below is now complete.


### Perfect-inflow hindsight diagnostic

```bash
uv run --group reproduction python scripts/analysis/perfect_inflow_mpc.py
```

This is an opt-in, validation-only experiment, separate from deployable MPC. Only
inflow is replaced with actual future values within the existing seven-day
horizon. Demand climatology, objective, solver/settings/fallback, float32 state
observations, and annual resets remain fixed. The original baseline is reused
with source/input/config/version checks and remains unchanged. No test evaluation
or candidate selection is performed. Results and PNGs stay ignored under
`results/mpc/perfect_inflow_v1/`; use a new `--output` to rerun.

| Annual mean | Seasonal inflow | Perfect inflow (hindsight) |
|---|---:|---:|
| Full return | -50,274 | -28,962 |
| Flood days | 30.0 | 13.4 |
| Low-water days | 36.9 | 18.6 |
| Unmet demand (million m³) | 530.68 | 535.49 |

Full reward improved in all ten years. Inflow forecast error became zero; average
one-day level error fell from 0.03265 m to about 1e-8 m in this simulator. All
remaining 134 flood and 186 low-water endpoints were predicted as violations.
The median remaining low-water exceedance was only 0.000125 m, exposing the
sensitivity of the binary scoring penalty to very small accepted violations.
Demand forecast is still imperfect and unmet demand increased slightly.

Solver failures rose from 20 to 28 and fallbacks from 151 to 193: accurate inputs
also change the optimization landscape. This measures the effect of replacing
inflow for this controller, not a causal variance decomposition, deployable result,
or guaranteed upper bound. Next isolate objective/optimizer effects under the
fixed hindsight input; retain the original deployable baseline and avoid tuning
multiple factors together.

### Binary versus surrogate objective on common grids

```bash
uv run --group reproduction python scripts/analysis/compare_mpc_objectives.py
```

This hindsight-only audit exhaustively scores both objectives on identical finite
action grids, without a local solver or new annual rollouts. The same six fixed
calendar cases are taken from both prior trajectory sources (12 initial states),
with 1–3-day horizons. All forecasts, initial states, nonlevel costs, discounting,
and terminal costs are shared within each comparison. Actions are float32 requests.
Nested grids use 401/10,001 points for one day, 21/41 per action for two days, and
11/21 for three days. No deployed settings are changed.

On the finest grids, the surrogate-selected plan has materially higher binary
planning cost in 19/36 problems, with fewer predicted violation-days under the
binary choice in those 19. Fifteen change the first release; four change only
later planned releases. For example, a one-day hindsight flood case switches
307.50 to 308.24 m³/s, moving the predicted level from 1.240410 to 1.239973 m and
saving about 999.85 binary cost units. This establishes preference misalignment
on shared inputs and grids; it is not explained by SLSQP selection.

The binary objective still uses forecast demand and is not realized annual reward.
These are failure-focused finite problems, not a generalization estimate or a
seven-day global-optimality certificate. Results, two PNGs, winning plans, grid
refinements, and a report remain ignored under `results/mpc/objective_grid_v1/`.
Next compare objectives in closed loop with a common search method before claiming
an improvement to deployable MPC. Preserve the original surrogate baseline.

### Closed-loop objective comparison with common search

```bash
uv run --group reproduction python scripts/analysis/closed_loop_mpc_objectives.py
```

Both diagnostic arms enumerate the same 10,001 float32 daily requests, use perfect
current-day inflow and training-only demand climatology, and reset annually. Only
level-penalty shape differs. Both use a **one-day horizon**, not the deployed
seven-day SLSQP controller. Batched hourly predictions are checked against scalar
physics, with scalar rechecks at switches and near both minima. Results/PNGs stay
ignored under `results/mpc/objective_closed_loop_v1/`; preserve existing outputs.

| Annual mean | Surrogate | Binary |
|---|---:|---:|
| Full return | -50,066 | -15,976 |
| Flood days | 30.0 | 5.0 |
| Low-water days | 36.8 | 0.0 |
| Unmet demand (million m³) | 532.94 | 538.52 |
| Flood exceedance integral (m·days) | 2.72684 | 2.79651 |

Binary improves reward in all ten years, but not every physical metric. Most
surrogate flood days were less than 1 mm above threshold. The binary arm has fewer
counted flood days yet slightly more cumulative flood-level exceedance and a
slightly higher mean annual peak. The exceedance integral uses daily endpoints;
it is not flood volume or calibrated damage. These results establish a reward
alignment effect within this paired hindsight experiment, not a better deployable
seven-day MPC or a fair comparison against PPO.

### Non-oracle forecasts under the same search

```bash
uv run --group reproduction python scripts/analysis/closed_loop_mpc_objectives.py --inflow-source seasonal
```

This repeats the **one-day, 10,001-action search** on all ten validation years,
using inflow and demand climatology fitted only on 1946–1987. Actual daily inflows
enter the simulator, never the controller. Both objectives keep identical physics,
action grids, resets, and forecasts. No PPO training or chronological test runs.
The seven-day MPC is unchanged.

| Annual mean, seasonal forecast | Surrogate | Binary |
|---|---:|---:|
| Full return | -50,716 | -50,463 |
| Flood days | 30.4 | 30.2 |
| Low-water days | 37.2 | 37.0 |
| Unmet demand (million m³) | 530.39 | 530.25 |
| Flood exceedance (m·days) | 2.98780 | 2.98242 |
| Low-water exceedance (m·days) | 0.33147 | 0.32900 |

Binary improves reward in nine of ten years, but the mean gain is only 253 points
(0.5% of the surrogate cost magnitude), versus 34,089 under perfect inflow. Severity
and demand improve only slightly; mean annual peak level is essentially unchanged
(1.31514 versus 1.31518 m). Exceedance uses daily endpoints, not calibrated damage.

The large hindsight benefit does **not** survive these seasonal forecast errors.
Accurate inflow makes threshold targeting effective for event-count reward; under
uncertainty, changing the penalty alone offers little improvement. This is evidence
for investigating forecast uncertainty and safety margins after choosing an explicit
engineering-risk criterion. It does not establish which forecasting method will work,
or that a one-day grid is a mature engineering baseline.

Outputs and three PNGs are ignored in `results/mpc/objective_closed_loop_seasonal_v1/`.
The four-arm comparison reuses the preserved hindsight runs after verifying source
hashes and identical search logic. Saved metadata confirms both reference and initial
MPC results were unchanged. Use a new `--output` directory to rerun.

### Causal past-inflow forecast comparison

```bash
uv run --group reproduction python scripts/analysis/compare_mpc_forecasts.py
```

All six arms use the same one-day 10,001-action grid, annual resets, seasonal
demand, and validation years 1988–1997. Three inflow forecasts are crossed with
surrogate/binary penalties:

- **Seasonal:** training-year daily climatology, reproducing the earlier result exactly.
- **Persistence:** yesterday's inflow predicts today's inflow.
- **Seasonal anomaly:** today's climatology plus 0.7572 times yesterday's departure
  from climatology, clipped at zero. The coefficient minimizes one-step squared
  prediction error using within-year pairs from the 42 training years only.

DamEnv does not expose inflow in its default policy observation or step info.
With evaporation disabled, the controller reconstructs only the **completed** day's
inflow from `(new_storage - previous_storage) / 86400 + actual_release`. The runner
checks these estimates against past forcing after each rollout. January 1 uses
climatology; no previous-year state or current/future inflow is supplied. This
assumes exact storage/release telemetry and provides additional history beyond
current PPO inputs; it is not a matched PPO comparison or verified operator setup.

| Forecast | Penalty | Return ↑ | Flood days | Low days | Unmet demand, million m³ | Flood severity, m·days | Low severity, m·days |
|---|---|---:|---:|---:|---:|---:|---:|
| Seasonal | Surrogate | -50,716 | 30.4 | 37.2 | 530.39 | 2.988 | 0.331 |
| Seasonal | Binary | -50,463 | 30.2 | 37.0 | 530.25 | 2.982 | 0.329 |
| Persistence | Surrogate | -30,051 | 12.5 | 25.4 | 549.10 | 2.812 | 0.202 |
| Persistence | Binary | -28,508 | 12.5 | 18.5 | 558.64 | 2.827 | 0.126 |
| Seasonal anomaly | Surrogate | -37,087 | 18.1 | 32.0 | 537.64 | 2.842 | 0.273 |
| Seasonal anomaly | Binary | -35,364 | 17.5 | 27.1 | 543.05 | 2.869 | 0.212 |

Both history forecasts improve return over seasonal climatology in all ten years
for either penalty. Persistence/binary gains 21,955 points but increases unmet
demand by 28.38 million m³/year. Flood severity falls about 5%, far less than the
59% decline in flood-day counts. Changing from persistence/surrogate to binary
reduces low-water events but slightly increases flood severity and unmet demand.
Daily-endpoint severity is not calibrated damage.

Forecast MAE/RMSE (m³/s) are 65.12/117.60 for seasonal, 40.75/84.69 for persistence,
and 39.51/79.78 for seasonal anomaly. **The lowest prediction error does not produce
the best control score.** A regression objective averages errors differently from
threshold-sensitive reservoir decisions. These experiments do not isolate the
specific error regimes responsible, establish a final risk preference, or prove
seven-day performance.

Next extend these causal forecasts across the seven-day planning horizon, keeping
solver and costs fixed; compare physical trade-offs before adding safety margins.
This requires horizon-specific training fits and timing tests, not simply giving
the planner future observed inflows. Preserve the original baseline and reserve
chronological test years until the protocol is frozen. Any subsequent PPO comparison
must explicitly match the extra history or label the information difference.

Ignored outputs: `results/mpc/forecast_study_v1/` contains annual/daily metrics,
forecast errors, all pairwise annual differences, provenance, and a PNG. All 60
annual episodes completed; no training of PPO or test evaluation occurred. The
archived execution source and formatted current source have identical Python ASTs.

### Causal forecasts in seven-day MPC

```bash
uv run --group reproduction python scripts/analysis/compare_causal_mpc.py
```

`CausalMPC` supplies forecasts to the original `DeterministicMPC` without changing
its surrogate cost, seven-day horizon, SLSQP options, initialization candidates,
warm starts, fallbacks, hydraulic model, or precision. The planning horizon still
shrinks near the end of each 365-step episode. Demand remains training climatology.
This is a forecast-only ablation; binary grid search is not used here.

- **Seasonal:** unchanged daily training climatology.
- **Persistence:** the most recent completed day's reconstructed inflow is held
  constant across every remaining planning lead.
- **Seasonal anomaly:** each lead's climatology is adjusted by a direct regression
  on that last observed anomaly. Separate coefficients are fitted at lags 1–7,
  using only within-year pairs from 1946–1987 and constrained to [0, 1]. They are
  0.7572, 0.5610, 0.4670, 0.4092, 0.3647, 0.3417, and 0.3251. These are direct
  horizon fits, not powers of the one-day coefficient.

On January 1, all leads use climatology because there is no preceding observation
within the reset episode. Thereafter, only completed-day storage changes and actual
releases are used to reconstruct past inflow. Current/future observed inflows are
never controller inputs; actual forcing is used afterward to score forecast error
at each lead. Exact storage/release telemetry and disabled evaporation are assumed.
This adds history beyond the current PPO policy and is not a matched PPO comparison.

The runner checks common forcing/reset conditions, past-inflow reconstruction,
and exact request/level/reward reproduction of the preserved seasonal baseline.
Outputs stay ignored under `results/mpc/causal_seven_day_v2/`. No PPO training,
chronological test evaluation, or baseline replacement occurs.

| Annual mean | Seasonal | Persistence | Seasonal anomaly |
|---|---:|---:|---:|
| Full return ↑ | -50,274 | -28,931 | -35,473 |
| Flood days | 30.0 | 13.8 | 17.7 |
| Low-water days | 36.9 | 14.7 | 27.1 |
| Unmet demand (million m³) | 530.68 | 564.71 | 538.48 |
| Flood exceedance (m·days) | 2.98678 | 2.76241 | 2.85080 |
| Low-water exceedance (m·days) | 0.32899 | 0.11147 | 0.21407 |
| Mean annual peak level (m) | 1.31521 | 1.30360 | 1.31229 |

Both causal forecasts improve return in all ten years. Persistence gains 21,343
points/year, reduces flood/low-water severity by about 8%/66%, but increases unmet
demand by 34.03 million m³/year (6.4%). Its 54% reduction in flood-day counts is
much larger than its severity improvement. Seasonal anomaly offers a smaller reward
gain with less additional unmet demand. These are engineering trade-offs rather
than a uniformly better controller.

The anomaly forecast has lower RMSE at every lead: 79.78–110.06 m³/s versus
84.69–131.49 for persistence. Persistence nevertheless has better control reward.
Persistence becomes worse than climatology on RMSE at leads 3–7, so its successful
closed-loop score is not evidence that flat persistence is an accurate week-ahead
forecast. Neither forecast-error ranking nor reward alone establishes a final
engineering baseline.

| Optimizer, all 3,650 decisions | Seasonal | Persistence | Seasonal anomaly |
|---|---:|---:|---:|
| SLSQP failure days | 20 | 149 | 52 |
| Fallback days | 151 | 299 | 152 |

The solver and fallback logic are unchanged; different forecasts and trajectories
change optimization difficulty. Fallbacks include successful solver results that
are worse than an initialization candidate. These results evaluate the complete
controller with its fallback, not globally optimal seven-day plans. Persistence
cannot be promoted without inspecting those additional failures and resolving the
supply-versus-level-risk criterion. Current results also do not isolate the effect
of horizon length from search method relative to the earlier one-day grids.

All 30 annual episodes completed; the seasonal arm reproduced original daily
requests, levels, rewards, inflows, and demands exactly. Saved metadata verifies
unchanged sources, inputs, and original baseline. `paired_metrics.csv` includes
annual differences; `forecast_predictions.csv` and `forecast_accuracy.csv` retain
all planning leads. Tests cover training membership, lead fits, timing, reset,
year-end horizon handling, unchanged seasonal solver behavior, and severity merges.
An initial run in `causal_seven_day_v1/` failed during summary aggregation after
rollouts; it is labeled incomplete. The corrected run uses `v2` and saves each
completed year before aggregation.

### Audit of additional causal MPC solver failures

```bash
uv run --group reproduction python scripts/analysis/audit_mpc_solver.py
```

This is an offline audit of saved seven-day decisions, not another annual policy
comparison. It replays all three forecast variants and verifies emitted requests,
solver flags, iteration counts, initialization/planned costs, and every forecast
lead against the saved run. Actual historical states are replayed without applying
alternative actions. The original cost, solver settings, and fallback are unchanged.

The audit inspects the final iterate returned by SLSQP, including those discarded
when its success flag is false. "Bounded" here means nominal request bounds; it
does not certify acceptable lake levels or realized supply. A 401-point first-action
slice, also including the executed first action, holds the rest of the fallback
plan fixed. It covers every persistence failure and the nearest same-year successful
nonfallback state, with ties going to the earlier day and duplicate controls removed.
These controls are descriptive, not matched causal counterfactuals.

Outputs remain ignored in `results/mpc/causal_solver_audit_v1/`: exact replay,
discarded iterates, selections, per-year/status counts, paired failure flags,
first-action slices, and PNG. Planning costs are divided by 1,000; a material gain
here means more than 0.001 in that scale. Overlapping horizon gains cannot be added
or interpreted as annual reward gains, and fixed-tail slices do not establish a
global optimum. Lower surrogate cost alone need not improve binary event reward,
unmet demand, or exceedance severity.

All 10,950 decisions replayed exactly. Every reported failure is SLSQP status 9
(iteration limit, 35 iterations), not a reported infeasible-constraint status.
Persistence has 149 failures: 47 in 1989. Only 3 failure dates overlap the 20
seasonal failures; there are 146 persistence-only and 17 seasonal-only dates.
These policies follow different states, so overlap is descriptive, not causal.

| Failed decisions | Seasonal | Persistence | Seasonal anomaly |
|---|---:|---:|---:|
| Iteration-limit days | 20 | 149 | 52 |
| Final iterate within request bounds | 20 | 149 | 52 |
| Final iterate materially lowers initialization cost | 18 | 148 | 51 |

The existing controller rejects every unsuccessful result and uses its best
initialization candidate instead. For persistence, the discarded final iterate
improves predicted cost on 148/149 days; one is worse by 0.04929 in normalized
planning cost. The median improvement is 0.11154 (111.54 planning-cost units before
division by 1,000). Request-bound feasibility does not establish acceptable flood
risk or supply, and a success flag alone is not a global-optimality certificate.

Holding each remaining six-day tail fixed, first-action slices improve predicted
cost materially on **141/149 failure states**, versus **21/137 distinct nearby
successful nonfallback controls**. Median gains are 0.06777 versus zero. The
largest failure-state gain is 0.19923. These are selected local search diagnostics,
not population estimates or annual performance gains. Actual failure days include
30 floods and 47 low-water days, but those associations do not establish that the
solver failure caused the violations.

The evidence points to useful candidates being discarded at the iteration cap;
it does not establish the numerical mechanism preventing convergence. The next
focused experiment is to retain the lower-cost candidate among the initialization
and a finite, bounded final iterate even when status is 9, while keeping the
failure diagnostic visible. Keep the existing iteration budget, forecast, and cost
fixed. Reject nonfinite, out-of-bound, and worse iterates; test the change in full
validation trajectories with severity and supply metrics before adopting it.
Blindly accepting all unsuccessful iterates or increasing the budget is not yet
supported as the best intervention. No such controller change occurred in this audit.

### Cost-checked retention of iteration-limit iterates

```bash
uv run --group reproduction python scripts/analysis/compare_mpc_retention.py
```

This pairs original fallback and experimental retention across all ten validation
years with the **same seven-day persistence forecast**. Costs, demand climatology,
SLSQP settings (35 iterations), initialization candidates, and hydraulic model stay
fixed. Subsequent warm starts and trajectories can change after an iterate is kept.
The original fallback arm must reproduce saved daily actions, levels, rewards,
solver flags, iteration counts, and planning costs exactly.

`RetainingMPC` is experimental; the original `DeterministicMPC` and historical
sources remain unchanged. Its planning/solver setup is checked against the original
Python AST, and the disabled arm is tested against the reference. Candidate
selection is the only intervention. A failed iterate may be kept only if:

- The status is **9 (iteration limit)**; other unsuccessful statuses keep the fallback.
- Coordinates are finite and within nominal bounds to the existing 1e-8 tolerance.
- Recomputed cost after clipping coordinates to bounds is finite and improves on
  the best initialization by more than 1e-9 in normalized planning cost.

`solver_success=False` and the original solver status remain recorded. Separate
`retained_iteration_limit` and `selected_source` fields identify retained iterates;
`used_fallback` means the initialization was selected. No failure is relabeled as
successful optimization. Request bounds do not guarantee preferred lake levels.
Unit tests cover nonfinite/out-of-bound/equal/worse candidates, other failure
statuses, recomputation rather than trusting `result.fun`, and preserved flags.

Ignored outputs in `results/mpc/iterate_retention_v1/` include annual/daily metrics,
solver diagnostics, paired annual differences, runtime, provenance, and a PNG.
Each completed year is saved before final aggregation. This is validation only,
with extra past-inflow information beyond current PPO; no default-controller
replacement, training, or chronological test evaluation is performed.

| Annual mean | Original fallback | Cost-checked retention |
|---|---:|---:|
| Full return ↑ | -28,931 | -26,361 |
| Flood days | 13.8 | 12.3 |
| Low-water days | 14.7 | 10.4 |
| Unmet demand (million m³) | 564.71 | 565.60 |
| Flood exceedance (m·days) | 2.76241 | 2.75032 |
| Low-water exceedance (m·days) | 0.11147 | 0.06760 |
| Mean annual peak level (m) | 1.30360 | 1.30393 |

All 20 annual episodes completed; the original arm reproduced the saved run
exactly. Retention improves return in every validation year (mean +2,569.79;
range +293.31 to +7,048.43). Low-water severity falls about 39%, whereas flood
severity improves only 0.44%. Unmet demand increases by 0.88 million m³/year
(0.16%). Flood severity increases in some years, and the largest annual peak
increase is 1.24 cm in 1992; mean annual peak increases about 0.33 mm. This is
not uniform physical-risk dominance.

| All 3,650 decisions | Original fallback | Cost-checked retention |
|---|---:|---:|
| Reported solver failures | 149 | 131 |
| Retained iteration-limit iterates | 0 | 131 |
| Initialization fallback days | 299 | 149 |
| Mean decision time (ms) | 13.82 | 13.63 |

Every retained iterate still has `solver_success=False` and status 9; all passed
finite/bounds/cost checks. Fewer reported failures reflect changed warm starts
and trajectories, not a larger iteration budget or a relabeled success flag.
There are still 149 initialization fallbacks from unsuccessful candidate selection
on otherwise successful solver calls. Runtime is descriptive of these CPU runs;
no speed improvement is established.

This supports retention as a promising experimental candidate under persistence,
not automatic adoption across forecasts or a globally optimal MPC claim. The
next decision is the engineering supply-versus-level-risk criterion before tuning
safety margins or freezing a baseline. A final PPO comparison must match the added
history or explicitly separate the information conditions. The original controller,
configs, earlier studies, and chronological test years remain unchanged.

### Matched MPC horizon experiment

```bash
uv run --group reproduction python scripts/analysis/compare_mpc_horizons.py
```

All 30 annual validation runs use persistence inflow, training-only demand,
the same surrogate cost, SLSQP settings/cap, initialization recipe, and cost-checked
iteration-limit retention. Only the planning horizon changes (1, 3, or 7 days).
Each applies the first request and replans daily; all have zero terminal value,
annual resets, and a shrinking horizon near year end. Seven days reproduced the
saved retention experiment exactly, including daily solver diagnostics.

Annual averages:

- **1 day:** reward **−36,015**; **18.4** flood days, **25.6** low-water days;
  unmet demand **549.03 million m³**. Flood/low-water exceedance:
  **2.84787 / 0.20397 m·days**.
- **3 days:** reward **−28,077**; **12.8** flood days, **15.7** low-water days;
  unmet demand **557.83 million m³**. Exceedance: **2.80220 / 0.10145 m·days**.
- **7 days:** reward **−26,361**; **12.3** flood days, **10.4** low-water days;
  unmet demand **565.60 million m³**. Exceedance: **2.75032 / 0.06760 m·days**.

Seven days beats one day on reward in all ten years and three days in nine.
The exception is 1994: three days gains 10,718 reward points with 11 fewer flood
days, but 15.27 million m³ more unmet demand. This supports retaining seven days
as the best mean-reward setting among these three, not a universal optimum or
physical-risk dominance. Longer planning improves average level metrics while
leaving more demand unmet in this experiment.

Mean decision times were **1.55 / 3.73 / 13.68 ms** for 1/3/7 days. The corresponding
solver failure counts were **0 / 5 / 131**, all failure iterates retained by the
cost/bounds checks; initialization fallbacks were **0 / 41 / 149**. Average function
evaluations were **20.1 / 34.7 / 79.2**. The same 35-iteration cap does not equalize
computation, and success is not global optimality. Changing horizon also changes
optimization dimension and warm-start length. These results describe this practical
controller, not exact optimal control at each horizon. Earlier one-day exhaustive
grids used a different search and do not replace this matched comparison.

The prototype horizon is therefore supported by this validation ablation rather
than assumed. It remains possible that another horizon, improved forecasts,
terminal value, or a different search changes the ranking. No new horizon sweep
or default-controller change is implied. Next investigate matched past-inflow
observations for PPO, preserving the four-feature control and all five seeds.

Ignored `results/mpc/horizon_study_v1/` contains per-year differences versus seven
days, full daily/annual metrics, timing, failure/retention counts, source hashes,
and the comparison PNG. No chronological test evaluation or training occurred.
