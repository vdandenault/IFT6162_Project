# Simulator and training gym

All six development directions now have working research implementations and
bounded checks. **Operational calibration remains incomplete**: timestamps/datum,
measured aligned outlet flows, actuation and river parameters still need evidence.
The original annual benchmark and saved PPO/MPC/FQI/SAC results remain unchanged.

**Source audit update:** [calibration evidence](CALIBRATION.md) now supports the
Malgrate rating-curve reference and dated historical crossmatches. It also finds
that the benchmark target tracks historical outlet flow, and identifies a nearly
exact data-balance coefficient. These are not independent irrigation or operational
calibration. Unidentifiable ramp/routing/threshold parameters remain rejected for
promotion.

**Optional outlet variant:** [implementation and findings](CALIBRATION.md#outlet-ablation-implemented-findings)
separate storage reference from maximum capacity. This variant remains experimental:
fixed release and SAC warmup can leave its declared low-water evaluation range.
The [optional reserve operating rule](CALIBRATION.md#explicit-low-water-operating-protection)
addresses that low-water failure in the tested runs and completes three short SAC
smokes. Constant-release performance is mixed, and a high-water domain exit remains.
The [high-water replay](CALIBRATION.md#high-water-diagnosis-release-timing-and-model-limits)
shows that earlier release can avoid this range exit, while late requests are
ineffective under forced opening. Finer integration still crosses the limit;
avoiding that limit is distinct from avoiding flood penalties.
The [fixed-cost protected-MPC screen](#protected-mpc-with-causal-forecasts) now
tests this anticipation with prior-only forecasts. MPC stays in range, but feedback
still has lower cost in the tested windows.

## Start here

The current configuration is [simulator.json](../configs/simulator.json).
Use the operational environment for new experiments:

```python
import numpy as np
from reservoir_control.systems.lake_como.simulation import make_operational_env, Operations

env = make_operational_env(
    "train", years=(1946, 1947), decision_hours=6,
    operations=Operations(ramp_m3_s_per_hour=5, river_residence_hours=6),
)
observation, info = env.reset(seed=901)
while True:
    observation, reward, terminated, truncated, info = env.step(
        np.array([100.0], dtype=np.float32)
    )
    if terminated or truncated:
        break
env.close()
```

These ramp/routing values are **illustrative sensitivity settings**, not measured
Olginate limits. The new environment defaults to a physical objective, with explicit
weights; its return is not the benchmark return or a monetary damage measure.

Run the complete declared suite into a fresh directory:

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.study \
  --output results/simulator/new_operational_study
```

Stages can be run separately: `prepare`, `validate`, `forecast`, `stress`, `controls`,
`train`, `aggregate`. `prepare` freezes sources/configuration. Existing outputs are
protected. The completed active run is `results/simulator/operational_v2/`.
The command includes fresh, bounded SAC integration smokes; it is not just a plotter.

For intuition, start with the local [operational-gym walkthrough](../background/OPERATIONAL_GYM.md)
and PNGs 37–39. `background/` and generated results remain ignored.

## 1. Continuous training protocol

`TaskProtocol` and `TrainingEpisodes` declare the sequence length, initialization,
spin-up, action interval, discount and terminal behavior. The default smoke uses
random contiguous **two-year training-only windows**, fixed initial level 0.35 m,
and **30 days at 100 m³/s** before scored interaction. Spin-up advances lake, river,
actuator and demand memory; it is counted as interaction but excluded from replay.
An optional initial-level range is a declared experiment distribution, not an
empirically estimated operating-state distribution.

Eight operational observations are exposed: **level, seasonal sine/cosine, past-demand
EMA, completed-decision mean inflow, actuator command, river storage, time-to-go**.
Command/river telemetry is idealized simulated state. It is not established that
all of it would be directly measured in deployment. Hidden hydrological conditions
remain; time-to-go alone does not make the entire system fully observed.

Two terminal modes are explicit:

- **`finite_horizon`:** the declared task ends with `terminated=True`; remaining-time
  fraction is observed. An optional terminal reserve penalty applies once at the end.
- **`continuing_record`:** exhausting supplied data gives `truncated=True`; time-to-go
  stays constant. The learner bootstraps from the correct final physical state,
  then resets. A finite terminal-reserve penalty is rejected in this mode.

The new SAC loop masks **termination only**, while resetting on either end flag.
It does not store a reset state as the preceding transition's next state. The old
SAC study's finite episodic rule is preserved in its original code.
[Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
supports separating intrinsic finite horizons from external truncation.
Discount scales with time: `gamma_per_decision = gamma_per_day ** (hours / 24)`.
Unit scales are fixed, with reward scale 1 for the new physical-cost smoke; no
normalization is fitted on validation/test data. Update counts still differ across
control frequencies, so equal elapsed time is not equal computation.

`ContinuousMPC` carries its warm-start plan across model-year boundaries and forecasts
past December 31 using the declared seasonal model. `reset()` is explicit, never
inferred from January 1. `ContinuousResidual` adapts a frozen six-feature proposal-aware
checkpoint, using the carried past inflow; it deliberately ignores the new extra
features and is a transfer diagnostic, not a matched-information retraining result.
Daily legacy MPC assumes no new ramp/routing dynamics; use the operational scenario
planner when those mechanisms are included in prediction.

## 2. Physical validity and observed-data checks

The operational model reuses the pinned level/storage and release curves. A complete
single-year compatibility run reproduces the old storage trajectory exactly, with
reward discrepancies below 10⁻⁹. Tests also check routing balance, evaporation units,
actuator/forced-release distinction, cross-year state continuity and API behavior.

The sampled rating-curve audit evaluates **36,865 level/day combinations** and finds
finite nonnegative bounds. This is not calibration or proof over every possible state.
The active-storage datum is not the lake's geological bottom: levels outside the
pinned operating range are reported, not silently repaired by creating water.

A separate training-only observed-level check keeps inflow offsets **−1, 0, +1**.
It infers a release proxy from storage changes and compares observed next storage
with model trajectories under zero and maximum nominal requests. Under zero offset,
**12.86%** of 15,288 transitions lie outside that extreme-request envelope; ±1 offsets
also show discrepancies. No offset is selected. The envelope is not an exhaustive
reachability proof, and inferred releases are not measured gate decisions.
Unverified timing, datum, averaging conventions and model mismatch all matter.

A strict `measured_balance(frame, metadata)` gateway supports supplied observations
with columns `timestamp, level_m, inflow_m3_s, outflow_m3_s`. It requires a source,
timezone, gauge datum, SI units, and explicit confirmed alignment: levels at the
timestamp, flows averaged until the next timestamp. It checks actual elapsed seconds,
including timezone effects. It does not invent missing measurements or certify a
provider's alignment declaration. Evaporation and unmodeled inputs must be considered
before interpreting residuals as model error.

Sources checked: [Gym4ReaL Appendix C.1/D.1](../papers/Gym4ReaL-%20A%20Suite%20for%20Benchmarking%20Real-World%20Reinforcement%20Learning.pdf) confirms daily records and the pinned model
parameters but does not settle the CSV date anchor or gauge mapping. Its expert-action
reconstruction suggestion does not turn a balance-derived proxy into measured releases.
The [operator's publications](https://www.addaconsorzio.it/attivita/) include historical
lake-level/inflow/outflow studies. [ARPA Lombardia's hydrology page](https://www.arpalombardia.it/temi-ambientali/acqua/acque-superficiali/quantita-idrologia/)
identifies hydrometric data, gauge zeros and rating curves as relevant sources.
These leads are not yet a verified join to the Gym4ReaL rows.

## 3. Forecast uncertainty and stress inputs

`HistoricalScenarios` accepts exactly the chronological training years. It draws
**contiguous historical blocks** near the model season, conditioned on the last
observed flow. The nearest-history pool is fixed independently of requested scenario
count. Cross-year blocks stay within training records. This retains within-block
dependence; it is not a calibrated stochastic hydrological model.

The declared check fits on training years and scores 1,820 seven-day lead values on
validation years. Mean absolute errors are **53.23 m³/s** for the historical scenario
mean, **61.30** persistence, **67.01** climatology. Empirical 10–90% scenario ranges
cover **71.04%**, demonstrating undercoverage relative to an 80% nominal interpretation.
No forecasting method is promoted from this single development check.

Separate labeled stresses include 0.5× daily inflow, 1.5× daily inflow, and early/late
six-hour pulses. Intraday shapes preserve each supplied daily volume. Scaling daily
inflow changes the water budget intentionally; none of these scenarios is assigned
a real-world probability or called measured hourly weather.

The [scenario-MPC paper](https://arxiv.org/html/2309.00373v1) motivates representing
inflow uncertainty in decisions. Our block generator and finite candidate-schedule
planner are new adaptations; they do not reproduce its Prophet/CVX implementation.

## 4. Actuation and downstream water

Optional command ramping limits changes in **requested flow command per hour**.
Actual flow remains subject to lake-level hydraulics and can exceed or change faster
than that command. This is a flow-command proxy, not calibrated gate-position dynamics.

Optional routing uses a linear reservoir. For constant hourly inflow, the river store
updates analytically; outlet volume equals input volume minus storage increase.
The residence parameter is a response time, **not a fixed travel-time delay**.
Bypass routing recovers direct release. Environmental flow stays in the river while
available water is diverted toward demand: it is not subtracted twice from conservation.
Diversions leave this modeled system; return flows and individual canal networks
are not represented.

The demonstration uses 5 m³/s/hour ramping, 6-hour residence and a 300 m³/s downstream
reference. These are uncalibrated sensitivities, not legal limits or verified channel
capacity. The downstream reference measures exceedance after diversion; it does not
cap physical discharge or model an inundation map. Optional supplied evaporation is
converted from mm/day to volume using the pinned surface; no measured series is bundled.
Hydropower is not computed, and surplus water is not electricity.

## 5. Objectives, finite constraints and recovery

`PhysicalWeights` separates shortage volume, flood/low-water excess integrated over
time, downstream excess volume and optional final reserve. Weights carry their units
in their names; equal numerical weights do not establish stakeholder priorities.
Exposure is based on piecewise-linear interpolation of hourly model levels, not
measured flood damage. The physical cost is additive across decision intervals for
the same realized hourly requests.

The benchmark objective is available for daily, bypass-routing compatibility checks.
It is rejected for subdaily control or added terminal reserve. These scores must not
be mixed with the new physical objective or compared to the old results table.

`ScenarioConstraintFilter` searches bounded action candidates under sampled causal
inflow paths. Feasible candidates satisfy modeled lake and configured downstream
bounds over that interval. It chooses a nearby proposal; when no sampled candidate
is feasible, it explicitly reports recovery and chooses the least normalized violation,
with endpoint/cost/distance tie-breaks. This is **not** a no-flood guarantee: scenarios
are finite, states/plans have numerical precision limits, and no grid witness is not
proof that every continuous action is infeasible. Raw commanded and delivered flow
remain distinct. Actual violations must still be logged.

## 6. Subdaily decisions and results

The same hourly model supports decisions every **1, 3, 6, 12 or 24 hours**.
Demand EMA updates after each completed day; observed inflow uses only the completed
decision interval. Synthetic profiles are identified explicitly. No extra observed
weather is created by making the policy act more frequently.

Twelve declared cases simulate **210,240 total model hours**, including shared 30-day
spin-up periods. All tested lake/river balances remain below 10⁻⁶ m³ residual. Holding
the same 100 m³/s request produces the same trajectory at daily/six-hour/hourly cadence.

The causal feedback rule performs worse at higher cadence on the synthetic front-loaded
pulse: flood exposure is approximately **118 / 783 / 794 hours** for daily / six-hour /
hourly decisions over 700 scored model days. Shortage also rises. The rule reacts to
recent inflow; this sensitivity does not prove subdaily control is generally worse.
It shows that extra action opportunities require appropriate forecasting/control.
No trained-method ranking or real-event benefit is claimed.

Three fresh daily SAC seeds each complete **700 scored steps, 30 spin-up steps and
636 updates**. Two six-hour smokes exercise finite/continuing modes for 128 steps
each. Total: **2,036 gradient-update rounds**, with all collection/spin-up charged.
These are integration smokes, not converged results. Controller checks also exercise
cross-year MPC warm start and scenario planning/filtering at 24/6/1-hour intervals.

## Protected MPC with causal forecasts

`reservoir_control.systems.lake_como.simulation.protected_mpc` implements a finite-candidate MPC prototype
using the protected outlet physics. It is separate from frozen benchmark MPC and
does not reproduce the published continuous/scenario optimization algorithm.
The batched planning kernel is tested against `ProtectedOutletModel`; every
selected first-day scenario prediction is also checked against scalar protected
rollout at runtime. Costs are the same physical target-deficit/flood/low-water
costs and weights as the preceding experiments, with no discount or terminal cost.
No actor or critic is trained.

The declared screen is `results/simulator/protected_mpc_v1/`:

- **1986–1987 training-event replay:** forecasts fit only 1946–1985, excluding
  both replay years. This event was already investigated; it is not untouched
  validation.
- **1988–1989 chronological validation screen:** forecasts fit 1946–1987 only.
  No configuration was selected or retuned using these results. This is two years,
  not the full validation split or a definitive algorithm ranking.
- **68-day same-state event branch:** all controllers start from the saved constant
  controller state 61 days before its rejected trial and run until seven days
  afterward. The branch timing is selected in hindsight; forecast/action inputs
  themselves remain causal.

Each case compares constant request, causal feedback, and **persistence, seasonal
and conditional historical forecasts, each at one- and seven-day MPC horizons**:
24 arms total. Conditional forecasts draw five contiguous historical paths, using
season and previous observed inflow, from a fixed nearest-128 pool. The source
blocks are chosen as seven-day blocks for both horizons so their first-day forecasts
match exactly. These are empirical samples, not calibrated uncertainty bounds.
All modes use the same prior seasonal benchmark-target forecast; actual irrigation
demand is still unverified. Forecasts use no operational weather/inflow outlook.

Search uses nine constant schedules, a forecast service schedule, the current
feedback request, first-day grid/feedback requests followed by service, and a
shifted previous plan when available. Requests are bounded native float32 values.
This is a restricted schedule search, not a global optimization claim. Predicted
domain exits make a schedule infeasible in any sampled path; they do not receive
a fabricated finite penalty. When no schedule survives, maximum nominal request
is a labeled **uncertified recovery**, not a safety guarantee. Costs stay fixed,
but this explicit model-domain feasibility rule is part of the planner.

All arms receive the same eight observations. At full-record reset, the last-inflow
fallback comes from prior fitted last-model-day (sample 365) climatology, not the
target replay year. The plan's "Dec-31" label refers to that model-day sample;
it does not establish December 31 alignment for the truncated leap-year records.
The shared-state branch receives the actual available previous-day inflow and saved
demand history. The planner receives physical parameters and a fitted bank, not
live future record arrays. Forecast errors are scored only after actions are frozen.
Both model years retain the existing first-365-samples convention, including 1988.

**Outcome: staying in range is achievable; a cost advantage is not demonstrated.**

- All **18 MPC arms** complete their declared windows without an actual domain
  exit. Constant request still exits in the 1986–1987 replay and shared-state branch;
  feedback completes all three cases. Avoiding 2.64 m does not eliminate flood
  exposure above 1.24 m.
- In the same-state event branch, conditional MPC's peak falls from **2.3506 m
  (one day) to 2.2754 m (seven days)**. Feedback peaks at **2.2768 m**. Seven-day
  MPC cost improves from **453.13 to 429.42**, but feedback is better at **378.36**.
  All three costs cover the same 68 days; lower is better.
- Over complete 1986–1987, physical cost is **1245.96 for feedback**, versus
  **1632.86 / 1638.77** for conditional one-/seven-day MPC. Over 1988–1989 it is
  **1048.65**, versus **1230.75 / 1348.43**. Feedback beats all six fixed MPC
  variants on total cost in both complete two-year windows. These costs are not
  the annual benchmark returns reported in `RESULTS.md`.
- On 1988–1989, conditional seven-day MPC reduces low-water exposure versus one
  day (**54.75 → 2.77 m·hours**) but increases unmet benchmark target
  (**1176.00 → 1345.66 million m³**). Neither eliminates target-delivery tradeoffs.
  Recorded environmental shortages are zero in these windows; the operating rule
  still permits them in other sufficiently dry states.
- Forecasts **underpredict the extreme surge**. The day before the local maximum
  begins, the conditional seven-day mean-path peak is **386.52 m³/s** and the largest
  of five sampled peaks is **800.4 m³/s**, versus the later observed daily inflow
  **1835.6 m³/s**. In the shared-state branch the seven-day conditional controller
  enters that surge **0.1570 m lower** than one-day MPC, reconciling with
  **22.91 million m³** additional cumulative prior release. This is post-hoc
  water accounting, not proof of successful extreme-event warning. One-day MPC
  also stays in range.
- Persistence needs labeled predicted-domain recovery on **seven decisions across
  its arms**; seasonal/conditional arms need none. The maximum selected first-day
  batch/scalar level difference is **zero at recorded precision**, and physical
  balance errors stay below **3 × 10⁻⁸ m³**.

The interpretation is therefore limited: causal model-based control can maintain
headroom in these examples, and longer planning reduces the event peak, but the
added forecast/search machinery does not outperform feedback at the fixed costs.
The common seasonal target forecast and restricted schedule search were subsequently
diagnosed separately in the [fixed-cost target/search experiment](#target-forecast-and-search-diagnosis).
Do not attribute the outcome to forecast quality alone or claim this model is
operationally calibrated. Final test years remain unused.

Artifacts include `summary.csv`, equal-length `paired_vs_feedback.csv`, per-lead
`forecast_skill.csv`, per-decision forecast/search logs and resumable arm outputs.
Incomplete runs are compared only on matched prefixes, never by mixing their
partial total returns with full-window returns. `headroom_inspection.json` is a
post-hoc audit; `supplementary_inputs.json` captures resolved physical settings and
additional configuration hashes after execution, not as an extra preregistration.

```bash
uv run --group reproduction python scripts/analysis/protected_mpc.py \
  --output results/simulator/new_protected_mpc
```

Stages `prepare`, `run`, `aggregate` can be used separately with `--stage`.
Repeating `run` skips completed arms and preserves interrupted attempts; source
hash changes stop resumption. See the [protected-MPC PNG](../background/images/44_protected_mpc.png)
for the event, validation costs and forecast shortfall.

## Target forecast and search diagnosis

`reservoir_control.systems.lake_como.simulation.mpc_diagnosis` runs a **two-by-two diagnostic** on the same
three cases above, holding conditional inflow forecasts, seven-day horizon,
protected physics, initial states, domain feasibility and physical costs fixed.
There are four MPC variants plus feedback per case: **15 arm results**, including
six reused frozen seasonal/restricted and feedback references. All complete.
Only the nine new intervention arms are simulated again.

The target change uses the **existing past-target EMA observation**, not a new
measurement or future target. It adds a per-lead correction to seasonal target
means based on the deviation of observed EMA from expected seasonal EMA. The
zero-initialization mass is corrected explicitly; at reset, with no history, the
forecast equals the original seasonal prior. Seven linear coefficients are fitted
once using only prior training windows, then kept fixed. Their training observations
at time n contain targets through n−1, while response labels are future targets
inside those prior windows. No replay or validation targets enter fitting.
Window sample counts overlap and do not represent independent observations.

The search change adds a **17-point daily coordinate grid and two fixed sweeps**,
retaining the restricted solution as incumbent and executing only the first daily
action. It jointly increases grid resolution, schedule flexibility and compute;
this study does not distinguish those three ingredients. Expected cost cannot
increase relative to its retained seed under the same forecast. It is still a
bounded local search, not a global optimum. No costs, terminal terms, uncertainty
margins or physical parameters are changed.

The frozen study is `results/simulator/mpc_target_search_v1/`. All comparisons use
the same prior-only fitting cutoffs and already-inspected evaluation windows as
the protected-MPC screen. One fixed scenario-sampling seed is used. No alternative
forecaster, regression regularization or sweep count was selected using validation.
The target remains the historical benchmark proxy, not independently verified
irrigation demand.

**Target forecasting is the larger identified contributor to this MPC gap.**

- In 1986–1987, correcting targets with the original search reduces cost from
  **1638.77 to 1319.89**. Feedback remains better at **1245.96**.
- In 1988–1989, it reduces cost from **1348.43 to 1043.81** (about **22.6%**).
  Feedback costs **1048.65**, so the difference versus feedback is only **4.84**,
  about **0.46%**, in this small screen—not a demonstrated general advantage.
- Validation first-day target MAE falls from **45.68 to 21.87 m³/s**; seventh-day
  MAE falls from **46.01 to 35.75 m³/s**. Matching errors across search variants
  confirm that search does not alter target predictions.
- Delivery improves at a low-water tradeoff: validation unmet benchmark target
  falls from **1345.66 to 1028.46 million m³**, while low-water severity rises
  from **2.77 to 15.35 m·hours**. Flood exposure is zero for both variants there.
  Operating-floor protection remains unchanged; it does not guarantee zero
  exposure below the −0.45 m penalty threshold.
- In the shared 68-day event branch, target-only correction reduces cost from
  **429.42 to 404.34**, still worse than feedback's **378.36**. This is not a
  forecast-only accuracy claim: the controllers are evaluated over trajectories.

**Expanded search provides small, inconsistent closed-loop benefits.**

- With seasonal targets, cost changes **1638.77 → 1624.38** in the training event,
  but **1348.43 → 1350.35** in validation and **429.42 → 431.33** in the branch.
- With corrected targets, expanded search worsens cost in all three cases:
  **1319.89 → 1329.49**, **1043.81 → 1049.30**, and **404.34 → 404.56**.
- The expanded full-window arms take about **12 times the runtime** of the
  restricted arms in this screen. This is descriptive compute accounting, not a
  portable speed benchmark or a compute-matched algorithm comparison.

**Same-state checks separate optimizer behavior from realized outcomes.** Fresh
controllers are evaluated on 60 fixed reference-MPC states (every 30 days in full
windows and every seven days in the event branch). All plans are frozen before
realized future inputs are accessed for ex-post scoring. Expanded search never
increases expected cost in any of the 120 paired search comparisons across target
modes. Gains exceed 1e−8 in **41** pairs. Realized seven-day open-loop plan costs
improve in **16** pairs, worsen in **6**, and remain unchanged within 1e−8 in the
other **98**. All audited changes in the first requested action have gains above
that numerical threshold; the effect is not explained by zero-gain tie changes.

Target-only effects in these static open-loop audits are also mixed (20 better,
14 worse, 26 unchanged). These states come from the original MPC, and entire plans
are scored without replanning. They are not the new controller's state distribution
or closed-loop performance. Expected costs under different target forecasts are
not treated as optimizer gains; only same-forecast search pairs support that claim.
Lower predicted cost therefore does not imply better realized control when
forecasts are imperfect, and static plan scores need not rank adaptive rollouts.

The history-aware target forecast with the original bounded search was subsequently
checked over [all validation years](#broader-year-validation-of-the-target-forecast),
without automatically promoting it as a new best controller. Extra search is not the first improvement to prioritize
based on this screen. Leave the costs fixed and keep physical calibration and
operating-priority limitations explicit. Final test years remain unused, and
the original benchmark results and controller configurations are unchanged.

Artifacts: `summary.csv`, within-case `paired_factors.csv`, per-lead
`target_forecast_skill.csv`, `search_gains.csv`, `same_state_audits.csv`, prior-fit
coefficient records, and per-arm trajectories/decision logs. All physical cost
identities and unchanged inflow forecasts can be checked against the reference.

```bash
uv run --group reproduction python scripts/analysis/mpc_target_search.py \
  --output results/simulator/new_mpc_target_search
```

Stages `prepare`, `run`, `audit`, `aggregate` support resumption with frozen-source
checks. The [target/search PNG](../background/images/45_mpc_target_search.png)
summarizes validation costs, target errors and paired planning diagnostics.

## Broader-year validation of the target forecast

`reservoir_control.systems.lake_como.simulation.target_year_check` freezes the history-aware target forecast
and original search and checks **all ten validation model years, 1988–1997**.
The primary expansion is **1990–1997**, reported separately from the previously
inspected 1988–1989 screen. Three controllers are compared: seasonal-target MPC,
history-aware MPC, and causal feedback. The six annual reference results for the
first two years are reused unchanged; the other 24 annual results are new.

All inflow/target fits remain frozen at **1946–1987** for every validation window.
The seven-day horizon, nine-point original search, five conditional inflow paths,
sampling seed, physical costs, operating priorities and model-domain limits are
unchanged. No validation refitting, hyperparameter changes or new policy inputs
are introduced. This is a validation expansion after exploratory model selection,
not independent final-test confirmation or a new benchmark-paper reproduction.

The protocol remains **five non-overlapping two-year episodes**, initialized at
level 0.35 m with target EMA zero and the prior last-model-day inflow fallback.
Storage, target history and the previous plan carry across the year boundary
inside each pair. This is not one continuous ten-year trajectory. The original
365-samples-per-model-year convention is retained, including leap-year labels.
Annual cost is the sum of that model year's existing physical stage costs.
Flood/low-water days count days with positive interpolated exposure severity.

The completed run is `results/simulator/target_year_check_v1/`. All **30 controller
years complete**, with zero actual domain exits. Missing/partial years would have
no full-year score; primary scope means and paired deltas are suppressed if any
required year/pair is incomplete. Failed prefixes are not silently discarded or
scored as cheap full years. Tests cover these failure cases.

**On the eight additional years (1990–1997):**

- History-aware MPC mean annual cost is **592.42**, versus **705.35** for seasonal
  MPC (**16.0% lower**). It wins **7/8 years** and all four two-year blocks against
  seasonal MPC. Next-day target MAE falls from **53.42 to 23.00 m³/s**; seventh-day
  MAE from **53.76 to 39.65 m³/s**. Both improve in every additional year.
- Feedback costs **607.59/year**, so history-aware MPC is **2.5% lower on average**.
  It nevertheless wins **4 years and loses 4**, and wins 3/4 two-year blocks.
  Its annual cost difference versus feedback ranges from **−70.61 to +41.91**.
  This supports a competitive candidate, not annual dominance or a universal
  best-controller claim.
- The risk tradeoff persists. Mean flood exposure is **79.58 m·hours/year** versus
  feedback's **99.23**, with **5.88 versus 8.63 flood days/year**. Mean low-water
  exposure is **3.51 versus 0 m·hours/year**, with **33.38 versus 0 low-water days**.
  Mean unmet benchmark target is **509.34 versus 508.36 million m³/year**—nearly
  equal, slightly worse for MPC. Environmental shortages are zero in these runs,
  not guaranteed zero by the operating rule in general.
- On the full ten-year validation set, including the earlier two years, mean
  annual cost is **578.32 / 699.13 / 590.94** for history-aware MPC / seasonal MPC /
  feedback. History-aware MPC beats feedback in **5/10 years**. The additional-year
  result above remains the primary expansion rather than blending old and new
  evidence without distinction.

Annual paired comparisons are descriptive: years share state within two-year
blocks and are not ten independent random trials. No p-value or confidence claim
is made. Only the original scenario-sampling seed is used, so forecast-sampling
robustness remains open. These physical costs are not comparable with frozen
annual Gym4ReaL/PPO rewards in `RESULTS.md`; the target remains an outflow-related
benchmark proxy, not verified irrigation demand.

**Carryover and conditions matter.** On additional-year first positions in each
episode, history-aware MPC costs **484.10/year** versus feedback **502.87**; on
second positions, **700.75 versus 712.32**. The mean advantage appears in both
positions, but calendar conditions and record-end effects differ too. This is
not a randomized test of initialization sensitivity.

Using only prior-training annual-inflow quartiles to define ex-post context, both
additional dry years (1990, 1995) favor feedback. These annual labels require the
realized whole year and are **not** online policy inputs or a proposed oracle
controller-switching rule. The sample is too small for general regime guarantees.

The 1994–1995 pair provides a concrete storage-carryover hypothesis. MPC costs
**531.80 versus 553.27** for feedback in wet 1994, but **525.97 versus 484.06** in
dry 1995. It enters 1995 at **−0.463 m**, versus feedback **+0.082 m**, about
**79.60 million m³ less relative storage**. It has 43 low-water days that year
versus feedback's zero. This association does not isolate the causal contribution
of initial storage from the controllers' subsequent actions; the
[four-way replay below](#controlled-1995-storage-carryover-replay) separates these
conditional effects. No reserve term or cost weight is adjusted here.

Artifacts include all annual rows in `annual.csv`, separate old/new/full scope
means in `scope_summary.csv`, annual/block comparisons in `paired_summary.csv`,
`episode_position.csv`, per-year forecast skill, domain-failure records and
explicit matched-prefix comparisons. No controller defaults or frozen reference
results are replaced. Final test years **1998–2010 remain unused**.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.target_year_check \
  --output results/simulator/new_target_year_check
```

Stages `prepare`, `run`, `aggregate` support resumption and source checks. The
1988–1989 reuse requires the preceding target/search study artifacts; reproduce
those first in a fresh checkout. See the [broader-year PNG](../background/images/46_target_year_check.png)
for annual variation, cost components and the 1994–1995 carryover example.

## Controlled 1995 storage carryover replay

`reservoir_control.systems.lake_como.simulation.carryover_check` compares MPC and feedback from **both**
1995 starting inventories. The four primary branches hold exogenous history,
forecast fits/paths, physics, constraints and costs fixed. Two additional MPC
branches reset its carried plan to check numerical warm-start dependence.

The experiment reconstructs the original 1994 MPC and feedback trajectories and
checks them against the saved year-check records. Their target EMA and last inflow
at the boundary are identical. The branch keeps the original **(1994, 1995)**
episode, hour **8760**, 365 remaining days, target-history clock and forecast seed
offset. It does not initialize a new one-year task. All initial observations except
lake level are identical. Initial command is normalized to a common value; ramping
is disabled, and both own-storage replays match their original trajectories/costs
within 1e−9. Primary MPC branches share the reconstructed 1994 MPC plan. The two
cold-plan controls produce **identical daily trajectories and costs** in this case.

The frozen run is `results/simulator/carryover_check_v1/`. All six branches complete
1995 without a domain exit. The four primary annual physical costs are:

- **Low start (−0.463 m):** MPC **525.97**, feedback **562.98**.
- **High start (+0.082 m):** MPC **501.16**, feedback **484.06**.

At the same low inventory, MPC has 37.01 lower cost. At the same high inventory,
feedback has 17.11 lower cost. Giving the higher inventory to MPC reduces its cost
by **24.81**; giving it to feedback reduces cost by **78.92**. This is a strong
storage/controller interaction, not a single policy-independent storage penalty.
The original 41.91 gap has two exact conditional decompositions:

```text
41.91 = 24.81  + 17.11   # MPC's storage benefit, then controller gap at high storage
41.91 = −37.01 + 78.92   # controller gap at low storage, then feedback's storage benefit
```

The saved decomposition also supplies a symmetric averaging convention. It must
not be interpreted as a uniquely identified percentage of the gap caused by each
factor. These are deterministic interventions within the assumed model, not
estimates of real Lake Como operating effects.

**How the extra inventory is used matters.** For each controller separately,
high minus low initial inventory is **79.60 million m³**. Mass accounting gives:

- MPC converts **24.39 million m³** into additional modeled target delivery and
  **55.21 million m³** into additional remaining river flow.
- Feedback converts **78.85 million m³** into additional modeled target delivery
  and **0.76 million m³** into additional remaining river flow.

Almost none of this marginal initial inventory remains at year end under either
controller. Remaining river flow includes flow not diverted to the modeled target;
it may provide ecological or other benefits outside the current objective and is
not classified here as wasted water. The target itself remains an outflow-related
benchmark proxy, not verified irrigation demand. The budget reconciles extra
release plus extra final storage with initial inventory; no evaporation, water
injection during replay or post-transition storage clamping is introduced.

**End reserves complicate controller ranking.** From the same low start, MPC ends
1995 with **4.38 million m³** of relative storage, versus feedback **98.04 million m³**.
Their high-start runs end at essentially those same controller-specific values.
The finite score gives no terminal value to this ending reserve. MPC's lower
annual cost from the low start therefore does not establish better control of a
continuing reservoir; the value of feedback's extra ending reserve depends on the
subsequent conditions. All four primary branches have zero flood exposure in 1995.
MPC has 43/37 low-water days from low/high starts; feedback has 1/0.

The extra initial water is also not free. Under the original 1994 paths, feedback
costs **553.27** versus MPC **531.80**. The original complete two-year MPC cost is
**20.44 higher** than feedback after accounting for both years. Swapping the
boundary stock is a diagnostic, not a new feasible low-cost 1994–1995 policy.

The result supports investigating the value and allocation of retained water,
but **does not isolate the seven-day horizon or missing terminal value as the sole
cause**. Forecast errors, release timing and the objective's treatment of remaining
river flow can also influence reserve use. No terminal value or reserve penalty
is fitted or added here. This is one selected dry validation case with one
forecast-sampling seed; final test years remain unused and no defaults are promoted.

Artifacts: `boundary.json` and reconstructed prefix trajectories; six branch
trajectories/decision logs; `summary.csv`, `decomposition.json`,
`inventory_accounting.csv` and `memory_controls.json`. Units are explicit; storage
is relative to the model datum, not total lake volume. Incomplete primary branches
would suppress the full-year decomposition rather than create a cheap failure score.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.carryover_check \
  --output results/simulator/new_carryover_check
```

Stages `prepare`, `prefix`, `run`, `aggregate` support frozen-source checks and
resumption. The preceding broader-year study artifacts are required for reference
reproduction. The [carryover PNG](../background/images/47_carryover_check.png) shows
the four costs, water-level trajectories, marginal inventory use and ending reserves.

## Reserve-release timing and perfect-target diagnostic

**The extra reserve is mostly released in the first weeks of 1995.** In the
controlled high-minus-low MPC replay, cumulative additional remaining river flow
reaches half of its annual net total by model day **8**, 90% by day **20**, and
99% by day **67**. The final total is **55.21 million m³**. This is signed
counterfactual volume accounting, not tracking particular water parcels: positive
daily differences sum to 56.02 million m³ and negative differences to −0.81.

The frozen `results/simulator/reserve_use_check_v2/` experiment reproduces both
causal MPC trajectories exactly, then repeats each initial storage with perfect
future **benchmark targets** supplied only to the planner. Every decision keeps
the same five conditional inflow forecast paths, original clock/history, physics,
costs, constraints and search rule. Actual simulator targets and rewards are
unchanged. Target-dependent service proposals naturally change when their input
forecast changes; this is not a fixed candidate-set experiment. Forecast fitting
still uses only 1946–1987; the privileged target record is restricted to 1995.

Perfect target information reduces annual physical cost:

- **Lower initial storage:** 525.97 → **389.51**; low-water days rise 43 → **89**.
- **Higher initial storage:** 501.16 → **355.93**; low-water days rise 37 → **81**.

There is no flood exposure or domain recovery in these four runs. Low-water
severity also increases: 2.81 → 12.66 m·hours from the low start and
2.39 → 11.63 from the high start. Final storage is about 4.34 million m³ in
the perfect-target runs versus 4.38 in the causal runs. These are finite-year
cost improvements with worse low-water outcomes, not uniform physical improvements.

Perfect targets increase the modeled delivery attributable to the extra initial
79.60 million m³ from **24.39 to 32.55 million m³**. Additional remaining river
flow falls from **55.21 to 47.05 million m³**; none of the marginal initial
inventory remains at year end. Target forecast errors therefore matter, but
removing them does not eliminate early reserve release. This privileged experiment
is neither a deployable controller nor an optimal performance bound: inflows are
still imperfect forecasts and the bounded search is unchanged.

**The saved schedules reveal a flat-cost selection problem.** From the higher
start on day 1, MPC chooses a constant **122.90 m³/s** release, although its
predicted target plus environmental flow is only **78.32 m³/s**. All five chosen
seven-day trajectories have zero predicted cost, and their maximum level is
0.0825 m, far below the 1.24 m flood threshold. The actual target is 74.70 m³/s,
close to the forecast's 73.32. This early release is not explained by an anticipated
flood or a large first-day target forecast error.

The original search evaluates constant grid schedules before target-service
schedules and retains the first minimum. In addition, rounding a target-service
request to float32 can put it slightly below the exact required rate, creating a
tiny shortage cost. On day 1 its expected cost is only **0.000000564**, versus
zero for the larger constant request. Even exact ties favor the earlier candidate.

A separately frozen **post-hoc prediction audit** rounds service requests upward
to the next sufficient representable value, changing requests by at most about
0.000008 m³/s in the audited window. It finds a lower first-day release with equal
or better predicted cost on **17 of the first 20 causal high-start decisions**
(15/20 with perfect targets). The day-1 service plan has zero predicted cost while
retaining more storage in all five scenarios. These are comparisons at saved
states, not new controller trajectories or an established annual benefit.

The [following experiment](#precision-safe-service-and-exact-cost-ties) tests a
precision-safe service proposal and an explicit conservative tie-break at unchanged
costs, separately and together, before considering continuation-value terms. This
diagnosis does not identify missing terminal value as the sole cause, and
forecast-sampling robustness remains open. No controller default or frozen benchmark
result changes; final test years remain unused.

Artifacts include four daily/decision traces with full selected schedules and
predicted levels, `release_timing.json`, `marginal_inventory.csv`,
`selected_decision_audits.csv`, `posthoc_rounding_audit/` and independent information,
cost and source-hash checks. The first attempt is preserved as superseded: default
CSV parsing caused a 1.86×10⁻⁹ m³ comparison discrepancy; round-trip parsing confirms
exact trajectory parity. The controller itself was unchanged.

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.reserve_use_check \
  --output results/simulator/new_reserve_use_check
```

Stages `prepare`, `run`, `analyze` support resumption with source hashes frozen.
The separate post-hoc audit is `python -m reservoir_control.systems.lake_como.simulation.reserve_tie_audit`
and uses the recorded default study. The [reserve-use PNG](../background/images/48_reserve_use_check.png)
shows timing, information effects and the saved-state schedule comparison.

## Precision-safe service and exact-cost ties

**Both changes improve the complete 1995 replay from either initial storage.**
The frozen `results/simulator/precision_check_v1/` factorial contains eight
365-day runs: original, precision only, tie-breaking only and both changes, each
from the same lower/higher starting stocks as the preceding diagnosis.

The interventions are deliberately small and independent:

- **Precision-safe service:** round each target-plus-environmental-flow request
  upward to the smallest sufficient float32 value. Respect the representable
  action ceiling even if the target cannot be met. Apply this to the existing
  service proposal and schedules derived from it; add no water-volume margin.
- **Conservative ties:** among **exactly equal finite minimum costs**, select the
  smallest first-day request, then the smallest sum of scheduled requests, then
  original candidate order. There is no cost tolerance or new penalty. A smaller
  command is a reserve preference, not a guarantee of safety or minimum realized
  release under hydraulic constraints.

The original causal target/inflow forecasts, seed, eight observations, initial
history/carried plan, seven-day horizon, grid, hourly physics and physical cost
weights remain fixed. Different selected actions naturally change subsequent
states and warm plans. With both flags disabled, all saved daily quantities match
the original low/high trajectories **bit for bit**. Forecasts and exogenous history
match across every arm at every decision; actual targets are never supplied to
the planner. No full-open/domain rule or terminal value changes.

Annual physical cost, with lower being better:

- **Original:** lower start **525.97**; higher start **501.16**.
- **Precision only:** lower **519.06**; higher **484.98**.
- **Tie-breaking only:** lower **518.93**; higher **480.84**.
- **Both changes:** lower **488.88**; higher **413.86**—reductions of
  **7.1%** and **17.4%** against the original at the same start.

The combined result also improves the recorded physical outcomes in this case:
unmet target volume falls **523.16 → 486.65** million m³ from the lower start,
and **498.77 → 412.64** from the higher start. Low-water days fall **43 → 39**
and **37 → 25**; severity falls **2.81 → 2.23** and **2.39 → 1.21 m·hours**.
All eight branches have zero flood exposure, environmental shortage and domain
recovery. Tie-breaking alone has a tradeoff from the lower start: its cost falls,
but low-water severity rises **2.81 → 3.20 m·hours**. The changes are therefore
not interchangeable or uniformly better on every metric in isolation.

**Water accounting explains the improvement.** In the combined higher-start run,
first-20-day remaining river flow falls **64.75 → 12.00 million m³**. At year end,
both starting stocks retain **16.92 million m³**, compared with **4.38 million m³**
originally: an extra 12.55 million m³ despite better modeled target delivery.
Precision alone ends with 8.22 million m³; ties alone with 4.38.

Within each variant, the extra high-minus-low initial stock is still 79.60 million
m³. Its allocation changes from **24.39 delivered / 55.21 remaining river** in
the original to **74.01 delivered / 5.60 remaining river** with both changes.
Precision alone gives 33.58 / 46.02; ties alone 37.15 / 42.45. Marginal ending
storage is zero because each variant's two starts converge to its own common
ending inventory. These signed ledgers reconcile to the extra initial stock.
Remaining river flow may support benefits outside the modeled target; it is not
classified as wasted water. The benchmark target remains an outflow-related proxy,
not verified irrigation demand.

The precision change makes an accurately served target available at zero shortage
cost; the tie rule can then choose it over an earlier, larger grid request.
The combined closed-loop benefit is substantially larger than either change alone.
This demonstrates an annual effect of search selection in this selected case;
it does not remove the possible need for better forecasts or continuation value.

The [broader matched check below](#precisiontie-validation-over-full-episodes-and-forecast-seeds)
extends this comparison to complete validation episodes and forecast seeds.
These conditional 1995 boundary interventions alone do not establish feasible
1994 operating improvements or general controller superiority. No default is
promoted; the final 1998–2010 test remains reserved. The implementation is opt-in
in `precision_mpc.py`; the original controller and frozen studies are unchanged.

Artifacts: eight daily/decision traces, `summary.csv`, `paired.csv`, four marginal
inventory traces, `marginal_inventory.csv`, source manifest and information/cost
identity checks. Reproduce into a fresh output directory:

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.precision_check \
  --output results/simulator/new_precision_check
```

Stages `prepare`, `run`, `aggregate` permit resumption while checking source hashes.
The [precision/tie PNG](../background/images/49_precision_check.png) compares annual
cost, lake levels and the marginal reserve allocation. The suite runs 215 tests:
214 pass and one is skipped.

## Precision/tie validation over full episodes and forecast seeds

**The cost improvement extends across the validation episodes and forecast seeds,
with increased flood exposure.** The frozen
`results/simulator/precision_validation_v1/` study compares the original and combined
precision/tie variants over five complete pairs: 1988–1989, 1990–1991, 1992–1993,
1994–1995 and 1996–1997. Each is repeated with forecast seeds **901–905**: **50
two-year runs**, producing 25 paired episode comparisons and 50 paired annual
comparisons. All runs complete; none exits the declared model domain.

Both controllers reset to 0.35 m with zero target EMA and prior inflow climatology
at the start of each episode. Storage, observations/history and MPC plans carry
through its year boundary. Seven-day planning continues across that boundary and
shrinks only at the end of the complete record. Forecasts are fitted only on
1946–1987, with no validation refitting. Costs, observations, physical constraints,
scenario count and grid remain fixed. All five original seed-901 trajectories
match the earlier frozen reference bit for bit. Paired forecast paths and target
forecasts match at every decision, and physical balance/cost identities verify.

Across all ten validation years and all five seeds, annual means are:

- **Physical cost:** **576.26 → 516.75**, a **10.3% reduction**.
- **Unmet benchmark target:** **507.67 → 445.11 million m³**, down **12.3%**.
- **Low-water days:** **34.16 → 23.74**; severity **4.42 → 3.10 m·hours**.
- **Flood days:** **4.86 → 5.26**; severity **64.17 → 68.54 m·hours**, up **6.8%**.
- **Environmental shortage:** zero for both variants.

The combined variant has lower total cost in **all 25 paired two-year episodes**,
with reductions of **75.71–189.37** per episode. It wins **46 of 50** annual
comparisons. The four annual losses are in 1989 under seeds 901, 903, 904 and 905;
the same two-year episodes still improve. Mean cost reduction by seed ranges
**9.7%–11.1%**; no seed is selected or discarded.

The result is not driven solely by the diagnosed 1994–1995 case. Excluding that
pair, annual cost is **588.13 → 525.44** across the other eight validation years
and five seeds. Full 1994–1995 replays from reset also improve under every seed:
their mean annual cost is **528.78 → 482.02**. This latter check uses each
controller's own inherited 1995 stock, rather than an imposed boundary swap.

Ending inventory also increases: mean **episode-ending** stock is **13.26 → 34.23
million m³**. These are ending stocks after the second year, distinct from the
annual-end mean that also includes the first-year boundary. The finite objective
still assigns no terminal value to this inventory. The 1989 annual-cost losses
coincide with substantially higher ending reserves; they are not evidence that
all outcomes improve independently at every annual boundary.

**The flood tradeoff is consistent across seeds.** Each seed has higher mean
flood severity under the combined variant. About 95% of the aggregate severity
increase comes from **1993 and 1997**. Additional flood days occur in 1991, 1996
and 1997; 1993 has deeper exposure without more flood days. Retaining water may
reduce surge headroom, but establishing the decision-level cause requires a
matched-state event diagnosis. Smaller requested releases do not enforce flood
safety, and lower scalar cost does not establish uniformly better physical control.

There are also **16 original / 17 combined maximum-request fallback decisions**,
all during 1993. These occur when every tested schedule leaves the declared domain
under at least one forecast scenario. The executed trajectories remain in range,
but the fallback is still uncertified. All five 1992–1993 episodes in each arm
contain fallback decisions; the combined arm adds one decision under seed 902.
Completion must not be described as absence of planner infeasibility.

The [event diagnosis below](#flood-events-matched-states-and-forecast-infeasibility)
examines the additional 1993/1997 flood severity and the 1993 infeasible forecast
schedules with fixed costs, using matched states and pre-surge headroom.
The combined variant is a robust validation improvement for the current scalar
objective, with a documented flood tradeoff. Defaults remain unchanged pending
that review. No final 1998–2010 test is run, no new safety margin or continuation
value is fitted, and no operational calibration claim is made.

Seeds resample scenarios for the same historical hydrology; they are not independent
reservoir histories. Years within a two-year episode are dependent, and all these
validation years have been available during development. Results therefore use
paired descriptive summaries and seed ranges, without independent-sample confidence
or final-test claims. The target remains an outflow-related benchmark proxy.

Artifacts: 50 daily/decision traces; `annual.csv`, `blocks.csv`, `paired_annual.csv`,
`paired_blocks.csv`, `scope_summary.csv`, `seed_summary.csv`, `episode_position.csv`,
`matched_prefixes.csv`, `failures.json`, `recovery_decisions.csv` and independent
source/reference checks. Incomplete years/episodes would retain their completion
status and prefix scores while suppressing required-scope full averages.

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.precision_validation \
  --output results/simulator/new_precision_validation --workers 2
```

Stages `prepare`, `run`, `aggregate` support resumption with frozen source checks.
The [validation PNG](../background/images/50_precision_validation.png) shows cost
by seed and episode, the physical tradeoff, and ending reserves. The test suite
runs 220 tests: 219 pass and one is skipped. Original benchmark studies and defaults
are preserved.

## Flood events, matched states and forecast infeasibility

**Earlier reserve choices and limited inflow scenarios explain different parts of
the flood tradeoff. The 33 planning failures are certified forecast infeasibility
at the sampled states, not missed feasible solutions within the model.** No cost,
physics, controller default or final-test setting changes in this diagnosis.

`results/simulator/flood_diagnosis_v1/` uses all five seeds from the preceding
validation study. The union of factual flood days is **268–301 in 1993** and
**178–191 in 1997**, under the unchanged model-year convention. Each window extends
seven days after flooding ends. Branches start **14 and 7 days before** the first
flood day, without changing the original episode clock or shrinking MPC's horizon
at the diagnostic window boundary.

At each anchor, both controllers run from each inherited stock, with common
original command, observed history and original warm plan. A fifth branch restores
the combined controller's own plan as a factual/memory control. This gives **100
complete short replays**, with no domain exits. Prefixes replay the saved requests
from the full two-year reset to recover exact storage and unrounded EMA. All **40
factual window controls match bit for bit**, and the 20 common/own-plan controls
have identical complete daily traces. The four main branches therefore separate
stock and subsequent control without an observed warm-plan effect in these events.

**1993: choices during the pre-flood fortnight matter.** The combined controller
starts 14 days before flooding with about **6.87 million m³ more water**; that gap
has grown to **21.04 million m³** one week before onset. Mean event flood severity
is **492.51 → 516.00 m·hours**. At the 14-day anchor, replacing only the inherited
stock under the original controller adds **2.94 m·hours**; switching controllers
at that higher stock adds **20.55**. Reversing the intervention order gives **19.48**
for the controller change and **4.00** for stock. At the 7-day anchor, the analogous
original-controller stock effect is **13.96**, with **9.53** from the controller
change at the combined stock. Effects are conditional on anchor and order, not
unique percentages of the gap.

**1997: much of the gap is inherited earlier.** The extra starting inventory is
**44.31 million m³** at the 14-day anchor and **42.71 million m³** at the 7-day
anchor. Mean severity is **148.96 → 167.13 m·hours**. At 14 days, the original-
controller stock effect is **12.97** and the controller effect at combined stock
is **5.21**; the other order gives **3.74** for control and **14.43** for stock.
At 7 days, the first decomposition is **17.56** for inherited stock and **0.61**
for subsequent control. The saved contrasts also report window cost and target
shortage. Stock swaps remain conditional experiments, not free-water policies.

The flood cost does not determine the entire objective: in the 14-day event
windows, combined-minus-original total cost is **+17.88 in 1993** but **−18.97 in
1997**, where improved target delivery outweighs extra flood exposure under the
unchanged weights. These window results do not replace the complete two-year
comparison or establish a preferred flood/irrigation priority.

**The pre-flood forecasts often miss the incoming volume.** In the last seven
days before onset, realized seven-day inflow volume exceeds the maximum of all
five forecast scenarios in **29/35 day-seed forecasts for 1993** and **31/35 for
1997**. These are retrospective, event-selected coverage checks, not a general
forecast calibration rate. Realized future data is used only for offline labels.

A separate matched one-decision audit covers both factual levels, every seed
and all 14 pre-flood days: **140 observations per event**. The combined variant
chooses lower expected actual release while **both controllers predict zero cost
and neither predicts flooding** in **82/140 cases in 1993** and **33/140 in 1997**.
For example, at the original 1993 seed-901 state on model day 254, expected first-day
release is **172.31 → 99.10 m³/s**; both seven-day costs are zero and predicted
peaks are below 1.24 m. This exposes the flat-cost reserve preference and limited
forecast view. These repeated observations share hydrology and are not independent
statistical samples. Once level exceeds the inherited 1.25 m forced-open threshold,
different nominal requests can produce identical actual release; the graphics
show both quantities rather than interpreting a low request as a closed outlet.

**All 33 failed planning decisions contain an impossible forecast path at that
state.** A fixed expanded audit tests **563–572 distinct bounded schedules per
decision**: finer constants, binary low/high plans, nine-grid two-stage plans,
service proposals and maximum-first/service-tail plans. It finds zero feasible
alternatives. Bounded search alone would be inconclusive; the capacity check
provides the stronger result.

The hourly lower-bound trajectory uses
`r_max(h) = min(full_open_capacity(h), low_water_reserve_cap(h))`.
Any admissible release is at most this value at the same level. The map
`h_next = h + 3600 / area * (inflow - r_max(h))` is order preserving over the
declared domain: its release-slope factor is at most **1/24**, below one.
Induction therefore makes this a lower bound on every admissible level trajectory
until its first guard crossing. If even this bound exceeds **2.64 m**, no allowed
schedule can keep that forecast path inside the domain from the given state.
The diagnostic stops at the crossing and never evaluates the curve beyond it.

In this configuration the capacity at the forced-open threshold is **478.66 m³/s**,
below the maximum nominal request of **491.61 m³/s**. Below that threshold, maximum
request fully opens the outlet; above it the inherited rule forces full opening.
Thus the initially declared capacity relaxation actually matches maximum nominal
requests throughout this model's domain. Scalar maximum-request guard exits agree
with the bound for every audited path.

There are **37 certified upper-exit scenario paths across the 33 decisions**:
16 original decisions and 17 combined. Each failed decision contains at least
one certificate. In contrast, feeding the realized seven-day inflows into this
offline capacity check produces **zero upper-domain exits**. Staying below 2.64 m
does **not** mean avoiding the 1.24 m flood threshold. These certificates concern
the sampled forecasts and current starting states in the assumed hourly model;
they do not establish that earlier anticipation cannot help, or that actual
Lake Como floods are physically unavoidable.

The [causal forecast study below](#causal-history-forecasts-training-holdouts-and-event-rechecks)
tests history/trend information and wet-event coverage using training-only
chronological holdouts, then reruns matched events at unchanged costs. Broader
validation remains necessary before promotion. More candidate search cannot resolve
the certified 33 cases from the same states and scenarios. Defaults remain unchanged;
final test years, safety margins and continuation-value fitting remain untouched.

Artifacts include exact snapshots, 100 branch traces, `branches.csv`,
`attribution.csv`, `forecast_skill.csv`, 33 detailed `forecast_audits/`,
`feasibility_summary.csv`, the separately frozen `posthoc_decision_audit/`,
and independent source, replay, ledger and certificate checks. See the
[event/stock PNG](../background/images/51_flood_event_diagnosis.png) and
[forecast/feasibility PNG](../background/images/52_flood_forecast_feasibility.png).

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.flood_diagnosis \
  --output results/simulator/new_flood_diagnosis
```

Stages `prepare`, `prefix`, `branches`, `audit`, `aggregate` allow resumption with
source hashes frozen. The test suite runs 226 tests: 225 pass and one is skipped.
The public-source calibration limits and benchmark-target interpretation remain.

## Causal history forecasts: training holdouts and event rechecks

**A causal history-based forecast improves the separate training confirmation
scores, but produces only modest mean flood improvements in the selected events
and more planner fallbacks. Defaults remain unchanged.** The active study is
`results/simulator/forecast_study_v2/`.

Four fixed methods are compared with five scenarios and seeds 901–905:

- The existing seasonal/last-inflow historical sampler.
- History analogues using the last flow, three-/seven-day means and a recent
  trend in log-flow space, standardized using fitting-period data only.
- Those history analogues sampled from five near-equal ranks of historical
  seven-day volume, retaining each path's temporal shape.
- Stratified history analogues with bounded level adjustment: multiply each
  path by `(last_observed + 1) / (historical_predecessor + 1)`, clipped to 0.5–2.

All new methods match within the existing seasonal window and retain a 128-member
analogue pool. The chosen forecast is not a meteorological forecast or calibrated
probability model. The controller keeps only the last seven inflows actually
returned in observations; reset climatology is not counted as a measurement.
Before seven observations are available, it uses the unchanged current sampler.
The observation vector, target forecaster, MPC objective and physical model stay fixed.

**Chronological selection uses training data only.** Fitting uses **1946–1975**;
selection uses **1976–1981**; confirmation uses **1982–1987**, without refitting
the bank or revising the chosen method. Each method scores all origins with seven
past observations and seven future labels: model days 8–359 in each holdout year,
with all five seeds. This gives 2,112 distinct day origins per phase, repeated over
five scenario samples per method; repeated and overlapping origins are dependent.

The selection rule is frozen before scoring: strictly improve seven-day volume
ensemble CRPS, maintain or improve wet upper-envelope coverage, and keep first-day
mean MAE within 110% of baseline; choose the eligible method with lowest CRPS.
Wet labels mean future seven-day volume above the fitting-period 90th percentile.
Only evaluation labels use future values. If no method qualifies, baseline is kept.

The **scaled, stratified history method** wins selection and passes the separate
confirmation gates. On confirmation, against the current sampler:

- **Seven-day volume CRPS:** **23.37 → 20.35 million m³**, down **12.9%**.
- **First-day mean MAE:** **49.29 → 44.59 m³/s**.
- **Wet upper-envelope coverage:** **44.6% → 67.7%**.
- **Full envelope coverage:** **65.9% → 78.8%**; mean envelope width grows
  **69.44 → 90.69 million m³**.

CRPS evaluates the entire equally weighted five-path distribution, not just its
mean. Upper-envelope coverage is the fraction of wet truths below the largest
scenario, not a nominal confidence level. The remaining wet misses are material.
All four methods' tuning and confirmation metrics are retained; confirmation does
not trigger a second choice or hyperparameter adjustment.

**The selected method is then refitted only on 1946–1987 and frozen for events.**
Eighty matched event replays compare old versus selected inflow forecasts using
the same combined precision/tie MPC, two inherited stocks, two anchors and five
seeds. Common initial history, warm plan and original full-record clock are retained.
All 80 complete, all 40 old-forecast controls reproduce the prior daily traces
bit for bit, and every selected forecast's seven observations are independently
checked against the actual past. Target forecasts are identical between arms.

Mean event flood severity, pooling the two stocks and five seeds at each anchor:

- **1993, 14-day anchor:** **514.00 → 506.42 m·hours**; 7-day anchor
  **509.11 → 508.09**.
- **1997, 14-day anchor:** **159.92 → 155.60 m·hours**; 7-day anchor
  **158.59 → 155.12**.

These are small mean improvements, not uniformly better runs: flood severity
falls in **31/40** paired comparisons and window cost in **35/40**. Mean window
cost improves in all four year/anchor groups, but target shortage increases at the
14-day anchors by **0.72 million m³ in 1993** and **0.52 in 1997**. There is no
low-water exposure in these windows. No full-year controller ranking follows from
overlapping, retrospectively selected flood windows.

**Extreme-event coverage remains poor.** In the final pre-flood week, realized
seven-day volume still exceeds every selected scenario in **27/35** day-seed
forecasts in each event, versus 29/35 previously in 1993 and 31/35 in 1997. Better
aggregate holdout scores therefore do not establish adequate anticipation of these
extremes from inflow history alone.

**Broader scenarios also increase hard-domain warnings.** At the 14-day anchor,
mean maximum-request fallback counts per replay rise **3.4 → 15.7 in 1993** and
**0 → 2.7 in 1997** (7-day anchors: 3.4 → 15.5 and 0 → 2.8). Independent capacity
bounds certify every logged fallback in both arms; these are forecast paths that
cannot remain inside the outlet domain from their current states, rather than a
numerical search failure. The 435 logged decisions include repeated branches and
must not be treated as independent events.

At the **33 previously failed states**, changing only the forecast instead leaves
**24 certified infeasible** and gives **9 states with feasible schedules**. This
does not contradict the higher replay fallback counts: the new forecast changes
the risk paths and subsequent states at other times. The 2.64 m model guard remains
distinct from the 1.24 m flood penalty threshold; neither scenario coverage nor
successful execution supplies a flood-safety guarantee.

The [support/sampling diagnosis below](#tail-support-scenario-sampling-and-additional-fallbacks)
investigates the residual extreme-volume misses and extra domain warnings without
changing the selected forecast. Further forecast changes must be frozen before
complete two-year validation from reset. The
current method is an opt-in forecasting candidate, not a promoted controller;
cost weights, physical rules, defaults and final 1998–2010 tests are unchanged.

Artifacts include all holdout scores, tuning/confirmation summaries, the selection
checksum written before confirmation, `event_plan.json`, 80 event traces,
`paired_events.csv`, 33 `current_state_audits/`, `preflood_coverage.csv`,
`event_fallback_certificates.csv` and independent causal/source/reference checks.
The [forecast-study PNG](../background/images/53_causal_forecast_study.png) shows
confirmation performance and the event tradeoff.

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.forecast_study \
  --output results/simulator/new_forecast_study
```

Stages `prepare`, `forecasts`, `events`, `audit`, `aggregate` support resumption.
The first pass completed selection and all trajectories, then stopped on a missing
runner import during the follow-up audit. Version 2 fixes that import and preserves
the unchanged, hash-protected scores, selection and trajectories; the superseded
source snapshot is retained. No result-driven forecast changes were made. The
test suite runs 231 tests: 230 pass and one is skipped.

## Tail support, scenario sampling and additional fallbacks

**The 1993 and 1997 misses have different causes; more sampling alone cannot fix
both. Extra fallbacks are mainly a forecast-distribution effect, not a consequence
of holding more water in the new trajectories.** This is a read-only diagnosis in
`results/simulator/tail_diagnosis_v1/`: no new controller runs, forecast selection,
costs, physical rules or defaults change.

The audit exactly reconstructs the frozen scaled-history generator through three
stages: every seasonally eligible historical path after the same bounded scaling;
the nearest 128 paths retained by history conditioning; and the five sampled
scenarios. Future volume is used only to label where a completed forecast misses,
never to select an operational path. This separates lack of support, conditioning
that excludes suitable paths, and finite sampling within available support.

The previously inspected training holdouts are reused as **development diagnostics**,
not new confirmation. The fixed fit remains 1946–1975. Among wet upper-volume misses:

- **1976–1981:** 564 misses comprise **364 sampling misses**, **160 exclusions by
  history conditioning**, and **40 outside seasonal scaled support**.
- **1982–1987:** 519 misses comprise **449 sampling misses**, **60 conditioning
  exclusions**, and **10 outside seasonal scaled support**.

These are day-seed forecasts, not independent hydrological observations. The
reconstructed five paths and their coverage reproduce the prior study exactly.

**1993: suitable paths exist, but rarely appear in five samples.** All **27/35**
misses in the final pre-flood week are within the retained 128-path pool. None is
outside seasonal support or removed by conditioning. Across those seven days,
the algorithm's conditional chance that a five-path draw reaches the realized
volume averages only **25.7%**, close to the observed **22.9%** across seeds. The
misses are therefore consistent with sparse tail representation, not simply an
unusually unlucky seed. This sampling probability is conditional on the finite
candidate population; it is not the probability of an actual storm.

**1997: history conditioning removes most suitable paths.** Of the **27/35**
misses, **25** are beyond every retained conditional path, although suitable
scaled paths exist in the seasonal population; only **2** are sampling misses.
Even evaluating all 128 retained paths would cover only **28.6%** of these selected
forecasts. Drawing more often from this same pool cannot recover excluded paths.
The seasonal population can cover all event volumes in both years, but that is
only a retrospective support bound, not evidence that those paths deserve high
forecast probability. Support here means seven-day volume; it does not establish
matching pulse timing or hydraulic feasibility.

**Nested scenario-count sensitivity is offline only.** Keep the original five
paths and add independent stratified batches of ten, yielding 15 and 25 paths.
Upper-volume coverage becomes:

- Inspected 1982–1987 wet origins: **67.7% → 80.1% → 84.2%**; full conditional
  pool support covers **95.6%**.
- 1993 pre-flood week: **22.9% → 37.1% → 42.9%**; pool support **100%**.
- 1997 pre-flood week: **22.9% → 28.6% → 28.6%**; pool support **28.6%**.

No larger-set controller performance is measured or inferred. A nested larger
set also retains every already-infeasible path under the current all-path domain
criterion; more scenarios cannot cure those existing feasibility failures at the
same state. Removing the level-adjustment clipping does not change conditional-pool
upper support coverage in these holdout/event checks, so these results do not justify uncapping
the multiplier. Uncapped paths are diagnostic only, never controller inputs.

**Crossed states isolate the extra fallback warnings.** For every one of the
1,660 matched time points in the prior event replays, evaluate old and selected
forecasts at both saved states, with the same clock and observed inflow history.
The fixed hourly maximum-outlet bound determines whether any scenario necessarily
leaves the declared domain; the diagonal results match all logged fallbacks:

```text
old state + old forecast:  68 warnings
old state + new forecast: 374 warnings
new state + old forecast:  68 warnings
new state + new forecast: 367 warnings
```

One exact conditional order is **68 + 306 − 7 = 367**: replacing forecasts at old
states adds 306 net warnings, then using the new states removes 7. The other is
**68 + 0 + 299 = 367**. The forecast swap creates 326 warnings and removes 20 at
old states; **316 of the 326** creations occur at or after the first factual flood
day. The new-state swap removes seven more and creates none under the new forecast.
These orders are not unique percentage attributions. Counts repeat seeds, stocks
and overlapping event windows; they are not independent incidents.

This explains how poor early extreme-volume coverage can coexist with many later
infeasible scenarios. The warnings identify forecast/model-domain conflicts at
their current states, not failed numerical optimization. Suppressing warnings by
discarding wet paths would not establish better flood protection. The 2.64 m
model guard remains distinct from the 1.24 m flood threshold.

The [explicit-probability study below](#explicit-tail-probabilities-and-matched-event-control)
tests a tail-aware forecast representation on chronological training folds, then
checks whether earlier risk awareness creates headroom. Scenario probabilities
and feasibility handling are isolated: rare paths are not silently given one fifth
of expected-cost weight or removed to obtain a finite score. Complete two-year
validation remains necessary. Costs, physics, defaults and final test years remain
untouched in the current diagnosis.

Artifacts: `tuning_support.csv`, `confirmation_support.csv`, `event_support.csv`,
`coverage_summary.csv`, `miss_categories.csv`, `fallback_cells.csv`,
`fallback_contrasts.csv`, source hashes and independent reconstruction/partition
checks. The [tail/fallback PNG](../background/images/54_tail_and_fallback_diagnosis.png)
shows miss causes, sampling limits and the crossed forecast/state results.

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.tail_diagnosis \
  --output results/simulator/new_tail_diagnosis
```

Stages `prepare`, `holdouts`, `events`, `fallbacks`, `aggregate` permit resumption
with frozen sources. The test suite runs 235 tests: 234 pass and one is skipped.

## Explicit tail probabilities and matched event control

**Explicit tail representation creates additional pre-flood headroom and reduces
mean flood severity in the selected events, with increased target shortage and
more domain warnings. It remains an opt-in research candidate.** The frozen study
is `results/simulator/weighted_study_v1/`; no default or physical reward weights
change, and final test years remain unused.

The forecast starts with a finite mixture of the existing scaled history-conditional
pool and the full seasonal population. Seasonal mixture mass is tested at
**0%, 5%, 10% and 20%**; zero is a conditioning-only ablation. Each mixture is
represented by five probability bands, with either **1% or 5%** reserved for the
last band. A source path is drawn conditionally within each band and receives
that band's probability mass. Atoms crossing band boundaries may appear twice;
total represented mass still sums to one. All source paths and features use
strictly prior training records and observed history.

**Weights are explicit model probabilities, not calibrated storm probabilities.**
For the selected 5% tail, band boundaries are **0/25/50/75/95/100%**, and expected-
cost weights are **25/25/25/20/5%**. The last representative therefore carries 5%,
not 20%. For a fixed candidate schedule this is stratified probability integration
under the fitted finite mixture. The underlying mixture and its five-point
approximation are different objects; neither supplies a real-world safety guarantee.

Selection uses two chronological development folds:

- Fit **1946–1965**, score **1966–1975**.
- Fit **1946–1975**, score **1976–1981**.

Pool the same complete-history/full-horizon day origins and five seeds across
both folds. Require lower weighted seven-day-volume CRPS, no worse wet upper
coverage, and first-day mean MAE within 110% of legacy; select lowest eligible
CRPS, otherwise retain legacy. A zero seasonal mixture is allowed to win—positive
mass is not forced. The chosen configuration is locked before the later
**1982–1987** training check. Those later years were inspected previously and are
not independent confirmation. No reselection follows that check or event outcomes.

The best declared grid candidate is **80% conditional / 20% seasonal**, with a
**5% upper-tail band**. Twenty percent is the grid's upper boundary, not an estimate
of an optimal or true storm probability. Development five-point volume CRPS is
**19.35 → 19.02 million m³**. In the later training check:

- Five-point CRPS: **20.35 → 20.01 million m³**, about **1.7% lower**.
- First-day mean MAE: **44.59 → 44.42 m³/s**.
- Wet upper-envelope coverage: **67.7% → 86.3%**.

The 1% bands give higher raw upper coverage, but fail the primary CRPS improvement
gate; they are retained in the results rather than selected for coverage alone.
Changing the represented upper tail naturally changes maximum-scenario coverage,
so this number is not a fixed-nominal calibration comparison.

**Probability diagnostics separate the full model from control quadrature.**
Supplemental checks, unused for selection, score both the complete finite mixture
and its five representatives. In the later check, full-mixture CRPS improves only
**19.10 → 18.98 million m³** (about 0.6%); wet-event Brier score improves
**0.07175 → 0.07093**. Full-law Brier does not improve in the pooled development
folds, so probability improvement is not uniform. Much of the represented forecast's
gain comes from the reduction/sampling scheme, not a large new predictive signal.

The selected full-mixture 50/90/95% quantiles cover about **48.6/88.9/94.4%** of
later truths. Inverting the five weighted atoms instead gives **36.9/83.6/83.6%**.
This is a material reduction artifact: **do not display those five atoms as
calibrated confidence bands**. Full-mixture uncertainty diagnostics and weighted
control representatives must remain separate. No nominal probability guarantee
is claimed from these retrospective training checks.

**The controller changes expected-cost integration only.** The existing physical
scenario cost and candidate search are preserved; apply the declared probabilities
before selecting a plan. Exact ties retain the conservative rule. A separate
uniform-weight ablation evaluates the *same selected paths* with 20% each. It is
an intentionally different probability model, not a calibrated competing forecast.
All five represented positive-mass paths must still stay within the same outlet domain: a 1% invalid
path makes the schedule infeasible. No path is discarded or renormalized over
survivors, and no cost is evaluated after extrapolation. Scalar-model parity is
checked on every selected first-day plan; uniform legacy controls preserve exact
old arithmetic and trajectories.

After refitting the selected forecast on **1946–1987 only**, **120 complete matched
event replays** compare legacy, weighted and uniform-ablation arms at both inherited
stocks, anchors and all five seeds. All 40 legacy controls match earlier traces
bit for bit; observed histories, target forecasts and physical costs verify. Weighted
and uniform arms have identical forecast paths at every paired time.

Mean weighted event outcomes, relative to the prior history forecast:

- **1993, 14-day anchor:** flood severity **506.42 → 483.45 m·hours**, window cost
  **607.24 → 599.18**, but unmet target **100.82 → 115.74 million m³**. Additional
  actual pre-flood release creates **26.84 million m³** of headroom at factual onset.
- **1993, 7-day anchor:** severity **508.09 → 490.48**, cost **600.70 → 593.88**,
  additional headroom **19.86 million m³**; shortage again increases.
- **1997, 14-day anchor:** severity **155.60 → 151.83**, cost **274.34 → 273.92**,
  additional headroom **4.99 million m³**; shortage increases by **3.35 million m³**.
- **1997, 7-day anchor:** severity **155.12 → 154.92**, cost **251.30 → 251.06**,
  with only **0.19 million m³** of extra headroom.

Headroom gains reconcile exactly with extra pre-flood release at common inflows
and initial stock. Weighted control improves cost in **32/40** paired event windows
and flood severity in **34/40**, not every case. These selected, overlapping
windows are not an annual controller ranking.

Pre-flood upper-volume coverage rises **8/35 → 24/35 in 1993**, but only
**8/35 → 11/35 in 1997**. The seasonal mixture helps the conditioning gap, yet
does not resolve the extreme 1997 misses. At 14-day anchors, mean fallback requests
increase **15.7 → 25.0 in 1993** and **2.7 → 5.7 in 1997**; capacity bounds certify
the sampled forecast infeasibility. Explicit small probabilities do not weaken
the all-path guard.

The uniform ablation has somewhat lower event costs/flood severity here, but
overrepresents the upper band and must not be promoted from rare, selected flood
windows. At the weighted controller's saved states, probability weighting changes
the nominal request on only **19/1,660** decisions. The scenario set, hard-domain
fallback and forced-opening physics dominate many event decisions; nonuniform
weights alone are not a solution to infeasibility or limited anticipation.

The [complete validation study below](#weighted-tail-mpc-full-two-year-validation)
compares this frozen candidate from reset, including shortages, ending stock and
fallback behavior. Full-mixture uncertainty diagnostics remain separate from
five-path control approximation; final 1998–2010 tests remain reserved.

Artifacts include both development folds, later training scores, selection and
event-plan checksums, 120 trajectories, `event_summary.csv`, `paired_events.csv`,
`headroom.csv`, `preflood_coverage.csv`, fallback certificates and source/reference
checks. The [weighted-tail PNG](../background/images/55_weighted_tail_scenarios.png)
shows band probabilities, score improvements, flood exposure and early headroom.

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.weighted_study \
  --output results/simulator/new_weighted_study
uv run python -m reservoir_control.systems.lake_como.simulation.weighted_probability_audit \
  --study results/simulator/new_weighted_study
```

Study stages `prepare`, `forecasts`, `events`, `aggregate` support resumption.
The separate probability audit is post-hoc and never changes selection. The test
suite runs 245 tests: 244 pass and one is skipped. Defaults, physical reward
weights, hydrodynamics and final test use are unchanged.

## Weighted-tail MPC: full two-year validation

**Under the current physical objective, the flood benefits do not offset the
shortage and low-water costs. Do not promote the weighted-tail candidate.**
The frozen `results/simulator/weighted_validation_v1/` comparison uses the previous
scaled/stratified history MPC as its legacy comparator and the selected
80% conditional / 20% seasonal, 5%-tail model with its declared probabilities.
This comparator is distinct from the older last-inflow-only MPC; no claim about
the best controller across every historical study follows here.

All **50 two-year runs** complete: both controllers over 1988–1989, 1990–1991,
1992–1993, 1994–1995 and 1996–1997, with forecast seeds 901–905. Each episode resets
to 0.35 m, zero target EMA and prior inflow climatology; storage, observed history
and plans carry across the year boundary. The forecast variant/probabilities are
frozen from training selection and fit only 1946–1987. Physical reward weights,
target forecasts, candidate search, constraints and model-year convention stay fixed.
No validation refitting, seed selection or default change occurs.

An additional full 1988–1989 seed-901 legacy replay with the preceding controller
implementation matches bit for bit. All 36,500 primary daily decisions pass causal
history/probability checks; the 350 initial cold-history decisions use the same
fallback forecast. Paired exogenous inputs and target forecasts match, cross-year
continuity verifies, and physical water/cost ledgers reconcile. No actual trajectory
leaves the declared model domain; sampled forecast infeasibility still occurs.

Annual means, previous history MPC → weighted-tail candidate:

- **Total physical cost:** **517.51 → 531.79**, **2.8% higher**.
- **Unmet benchmark target:** **448.85 → 466.77 million m³**, **4.0% higher**.
- **Flood severity:** **67.28 → 63.39 m·hours**, **5.8% lower**; flood days
  **5.06 → 4.70**.
- **Low-water severity:** **1.39 → 1.62 m·hours**; low-water days
  **17.12 → 21.30**, **24.4% higher**.
- **Maximum-request fallback days:** **2.56 → 5.54/year**. Every logged fallback
  is certified capacity-infeasible for at least one sampled forecast path.
- **Environmental shortage:** zero for both.

At unchanged reward weights, the annual cost difference reconciles as:

```text
+14.27 = +17.92 unmet-target contribution
         −3.88 flood contribution
         +0.24 low-water contribution
```

The weighted candidate wins **3/25 paired two-year comparisons** and **11/50
paired annual comparisons**. Every seed has higher mean cost, with annual increases
of **9.53–19.55**. Both episode positions worsen: first-year mean cost
**421.61 → 432.71**, second-year **613.42 → 630.87**. The result is therefore
not restricted to a single bad seed or a failure to complete a difficult year.

**Dry, middle and wet context uses training-defined thresholds.** Training annual
mean inflow Q25/Q75 are **140.42/165.00 m³/s**. Validation years are labeled only
afterward, never passed to either policy. A dry or middle annual mean can still
contain a flood surge. Mean cost rises in every group:

- **Dry: 1989, 1990, 1995** (3 unique years, 15 seed-year pairs):
  **451.63 → 452.26**. Neither controller floods in these years. Target shortage
  is nearly unchanged, while low-water severity rises **2.17 → 2.87 m·hours**.
- **Middle: 1988, 1991, 1996, 1997** (4 unique years, 20 pairs):
  **490.54 → 520.46**. Annual unmet target rises **32.55 million m³**, against
  a flood-severity reduction of **2.74 m·hours**.
- **Wet: 1992, 1993, 1994** (3 unique years, 15 pairs):
  **619.37 → 626.42**. Flood severity falls **9.29 m·hours/year**, but unmet
  target rises **16.40 million m³/year**. Low-water severity falls slightly.

These are descriptive conditional means from a small, reused set of historical
years, not independent climate-regime estimates. Seeds resample forecasts for
the same forcing, and years within an episode share inherited storage. No p-values
or independent-sample confidence intervals are asserted.

**Ending reserves also decline.** Mean episode-ending stock, after the second year,
is **37.28 → 32.93 million m³**. The annual-end mean, which also includes first-year
carryover boundaries, is **30.42 → 22.34 million m³** and should not be confused
with the episode-end statistic. Terminal water still has zero objective weight;
these finite scores do not establish continuing-task optimality. Remaining river
flow rises **490.63 → 510.73 million m³/year** while modeled target delivery falls.
That river flow may serve benefits outside this objective and is not labeled waste.

The selected 1993/1997 event windows had suggested mean cost gains. Across their
complete years and all five seeds, cost instead rises **801.86 → 808.59/year**;
across the other eight validation years it rises **446.43 → 462.59**. This directly
demonstrates why an improved forecast score or a selected-event result cannot
substitute for full-trajectory controller evaluation.

**Decision:** retain the previous history comparator for this comparison and
preserve the weighted-tail model as a labeled flood/shortage sensitivity, with
no default promotion. Stronger flood priority would require an explicitly declared
control objective/risk study; do not retune probabilities on these validation
outcomes to obtain a preferred answer. This result is useful evidence about the
interaction between forecast tails and the unchanged all-path domain rule, not
evidence that the rare paths should be silently removed. Final test years stay
reserved, and original PPO/FQI/SAC/DamEnv benchmark results are untouched.

Artifacts: 50 full daily/decision traces; `annual.csv`, `blocks.csv`,
`paired_annual.csv`, `paired_blocks.csv`, `scope_summary.csv`, `seed_summary.csv`,
`regime_summary.csv`, `episode_position.csv`, `matched_prefixes.csv`, `fallbacks.csv`,
`failures.json` and source/reference checks. Incomplete years/episodes would retain
prefix/completion status and suppress required-scope full averages. See the
[full-validation PNG](../background/images/56_weighted_full_validation.png).

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.weighted_validation \
  --output results/simulator/new_weighted_validation --workers 2
```

Stages `prepare`, `run`, `aggregate` allow resumption with frozen source checks.
The test suite runs 249 tests: 248 pass and one is skipped. No final 1998–2010
evaluation or operational calibration claim is added.

## Reproduction, prior adapters and remaining work

Current implementation lives in `src/reservoir_control/systems/lake_como/simulation/`; the configuration and
source hashes are frozen in the active run. The first prototype run is preserved
with original source snapshots and a supersession note: scenario-count-dependent
conditioning and candidate precision were corrected before the current rerun.
All source hashes for the active run verify. No original fork or benchmark file changed.

Earlier narrow adapters remain available:

```bash
uv run python -m reservoir_control.systems.lake_como.calendar_audit
uv run python scripts/analysis/continuous_environment.py
uv run --group reproduction python scripts/analysis/continuous_environment.py --train-smoke
uv run python scripts/analysis/hourly_environment.py
```

Their existing outputs are protected; use `--output` to rerun into a new directory.
The calendar audit confirms 49 ordinary/16 leap-year rows and all seven metadata
checksums. Every version here still consumes the first 365 samples per source year.
**If column 0 is January 1**, leap day is retained and the final leap-year sample
would be December 31 and omitted. The new source audit supports the calendar-date anchor and Malgrate interpretation,
but exact flow intervals and demand construction remain unconfirmed. A real-calendar version needs a separate mapping
and leap-day seasonal/environmental-flow rule.

Still open: align measured dates/levels/releases, validate/calibrate physical and
operational parameters and calibrate scenario coverage. The matched PPO–MPC
study below is now complete; the final 1998–2010 controller test remains reserved.

## Matched PPO and MPC protocol in the protected model

This is a fresh PPO experiment in the improved simulator. Earlier DamEnv PPO
checkpoints and scores use different dynamics and rewards and are not transferred.
The reference is the **legacy history MPC** from `weighted_validation_v1`; the
weighted-tail variant remains an unpromoted sensitivity experiment.

Both controllers start each two-year episode at 0.35 m, with zero target EMA and
the same training-climatology inflow prior. Storage and available history carry
through the annual boundary. There is no burn-in. Both use the protected hourly
outlet model, the same nominal action bounds and hydraulic constraints, and the
unchanged physical cost: unmet benchmark target in million m³ plus flood and
low-water exposure in m·hours. The target remains a proxy, not independently
verified irrigation demand. Neither controller receives future measured inflows.

MPC retains seven observed inflows. PPO receives the eight simulator observations,
six older inflows and a history-count indicator: **15 inputs** with fixed unit
scales. Padding at reset is explicitly unmeasured climatology. PPO has independent
64–64 tanh actor/critic networks and a bounded tanh-Gaussian release distribution.
It uses FP32, fixed learning rate 0.0003, gamma 0.995, GAE lambda 0.95, PPO clip
0.2, 2,048-step rollouts, ten optimization epochs, 64-sample minibatches, gradient
norm cap 0.5 and approximate-KL stop 0.03. Entropy coefficient is zero. A fixed
0.01 multiplier changes learning reward units only; reported costs remain physical.
The finite task terminates after 730 decisions, without a terminal reserve cost;
GAE does not propagate across resets. Artificial truncation would bootstrap from
the actual stored terminal observation, while true finite-task termination does not.

The bounded study declares **seeds 123–127, 200,000 decisions each**, all 41
contiguous training pairs from 1946–1987, uniformly sampled. Final checkpoints
only; no best-seed or validation-checkpoint selection. Evaluation freezes PPO and
uses tanh of the actor mean on all five nonoverlapping validation pairs,
1988–1989 through 1996–1997. MPC's existing forecast seeds 901–905 are retained.
These are different sources of randomness: compare historical years after
averaging each controller's own seeds, not by pretending the seed IDs are paired.
Validation has already informed development; these are not untouched test results.
Final 1998–2010 controller evaluation remains reserved.

A domain exit stops that seed or trajectory. No outside-domain reward, hidden
state clamp or exclusion of hard training years is introduced. Missing full-budget
seeds or incomplete validation episodes suppress the required-study mean; partial
costs remain diagnostics. This budget establishes an initial comparison, not an
upper bound on PPO performance. No default changes automatically.

The 4,096-step integration check completed five 730-day training episodes and six
internal annual boundaries, with finite actor/critic updates and maximum hourly
balance error below 3×10⁻⁸ m³. Generated smoke results are in
`results/simulator/ppo_smoke_dev2/`; an earlier development attempt stopped on a
logging-field error before training updates. The matched run freezes source/data
checksums and its protocol in `results/simulator/ppo_comparison_v1/plan.json`.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_study \
  --stage smoke --root results/simulator/new_ppo_smoke
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_study \
  --stage all --root results/simulator/new_ppo_comparison
```

Use fresh output directories. Two CPU training workers run concurrently; each
seed writes progress, update diagnostics, a final checkpoint and completion status.
Completed seeds then produce daily validation trajectories, annual physical metrics,
seed means and historical-year paired differences. Source manifests are verified
before and after the study; the original benchmark and fork remain unchanged.

## Matched PPO and MPC results

All five PPO seeds completed **200,000 decisions each** without domain or numerical
failures. Each completed 273 full training episodes plus a counted unfinished
prefix. All 25 subsequent validation episodes completed 730 decisions: **18,250
daily decisions**, 50 annual records. MPC's frozen 25 complete legacy episodes
provide the comparison. The physics, constraints and reported costs did not change.

Annual averages over the same historical validation years:

- **History MPC:** cost **517.51**; unmet model target **448.85 million m³**;
  flood exposure **67.28 m·hours**, **5.06 flood days**; low-water exposure
  **1.387 m·hours**, **17.12 low-water days**.
- **Fresh PPO:** cost **631.34**; unmet model target **545.34 million m³**;
  flood exposure **62.95 m·hours**, **5.10 flood days**; low-water exposure
  **23.05 m·hours**, **117.20 low-water days**.

PPO costs **22.0% more**, with **21.5% more unmet target**. Its flood exposure is
6.4% lower, but flood-day counts are almost unchanged and low-water exposure is
much higher. Neither controller incurs environmental-flow shortage in these runs.
PPO's individual seed mean costs are **626.01, 641.12, 635.42, 626.42, 627.72**
(seeds 123–127); MPC forecast-seed means range **514.67–523.67**. These ranges
summarize different sources of random variation, not paired seed experiments.

After averaging each controller's own seeds, **MPC has lower cost in all ten
historical years and all five two-year blocks**, and in dry/middle/wet groups.
PPO also ends episodes with less stock: **5.65 vs 37.28 million m³**, relative to
the model's storage datum. First-year mean costs are 569.56 PPO / 421.61 MPC;
second-year costs are 693.12 / 613.42. Years in an episode are dependent, and
forecast/learning seed repetitions do not create independent hydrological records.
No significance or general algorithm-ranking claim is made.

The trajectory plots show PPO drawing storage down earlier and spending long
periods near the protected low-water floor. This is an observed behavioral
pattern, not yet an isolated cause of the gap. The useful next diagnosis is a
matched-state release/storage-value audit, followed by training-curve inspection;
these results alone do not establish whether longer training would fix it.
PPO used a 200k budget, a fresh architecture and discounted learning objective
(gamma 0.995); evaluation reports undiscounted physical cost. It is not a reproduction
of the earlier 1M-step DamEnv PPO, whose reward and physics differ.

Artifacts are in `results/simulator/ppo_comparison_v1/`: frozen `plan.json`, all
five `seed*/final.pt` checkpoints, `training.json`, `updates.csv`, full daily
validation CSVs, `annual.csv`, `summary.json`, `seed_means.csv`, `paired_years.csv`,
`paired_blocks.csv`, `regime_means.csv` and `episode_position_means.csv`.
Independent checks verify all 18,250 validation decisions: cost identity, lake
balance, exact annual state continuity, common historical forcing, bounded
requests and frozen source hashes. Project tests pass: **253 passed, one skipped**
(the original 252-pass project suite plus the additional failure-summary check).

PNG walkthroughs in ignored `background/`:

- [Matched protocol and results](../background/images/57_matched_protected_ppo_mpc.png).
- [All five two-year level and release trajectories](../background/images/58_protected_ppo_mpc_trajectories.png).

The weighted-tail MPC sensitivity and this PPO baseline remain unpromoted. No
1998–2010 controller evaluation, default change, fork modification or commit was made.

## PPO drawdown diagnosis and controlled credit experiment

The baseline remains frozen in `ppo_comparison_v1`. Three follow-ups distinguish
inherited storage, action effects and the learner's valuation of future water.
No physical costs, constraints or final test years change.

**Same-state decisions.** `ppo_diagnosis_v1` fixes eight dates in each validation
block: episode days 30, 90, 150, 300, 395, 455, 515 and 665 (zero-based). Replay
saved commands from reset to reconstruct exact state, unrounded target EMA and
causal flow history. Each PPO seed is queried from both its own factual state and
the corresponding MPC901 state. This gives **400 matched PPO/MPC action pairs**.
MPC receives the same observed history; its preceding reference plan is held
fixed, with a separate cold-plan control. Warm/cold requests match in every case.
The shared MPC states repeat across learning seeds and are not independent samples.

PPO requests about **212 m³/s more** on average, but **95/400 request pairs produce
the same actual release** within 10⁻⁵ m³/s because of hydraulic limits. From
MPC-inherited states, its actual release is 50.29 m³/s higher on average; from its
own lower-storage states, 13.85 m³/s higher. Mean first-day cost differences
(PPO minus MPC) are **−0.218** and **−0.623**, respectively: the immediate reward
can favor the larger release. Much of the extra flow remains in the river after
serving the model target; this is not measured waste or electricity production.

For each action pair, apply only that first action, then use the **same frozen
PPO mean-action continuation** for 30 days from identical state/history/clock.
All branches complete. Mean cost differences become **+0.680** from MPC states
and **−0.088** from PPO states. Thus the local effect depends on inherited stock;
it does not show that every larger PPO action is bad. These are short conditional
branches without an ending-stock price, not full MPC-versus-PPO trajectories.
A retrospective refill label uses subsequent recorded inflow only for grouping;
it is never supplied to either controller or used to select the experiment.

**Training progress.** Regress completed stochastic episode cost against decision
count after removing training-window means, separately in the first/last 100k
steps. The late-half fitted improvements per 100k are **1.44%, 0.67%, 0.21%,
3.75%, 2.58%** for seeds 123–127. This controls weather-window composition, but
policy/exploration are changing and no earlier checkpoints were saved. It is not
a fixed-policy evaluation curve or proof of convergence. Logged PPO value loss is
against bootstrapped GAE targets; a small loss does not establish accurate
full-trajectory values.

**Critic and stored water.** Use training pairs 1960–1961, 1976–1977 and 1984–1985,
with fixed origins at episode days 30, 120, 395 and 485. For each of five frozen
policies, reconstruct a stochastic trajectory, then perturb only storage by
−0.15 / 0 / +0.15 m. Run eight stochastic continuations per stock to the original
finite task end, with paired random action draws across stock levels. All **1,440
rollouts** complete. Rewards use the original fixed 0.01 learning-unit multiplier;
no terminal bootstrap is added. The critic estimates its stochastic policy's
return, so deterministic evaluation returns are not treated as its target.

Of 60 origins, **43 negative-stock perturbations fall below the protected operating
floor**. Those central differences remain labeled out-of-support sensitivities.
For reachable-stock interpretation, use only **base → +0.15 m**, which adds
21.885 million m³ and stays in the declared model domain in every case.
Median predicted value per metre is **0.114**, versus **0.550** in paired realized
returns (scaled reward units); signs agree in **39/60** cases. In physical cost
units, the median slopes are about **11.37 predicted vs 54.96 realized per metre**.
This is a substantial storage-value discrepancy in these training-state checks.

Weather remains the recorded historical future, unknown to either policy. Action
replications estimate action-noise uncertainty only; they do not integrate unknown
weather or create independent hydrological evidence. A critic's conditional expected
value can differ from a single realized-weather return. This audit therefore
motivates a controlled test, not a claim of isolated approximation error. Reweighting
those same current-policy trajectories with gamma 0.999 gives a median positive-stock
slope 0.567 versus 0.550 at 0.995; it does not evaluate a newly optimized high-gamma policy.

**One frozen experiment.** Based on training progress and the reachable-stock
checks, `ppo_controlled_v1` changes **only GAE lambda 0.95 → 0.99**, retaining gamma
0.995, physical rewards, architecture, normalization, learning rate, observations,
seeds 123–127 and 200k decisions per seed. GAE mixes longer sampled returns into
critic targets **and** policy advantages; it is not a critic-only intervention.
At a 30-day separation, the trace weight `(gamma × lambda)^30` rises from 0.185 to
0.636; at 90 days, from 0.0063 to 0.258. Episode termination still stops the trace.
Longer traces may improve delayed credit but also increase target variance.

The new collector reproduces the old model, optimizer, configuration and counters
bitwise in a separate 4,096-step control. The study locks its rationale and source
hashes before training. Evaluate final checkpoints only on every two-year validation
block, pair treatment/baseline by **the same learning seed and historical year**,
and compare MPC after averaging its own forecast seeds. Report all seeds, compute
and physical metrics; no validation checkpoint selection or default promotion.
The baseline source and all previous results remain unchanged.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_diagnosis \
  --stage all --root results/simulator/new_ppo_diagnosis
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_diagnosis_report \
  --root results/simulator/new_ppo_diagnosis
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_controlled \
  --factor gae --root results/simulator/new_ppo_controlled \
  --rationale 'Record the training-only evidence and hypothesis before running.'
```

The controlled runner reads the frozen diagnostic root `ppo_diagnosis_v1`; its
new output directory must not exist. Diagnostic stages can also run separately.
Each source manifest is verified before and after execution. Independent checks
verify action/stock accounting and reproduce one entire stochastic-return sample
exactly from its original training-state prefix. Project tests: **259 pass, one
skipped**. See the [storage and delayed-credit PNG](../background/images/59_ppo_storage_and_credit_audit.png).

## Longer GAE: controlled result

The declared lambda experiment is complete: **five fresh seeds × 200,000 decisions**,
all 273 full training episodes per seed plus the counted unfinished prefix, and
**25 complete two-year validation episodes**. No domain/numerical failures occurred.
The sampled training windows match the baseline exactly by seed. Only GAE lambda
changed; source checksums and all 18,250 deterministic validation actions verify.

Annual validation means:

- **Original PPO, lambda 0.95:** cost **631.34**; unmet model target **545.34 million m³**;
  flood exposure **62.95 m·hours** and **5.10 flood days**; low-water exposure
  **23.05 m·hours** and **117.20 low-water days**.
- **Longer GAE, lambda 0.99:** cost **632.55**; unmet model target **546.40 million m³**;
  flood exposure **63.22 m·hours** and **4.86 flood days**; low-water exposure
  **22.93 m·hours** and **116.56 low-water days**.
- **Frozen history MPC reference:** cost **517.51**; unmet model target **448.85 million m³**;
  flood exposure **67.28 m·hours** and **5.06 flood days**; low-water exposure
  **1.387 m·hours** and **17.12 low-water days**.

The treatment costs **1.21/year more (+0.19%)**, with no material improvement in
the low-water problem. Paired seed mean cost differences, treatment minus baseline,
are **+1.97, +1.06, −1.85, +1.48, +3.37** for seeds 123–127. Only one seed improves
on average; 15/50 paired annual records improve. Averaged over seeds, three of ten
years and two of five blocks improve. Episode-end stock is almost unchanged:
**5.650 vs 5.652 million m³**. Environmental-flow shortage remains zero.
These are descriptive paired results, not a statistical significance claim.

The five runs took about **760 summed worker-seconds** versus **787** for the
original study on this machine (not controlled hardware timings). Minibatch
optimizer updates total **153,705 vs 151,139** because the unchanged KL stopping
rule can stop different epochs. Interaction budget is exactly matched; update
count/compute are reported rather than assumed equal. Training stochastic costs
also remain higher late in the longer-GAE run. Its training value-loss magnitude
is not directly comparable with the original because its target definition changed.

**Decision: retain lambda 0.95; do not promote this treatment.** The value-of-water
checks motivate further diagnosis, but simply extending GAE traces did not improve
this controller. We have not shown that critic fitting itself improved, that it
alone caused the policy gap, or that a longer training budget would solve it.
A useful next isolated check would hold the actor fixed while fitting critic
returns on training-only data, then measure the storage-value discrepancy on
separate origins. Separately, nominal-action saturation deserves investigation:
many large requests map to similar feasible releases. Neither experiment is
included or silently introduced here.

Study artifacts: `results/simulator/ppo_controlled_v1/plan.json`, `summary.json`,
`annual.csv`, `paired_seed_years.csv`, `paired_seed_means.csv`, `seed_means.csv`,
all five checkpoints, update/episode diagnostics and complete daily trajectories.
`independent_checks.json` verifies source hashes, one-factor configuration, exact
training-window pairing, physical cost/balance, year-boundary continuity and all
18,250 deterministic requests reconstructed from only exposed observations.
The original baseline/checkpoints remain untouched. Project tests pass:
**259 passed, one skipped**. Final 1998–2010 controller testing remains reserved.

See [the controlled-result PNG](../background/images/60_ppo_longer_credit_results.png).

## Frozen-actor critic fitting and action saturation

`ppo_identification_v1` completes the next two diagnoses with the original five
PPO actors frozen. No validation or final-test trajectories are collected.

**Full-return critic fitting.** Collect four stochastic 730-day trajectories per
block and retain every pre-action observation with its discounted Monte Carlo
suffix return (gamma 0.995, learning reward scale 0.01), plus the actual terminal
observation with target zero. Fitting blocks are **1946–1947, 1950–1951, 1954–1955,
1958–1959, 1962–1963, 1966–1967**. Checking blocks are **1970–1971, 1974–1975,
1978–1979, 1982–1983**. These whole-year sets are disjoint for the new supervised
fit; both actor and original critic had already used the broader training record.
This is not an untouched baseline-training holdout.

There are **146,200 observations/targets** across five seeds, including 200 terminal
observations. Fit only the existing 64–64 critic with a fresh Adam optimizer,
learning rate 0.0003, 3,000 minibatches of 256 and gradient norm cap 0.5. Use the
final fit, with no checking-based early stopping. Actor weights, log standard
deviation, normalization and bounds remain bitwise unchanged. The diagnostic
fitted-critic checkpoints are for evaluation, not resuming the original optimizer.

Average broad checking MSE changes **0.27906 → 0.28798 (+3.2%)**. Four of five seeds
worsen; one improves. Fitting error falls, but that does not establish generalization.
For a separate storage test, use three fixed checking origins per block (episode
days 60, 180 and 425), base and +0.15 m storage, eight paired stochastic continuations
to the original task end: **960 complete stock-branch rollouts**, 60 state contrasts.
Mean absolute storage-slope error improves **0.37511 → 0.27201 (−27.5%)** in scaled
return/metre; sign agreement rises **56/60 → 58/60**. This new state sample differs
from the earlier stock audit. It demonstrates a mixed result: the network can
learn more storage sensitivity, while broad checking accuracy does not improve.

Returns are conditional on the fixed recorded future weather, unknown to the
policy. Action-noise replication does not estimate uncertainty over future weather;
shared days/trajectories are dependent. These diagnostics do not isolate pure
approximation error or show a controller improvement: the actor never changes.

**Saturation of exploration.** At six fixed dates in each rep0 trajectory (days
30, 120, 240, 395, 485, 665), sample 64 actions from the frozen actor: **300 states,
19,200 sampled requests**. Check instantaneous release and the entire day's
physical release under the same recorded forcing. Full-day maximum means the
release produced by the original maximum nominal command, not a new flood-safe
capacity guarantee. Count samples within 10⁻⁵ m³/s of its daily mean; exclude a
collapsed full-day range from the rate average.

Across **180 noncollapsed fitting states**, the mean upper-plateau probability is
**97.03%**; checking states give about **97.4%**. Substantial nominal request noise
often produces almost no actual-flow variation. These are fixed sampled states,
not an estimate that 97% of all historical training actions were redundant.
Instantaneous clipping alone would not establish full-day equivalence because
hourly level and capacity change; the full-day simulation is why this audit matters.

The rule was frozen before collection: prioritize an action-transform experiment
if the fitting-state daily plateau rate is at least 50%. Otherwise, consider a
critic-training experiment only if both checking MSE and stock-slope error improve
by at least 20%. The saturation result selects the first branch. Checking/validation
controller scores do not select its hyperparameters.

Independent checks verify whole-year separation, frozen actor/scales, all dataset
sizes and terminal targets, exactly replay a full stochastic trajectory with every
suffix return, and reproduce one full-day saturation sample. Original baselines,
critic diagnosis and negative GAE experiment remain unchanged.
[Critic and saturation PNG](../background/images/61_ppo_critic_and_saturation.png).

## Full-range hydraulic PPO protocol

The selected `hydraulic_ppo_v1` experiment changes **only the deterministic mapping
from PPO's sampled fraction to a requested release**. It is a model-informed action
parameterization: it adds known outlet/reserve physics to action construction,
without adding policy observations or future forcing. The actor/critic networks,
initial latent distribution, all PPO hyperparameters (including lambda 0.95),
reward, geometry, constraints, reset, train split and five-seed 200k budget match
the original PPO. Its initial *physical* action distribution changes as intended;
this does not isolate a unique initialization-versus-learning mechanism.

Use only the observed level and season, with fixed outlet/environmental-flow and
reserve formulas, to calculate current effective releases from nominal minimum
and maximum commands. The **middle 80%** of policy fractions spans that interval.
The **lower and upper 10% tails** retain the rest of the original nominal request
range. When the interval collapses, retain the identity nominal map. All original
nominal commands remain accessible, including maximum requests that can matter
as the lake fills later in the day. No current or future inflow/demand is used to
construct the map; season is decoded from the already observed sine/cosine.

The chosen **one constant request is held for 24 hours**. The unchanged simulator
still recomputes hydraulic clipping hourly. Simply restricting commands to the
morning interval would remove some later-day opportunities; recomputing commands
hourly would introduce a different controller. Neither is done here.

PPO stores the original pre-warp latent actions and their log probabilities.
Because the state-only actuator map is fixed across policy updates, importance
ratios are valid for this latent-action policy, including many-to-one outcomes.
The entropy coefficient remains zero; no transformed-entropy formula is assumed.
Checkpoint loading requires an explicit `.action.json` contract to prevent silently
using trained weights with the wrong action mapping.

The 4,096-step transformed-policy smoke completes five full training episodes and
six internal year boundaries, with finite updates and hourly balance error below
3×10⁻⁸ m³. Unit checks verify monotonicity, full nominal endpoint coverage, no
future-input dependence, unchanged initial networks/latent likelihoods, checkpoint
contracts and bitwise identity-collector parity. Final-only evaluation uses all
five validation blocks and pairs learning seeds with the original PPO. The frozen
MPC reference remains separate. Domain failures stop a seed without fabricated
penalties, discarded hard years or extrapolation; incomplete required scopes
suppress mean controller scores. No default promotion is automatic.

Before endpoint results exist, a supplemental mechanism protocol is locked:
compare original actor/raw requests, that **same actor with the transform**, and
trained transformed actor from the same 180 fitting states, with common standard
Normal draws and identical recorded weather. This distinguishes immediate mapping
from subsequent policy changes descriptively, without a new training sweep.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_identification \
  --stage all --root results/simulator/new_ppo_identification
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.hydraulic_ppo_study \
  --stage smoke --root results/simulator/new_hydraulic_smoke
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.hydraulic_ppo_study \
  --stage all --root results/simulator/new_hydraulic_study
```

The controlled runner uses the frozen `ppo_identification_v1` selection; fresh
output directories do not replace it. Source manifests are checked before/after.
Project tests: **267 passed, one skipped**. The original fork and prior source
manifests remain intact. Final 1998–2010 controller testing is reserved.

## Full-range hydraulic PPO results

All **five seeds complete 200,000 decisions** and all **25 two-year validation
episodes** complete. No domain/numerical failures occur. The original hyperparameters
and sampled training-window sequence match by seed; only the state-dependent
request transform changes. No fitted-critic weights from the preceding diagnostic
are used in this policy study.

Annual physical metrics, original PPO → mapped PPO:

- **Total cost:** **631.34 → 622.31**, down **1.43%**.
- **Unmet model target:** **545.34 → 541.56 million m³**, down about **0.69%**.
- **Low-water exposure:** **23.05 → 17.11 m·hours**; days **117.2 → 101.0**.
- **Flood exposure:** **62.95 → 63.63 m·hours**; days **5.10 → 5.20**.
- **Episode-end stock:** **5.652 → 6.335 million m³**, relative to the model datum.
  Environmental-flow shortage remains zero.

Paired seed mean cost differences are **−0.21, −36.91, −3.77, −4.82, +0.54** for
seeds 123–127. Four seeds improve, but seed 124 contributes most of the pooled
gain. **34/50** seed-year records improve. After own-seed averaging, all ten
historical years, all five two-year blocks and dry/middle/wet groups improve versus
the original PPO. These are dependent, reused validation results, not significance
or generalization guarantees. The frozen history MPC still has lower annual cost:
**517.51**, versus **622.31** for mapped PPO. MPC also has far fewer low-water days.
We do not promote a default or select seed 124 as the representative policy.

**Fixed-state mechanism check.** Its protocol was locked before endpoint outcomes.
Use the same 180 original fitting states and paired standard-Normal action draws:

- Original actor with raw requests: **97.03%** of samples on the full-day upper plateau;
  mean within-state actual-release standard deviation **3.28 m³/s**.
- The same actor weights with only the map applied: **15.43%** plateau mass;
  effective standard deviation **19.22 m³/s**.
- Trained mapped actor: **48.78%** plateau mass; effective standard deviation
  **7.67 m³/s**.

The mapping initially creates much more physical exploration. Training partially
returns toward saturation but preserves more effective variation than the original
policy on these fixed states. These results support the mechanical effect, not a
unique attribution of the annual gain: the physical initialization, model prior,
state-dependent sampling and subsequent actor updates all belong to the treatment.
Lower saturation alone does not guarantee better control, as seed variability shows.

The five runs use **1,000,000 new training decisions** and **141,979 minibatch updates**
(vs 151,139 original updates; unchanged KL stopping can stop different epochs).
Summed worker time is about **782 seconds** on this machine, not a controlled
hardware comparison. Diagnostic collection is extra development cost, not free:
146,000 broad frozen-policy decisions plus terminal observations, 488,000 stock-branch
decisions, and 19,800 one-day saturation evaluations including endpoint controls.
Those diagnostic returns do not train the mapped policy. Its subsequent mechanism
audit adds 34,560 sampled one-day evaluations plus shared endpoint controls. No
comparative total sample-efficiency claim is made.

**Decision:** retain the mapped controller as a promising, modest experimental
improvement, preserve the original reference and negative GAE result, and keep
MPC as the stronger control baseline in this simulator. The broad critic-fit gap
remains open; neither saturation reduction nor the critic-only fit solved the
storage problem. A future change should isolate broad-versus-storage critic
training or inspect return/action tradeoffs before another parameter sweep.
No further policy experiment is included here.

Artifacts in `results/simulator/hydraulic_ppo_v1/` include frozen `plan.json`,
`mechanism_plan.json`, five checkpoints with required action contracts, complete
daily trajectories, `annual.csv`, `summary.json`, `paired_seed_years.csv`,
`seed_means.csv`, `mechanism.csv`, `mechanism_summary.json`, `collector_parity.json`
and `independent_checks.json`. The identity collector also reproduces the original
4,096-step smoke checkpoint bitwise, including its optimizer/counters.

Independent verification reconstructs **every one of the 18,250 deterministic
validation requests** from only exposed observations plus the frozen action
contract, checks physical cost and water balance, exact annual state continuity,
common forcing and training-window pairing, and reproduces the mechanism's
baseline saturation rates. Source hashes, prior baselines and the fork remain
unchanged. Tests: **267 passed, one skipped**. Final test years remain reserved.
See the [mapping and results PNG](../background/images/62_hydraulic_ppo_results.png).

For a fresh reproduction including the mechanism plan frozen before training:

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.hydraulic_ppo_followup \
  --root results/simulator/new_complete_hydraulic_followup
```

This wrapper uses the same frozen selection, training/evaluation and mechanism
functions. It prepares both plans up front, runs two CPU workers and requires a
new output directory. The recorded original run prepared its supplemental plan
while training was still underway, before any endpoint scores existed.

## Seed 124 and matched critic coverage

`ppo_coverage_v1` completes the seed-gain diagnosis, data-coverage check and one
matched critic experiment. Existing policies and all previous artifacts remain
frozen. The seed audit is retrospective validation analysis; the critic refits
and their checking data stay entirely within training years. No new actor is trained.

**Where seed 124's gain comes from.** Retain all five mapped endpoints. Its annual
cost improvement of 36.91 comprises **25.87 less unmet-target cost**, **11.71 less
low-water exposure**, offset by **0.68 more flood exposure**. Low-water days fall
by **33/year**. These contributions use unchanged physical weights; they are not
independent effects or measurements of real irrigation benefit. Seed 124 had a
relatively weak original PPO endpoint and a better mapped endpoint on the reused
validation record; the large paired gain does not establish a representative seed.

The two-year water ledger reconciles its average change exactly: **51.74 million
m³ more delivered to the model target**, **53.47 million m³ less remaining river
flow**, and **1.73 million m³ more ending stock**. Most of the changed allocation
serves the target at different times rather than remaining as permanent stock.
Remaining river flow is not measured waste; downstream benefits are not modeled.
All 25 seed/block ledgers reconcile, with common forcing and equal initial stock.

**Same inherited state.** Replay the frozen history-MPC901 trajectories at eight
fixed dates per validation pair, giving 40 common full states/history/clock. Query
all five mapped policies, and run each for 30 days from the same state: **200
complete conditional branches**. Seed 124 releases **13.01 m³/s less on the first
day** than the mean of the other four seeds, has **2.01 lower mean branch cost**,
and ends **0.0182 m higher**. It is not uniformly better: the strata contain 33
lower-stock states, five retained-stock states, one low-water and one flood state;
individual extreme-state outcomes cannot support broad claims. Thirty-day branches
do not price ending stock and are not new full-episode controller scores.

**Critics do not order these policies.** Use twelve common *training* states from
the original seed123 trajectory, base and +0.15 m stock, four paired stochastic
continuations for each mapped policy: **480 complete full-remaining-task rollouts**.
Each critic is compared only with its own stochastic-policy returns. Seed 124's
mean storage-slope absolute error is **0.481**, while seed 126 has **0.190**; yet
seed 126 does not have the best annual policy outcome. On these states, predicted
versus realized mean stock slopes are **0.203 vs 0.684** for seed 124 and **0.425
vs 0.614** for seed 126, in scaled-return/metre units. Endpoint critic accuracy
alone does not explain policy ranking or establish what happened during training.
The states can be off-distribution for an individual mapped actor, and weather
is a fixed recorded future. Raw values from different policies are not ranked.
Seed 124 was chosen post hoc for explanation, not promoted as a champion.
[Seed explanation PNG](../background/images/63_seed124_release_and_storage.png).

**Matched coverage experiment.** The original supervised fit used only twelve
early training years, while baseline PPO had trained across 42. Test that coverage
hypothesis without increasing samples or updates:

- **Concentrated:** original six two-year fitting blocks, four stochastic
  trajectories each (twelve years).
- **Diverse:** those same six blocks plus **1968–1969, 1972–1973, 1976–1977,
  1980–1981, 1984–1985, 1986–1987**, two trajectories each (24 years).

Both arms use **24 full trajectories / 17,544 observations per seed**, the same
initial critic/actor/scales, 3,000 critic updates, 256-sample batches, learning rate
0.0003, norm cap 0.5 and identical minibatch-index stream. The diverse arm trades
weather-year coverage for repeated action-noise trajectories; it does not hold
those two aspects independently fixed. Its later fitting years also mean this
internal comparison is not forward-in-time generalization. The same whole checking
blocks **1970–1971, 1974–1975, 1978–1979, 1982–1983** remain excluded from both
new refits; baseline training had already seen them. New data collection is only
43,800 frozen-policy decisions plus 60 terminal observations. Existing data and
stock-return labels are reused. Every concentrated checkpoint reproduces the
previous supervised-fit model/optimizer/configuration/counters **bitwise**.

**Result: broader coverage alone does not fix this critic at the matched budget.**
Mean checking MSE is **0.27906 original / 0.28798 concentrated / 0.29415 diverse**.
Diverse worsens by **2.14%** versus concentrated; only two of five seeds improve.
Mean absolute error improves modestly **0.44596 → 0.43196**, illustrating why a
single error statistic is insufficient. Storage-slope mean absolute error is
**0.37511 / 0.27201 / 0.50200**: diverse is worse than both references, by about
**84.6%** versus concentrated.

Checking observations are nevertheless closer to the diverse fitting set in fixed
scaled feature space: mean per-seed median nearest distance **0.59867 → 0.56315**;
p90 **1.17236 → 1.09722**. A post hoc analysis uses the *same* concentrated-distance
strata in both arms: nearer-half errors improve, while middle/furthest strata
worsen. Better geometric support does not guarantee accurate conditional return
estimates. This experiment does not rule out coverage effects generally: repetitions,
target variability, model capacity and finite fitting budget remain relevant.
No causal attribution or statistical-significance claim is made.

The predeclared gate requires both mean checking MSE and stock-slope error to beat
**both** original and concentrated critics, with MSE improving in at least four
seeds. It **fails**. Do not insert this diverse refit into PPO or launch a policy
run from it. The mapped PPO remains the same experimental candidate at 622.31
annual validation cost; history MPC remains stronger at 517.51. Rewards, physics,
defaults and final test years do not change.
[Coverage comparison PNG](../background/images/64_critic_coverage_comparison.png).

Artifacts under `results/simulator/ppo_coverage_v1/` include frozen `plan.json`,
sealed `dataset_manifest.json`, added data, both refit checkpoints and metrics,
`summary.json`, `checking_groups.csv`, `stock.csv`, `seed_gain_summary.json`,
`seed_common_actions.csv`, own-policy return checks, `seed_stock_ledgers.csv`,
`seed124_common_contrasts.csv`, `coverage_support_strata.csv` and derived-source
hashes. Independent checks verify source/data hashes, all five exact concentrated
controls, frozen actors, matched budgets, brute-force nearest-distance samples,
a complete newly collected trajectory and every suffix target, water ledgers,
annual cost decomposition, one exact common-state branch and one full own-policy
stochastic-return replay. Tests: **272 passed, one skipped**.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_coverage \
  --stage all --root results/simulator/new_ppo_coverage
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_seed_gain \
  --stage all --root results/simulator/new_ppo_coverage
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_coverage_report \
  --root results/simulator/new_ppo_coverage
```

Use a fresh output directory. Both studies retain the frozen reference endpoints
and prior data roots. No commits or fork changes are made; PNGs remain in ignored
`background/` and study outputs remain in ignored `results/`.

## Frozen PPO execution and mapped training progress

`ppo_execution_v1` completes both checks without training a new policy or changing
the existing reference checkpoints. Every original and mapped PPO seed is included.
Final 1998–2010 controller testing remains reserved.

**Execution alignment.** Evaluate each frozen checkpoint on all five validation
pairs, with one deterministic trajectory and five stochastic repetitions (action
seeds **2101–2105**). After checkpoint loading/reset, Torch uses seed
`action_seed + 10000*block_start`, giving common standard-Normal draws across
policies for comparisons. Keep physical state/history through the annual boundary.
This produces **300 complete 730-day episodes / 219,000 decisions**, with no
domain/numerical failures. All **50 deterministic reference trajectories replay
bitwise**, and model digests/counters remain unchanged throughout execution.

Deterministic means applying the existing squashing/action map to the Gaussian
mean. It is **not** generally the expected physical action after squashing,
mapping and hydraulic clipping. Stochastic execution samples the learned Gaussian
before those same transformations. No variance rescaling, new controller input,
reward modification or mode-specific training is introduced.

Annual cost, deterministic → stochastic:

- **Original PPO:** **631.34 → 637.80**, about **1.02% higher**.
- **Mapped PPO:** **622.31 → 632.15**, about **1.58% higher**.

Stochastic mean cost is higher in **every learner seed** for both variants. For
mapped PPO, unmet model target increases **541.56 → 552.35 million m³/year**;
flood exposure increases slightly **63.63 → 64.19 m·hours**. Low-water exposure
improves **17.11 → 15.61 m·hours**, and low-water days fall **101.0 → 96.13**.
Thus noise changes the physical tradeoff but does not improve the configured cost.
The stochastic action-seed cohort mean costs range **630.65–634.02** for mapped
and **636.65–638.96** for original PPO; no best action seed is selected.

The full-two-year **discounted** physical cost at gamma 0.995 also favors
deterministic execution: original **285.98 → 289.22**, mapped **278.37 → 284.11**.
Discounting runs through the annual boundary and is not restarted each year.
These discounted block scores are not annual undiscounted costs or a direct
measurement of PPO's surrogate optimization loss.

Average action repetitions within learner/year before pairing with its single
deterministic endpoint. There are **100 paired learner/year comparisons**, not
five independent deterministic replicas. Action seeds share the same ten reused
validation years and do not create independent hydrological records. The study
has no p-values, mode-selection promotion or general claim that deterministic
execution is universally better. It also does not test disabling exploration
during training. For these checkpoints, sampled deployment does **not** explain
away the PPO–MPC gap; retain the existing deterministic benchmark convention.
[Execution alignment PNG](../background/images/65_ppo_execution_alignment.png).

**Mapped training progress.** Inspect the saved completed stochastic training
episodes and update diagnostics. Each seed has **273 full 730-day episodes /
199,290 decisions**, plus a **710-decision unfinished prefix** excluded from
full-episode cost curves but counted in the original 200k budget. Remove sampled
weather-window means within each seed and estimate within-window linear trends
over early 100k, late 100k and last 50k decisions. This adjusts composition of
sampled years, not policy/exploration changes, temporal dependence or optimizer
noise. The plotted line is a 20-completed-episode moving mean. No intermediate
checkpoints were saved, so this is not a frozen-policy validation learning curve.

Mapped late-half fitted cost declines per 100k are **0.19%, 14.51%, 0.98%, 2.19%,
5.52%** for seeds 123–127. The corresponding last-50k trends, normalized per 100k,
are **−2.00%, −7.10%, −0.17%, +0.92%, −0.79%** (negative is improvement).
These fitted/extrapolated rates are not promised improvements over future steps.
Progress is heterogeneous and strongest in seed 124; seed 126 turns slightly up
in the last 50k despite an improving late-half slope.

The rule was frozen before inspecting curves: support a future duration probe
only if at least three seeds have late fitted decline >=2% per 100k and at least
three still decline in the last 50k. It **passes: 3/5 and 4/5**, respectively.
The rule applies to cohort counts in the two periods, not a claim that the same
three seeds meet both thresholds. It is a descriptive heuristic, not a statistical
convergence test or an automatic training launch.

Underlying Gaussian standard deviation decreases too: late-quartile means are
roughly **0.29–0.33**, versus **0.48–0.53** in the first quartile. These are latent
Gaussian spreads, not physical-release standard deviations. Update losses and
gradients remain finite. PPO's logged ratio-clipping fraction is distinct from
hydraulic action saturation; minibatch value loss is against changing bootstrapped
GAE targets, not a global Monte Carlo accuracy measure. Numerical stability and
low training loss do not establish policy convergence.

**Decision at that stage:** keep deterministic execution and the existing controller results.
The next justified policy experiment was a bounded **duration-only** extension of
mapped PPO, with all five seeds retained and the complete 200k prefix preserved
or reproduced before continuing. Freeze its budget/checkpoint rules first, monitor
training-only conditions, and evaluate declared endpoints rather than selecting
validation checkpoints. That execution study did not train longer. The subsequent 400k duration run and
inspection are recorded below; it did not close the MPC gap.
[Mapped progress PNG](../background/images/66_mapped_ppo_training_curves.png).

Artifacts include `plan.json`, all 300 daily CSVs/status records, `annual.csv`,
`blocks.csv`, `mode_summary.csv`, `paired_annual.csv`, `paired_learner_means.csv`,
`action_seed_means.csv`, `curve_episodes_*.csv`, `curve_slopes.csv`,
`curve_updates.csv`, `curve_summary.json` and `independent_checks.json` under
`results/simulator/ppo_execution_v1/`. Independent checks reconstruct **all
219,000 requests**, verify discounted/physical costs, water balance, annual state
continuity and action-seed aggregation, and reproduce all **30 curve slopes** using
an explicit weather-dummy least-squares regression. Source/checkpoint hashes
verify. Tests: **277 passed, one skipped**. No commits, default or fork changes.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.ppo_execution \
  --stage all --root results/simulator/new_ppo_execution
```

Use a fresh output directory. Stages `prepare`, `curves`, `execute`, `aggregate`
can also run separately against the frozen plan. Execution uses two CPU workers;
fresh roots protect prior studies, and model files are never modified. PNGs remain in ignored
`background/`; the approximately 60 MB study output remains in ignored `results/`.

## Mapped PPO duration runner

The duration-only runner is implemented at
`src/reservoir_control/systems/lake_como/simulation/mapped_ppo_duration.py`. The user subsequently completed
the five-seed 400k study; see the result and curve inspection below. For a fresh
reproduction from the repository root, run:

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.mapped_ppo_duration \
  --root results/simulator/new_mapped_ppo_duration \
  --workers 2
```

This uses all five seeds **123–127**, unchanged mapped-PPO settings and physical
costs. Each seed performs **400,000 total decisions**: freshly reproduce the
original 200k run, then add 200k in the same live collector. Across five seeds,
that is two million decisions including reproduction work. It does not load a
200k inference checkpoint into a new optimizer or reset reservoir episode.

At 200k, reproduce the original shortened final rollout before updating and saving
`prefix_200000.pt`. Require exact model/optimizer/configuration/counter equality,
the same action contract, episode accounting and optimizer update diagnostics
against the frozen `hydraulic_ppo_v1` reference. Verification uses `torch.load`
without constructing another agent, and checks that it has not changed Torch,
shuffle/environment RNG or the live physical/history state. A mismatch stops that
seed before extension. Current environment, causal history, optimizer and RNG
streams remain in memory for continuation; a normal episode reset occurs only
when a true 730-day task terminates.

Only the fixed **400k final checkpoint** is evaluated, deterministically on every
1988–1997 validation pair. Pair with the frozen mapped-PPO **200k** results, not
original raw-action PPO. No intermediate validation/checkpoint selection, cost or
physics change, default promotion, or final 1998–2010 evaluation is introduced.
A failed or missing required seed/year withholds the cohort comparison means;
CLI execution returns nonzero for incomplete stages.

Outputs include per-seed progress/update logs, prefix verification and checkpoint,
final checkpoint/action contract, training status, daily validation trajectories,
cached endpoint annual metrics, and root `annual.csv`, `paired_seed_years.csv`,
`paired_seed_means.csv`, `seed_means.csv` and `summary.json`.

The default stage is `all`, which requires a **fresh output directory**. Optional
stages are `prepare`, `run`, `evaluate`, `aggregate`. `prepare` only validates and
freezes the plan; after preparing, use the later stages on that same root instead
of `all`. Completed/failed recorded seeds are not retrained, and completed endpoint
evaluations are cached with integrity checks. An interrupted seed directory without
`training.json` is **not** a resumable state: preserve it and use a fresh root.
Prefix model files do not contain a full restartable environment/RNG snapshot.
`--workers 1` reduces CPU parallelism; the default is two one-thread Torch workers.

Tests use reduced budgets only: a 4,096-step reference followed by a short live
extension, a prefix exactly on a terminal boundary, a deliberately mismatched
checkpoint that must stop before extension, prepare-only behavior, staged caching,
interrupted-output protection, and incomplete/duplicate/nonfinite cohort guards.
The original runner tests used reduced budgets. The full run is now recorded in
`mapped_ppo_duration_v1`; inspection does not rerun training. The current suite
passes **288 tests, with one skipped**. Existing reference sources/checkpoints
and the fork remain untouched. Nothing is committed.

## Mapped PPO 400k result and curve inspection

The user completed `mapped_ppo_duration_v1`: all five 200k model/optimizer/config
prefixes verify exactly, all seeds reach **400,000 decisions**, and every one of
**25 two-year validation episodes** completes. The runner's `summary.json` and
endpoint files are preserved; the inspection only reads them.

Annual means at 200k → 400k:

- **Cost:** **622.31 → 617.91 (−0.71%)**.
- **Unmet model target:** **541.56 → 537.56 million m³**.
- **Flood exposure:** **63.63 → 62.76 m·hours**; days **5.20 → 4.86**.
- **Low-water exposure:** **17.11 → 17.59 m·hours (+2.8%)**; days **101.00 → 101.02**.
- **Mean annual ending stock:** **6.485 → 7.373 million m³**. This averages both
  year boundaries. The distinct *episode-final* stock mean is **6.335 → 7.932**.

Four seeds improve on mean cost, though one barely changes. Paired mean annual
cost differences (400k minus 200k) are **+0.206, −9.214, −6.329, −0.081, −6.547**
for seeds 123–127. **33/50** paired annual records improve. These are descriptive
paired development results, not a statistical significance or convergence claim.
PPO's annual cost remains about **19.4% higher** than MPC's **517.51**.
No controller default is promoted.

**The training curves were saved.** Each `seed*/updates.csv` contains **196**
rollout/update records through step 400k, including losses, gradients, accepted
approximate KL, ratio-clipping fraction, early-stop flags and Gaussian spread.
Each `training.json` holds **547 complete 730-day episodes** with historical years,
physical return and terminal level, plus a counted **690-decision unfinished
prefix**. Prefix and final model checkpoints exist; there are no intermediate
saved-policy validation curves.

`mapped_duration_inspection_v1` fits within-weather-window trends on complete
episodes. The primary convention requires each episode to lie wholly within its
phase. In particular, the episode ending at **200,020** contains **710 decisions
before 200k and only 20 afterward**. It is plotted in the overall history but
excluded from extension-only fits. The 200k–400k primary fit therefore uses **273
whole episodes / 199,290 decisions** per seed; 20 extension decisions complete the
straddling episode and 690 form the final unfinished prefix. The older
ending-in-period convention is also reported as a sensitivity, without choosing
whichever trend looks better.

Primary fitted cost changes, normalized per 100k decisions:

- **200k–400k:** **−0.346%, −0.136%, −0.965%, +0.447%, −1.126%** (seeds 123–127).
- **300k–400k:** **−0.020%, −0.916%, −0.225%, +0.540%, −0.069%**.
- **350k–400k:** **−0.570%, −2.984%, −0.958%, +1.289%, −0.315%**.

Negative means improving cost. The last-50k estimates are normalized/extrapolated
to 100k units, not promised future gains. End-time sensitivity gives the same broad
picture and trend signs. Local improvements remain, especially for seed 124, but
the strong earlier cohort progress mostly flattens out. Stochastic training costs
mix changing actor/critic behavior and shrinking exploration with correlated
weather trajectories; fixed effects remove weather-window composition, not every
source of noise. No curve proves a universal PPO performance ceiling.

Updates stay finite and gradients remain active. Mean Gaussian latent spread in
300k–400k is about **0.265–0.297**, versus **0.311–0.337** in 100k–200k. Late
accepted-minibatch approximate KL is around **0.0063–0.0069**; the unchanged
KL-based early-stop rule still activates. Declining GAE value loss does not itself
prove accurate full-return or storage-value prediction. These records show no
numerical breakdown that accounts for the remaining control gap.

**Decision: stop the duration sweep at 400k; do not run 800k.** The modest final
validation gain, persistent low-water behavior and weak/noisy late trends support
this scope decision. Record all seeds and the low-water-severity tradeoff, preserve
previous controls/negative findings, and leave test years reserved. This is not a
claim that more training could never help. Current understanding and the full
investigation overview now lead `RESULTS.md`; earlier DamEnv results are preserved
in a collapsed historical section and remain numerically incomparable.

[400k curves PNG — local](../background/images/67_mapped_ppo_400k_curves.png).
Inspection artifacts are in `results/simulator/mapped_duration_inspection_v1/`:
source hashes, per-seed episode curves, 40 primary/sensitivity trend rows, update
summaries, endpoint/episode-final stock metrics and `independent_checks.json`.
A separate explicit weather-dummy least-squares regression reproduces all forty
slope estimates. All **18,250** saved endpoint daily records pass physical-cost,
water-balance and annual-continuity checks; means match the user's completed run.
The original study files and model weights are unchanged. Tests: **288 passed,
one skipped**. No new training, final-test use or commits.

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.simulation.mapped_duration_inspection \
  --source results/simulator/mapped_ppo_duration_v1 \
  --output results/simulator/new_mapped_duration_inspection
```

Use a fresh inspection output directory; this command does not train or run a new
controller. PNGs remain in ignored `background/`, and generated metrics remain in
ignored `results/`.
