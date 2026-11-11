# Results and current understanding

## Ontario–St. Lawrence

**The Ontario simulation and first MPC comparison have completed.** The current
pilot matches the causal rule curve in 1995 and 2019. In 2017 it trades a small
reduction in downstream exceedance severity for greater Ontario severity and
one additional downstream exceedance period.

### Simulation and benchmark checks

- The native Python reference ran **5,808 quarter-month decisions over 1900–2020**,
  following the supplied initialization year. Adapter parity passed and all
  **seven native performance indicators** executed.
- The causal rule curve ran **5,760 scored decisions over 1901–2020**, following
  initialization and a past-only 1900 warmup.
- The MPC comparison contains **288 paired trajectory rows**, covering 48 periods
  per controller in each of 1995, 2017 and 2019. Each controller starts the year
  from the same continuous causal rule-curve state.

### Rule curve versus four-quarter-month MPC

The fixed study caps are **75.50 m Ontario end-of-period level** and **22.10 m
modeled Pointe-Claire level**, using their respective simulator level coordinates.
The Bv7 F limit retains its separate tiered thresholds. Operational threshold
and datum validation remain open.

- **1995.** MPC and the rule curve use identical releases. Peak Ontario
  end-of-period level is **74.828 m** and peak Pointe-Claire level is **21.78 m**.
  Both stay below the study caps throughout the year.
- **2017.** MPC changes **10 of 48 releases**. Ontario exceedance rises from
  **6.99 to 7.15 m·periods**, with **19 periods above the cap** for both controllers.
  Peak Ontario end-of-period level rises from **76.109 to 76.126 m**.
  Pointe-Claire exceedance falls from **2.98 to 2.95 m·periods**, while periods
  above its cap increase from **10 to 11**. Both peak at **22.94 m** downstream.
- **2019.** MPC and the rule curve use identical releases. Peak Ontario
  end-of-period level is **75.919 m**, with **14 periods above the cap** and
  **3.88 m·periods** of exceedance. Pointe-Claire peaks at **22.67 m**, with
  **9 periods above the cap** and **1.99 m·periods** of exceedance.

Exceedance in m·periods sums the amount above the study cap at each model period.
These trajectories resolve quarter-month conditions. Daily peaks require finer
observations and a suitable model time step.

### What the controller currently optimizes

MPC searches four future releases, executes the first and replans. Its objective
penalizes squared deviations from forecast rule-curve releases and from the
rule curve's forecast terminal Ontario level. This keeps its preferred schedule
close to the benchmark while searching for schedules that satisfy the study caps.

Both controllers use past-only persistence forecasts of Ontario supply and SLON,
plus a trailing annual supply estimate. They share the same estimated ice and
roughness inputs and the same Bv7 release limits. Realized supplies determine
storage and routed levels after the decision.

The beam search retains **12 schedules**. It finds no schedule meeting both
forecast caps on **0/48 decisions in 1995, 20/48 in 2017 and 17/48 in 2019**.
Those decisions use the retained schedule with the smallest peak exceedance,
then smallest total exceedance, then lowest tracking/terminal objective.
A finite search leaves global feasibility unresolved. Forecast compliance can
also differ from realized compliance. Pointe-Claire first-step forecast mean
absolute error is approximately **0.095 m, 0.150 m and 0.106 m**, respectively.

### Shared controller score

A single scorer now applies the same settings to every Ontario controller.
The initial flood-exposure score is the negative sum of Ontario and Pointe-Claire
exceedance, using equal weights and 1 m scales from
`configs/studies/loslr/scoring.json`. Higher is better. Exceedance is measured
above the experimental caps of **75.50 m for Ontario** and **22.10 m for
Pointe-Claire**, summed over the 48 quarter-month periods in each year.

**Current overview across 1995, 2017 and 2019:**

- **Plan 2014 rule curve:** mean annual score **−5.282**; mean annual Ontario
  exceedance **3.625 m·periods** and downstream exceedance **1.657 m·periods**.
- **Four-period MPC:** mean annual score **−5.325**; mean annual Ontario
  exceedance **3.678 m·periods** and downstream exceedance **1.647 m·periods**.
- **Ontario PPO and bilevel MPC:** results pending.

The rule curve scores slightly better overall. MPC reduces downstream
exceedance slightly but increases Ontario exceedance in 2017; the controllers
tie in the other two years. These three development cases do not establish
performance across the historical record.

Annual scores:

- **1995:** rule curve **0.000**, MPC **0.000**.
- **2017:** rule curve **−9.972**, MPC **−10.101**.
- **2019:** rule curve **−5.873**, MPC **−5.873**.

These are scores for the declared flood-exposure criterion. Navigation,
low-water and other service objectives still require their own terms if the
comparison is expanded. Equal upstream/downstream weights are an initial study
choice, rather than calibrated stakeholder costs. Peaks, exceedance counts and
terminal level are also retained alongside the scalar score. Ontario PPO can
use the identical scorer when its trajectories exist.
Saved annual and aggregate scores are in `results/loslr/controller_scores_v1/`.

### Forecast, search and objective diagnostics

All three controlled checks completed on the same 1995, 2017 and 2019 cases.
The reference costs, four-period horizon, study caps and release rules were
retained for the forecast/search checks. Objective weights changed only in the
explicit ablations.

- **Fallback feasibility.** Of 37 original fallback states, **32** have disjoint
  one-step release requirements for the Ontario and Pointe-Claire study caps,
  allowing for native rounding. This is conditional on the inherited state and
  forecast. Three have a feasible first action but no feasible four-step schedule
  in the complete candidate tree. Two late-2019 states have overlapping relaxed
  cap bounds, but the request grid only achieves releases up to **9,910 m³/s**,
  below the approximately **10,304 and 10,404 m³/s** needed for Ontario.
- **Search coverage.** Increasing the beam from 12 to **96** and enumerating
  **15,062 complete candidate schedules** across the 37 fallback states find
  **zero additional cap-feasible plans**. A larger search improves predicted peak
  exceedance at two 2017 states. Complete candidate enumeration covers the
  declared finite request tree. Continuous-action feasibility remains open.
- **Perfect inflow diagnostic.** Four realized Ontario supplies and SLON values
  replace persistence forecasts in planning and M/F-limit evaluation. The causal
  annual estimate and lagged ice/roughness estimates are retained. In **2017**,
  Ontario severity falls **7.15 → 6.54 m·periods** and downstream peak falls
  **22.94 → 22.50 m**, while downstream severity rises **2.95 → 3.15 m·periods**.
  In **2019**, Ontario severity falls **3.88 → 3.76 m·periods** and downstream
  peak falls **22.67 → 22.49 m**, while downstream severity rises
  **1.99 → 2.18 m·periods**. The 1995 case stays below both study caps.
- **Objective ablations.** Removing release tracking, terminal tracking or both
  changes **zero of 144 executed releases** in each arm. Re-ranking the common
  fallback candidate trees under those four objectives also selects the same
  first release. The current ranking prioritizes peak and total constraint
  exceedance before either tracking term. The tracking penalties did not explain
  the tested release decisions.

These results direct attention toward cap compatibility, operational release
limits and earlier storage preparation. One-step conflicts at the reached states
leave annual avoidability unresolved. Forecast improvement changes the upstream/
downstream tradeoff and needs evaluation on several physical metrics.

The complete records are in `results/loslr/mpc_diagnostics_v2/`, including annual
arm trajectories, request grids, saved fallback states, common-candidate scores,
settings and source/output hashes. Three explanatory PNGs are saved locally:

- [Fallback bounds](background/images/ontario_mpc_fallback_diagnosis.png).
- [Oracle inflow trajectories](background/images/ontario_mpc_oracle_inflows.png).
- [Objective ablations](background/images/ontario_mpc_objective_diagnosis.png).

The project suite passed **322 tests with one skip**. The persistence arm
reproduced the original pilot's decisions and levels.

### Interpretation and saved evidence

The diagnostics identify conflicts between the current study caps at reached
states and confirm sensitivity to the inflow information supplied to M/F limits.
Operational threshold validation and the recovery priority need review before
extending the comparison. The Ottawa follower
currently has a storage-balance scaffold and an evidence specification. Coupled
Stackelberg and Ontario PPO results remain pending.

The archived GLRRM comparison exposes different annual forecasts and release
regimes. The largest original release gap is **2,100 m³/s** in 1934, where GLRRM
uses `M-`. Replacing the Python annual forecast with the archived values reduces
the largest gap to **2,010 m³/s**. The declared benchmark is the pinned Python
rule curve with Bv7 limits and explicit past-only inputs. The archival configuration
still needs further verification.

Saved local artifacts include:

- `results/loslr/native_baseline_v4/` — native reference, adapter parity and indicators.
- `results/loslr/causal_baseline_v2/` — causal full-record replay and forecast audit.
- `results/loslr/mpc_v2/` — initial paired MPC pilot.
- `results/migration/ontario_adapter_split_v1/` — latest replay through the shared
  search and Ontario planning adapter, matching all physical outcomes of the pilot.

The paired `comparison.csv` contains requests, executed releases, levels, forecast
inputs, every predicted stage margin and fallback status. `summary.json` aggregates
metrics and records the study settings and source hashes.

## Lake Como

In the improved Como simulator, history MPC has annual cost **517.51**, compared
with **617.91** for mapped PPO trained for 400k decisions per seed. PPO costs
approximately **19.4% more**, with slightly lower flood exposure and substantially
more low-water exposure. These are development results over **1988–1997**, using
training/fitting years **1946–1987**. Final **1998–2010** testing remains reserved.

The two-system package migration preserved a deterministic PPO update exactly,
loaded existing checkpoints and reproduced Ontario trajectories. The project
suite passed **319 tests with one skip**. The later MPC adapter split passed
**24 focused checks**.

<details>
<summary>Lake Como results, investigations and earlier DamEnv benchmarks</summary>

**MPC remains the strongest controller in the improved simulator.** Our latest
mapped PPO improves with extra training, but still costs about **19.4% more** and
spends much longer below the preferred low-water level. We stop the duration sweep
at **400k decisions per seed**; no 800k run is planned.

These are development results on **1988–1997 validation years**. Training/fitting
uses **1946–1987**; final **1998–2010** controller testing remains reserved.

## Current matched comparison

Both controllers choose daily release requests in the same protected hourly
reservoir model, with the same physical costs, constraints and allowed observed
history. Storage and history carry across each two-year episode. Five learner
seeds are retained for PPO; MPC averages five forecast seeds over the same years.
Those seeds represent different randomness, not independent hydrological records.

Annual averages, lower is better:

- **History MPC:** cost **517.51**; unmet model target **448.85 million m³**;
  **5.06 flood days**, **17.12 low-water days**.
- **Mapped PPO, 400k:** cost **617.91**; unmet model target **537.56 million m³**;
  **4.86 flood days**, **101.02 low-water days**.

PPO has slightly less flood exposure (**62.76 vs 67.28 m·hours**), but much more
low-water exposure (**17.59 vs 1.39 m·hours**) and unmet target. A flood-day count
alone misses this tradeoff. The cost combines unmet target in million m³ with
flood/low-water exposure in m·hours using the declared weights; it is not money.

MPC uses the model and causal seven-day forecasts, executing one release before
replanning. PPO learns a policy from simulated experience; its experimental
state-aware action map uses known model hydraulics to make exploration more useful.
This is a model-informed PPO variant, not an entirely model-free comparison.
The 400k endpoint is recorded without changing defaults or selecting a best seed.

## What we investigated

- **Simulator and data:** audited units, information timing, finite-task endings,
  year carryover, water balance and outlet hydraulics; added an explicit low-water
  reserve rule. The mechanisms are tested, but operational calibration remains open.
- **MPC forecasts and search:** investigated causal inflow/target forecasts, horizon,
  schedule search, carryover, solver failures, precision-safe requests and exact-cost
  tie handling. More elaborate forecasts did not consistently improve decisions:
  weighted-tail MPC reduced flood severity but worsened cost **517.51 → 531.79**,
  so it was not promoted. [MPC investigations](docs/SIMULATOR.md).
- **PPO reference and credit assignment:** fresh PPO cost **631.34** at 200k.
  Increasing GAE lambda from 0.95 to 0.99 gave **632.55**, with no meaningful fix
  for low-water behavior. The original setting was retained.
- **Physical action exploration:** different requests often produced identical
  releases. On fixed sampled training states, upper-release saturation fell from
  **97% to 49%** with the trained full-range hydraulic map. Annual cost improved
  **631.34 → 622.31**; most of the average gain came from seed 124.
- **Critic fitting and coverage:** frozen-actor full-return fitting improved storage
  sensitivity but worsened broader checking error. A matched broader-year refit
  brought checking states closer to fitting data, yet worsened MSE
  **0.288 → 0.294** and storage-slope error **0.272 → 0.502**. Neither refit was
  inserted into policy training. [Coverage study](docs/SIMULATOR.md#seed-124-and-matched-critic-coverage).
- **Seed variability and release timing:** seed 124 served the model target better
  while retaining slightly more water. Its advantage persisted from common states,
  but it did not have the most accurate critic. All five seeds remain in summaries;
  it is not promoted as a champion.
- **Stochastic deployment:** at the frozen 200k checkpoints, sampling increased cost
  **631.34 → 637.80** for original PPO and **622.31 → 632.15** for mapped PPO.
  All 300 two-year episodes completed. Deterministic execution remains the reference;
  this does not test removing exploration during training.
- **Longer mapped-PPO training:** all five 200k prefixes reproduced exactly before
  continuation to 400k. Cost fell **622.31 → 617.91 (0.71%)**, with four seeds
  improving and **33/50** paired annual wins. Unmet target and flood severity fell,
  but low-water days stayed near **101** and low-water severity increased **2.8%**.
  This is a modest improvement with a physical tradeoff, not a convergence result.

FQI, SAC and residual RL were also implemented/studied in the earlier DamEnv task.
Their results remain below; they have not been rerun as matched competitors in the
improved simulator and cannot be ranked against the numbers above.

## How we currently understand the problem

**Water has value through time.** Serving a target today can reduce supply later;
retaining water can help dry periods but reduce flood headroom. Our paired ledgers
and same-state branches show why release timing and inherited storage both matter.
Remaining river flow is not automatically waste, and stored volume is not electricity.

**The action distribution must be understood physically.** Large nominal noise can
produce almost no discharge variation under hydraulic limits. The mapping improves
this, but better exploration alone does not solve long-term storage management.

**A smaller value loss is not necessarily better control.** Storage sensitivity,
broad return accuracy and actor decisions can improve differently. The best policy
endpoint need not have the best critic on a diagnostic state set. Broader geometric
coverage and more training have not been reliable shortcuts to controller quality.

Low-water days mean some exposure below the model's preferred **−0.45 m** threshold;
the reserve floor is **−0.5 m**. These are relative model levels, not an empty lake
or a claim of real operational violations. PPO often operates close to that floor.

## Saved curves and decisions

**Yes, the 400k training history is saved:** each seed has `updates.csv` and
`training.json` under `results/simulator/mapped_ppo_duration_v1/`. There are
**196 rollout/update records and 547 completed two-year episodes per seed**, plus
a counted 690-decision unfinished episode. Only the 200k and 400k checkpoints were
evaluated; there is no intermediate frozen-policy validation curve.

The 200k–400k weather-adjusted training trends are small and heterogeneous:
about **−1.13% to +0.45% per 100k decisions** across seeds. The most recent 100k
is mostly flatter still. We inspect whole episodes inside each interval and report
an end-time convention sensitivity; changing exploration and optimizer noise remain.
These logs support diminishing returns, not proof that PPO cannot improve further.
[Saved 400k curve graphic — local](background/images/67_mapped_ppo_400k_curves.png).

**Decision:** preserve the 400k experimental endpoint, original controls and negative
findings; **do not extend to 800k**. PPO’s remaining gap involves storage-aware decisions, but its root learning
mechanism has not been isolated. Recorded updates are finite, and sampling does
not explain away the gap. Further
work should start from a specific control hypothesis, rather than another duration
or critic-data sweep. No new controller or default is promoted here.
[Duration result and inspection](docs/SIMULATOR.md#mapped-ppo-400k-result-and-curve-inspection).

## Interpretation limits

The benchmark target labeled `demand` closely tracks historical outlet flow; unmet
target is **not independently verified irrigation deficit**. Geometry, timing,
operational metadata and outlet behavior still need independent calibration.
[Evidence and limitations](docs/CALIBRATION.md).

Validation has been reused for development; years within episodes and repeated
seeds are dependent. The two-year resets and first-365-sample calendar remain
modeling assumptions. These are simulator results, not demonstrated superiority
over Lake Como operators or a universal ranking of RL versus MPC.

<details>
<summary>Earlier DamEnv benchmark — historical results (expand)</summary>


These results use different dynamics/rewards and cannot be compared numerically
with the improved simulator above.

- **PPO** learns a daily release policy from simulated experience. The current
  past-inflow variant observes level, season, averaged demand, and yesterday's
  reconstructed inflow. Five seeds each train for 1M steps.
- **MPC** simulates seven days of releases, applies the first, and replans tomorrow.
  It uses persistence inflow forecasts, seasonal demand, and cost-checked retention
  of bounded iteration-limit results.

**Plain controllers — annual averages**

- **PPO:** reward **−16,505**; **4.3 flood days**, **3.5 low-water days**;
  unmet demand **525.24 million m³**.
- **MPC:** reward **−26,361**; **12.3 flood days**, **10.4 low-water days**;
  unmet demand **565.60 million m³**.

**With the same flood-scenario filter — annual averages**

- **PPO:** reward **−16,505**; **4.3 flood days**, **3.5 low-water days**;
  unmet demand **525.20 million m³**. Only eight interventions across five seeds.
- **MPC:** reward **−18,486**; **4.5 flood days**, **10.7 low-water days**;
  unmet demand **556.71 million m³**. It intervenes on 5.7% of decisions.

**First FQI pilot — not yet a settled benchmark**

Three forest seeds share 30,660 training transitions and a 21-action grid:

- **Plain:** reward **−20,062**; **5.43 flood days**, **2.77 low-water days**;
  unmet demand **625.90 million m³**.
- **Shared filter:** reward **−19,219**; **4.70 flood days**, **2.77 low-water days**;
  unmet demand **620.73 million m³**.

FQI has fewer low-water days but much more unmet demand. Its data budget, action
grid, and regression-seed design differ from PPO/MPC; this pilot does not establish
convergence, general method ranking, or comparative sample efficiency.

**First SAC study — 200k steps per seed**

SAC learns a continuous release policy using replay, twin critics, and a stochastic
actor. Five seeds use the same five observations, reward, physics, and splits:

- **Plain:** reward **−21,064** (seed SD **759**); **4.84 flood days**,
  **8.90 low-water days**; unmet demand **623.12 million m³**.
- **Shared filter:** reward **−20,871**; **4.70 flood days**, **8.90 low-water days**;
  unmet demand **620.34 million m³**.

SAC trails PPO on these validation metrics and leaves more demand unmet than MPC.
It beats plain MPC on reward, but trails filtered MPC. PPO used **1M steps per seed**
and smaller networks; this first SAC configuration is not a matched-budget method
ranking. No reference was replaced. See [SAC study](docs/SAC.md#first-validation-results).

**MPC + residual RL — first three-seed pilot**

A frozen MPC proposes each release; SAC learns a correction bounded to ±50 m³/s.
Each seed trains for only 3,650 steps, including 365 MPC-only warmup steps:

- **Plain:** reward **−17,310** (seed SD **968**); **4.70 flood days**, **5.80
  low-water days**; unmet demand **533.42 million m³**.
- **Shared filter:** reward **−16,924**; **4.40 flood days**, **5.87 low-water days**;
  unmet demand **528.11 million m³**.

Both tracks improve the MPC prototype on reported mean metrics. A matched full-action
actor receiving the same MPC proposal scores **−26,973 plain**, with **43.03 low-water
days** but less unmet demand (**499.13 million m³**). This is a short implementation
pilot, not convergence or a budget-matched comparison with old PPO/SAC. Both learned
pilot arms pay for daily MPC planning; no reference is replaced.
[Schematic and results](docs/RESIDUAL_RL.md).

Higher reward is better; lower physical penalties are better. The filter helps
MPC's event count much more than severity: flood exceedance falls only **2.4%**,
while low-water severity worsens slightly. Remaining floods and forecast-range
breaches mean this is **not a hard safety guarantee**. Keep the two tracks separate.

The original four-feature PPO scored **−17,232**, with seed SD **1,823**; past inflow
scores **−16,505**, with SD **238**. Three seeds improve and two worsen; most mean
gain comes from the previously weakest seed, while unmet demand rises about 2%.

These are tuned validation results, not proof of general or operator-level
superiority. Episodes reset annually and retain the 365-step leap-year approximation.
PPO and SAC figures average five seeds and ten years; residual results average three
seeds and ten years. MPC is deterministic over ten years.
PPO/MPC references and both tracks are frozen while FQI remains exploratory; existing
simulator/default configuration choices remain unchanged.

Details: [PPO observation study](docs/PPO.md#past-inflow-observation-ablation),
[common filter experiment](docs/EVALUATION.md#common-finite-scenario-safety-filter),
[FQI pilot](docs/EVALUATION.md#first-tree-based-fqi-pilot),
[frozen protocol](configs/experiments/control_protocol_v1.json).

**Duration follow-up — 36,500 steps per seed, same settings**

The longer residual controller **deteriorates in all three seeds**:

- **Residual plain:** reward **−21,799**; **4.87 flood days**, **21.67 low-water
  days**; unmet demand **549.93 million m³**.
- **Residual shared filter:** reward **−21,266**; **4.43 flood days**, **21.77
  low-water days**; unmet demand **543.55 million m³**.
- **Matched full-action plain:** reward **−21,353**; **4.27 flood days**, **18.80
  low-water days**; unmet demand **548.79 million m³**. Its reward improves in all
  three seeds; the filtered reward is **−21,343**.

Only training duration changed; all six short prefixes reproduce exactly. The
longer residual endpoint is not promoted. Its early advantage over full-action
control does not persist. Original reference results and test years remain intact.
[Duration study](docs/RESIDUAL_RL.md#duration-only-extension).

The additional **frozen-actor SAC critic diagnostic** improves absolute return
fitting but still fails to learn reliable release differences; it changes no
controller scores. [Diagnostic findings](docs/SAC.md#action-diverse-rollout-supervision).

**Residual regression diagnosis:** saved trajectories attribute about **88%** of
the longer run's reward loss to low-water penalties. It releases more water but
adds **32.03 million m³ surplus**, lowers terminal storage **15.52 million m³**,
and leaves **16.51 million m³ more demand unmet** annually. The later actor makes
larger positive corrections even on the same saved states. This identifies changed
behavior, not the causal training failure. [Details](docs/RESIDUAL_RL.md#diagnosing-the-duration-regression).

</details>

</details>
