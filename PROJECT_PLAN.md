# Reservoir control project plan

Lake Ontario–St. Lawrence is the main research case. We compare the Plan 2014
rule curve, MPC and Stackelberg MPC with Lake Ontario as leader and an aggregated
Ottawa reservoir operator as follower. PPO provides an additional learned
comparison. Lake Como remains the completed reference case for our controller
implementations and simulator investigations.

## Delivery plan for the coupled comparison

The current evidence supports two conclusions. A wider two-path search can
recover missed schedules in closed loop, and its overall advantage over
isolated-jump MPC is small on the synthetic holdouts so far. The sustained-rise
audit found modeled continuations at all 16 stopped states. Substituting
realized Ottawa forecasts reopened seven decisions under the same search and
reserve, which leaves nine cases for one final planning diagnostic. The next
milestone is a single comparable controller study.

### 1. Freeze the MPC variants after one targeted diagnostic

- [x] Budget half a day for the nine frozen states. Check the first five
  periods of the physical witness against the planner, then test the reserve
  and fixed fifth Ontario commitment separately and together. Repeat the
  search under the same settings and save a rejection reason for each case.
- [x] Allow one implementation and closed-loop validation pass for the
  strongest supported change. If the fifth commitment is binding, use a
  declared causal fifth Ontario forecast and give the same fifth-lead policy
  to both MPC arms. If the diagnostic is inconclusive or the change trades
  completions for realized violations, freeze the existing reference and
  report the failed cases.
- [x] Select one external-ranked MPC and one response-aware MPC for the main
  comparison. Lock forecasts, horizons, candidate eligibility, reserve,
  fallback rules, and the evaluation contract before opening reserved years.
  Keep the two-path controller as a separate sensitivity result unless it
  earns a place in the main comparison under the same checks.

### 2. Run a bounded PPO comparison alongside MPC

The existing five-seed Ontario PPO pilot ran for 4,096 decisions per seed in
the earlier responsive-follower simulator. That experiment established the
training pipeline. A stronger comparison needs the same coupled physics,
Ottawa response, and outcomes as the final MPC study.

- [x] Verify the Gymnasium adapter uses the final paired execution contract.
  Give PPO the same decision-time information and native release projection.
  Define and log rejected requests, hard-limit violations, and incomplete
  episodes explicitly. If a common feasibility filter mediates actions, apply
  it consistently and report its interventions separately from PPO's requests.
- [x] Smoke-test one training seed, then run one declared budget across the
  five existing seeds on training years only. Freeze the budget after the
  smoke test and before evaluating development results. Retain the final
  checkpoint and the learning curve for each seed. Keep the existing
  chronological split and shared operating cost.
- [x] Evaluate the final checkpoints on the development cases. Record whether
  PPO finishes runs that MPC stops, and whether that comes with changes in
  upstream or downstream exposure, Ottawa storage, or native projection
  frequency. Use this as the single PPO comparison for the project.

### 3. Run the paired evaluation and write the result

- [x] Freeze a common case manifest covering the original chronological
  protocol, the 2017 and 2019 spring development cases, and the separately
  labeled synthetic Ottawa uncertainty cases. Include Plan 2014 as the
  operational reference, external-ranked MPC, response-aware MPC, and PPO.
  If a run ends early, report the stop and compare horizon costs only on
  pairs that complete their full windows.
- [x] Report completion and physical violations first, then Ontario shoreline
  and Lake St. Louis flooding, Ottawa storage and service, energy, and solver
  time. Preserve water in transit at each endpoint. Show per-seed PPO
  variability and the upstream/downstream trade-offs in 2017 and 2019.
- [x] Update [RESULTS.md](RESULTS.md) and the visual replay with a common
  comparison. State which gains come from the follower response and which
  come from search or feasibility safeguards. Document the synthetic Ottawa
  assumptions and keep physical calibration as a separate workstream.

Work toward one results package in roughly five focused work blocks. Spend the
first on the frozen diagnostic and the evaluation contract, the next two on
the single MPC revision and PPO pilot/training setup, the fourth on paired
rollouts, and the last on the written results and replay. PPO training can run
while the frozen MPC comparison is evaluated on development cases.

The delivery study is complete. The diagnostic attributed most of the
nine rejections to the reserve on the tested forecasts. The reserve-fallback
sensitivity completed 18/36 versus 7/36 for the guarded wider two-path arm,
with no recorded physical violations. It remains a separate sensitivity.
The main comparison used the common hard follower and fixed fifth lead.
All four approaches completed the same 15 of 18 reserved annual cases.
PPO did not open any of the common stops, and Stackelberg and external
ranking gave the same operating cost. The common filter stopped all
synthetic stress runs. See [RESULTS.md](RESULTS.md) for the scored
comparison and replay.

### Shared recovery follow-up

- [x] Audit the stopped flood and stress decisions, then compare reserve-only,
  wider-profile, and proactive recovery through all four controllers.
  Wider recovery completes 2017 and 2019 with zero recorded violations.
- [x] Retrain five PPO seeds under the filtered profiles contract for the
  same 16,384-decision budget. The new checkpoints do not beat external MPC
  on the two flood years or average operating cost on annual replays.
- [x] Test the causal peak storage check on new hydrographs and noise seeds.
  Confirmation stress improves external MPC completion from 4/36 to 15/36
  and removes realized storage violations. Its operating cost is lower
  than filtered Plan 2014 on all 14 joint completions. External and
  Stackelberg costs match. Use external-ranked MPC as the research
  reference and keep the peak-protected recovery opt-in.
  See `results/loslr/final_shared_recovery_report_verified/report.md`.
- [x] Check whether anticipating Ottawa's response changes Ontario's schedule
  across three storage scales and three routing delays. The selected
  schedule agrees in all 150 eligible frozen comparisons. Keep external-ranked
  MPC with peak protection as the research reference and report the complete
  2017 and 2019 replays. See [sensitivity results](RESULTS.md#ottawa-storage-and-routing-sensitivity).
- [x] Screen downstream-sensitive states across the flood years, annual replays
  and synthetic confirmation windows. Anticipation changes predicted
  Pointe-Claire levels while preserving zero forecast exceedance and the
  same winning schedule. Package the final four-controller comparison.
  See [downstream diagnostic](RESULTS.md#anticipating-ottawa-near-the-downstream-limit).

### Follow-up controller experiments

The peak-protected four-controller comparison is the reference snapshot.
Use a new run directory and keep 2017, 2019 and the 18 reported annual
replays as fixed reference cases.

- [x] Assess observed Ottawa flows and routing. HYDAT covers the older
  Britannia–Carillon benchmark and supplies stage checks in both flood
  years. SLON provides the existing quarter-month aggregate comparator.
- [x] Benchmark daily Carillon persistence, Carillon history, and
  history plus Britannia on 1992–1994. Carillon history lowers average MAE
  across all tested leads. Britannia has mixed effects, including higher
  one-day MAE at high flow.
- [x] Benchmark quarter-month SLON at one to four leads, with the latest
  completed period available or delayed by one period. Seasonal history
  improves MAE across both latency cases. High-flow underprediction persists.
- [x] Use the 1987–1993 historical overlap for a downstream stage timing
  check. Same-day Carillon flow improves stage agreement with Pointe-Claire,
  while tested lag variants give no timing improvement that transfers.
  Physical routing and absolute flood-stage validation remain open.
- [x] Adopt the research rule for replacing the reference. Require lower
  total Lake St. Louis exposure, zero added Ontario exposure in every
  matched year, preserved completion, and no added physical violations.
  Report positive upstream trade-offs as sensitivities. The archived
  half-margin candidate fails the agreed rule at both data delays.
- [x] Test earlier release timing against standalone persistence MPC.
  Hindsight finds feasible improvements in 25 of 28 development cases
  with downstream exceedance. The causal forecast and four-period
  search recover no exposure benefit. Identify a stronger early warning
  signal on independent data before reopening this strategy.
- [x] Test an earlier high-SLON onset warning using observed Britannia
  flow. The gauge adds no onset recall over SLON history on the reserved
  years. Only three of 12 reachable hindsight shifts coincide with onset.
- [x] Forecast Pointe-Claire headroom under the issued Ontario plan and
  score local opportunities on the reserved years. The fitted alert
  reduces false alarms but misses more opportunities than the seasonal
  alert. Keep the existing search.
- [x] Obtain downstream flow and stage observations and compare stage
  under observed flows. The 2021–2025 benchmark adds 197 complete
  quarter-months, including three above the study threshold.
- [x] Check daily reach travel and the Prairies branch proxy using
  observed discharge. Development favors same-day inputs and the
  paired branch adjustment adds no benefit. Held-out near-threshold
  stage errors remain large enough to affect release decisions.
  Keep the frozen controller comparison for the handoff. Reopen
  routing calibration when matched Carillon and outlet flows or a
  validated hydraulic routing model are available.
- [x] Run an offline Ontario replay for persistence versus seasonal
  history using historical aggregate SLON as the realized forcing.
  Hold supply forecasts, constraints, costs, and candidate search fixed.
  All 160 runs complete. Seasonal history worsens aggregate downstream
  exposure in every cohort under both availability settings.
- [x] Compare the two SLON forecast paths at the same inherited states
  where realized downstream exposure rose. Identify which lead and
  limit influenced the first action. The immediate forecast swap explains
  5.565 of 6.390 metre-periods of gross increase across the selected decisions.
- [x] Calibrate a one-sided lead-one SLON forecast bound by latency on
  training data. Screen it over full development years with the same MPC
  settings. The half-margin candidate passed development and reduced
  reserved-year downstream exposure, but exceeded the upstream allowance
  at both delays. Keep persistence as the reference and close this
  forecast-margin search on the current split.
- [ ] Test one MPC change at a time. First compare causal supply forecasts
  at the existing horizons with objective and constraints fixed. Extend
  planning only once usable forecasts for the additional leads are
  available. Declare the evaluation windows and report completion, both
  flood exposures, operating cost and computation time on matched cases.
- [x] Audit archived PPO interventions and probe fixed flood-year states.
  Nine distinct requests selected the same first release at 29/90 states.
- [x] Enumerate fixed-state PPO menus on development trajectories. Only
  35 of 900 sampled states had one eligible first release. Direct menu
  selection adds a reachable choice at 10 states, so a new training run
  based on action availability alone is low priority.
- [x] Measure PPO probability across reachable releases. The most probable
  executed release differs from the deterministic choice at 379 of 860
  sampled development states with multiple reachable releases.
- [x] Compare deterministic PPO with the most probable executed release
  using the same checkpoints. Modal execution raises cost and leaves
  reported flood exposure unchanged, so keep deterministic execution.
- [x] Compare 16,384 and 32,768 PPO decisions under the same peak-protected
  training contract using five seeds. Mean cost rises on the 19 common
  complete development years, four seeds worsen, and one later rollout
  stops. Retain the frozen PPO comparison and close the duration sweep.

See [Ottawa and PPO audit results](RESULTS.md#observed-ottawa-benchmark-and-ppo-filter-audit).

## Current status

- [x] Diagnose the H=4 delayed follower failure using independent full-experiment
  and prefix feasibility LPs. The original state is feasible; loss follows the
  second action. H=5/6/8/12 complete with original constraints and weights.
  Record the short-window arrival-boundary effect and terminal transit before
  considering controller changes (`delay_diagnostic_verified_v2`).

- [x] Step 2: implement a standalone, hard-constrained Ottawa follower for an
  executable Ontario schedule; test pure QP solving, first-action-only execution,
  conservation, infeasibility, routing queues and noisy forecast departures.
- [x] Run six synthetic cases with perfect forecasts and three noisy seeds,
  preserving the historical datasets and earlier Ontario studies.
- [ ] Investigate recursive feasibility and uncertainty handling before coupling
  this new follower to any leader. The tight delayed case fails even with perfect
  forecasts; these demos do not establish operational flood protection.

- [x] Separate reusable MPC search from the Ontario planning adapter. Shared PPO,
  SAC and FQI code also lives under `controllers/`.
- [x] Reproduce the pinned native Python Ontario simulation over 1900–2020 and
  verify adapter parity for all 5,808 decisions. Run all seven native performance
  indicators.
- [x] Audit differences from the archived GLRRM results, including the annual
  forecast and release-regime differences. Retain the Python implementation as
  the declared research benchmark.
- [x] Build a past-only rule-curve replay with 5,760 scored decisions in 1901–2020.
- [x] Implement four-quarter-month MPC and run paired 1995, 2017 and 2019 cases.
- [x] Add one controller-independent Ontario flood-exposure scorer with explicit
  weights, complete-year checks and per-site physical metrics.
- [x] Specify the Ottawa aggregate model and test its storage balance.
- [x] Add the Ontario HTML experiment playback and dedicated MPC/PPO diagrams,
  with the Ontario PPO application labelled as planned.
- [ ] Validate operational stage limits, model datum and the effective time
  convention needed to couple Ottawa storage to the Ontario model.
- [x] Complete the fallback, perfect-inflow and objective diagnostics on the three pilot cases.
- [x] Implement a convex responsive Ottawa follower and nested Stackelberg MPC
  reference in a declared synthetic coupling; validate KKT residuals and response derivatives.
- [x] Add the Ontario Gymnasium adapter and complete a five-seed, 4,096-decision
  PPO pipeline pilot with a frozen chronological protocol.
- [ ] Calibrate the Ottawa decomposition/routing, expand the leader search and
  validate longer PPO training before a final reserved-year comparison.

The MPC pilot matches the rule curve in 1995 and 2019. In 2017 it changes ten
releases, with a small downstream severity reduction and an upstream severity
increase. The finite search finds no schedule meeting both forecast caps on
20 decisions in 2017 and 17 in 2019. Current numbers and their interpretation
are recorded in [RESULTS.md](RESULTS.md).

## Next coupled replay

- [x] Use a fixed Ontario release schedule for the first replay. Ottawa plans
  against committed total discharge in m³/s. Check each commitment against
  Ontario's operating limits using the coupled downstream flow. If a limit
  would change the release, record the rejected commitment and stop the replay.
- [x] Use five quarter-months as Ottawa's study look-ahead. Ottawa reads up to
  five releases from the fixed schedule and shortens its window at the replay
  endpoint. The current Ontario runner generates that schedule with its four
  available supply forecast leads.
- [x] Run the 12-period fixed-schedule verification and inspect the first
  conflict. With the saved Ontario H4 schedule and the synthetic delayed Ottawa
  case, period 1 executes and balances. Before period 2 the native F limit
  reduces the permitted Ontario flow from 6,930 to 6,220 m³/s. The replay stops
  with both period-2 actions unexecuted and the period-1 Ottawa release still in
  transit. See `results/loslr/fixed_ontario_ottawa_h5_verified/`.
- [x] Establish a fixed schedule under the coupled downstream conditions.
  Offline repair changed seven releases. A fresh replay using only the saved
  schedule completed all 12 periods with identical traces. This is a feasibility
  benchmark with perfect Ottawa forecasts. See
  `results/loslr/feasible_fixed_schedule_h5_checked/`.
- [x] Compare Ontario MPC with an external Ottawa forecast against Ontario MPC
  that solves the Ottawa response for each candidate. Both use Ontario H4,
  Ottawa H5 and the same fifth commitment from the verified schedule. The shared
  grid produced the same executed releases for two periods. Both planners
  exhausted the candidate set before period 3 because of projected native
  limits. See `results/loslr/ottawa_response_comparison_verified/`.
- [x] Diagnose and rerun the period-3 forecast rejection. Substituting only
  period-4 Pointe-Claire roughness (1.229 in the replay versus 1.294 carried
  forward in the forecast) admits 6,110 m³/s for both arms. The shared native
  limit candidate generator extends the external arm to 12/12 periods. The
  anticipating arm stops after two periods, with 38 candidates rejected by the
  forecast F limit. See `results/loslr/period3_forecast_diagnostic/` and
  `results/loslr/ottawa_response_native_limits_verified/`.
- [x] Align Ottawa's forecast cap with Ontario's F limit for the first four
  leads. Switching on alignment at period 3 preserved the prior state and
  admitted 30 anticipating candidates. Both policies completed the staged
  replay. With alignment from period 1, anticipating completed 12 periods and
  the external arm stopped after four when its chosen schedule made Ottawa's
  capped solve infeasible. See `results/loslr/ottawa_f_alignment_{from3,full}/`.
- [x] Test a causal four-observation roughness mean for future leads. Its
  mean absolute error was 0.0833 versus 0.0652 for persistence across the 30
  available forecast and observation pairs. Completion with alignment stayed
  12/12 for anticipating and 4/12 for external.
- [x] Screen the externally ranked schedules for Ottawa feasibility.
  The screened and anticipating arms both completed 12/12 periods. At period
  5, screening skipped the top three schedules and selected the fourth.
  Screening used 15 Ottawa candidate solves versus 393 for anticipation.
  See `results/loslr/ottawa_external_screening/`.
- [x] Stress both policies with the same noisy inflow forecasts and two
  downstream-capacity settings. Both complete only seed 33 in each setting.
  Native-limit-checked offline witnesses show a feasible continuation at
  all eight stops, including the tight seed-11 pair when the first Ontario
  commitment is lowered. See `results/loslr/ottawa_noisy_stress_verified/`.
- [x] Test recovery at the failed decision. The wider candidate set alone
  completed 2/6 cases, matching the screened baseline. Retrying with causal
  inflow persistence completed 5/6. The tight seed-22 run exceeded the
  Ottawa storage limit at period 6, after returning to the original forecast
  bank. See `results/loslr/ottawa_noisy_recovery_verified/`.
- [x] Check storage headroom during recovery for one or two periods of
  +700 m³/s inflow. The one-period check completes 6/6 cases without
  realized limit violations. The two-period check completes 5/6 and rejects
  the tighter seed-11 case at its original stop. See
  `results/loslr/ottawa_retry_storage_guard/`.
- [x] Freeze the one-period +700 m³/s check and validate against seeds
  101–106 in the original Ontario window and the April–June 2017 and 2019
  windows. With the same two synthetic Ottawa scenarios, unguarded retry
  completes 34/36 cases and guarded retry completes 32/36. Both exceed
  Ottawa storage in the same two spring cases. The guard adds two stops
  with feasible offline continuations. See
  `results/loslr/ottawa_guard_holdout_{1961,2017,2019}/`.
- [x] Apply the +700 m³/s check to the ordinary decision after a retry,
  and compare with checking every decision. On the same evaluation cases,
  those variants complete 32/36 and 33/36 respectively, with zero storage
  violations. The targeted check completes 2017 seed 101 and stops 2019
  seed 101 before its period-6 overflow. See
  `results/loslr/ottawa_replan_guard_{1961,2017,2019}/`.
- [x] Diagnose the 2019 seed-101 stop from its inherited state. The original
  candidates allow at most 4.271 billion m³ of Ottawa release over the next
  three periods, below the 4.409 billion m³ needed under the storage check.
  An opt-in search for lower Ontario commitments in the arrival period
  restores feasibility and completes all 12 periods in this replay. The
  replay pairs historical Ontario inputs with synthetic Ottawa conditions.
- [x] Evaluate the targeted arrival-period search on the frozen multi-seed
  matrix. Completion improves from 32/36 to 34/36 with no realized storage
  or downstream violations beyond tolerance. The search recovers seed 101 in
  both 1961 and 2019 and is ineligible at the two remaining stops, which
  have no preceding retry. The default stays as-is.
  See `results/loslr/ottawa_delay_headroom_holdout_verified/report.md`.
- [x] Diagnose the two period-6 stops without a preceding retry and test a
  broader trigger. At the frozen state, the wider candidate set needs guarded
  persistence in 1961 seed 102 and works with the original forecast in 2019
  seed 103. The opt-in trigger completes both cases. For fresh seeds 107–112,
  it completes 36/36 versus 32/36 for the after-retry trigger, with zero
  realized storage or downstream violations beyond tolerance. This varies
  forecast noise on the same synthetic Ottawa scenarios and Ontario windows.
  See `results/loslr/ottawa_remaining_stops_verified/report.md` and
  `results/loslr/ottawa_broader_headroom_validation_v1/summary.csv`.
- [x] Compare finite-search Stackelberg MPC with the improved external-ranked
  baseline using the same candidate eligibility and recovery rules. The added
  native check on the baseline reproduced all 36 archived trajectories. Both
  arms completed 36/36, with identical schedules and realized outcomes.
  Scoring the routed best response changed at least one eligible score at
  384 of 432 decisions, while the top schedule stayed the same at every
  decision. Anticipation required 23,329 Ottawa candidate solves versus
  11,034 for external ranking. This comparison holds candidate eligibility
  fixed and isolates ranking on the tested synthetic Ottawa scenarios.
  See `results/loslr/ottawa_response_ranking_v1/report.md`.
- [x] Compare generic and Ottawa-guided local candidate expansions at 12
  proposals per feasible decision. All three arms complete 6/6 saved spring
  cases and 33/36 fresh cases. On the 33 fresh complete pairs, the guided
  arm improves the existing scalar Ontario objective over the generic arm
  in 10 cases and worsens it in 21, with two ties. It improves Ottawa
  storage tracking in 25 cases. All three arms exceed Ottawa storage in
  the same three period-6 seed-115 cases. Keep the expansions opt-in.
  See `results/loslr/ottawa_candidate_proposals_{saved,fresh}_verified/report.md`.
- [x] Diagnose the period-6 seed-115 overflows and compare storage guard
  triggers on the same 36 cases. The selected Ontario schedules cap Ottawa's
  first release at 1,500 m³/s, about 113 m³/s short. Offline schedules that
  free capacity at the next downstream arrival complete all three windows.
  Guarding period 5 alone leaves the period-6 storage and overflows unchanged.
  Always-on guarding completes 36/36. Tight-headroom guarding completes
  35/36. Both avoid the three overflows. Keep the variants opt-in while
  testing on different hydrology and forecast seeds.
  See `results/loslr/ottawa_storage_guard_seed115_final/report.md`.
- [x] Validate the frozen +700 m³/s guard on three new Ottawa inflow shapes
  and forecast seeds 119–124. Always-on guarding prevents two sustained-wet
  overflows and passes all dry cases, but adds three stops in the sharp-peak
  cases. All guarded stops have feasible offline continuations. Completion
  is 39/54 for always-on and 40/54 for after-retry. Keep after-retry as the
  common comparison baseline and always-on as a sensitivity case.
  See `results/loslr/ottawa_storage_guard_hydrology_v1/report.md`.
- [x] Separate the forecast tail from the reserve at the three extra
  sharp-peak stops. Removing the reserve from the saved forecast admits
  1, 55 and 10 schedules. Leaving the reserve in place and replacing only
  Ottawa leads 2–5 with realized inflows offline admits 921, 572 and 311.
  The saved tails overstate four-period inflow by 3.09–4.20 billion m³.
  Persistence without the reserve admits at most one schedule in each case.
  Use these cases to motivate a causal receding-inflow forecast experiment.
  See `results/loslr/ottawa_sharp_peak_diagnostic_v1/report.md`.
- [x] Test a causal recession forecast with the always-on +700 m³/s guard.
  Preserve lead one and the noise draws, then decay positive departures
  from recent inflow with a one-period half-life. Sharp-peak completion
  improves from 3/18 to 10/18 on archived seeds and from 0/12 to 6/12
  on fresh seeds 125–128. All three diagnosed stops complete. Both
  forecasts complete every sustained-wet and dry case, with zero realized
  limit violations across the comparisons. Recession increases the negative
  forecast bias during sustained wet spells. Keep forecast defaults fixed.
  See `results/loslr/ottawa_recession_{archived,fresh}_v1/report.md`.
- [x] Compare persistence, recession, and recession after an isolated jump
  under the same always-on +700 m³/s guard. On seeds 129–132 the three rules
  complete 27/48, 30/48, and 31/48 respectively, with zero realized limit
  violations. Isolated-jump recession retains the six sharp-peak completions
  and recovers the one sustained-rise case completed by persistence. All
  three rules still struggle with sustained rises even though offline
  schedules exist from each initial state. Keep forecast defaults fixed.
  See `results/loslr/ottawa_isolated_jump_v1/report.md`.
- [x] Test a two-path controller with shared first Ontario and Ottawa
  releases and branch-specific suffixes. Offline checks find a common
  first action in all 12 archived states. With the always-on guard, the
  causal planner completes 10/36 new cases versus 16/36 for isolated-jump
  recession. It completes one additional sustained-rise case in aggregate
  and seven fewer pulse cases. No realized limits are exceeded. Offline
  checks find a common action at 11 of 12 held-out pulse branch stops.
  Keep the two-path planner experimental.
  See `results/loslr/ottawa_two_path_recourse_v1/report.md`.
- [x] Compare current and wider Ontario schedule searches with and without
  the reserve at all 12 held-out pulse stops. The four cells find 0/12
  (current, with reserve), 4/12 (current, without), 6/12 (wider, with),
  and 8/12 (wider, without) shared actions. The sustained forecast has
  no feasible first-release interval among the wider candidates in the
  remaining four stops. The wider search is eligible for a fresh online
  evaluation with the reserve intact.
  See `results/loslr/ottawa_two_path_factorial_v1/report.md`.
- [x] Evaluate wider two-path search in closed loop with the reserve intact.
  All six frozen pulse recoveries complete their windows. On the archived
  cases, isolated-jump, original two-path, and wider two-path MPC complete
  16/36, 10/36, and 16/36. On fresh pulse heights, durations, and noise seeds
  they complete 6/36, 0/36, and 7/36. No realized storage or downstream
  limits are exceeded. Relative to isolated-jump, the wider search adds
  three fresh completions and loses two, so it remains experimental.
  See `results/loslr/ottawa_two_path_wider_{archived,fresh}_v1/report.md`.
- [x] Check every executed prefix of the 18 fresh sustained-rise cases.
  All have a modeled continuation from the initial state, and all 16
  stopped runs still have a native-limit-checked continuation from their
  stopped state. With the reserve and search fixed, substituting realized
  Ottawa inflows at the stopped decision opens 7 of 16 decisions offline.
  Nine remain blocked despite full-window physical witnesses. The study
  does not isolate the planning reserve, fixed fifth lead, or finite
  candidate restrictions in those nine.
  See `results/loslr/ottawa_plateau_recoverability_verified/report.md`.
- [ ] Before comparing coupled planners at a common five-period horizon,
  specify and validate a causal fifth Ontario forecast lead.

## Architecture and data

```text
src/reservoir_control/
  controllers/                   shared MPC search, PPO, SAC and FQI
  core/                          time, state and information contracts
  adapters/                      shared planning and Gymnasium interfaces
  evaluation/                    rollout and provenance checks
  systems/
    lake_como/                   Como physics, observations and studies
    loslr/                       Ontario–St. Lawrence physics and studies
      planning.py                Ontario MPC planning adapter
    ottawa/                      aggregate storage model
```

The **Adapter pattern** connects the physical systems to their application
interfaces. MPC uses forecast-driven planning through a system adapter. PPO uses
observations, actions, rewards and episode boundaries through Gymnasium. The
application runner composes the system and selected controller.

The Ontario adapter supplies the 48-period calendar, candidate releases, Bv7
limits, stage calculations and objectives to
`controllers/mpc/receding_horizon.py`. MPC and PPO have their own implementations
and execution lifecycles.

Como data lives in `external/gym4real/gym4real/data/dam/`. Ontario historical
inputs live in `external/loslr_regulation_optimization/input/historic/1900_2020/`.
The current reference input is `12month_sqAR.txt`. The Ottawa model still requires
an identified aggregate boundary, storage parameters and suitable forcing data.
Historical SLON already contains regulated tributary contributions, so coupling
must account for those contributions before adding an optimized follower release.

## Approximate Baskatong physical scenario milestone

- [x] Implement a separate linear storage/level scenario mapping with three
  documented positive area scales; retain strict reconstruction unchanged.
- [x] Apply explicit daily-center timing, adjacent-discharge averaging and
  source-flag propagation to the preserved 1970 records, including context.
- [x] Reconstruct fixed effective-input interpretations separately by area and
  discharge proxy; retain negative values and unflagged-only diagnostics.
- [x] Run historical-proxy accounting replay and prescribed release retention/
  repayment; report remaining volume, conservation errors and unclipped bounds violations.
- [x] Save all six case interpretations, twelve trajectories and sensitivity PNGs
  under `results/ottawa/baskatong_linear_scenario_final_v2/`.
- [ ] Resolve actual storage mapping, physical timing and complete outlet boundary
  before interpreting the scenario inputs or constraints operationally.

This is an offline research experiment, not an optimized controller or calibrated
Ottawa model. Reproduce with `python -m reservoir_control.systems.ottawa.baskatong_scenario`;
see [the guide](configs/studies/ottawa/README.md).

## Actual one-year Ottawa observation milestone

- [x] Acquire official ECCC Baskatong/Mercier daily observations for 1970 and
  retain raw station/collection metadata, permissions, source symbols and hashes.
- [x] Process 365 paired dates; distinguish the 28 Partial Day discharge values
  from missing days and preserve the separate reservoir-station discharge.
- [x] Produce observed level/discharge, normalized water-level-position and
  coverage plots using the existing pipeline. Verify offline source replay.
- [x] Compare both 1970 discharge records and source flags; retain 02LG006 as a
  labelled downstream proxy, without claiming total-reservoir outlet completeness.
- [ ] Resolve midnight boundary levels, historical daily clock, all-outlet
  definition and a storage–elevation mapping before estimating volumetric inflow.

Run: `uv run python -m reservoir_control.systems.ottawa.historical_pilot --output results/ottawa/baskatong_1970_review`.
See [the pilot guide](configs/studies/ottawa/README.md). This is an actual
observation milestone; it does not calibrate the synthetic Ottawa controller.

## Baskatong measurement milestone

- [x] Assess the reservoir boundary, inventory volume, operating elevations,
  owner access routes and actual ECCC archive coverage for the 2016–2019 target.
- [x] Implement a permission-aware daily ingestion and net-balance interface,
  with explicit timestamps, quality checks and separate storage modes.
- [x] Save the assessment and input templates under
  `results/ottawa/baskatong_assessment_v2/`; no historical observations were fabricated.
- [ ] Obtain owner-permitted boundary-instantaneous levels and matching total
  outlet means, including the 2020-01-01 closing level. Resolve datum/clock and
  Mercier/Castor/spill completeness, then source the storage–elevation mapping.
- [ ] Run and inspect the actual reconstruction before using it in a follower.

See [the data milestone guide](configs/studies/ottawa/README.md). The synthetic
coupled parameters and existing Ontario experiments remain unchanged.

## Coupled study implementation

The five requested implementation areas now have an executable research pilot:
threshold/timing audit, explicit common operating cost and fallback priority,
responsive Ottawa follower, nested bilevel MPC and Ontario PPO. See the
[protocol and source evidence](configs/studies/loslr/README.md) and
`configs/studies/loslr/coupled_v1.json`. The run is saved in
`results/loslr/coupled_v1_information_matched/` with 24 annual controller/seed records.

The F-limit targets are tiered by beginning Ontario level. The 75.50 m Ontario
cap remains a research choice, not a verified universal operational limit. The
Ottawa capacity, initial storage, historical-SLON split and zero-lag routing are
explicit assumptions. Shared Bv7 projection applies to all controllers; MPC
searches for additional forecast stage-cap feasibility, whereas PPO optimizes the
same stage-exposure cost without a hard stage-safety certificate. A nested QP
reference and independently checked KKT residuals replace the need to embed KKT
conditions in a first leader solver. Local active-set sensitivity matches finite
differences. The finite leader candidate family is not globally optimal.

Training: 1901–1980; development: 1981–2000 plus 2017/2019. Other 2001–2020 years
are reserved. Five PPO seeds completed 4,096 decisions each, with final-only
pilot evaluation, finite gradients and explicit annual terminal storage costs.
This is a training-pipeline check, not a converged PPO benchmark.

## Remaining validation and extensions

### 1. Establish the Ontario MPC comparison

The current controller plans four quarter-month releases and executes the first
before replanning. Ontario supply and SLON forecasts persist the last completed
period. The annual supply estimate uses the trailing 48 periods. Both controllers
use the same decision-time ice/roughness estimates and Bv7 release limits. Both
start each selected year from the continuous causal rule-curve state, then carry
storage forward independently through the year.

The pilot uses fixed experimental caps of **75.50 m Ontario end-of-period level**
and **22.10 m modeled Pointe-Claire level**. Bv7 retains its tiered F limit. The
objective tracks rule-curve releases and its forecast terminal Ontario level.
The beam search retains 12 schedules and records forecast margins and fallback
use. Its failure to find a feasible schedule establishes a search outcome.
Global feasibility remains unresolved.

- [x] Source the Plan 2014 IGLD 1985 F-tier table and distinguish operational
  weekly releases from the model's quarter-month convention.
- [ ] Independently verify historical gauge metadata and season-specific research
  safety thresholds; operational calibration remains open.
- [x] Diagnose fallback decisions using first-action grids, relaxed cap bounds,
  a 96-wide beam and complete candidate enumeration. Thirty-two of 37 states
  have incompatible one-step forecast cap requirements. No finite candidate
  schedule meets both caps in the 37 original fallback states.
- [x] Run a labelled perfect-inflow diagnostic with costs and search fixed.
  Ontario severity and downstream peaks improve in both flood years, while total
  downstream exceedance increases.
- [x] Remove release tracking, terminal tracking and both in matched ablations.
  Each arm reproduces all 144 reference releases. Common candidate re-ranking
  also preserves the selected first release at all 37 fallback states.
- [x] Declare tiered downstream targets, shared native Bv7 bounds and explicit
  least-exceedance recovery priority in the separate coupled study.
- [x] Define shared upstream/downstream exposure, low-water, release-change and
  finite-task terminal-storage costs; retain follower service and storage metrics.
- [ ] Extend native performance-indicator reporting to the coupled trajectories;
  all seven already run for the native reference.
- [ ] Extend paired continuous trajectories across ordinary, wet and dry conditions
  once the objective and constraints are declared.

### 2. Implement the Ottawa follower

The current pure storage balance requires explicit capacity and release bounds.
The next implementation needs an independent operating objective and a justified
mapping from controlled releases and uncontrolled runoff to Lake St. Louis.

- [x] Declare aggregate storage, synthetic inflows and outlet/routing assumptions,
  distinguishing published reservoir capacities from usable aggregate capacity.
- [ ] Replace the synthetic SLON decomposition with independently supported inputs
  and test nonzero travel-time assumptions.
- [x] Define follower storage targets, release/service requirements and terminal storage value.
- [x] Implement a strictly convex best response to Ontario's announced request schedule.
- [x] Test conservation, storage feasibility, causal response timing, KKT optimality
  and local announcement sensitivity.
- [ ] Run declared capacity, controlled-fraction and objective-weight sensitivities.

### 3. Implement Stackelberg MPC

- [x] Define request announcements, advisory future schedules and Ottawa's decision-time information.
- [x] Implement and run the nested follower best-response reference on three complete annual cases.
- [x] Check follower primal, stationarity and complementarity residuals, plus
  active-set KKT sensitivities against independent finite differences.
- [ ] Investigate a leader optimizer using those derivatives; the current nested
  finite search remains the reference.
- [x] Compare external-flow and anticipating MPC in the same responsive simulator with matched forcing and starts.
- [x] Report both operators' storage, service and flood outcomes in the coupled pilot.

### 4. Add PPO and complete evaluation

- [x] Define causal Ontario observations, residual requests, common reward and finite annual episodes through Gymnasium.
- [x] Complete a bounded five-seed training check and matched pilot comparison; longer training remains open.
- [x] Freeze chronological fitting/development/evaluation membership before fitted
  forecast or policy selection. The 1995, 2017 and 2019 cases are development
  evidence; the other 2001–2020 years remain reserved in the declared protocol.
- [x] Report Ontario/Pointe-Claire peaks, exceedance counts and severity, release
  changes, terminal storage, forecast/realized flows, fallback frequency and runtime.
- [ ] Include supported native performance indicators with their declared units
  and aggregation. Use the same physical execution rules across controllers.

## Reproducibility and references

Frozen historical manifests retain their original source hashes. Reproducing an
older study requires its recorded source revision and environment.

The three diagnostics and their source/output manifests are saved under
`results/loslr/mpc_diagnostics_v2/`. The coupled study records its protocol,
source hashes, trajectories and saved PPO endpoints under
`results/loslr/coupled_v1_information_matched/`. Details are in [RESULTS.md](RESULTS.md).

Detailed model decisions and implementation milestones are in
[the two-system design](docs/MULTI_SYSTEM_PLAN.md). Ottawa evidence and remaining
identification questions are in [the follower specification](docs/OTTAWA_MODEL_SPEC.md).

<details>
<summary>Lake Como historical plan and completed investigations</summary>

This section retains planning context from the earlier Como studies. The active
Ontario phases are listed above.

## Research question and current status

Can a learned daily release policy improve the trade-off between downstream water
supply and flood/low-water risk on held-out Lake Como years?

The core comparison is **literature-based MPC versus learning-based control** in
DamEnv, with matched decision-time information, physical constraints, and evaluation
metrics. PPO/MPC references are frozen, the common filter experiment is complete,
and FQI/SAC pilots are implemented. A first MPC-residual pilot improves the MPC
prototype on validation, with a matched proposal-aware full-action control. See [current results](RESULTS.md). The
chronological test split remains reserved while experimental choices are settled.
Actual operator use of the proposed MPC remains unverified.

PPO and FQI are already model-free RL methods. The **hands-on SAC learning phase**
implemented an online, replay-based continuous actor–critic in PyTorch and completed
bounded diagnostics. Its purpose is understanding and a controlled comparison;
the original SAC pilot did not beat PPO. Existing frozen
references remain intact; FQI refinement and SDP are not prerequisites for learning
SAC. A subsequent literature exploration phase connects SAC to residual control,
history representations, offline-to-online learning, and risk-sensitive objectives.
Hydropower still requires justified turbine, head, efficiency, and price
assumptions; stored water or surplus discharge is not measured generation.

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

- [x] Document hourly storage balance, hydraulic clipping/forced spill, all reward terms, information timing, no evaporation, resets, and endpoint versus hourly safety. See [equations and safety definitions](docs/EVALUATION.md#reservoir-equations-and-safety-definitions).
- [x] Compare MPC horizons **1, 3, and 7 days** with persistence, surrogate, retention, initialization recipe, and solver cap fixed: validation returns -36,015 / -28,077 / -26,361. Seven days beats one day in all years and three days in nine; shorter horizons use less CPU and leave less demand unmet. Keep seven days as the current best mean-reward setting, not a universal optimum. See [matched horizon study](docs/EVALUATION.md#matched-mpc-horizon-experiment).
- [x] Add **last completed-day inflow** through `PastInflowObservation`, with training-only January 1 fallback, reset isolation, terminal handling, and causality tests. Dynamics/reward and the first four observation values are unchanged; no current/future forcing is supplied before decisions.
- [x] Train five past-inflow PPO seeds for 1M steps each, reusing the verified five 1M four-feature controls with exact daily evaluation replay. Mean reward improves -17,232 → -16,505 and seed SD 1,823 → 238, while unmet demand rises 2.0%. Three seeds improve; two worsen. No automatic promotion. See [observation ablation](docs/PPO.md#past-inflow-observation-ablation).
- [x] Analyze changed decisions and wet/dry outcomes with years selected solely by total validation inflow: dry 1990 and wet 1993. Past-inflow PPO reduces low-water events in both, but unmet demand rises in every validation year and wet-year flood days barely change. See [decision analysis](docs/PPO.md#what-changes-in-wet-and-dry-years).
- [ ] Confirm project expectations with the instructor; this controlled engineering analysis is stronger than adding a feature alone, but no particular grade or acceptance is assumed.
- [x] Audit one-day flood feasibility from 40,150 saved states under known-inflow and training-calibrated stress diagnostics. Past-inflow PPO has 208/215 flood endpoints certified infeasible from that day’s state; MPC has daily-safe grid alternatives for 79/123. Distinguish endpoint/hourly safety and grid witnesses from certificates; this does not establish avoidability from the annual reset. See [feasibility audit](docs/EVALUATION.md#flood-safety-feasibility-audit).
- [x] Specify a bounded common filter experiment: float32 observed level, exact past storage/release telemetry, five training-residual interval scenarios plus persistence, hourly flood priority, and explicit recovery. These finite scenarios are not a robust inflow bound or unconditional guarantee.
- [x] Complete one common-filter experiment (120 validation episodes). MPC reward improves -26,361 → -18,486, flood days 12.3 → 4.5, and unmet demand falls; low-water severity worsens slightly. PPO barely changes (8 interventions). Actual forecast-range breaches prevent a safety guarantee. See [filter findings](docs/EVALUATION.md#common-finite-scenario-safety-filter).
- [x] Freeze the past-inflow PPO and seven-day persistence/retention MPC references with plain and shared-filter evaluation tracks in `configs/experiments/control_protocol_v1.json`. Preserve four-feature PPO as an ablation, stop PPO/MPC tuning, and reserve the test split while FQI is developed.
- [x] Implement and run a first **tree-based FQI pilot** using the frozen five-feature observation, 30,660 training-only transitions, 21 actions, and 60 Bellman fits. Three forest seeds share one dataset. Validation rewards: plain -20,062; shared-filter -19,219. This is an Ernst et al. adaptation, not a reservoir-paper reproduction or mature benchmark. See [FQI pilot](docs/EVALUATION.md#first-tree-based-fqi-pilot).
- [ ] Audit FQI conditional state/action coverage, target/fit behavior, and discretization before selecting a larger dataset or backup budget. Preserve pilot results and the frozen PPO/MPC controls; no test evaluation until FQI is settled.
- [ ] Use stochastic DP only as an optional reduced-state, coarse-grid reference or FQI sanity check. Fit inflow transition probabilities from training years, account for season, and check grid refinement. Do not make both SDP and FQI prerequisites: PPO plus a credible MPC already covers the main comparison, and extra methods must answer a concrete question.

### Phase 4B — expand the simulation and training gym (incremental work)

Start with one reviewed change at a time; preserve the frozen annual benchmark.
[Simulator development notes](docs/SIMULATOR.md) define scope and commands.

- [x] Add an opt-in continuous environment using the existing DamEnv equations: carry storage, level, last release, demand EMA and past-inflow telemetry across consecutive years within one split. Only the supplied record end truncates the episode; explicit reset rewinds it.
- [x] Verify exact single-year benchmark parity, correct year rollover/final observations, no future-year observation leakage, split isolation and Gymnasium compatibility. Check replay continuity and the final-record end mask.
- [x] Complete a bounded training-only integration smoke: fresh SAC, 730 steps including 64 warmup, 666 updates, two model years 1946–1947. Finite values and balance checks pass. This establishes integration, not better control.
- [x] Audit source calendar structure and metadata integrity: all seven hashes match; 49 ordinary / 16 leap-year rows have 365 / 366 samples for each signal. The 365-step implementation omits the last leap-year sample, not explicitly February 29. Keep the date anchor/datum/demand-construction questions unresolved and the current model-year policy unchanged.
- [ ] Confirm original timestamps, gauge datum and demand construction before defining a real-calendar version and leap-day MEF/seasonal mapping.
- [x] Add opt-in hourly telemetry delegated to the original integrator, with unchanged observations/rewards/trajectories. Log hourly balance, sampled peaks, interpolated threshold duration/severity, and physical shortages. Verify exact two-year parity and controlled within-day exposure; training-only check saves 17,520 hourly rows with zero learner updates.
- [x] Implement a separate operational research gym with configurable command ramping, analytic water-conserving river routing/diversion, supplied evaporation and volume-preserving intraday shapes. Parameters remain labeled sensitivity assumptions; no calibrated operational claim.
- [x] Trace primary daily-flow sources and gauge evidence; crossmatch training dates and documented extremes. Fit the effective balance coefficient on 1946–1975 and check 1976–1987. Record target/outflow semantics and circularity; no benchmark parameter is silently replaced. See [calibration evidence](docs/CALIBRATION.md).
- [x] Acquire and audit public ten-minute 2021 Malgrate/Lavello level and Lavello discharge records, retaining raw files, quality codes, fixed-UTC+1 timing and source hashes. Keep this modern observational audit separate from benchmark fitting; see [measurement evidence](docs/CALIBRATION.md#public-subdaily-measurements-separate-2021-audit).
- [x] Implement the opt-in [lake-balance/outlet ablation](docs/CALIBRATION.md#outlet-ablation-implemented-findings): separate storage reference from operating limits/capacity, compare legacy versus published fully-open capacity at fixed area, test area sensitivity separately, check conservation/boundaries/integration convergence, and run fixed/feedback controls plus a bounded fresh SAC attempt. The candidate reduces low-water proxy/capacity inconsistencies from 112 to 1; feedback completes identically, while constant release and SAC warmup leave the declared domain. This is a structural diagnosis, not independent calibration or successful RL training.
- [x] Add and test an explicit level-only low-water reserve rule independently of outlet capacity (experimental −0.5 m floor / 24-hour reserve). Compare three training windows and all three declared SAC seeds. All protected SAC smokes finish with 64 updates each; feedback is unchanged, constant-release performance is mixed versus legacy, and the 1986–1987 constant-release run still exits the upper domain. Log environmental priority tradeoffs; no operational calibration or performance-win claim. See [protection findings](docs/CALIBRATION.md#explicit-low-water-operating-protection).
- [x] Diagnose the high-water exit at fixed costs/rules: exact hourly replay, same-state early-release branches, capacity/request audit and 1–32 substeps/hour checks. The exit persists under refinement; maximum release four days earlier or feedback eight days earlier avoids the upper guard in the event replay, while flooding remains. Late requests lose authority under the inherited forced-full-open rule. No range widening or optimizer/cost tuning. See [high-water diagnosis](docs/CALIBRATION.md#high-water-diagnosis-release-timing-and-model-limits).
- [x] Test fixed-cost causal protected MPC: persistence/seasonal/conditional forecasts, one-/seven-day horizons, prior-only fitting for the 1986–1987 event, a two-year 1988–1989 validation screen and a shared-state event branch (24 arms). All 18 MPC arms stay in range; feedback still has lower total cost. Longer planning lowers the event peak, but forecasts underpredict the surge and target-delivery tradeoffs remain. See [protected MPC findings](docs/SIMULATOR.md#protected-mpc-with-causal-forecasts).
- [x] Diagnose target forecasting versus search in a fixed conditional-seven-day MPC factorial: prior-only correction using the existing target EMA, original versus 17-point/two-sweep daily-coordinate search, three matched cases, and 60 shared-state audits. Target correction cuts validation cost 1348.43 → 1043.81, close to feedback's 1048.65; extra search has small/inconsistent realized benefits despite non-increasing expected cost and about 12× runtime. Keep costs/physics fixed and report increased low-water exposure. See [target/search diagnosis](docs/SIMULATOR.md#target-forecast-and-search-diagnosis).
- [x] Check the fixed history-aware target forecast/original search over all validation years in the same five two-year episodes, separating prior 1988–1989 from additional 1990–1997. All 30 controller years complete. On the eight additional years, MPC costs 592.42/year versus seasonal MPC 705.35 (7/8 wins) and feedback 607.59 (4/8 wins). Report fewer flood days but more low-water exposure, paired blocks, first/second-year results and the 1994–1995 carryover loss. No best-controller promotion, objective changes or final-test use. See [broader-year check](docs/SIMULATOR.md#broader-year-validation-of-the-target-forecast).
- [x] Run the controlled 1995 storage/controller replay with unchanged history, forecasts, costs and physics, plus two MPC cold-plan controls. Low-start costs: MPC 525.97 / feedback 562.98; high-start: 501.16 / 484.06. Higher inventory helps MPC by 24.81 and feedback by 78.92; warm-start controls are identical. Reconcile the marginal 79.60 million m³ inventory, report different end reserves and both exact gap decompositions, and avoid a unique percentage attribution or claim that horizon alone explains the gap. See [carryover replay](docs/SIMULATOR.md#controlled-1995-storage-carryover-replay).
- [x] Trace 1995 reserve use and run a labeled perfect-target information diagnostic at fixed inflow forecasts, costs, physics and search. Most extra river flow leaves by day 20; perfect targets lower cost but worsen low-water exposure and leave 47.05 of the original 55.21 million m³ marginal river allocation. Saved-state audits expose first-minimum candidate ordering and float32 service undershoot: lower-release service has equal/better predicted cost on 17/20 early causal high-start decisions. No annual benefit of a search fix is claimed. See [reserve-use diagnosis](docs/SIMULATOR.md#reserve-release-timing-and-perfect-target-diagnostic).
- [x] Test precision-safe service proposals and conservative exact-cost tie-breaking separately/together at unchanged costs in eight complete 1995 replays. Combined costs fall 525.97 → 488.88 (lower start) and 501.16 → 413.86 (higher); low-water exposure falls and ending stock rises 4.38 → 16.92 million m³. Original-off controls reproduce bitwise trajectories, all forecast paths match, and marginal stock ledgers reconcile. Record the isolated tie-only low-start severity tradeoff and selected-case limitation. See [precision/tie experiment](docs/SIMULATOR.md#precision-safe-service-and-exact-cost-ties).
- [x] Check original versus combined precision/tie MPC over all five validation two-year episodes and forecast seeds 901–905 (50 complete runs). Cost falls 576.26 → 516.75/year; all 25 paired episodes and 46/50 paired years improve. The benefit extends beyond 1994–1995, while mean flood severity rises 6.8% and flood days 4.86 → 5.26/year. Report 16/17 planner fallback decisions, ending reserves, paired seed ranges and reused-validation limits. No default change. See [full-episode/seed validation](docs/SIMULATOR.md#precisiontie-validation-over-full-episodes-and-forecast-seeds).
- [x] Diagnose 1993/1997 flooding with 100 matched-state event branches across five seeds and two pre-flood anchors. Warm-plan controls are identical; 1993 choices during the pre-flood fortnight matter, while much of 1997's excess is inherited stock. Audit 280 matched pre-flood observations and selected-window inflow undercoverage. Expanded search finds no feasible alternative in 33 failed states; monotone full-capacity bounds certify at least one infeasible forecast path in every case (37 paths total). Keep model, costs and defaults fixed. See [event and feasibility diagnosis](docs/SIMULATOR.md#flood-events-matched-states-and-forecast-infeasibility).
- [x] Compare four causal inflow samplers with fit 1946–1975, selection 1976–1981 and separate confirmation 1982–1987. Freeze the scaled/stratified history analogue method before confirmation/event outcomes: confirmation volume CRPS 23.37 → 20.35 million m³ and wet upper coverage 44.6% → 67.7%. Verify no holdout fitting, causal observation history and unchanged baseline outputs. See [forecast study](docs/SIMULATOR.md#causal-history-forecasts-training-holdouts-and-event-rechecks).
- [x] Refit the selected forecast on 1946–1987 only and repeat 80 matched event checks at unchanged costs/physics. Mean flood severity falls modestly in all four year/anchor groups, but only 31/40 paired comparisons improve and maximum-request fallbacks increase. Reaudit the original 33 failed states: 24 remain capacity-infeasible and 9 have feasible schedules. Record persistent extreme-volume undercoverage; preserve defaults and final test years.
- [x] Separate seasonal support, history conditioning and finite sampling in reused training holdouts and event forecasts. All 27 pre-flood misses in 1993 are sampling misses; 25/27 in 1997 arise from conditioning exclusions. Nested 5/15/25-path sensitivity cannot fix missing pool support. Cross both forecasts with both saved states at 1,660 matched times: warning counts 68/374/68/367 show a +306 forecast effect and −7 state effect in one conditional order. No new policy runs or defaults. See [tail/fallback diagnosis](docs/SIMULATOR.md#tail-support-scenario-sampling-and-additional-fallbacks).
- [x] Test explicit-probability tail representation on chronological development folds (1946–1965 → 1966–1975; 1946–1975 → 1976–1981), then lock before the reused 1982–1987 check. Select 80% conditional / 20% seasonal mixture and a 5% tail band; represented CRPS falls about 1.7%. Complete 120 matched event replays with uniform-probability ablation and unchanged all-path feasibility/reward weights. Record early headroom, shortage/fallback increases and the distinction between full-mixture calibration and coarse control atoms. See [weighted-tail study](docs/SIMULATOR.md#explicit-tail-probabilities-and-matched-event-control).
- [x] Evaluate the frozen weighted-tail candidate against the preceding history MPC over all five complete validation pairs and seeds 901–905 (50 completed runs). Cost worsens 517.51 → 531.79/year despite 5.8% less flood severity; unmet target rises 4.0%, low-water days 17.12 → 21.30 and ending stock falls. Only 3/25 paired episodes improve, with higher mean cost under every seed and dry/middle/wet group. Preserve this negative full-record result; do not promote the candidate. See [full weighted-tail validation](docs/SIMULATOR.md#weighted-tail-mpc-full-two-year-validation).
- [ ] Keep the weighted-tail variant as a labeled sensitivity. Any stronger flood-priority objective or probability/feasibility change needs a separate frozen protocol before new controller comparisons; do not retune it on these validation outcomes. Preserve full-mixture uncertainty diagnostics, fixed physical reward weights and unsupported-ramp/routing exclusions. Final test years remain reserved.
- [ ] Complete independent operational calibration: public subdaily records are now available, but synchronized independent outlet flow, gate openings/commands, gauge-zero/rating validity, diversions and actual irrigation demand remain missing. The evidence profile explicitly rejects promotion of unsupported 5/6/300 parameters. A draft data request is prepared locally, not sent; the user has no additional operator records.
- [x] Declare the operational task protocol: seeded two-year training windows, initial-state range, counted 30-day spin-up, eight explicit state/time observations, finite versus continuing-record terminal modes, time-adjusted discount and fixed unit normalization. Fresh daily/subdaily SAC smokes verify integration; original episodic masks remain unchanged.
- [x] Add explicit-reset continuous MPC/residual adapters with cross-year planning/warm start. Add a bounded scenario planner and finite-scenario constraint filter with reported recovery, not unconditional safety claims.
- [x] Fit a training-only conditional historical-block scenario generator with scenario-count-independent conditioning; report chronological validation forecast error/coverage. Run labeled drought/wet/pulse sensitivities without assigning climate probabilities.
- [x] Add physical shortage/exposure/downstream/reserve objectives and subdaily 1/3/6/12/24-hour actions. Twelve declared cases verify balance and same-request cadence parity; extra action opportunities can worsen the example feedback rule. New weights are explicit assumptions, not a benchmark-score replacement.
- [x] Complete the six-track implementation suite in `results/simulator/operational_v2/`, preserve the superseded prototype/source snapshot, add PNG-only physical explanations, and retain all seeds. No new controller ranking or held-out test evaluation.
- [ ] Calibrate forecast coverage; operational measurement validation remains open.
- [x] Freeze a separate matched protected-model PPO–MPC protocol: common physical costs, constraints, reset and causal seven-inflow history; five PPO training seeds × 200,000 decisions, final checkpoints, all five two-year validation blocks. Keep final test years reserved. See [matched protocol](docs/SIMULATOR.md#matched-ppo-and-mpc-protocol-in-the-protected-model).
- [x] Pass the 4,096-step training-only PPO integration check, including finite actor/critic gradients, physical balance, annual carryover and finite-task termination.
- [x] Complete the declared five-seed PPO study (5 × 200,000 decisions) and all 25 two-year PPO validation episodes. MPC has lower mean annual cost (517.51 vs 631.34) in every historical year and block after own-seed averaging. PPO has slightly less flood severity but more unmet target and low-water exposure. Preserve the negative RL result, common costs and reserved final test; no promotion. See [matched results](docs/SIMULATOR.md#matched-ppo-and-mpc-results).
- [x] Audit 400 PPO/MPC action pairs from identical states, with realized first-day flows, warm/cold MPC controls and common-PPO 30-day continuations; distinguish inherited stock from first-action effects.
- [x] Inspect weather-adjusted PPO training progress and 1,440 frozen-policy stochastic returns at 60 training states. Use reachable base-to-higher-stock differences; retain below-floor perturbations only as sensitivities. Predicted storage value is much weaker than paired realized value. See [drawdown diagnosis](docs/SIMULATOR.md#ppo-drawdown-diagnosis-and-controlled-credit-experiment).
- [x] Run the frozen one-factor GAE experiment (lambda .95 → .99; gamma/costs/budget unchanged), five seeds × 200k and all 25 complete validation episodes. Cost is 632.55 vs 631.34 (+0.19%); only one seed improves and low-water days barely change. Retain lambda .95; no default promotion or final-test use. See [controlled result](docs/SIMULATOR.md#longer-gae-controlled-result).
- [x] Fit only the frozen PPO critic on full-return targets and check disjoint whole training-year blocks: 146,200 labels, unchanged actor/scales, 960 complete stock branches. Storage-slope error improves 27.5%, but broad checking MSE worsens 3.2%; no policy improvement is inferred.
- [x] Measure full-day action saturation from 300 training states × 64 draws. Fitting-state daily upper-plateau mass is 97.03%; the predeclared >=50% rule selects a model-informed full-range action transform. See [identification](docs/SIMULATOR.md#frozen-actor-critic-fitting-and-action-saturation).
- [x] Complete the five-seed, 200k full-range hydraulic-action-transform study and all 25 validation episodes. Cost improves 631.34 → 622.31; low-water days 117.2 → 101.0, with slightly more flood exposure and most average gain from one seed. Fixed-state plateau mass falls 97.03% → 15.43% with mapping alone, then 48.78% after training. MPC remains stronger (517.51). Preserve both PPO versions without default promotion. See [hydraulic results](docs/SIMULATOR.md#full-range-hydraulic-ppo-results).
- [x] Diagnose seed124 with all five mapped policies retained: physical allocation ledgers, 200 common-state validation branches and 480 own-policy stochastic training returns. Its gain is mainly better target service/less low-water exposure, not the best endpoint critic. No champion selection.
- [x] Run matched critic coverage: six blocks × four trajectories versus twelve blocks × two, identical 17,544 samples and 3,000 updates per seed, actors frozen and checking years excluded. All concentrated controls reproduce bitwise. Diverse support is closer, but MSE worsens 0.288 → 0.294 and stock-slope error 0.272 → 0.502. Gate fails: no refit-to-PPO promotion or new actor training. See [seed and coverage findings](docs/SIMULATOR.md#seed-124-and-matched-critic-coverage).
- [x] Compare frozen original/mapped PPO deterministic and stochastic execution across every validation block: five learner seeds × five action seeds, 300 complete episodes, all 50 deterministic trajectories replay bitwise. Sampling raises original/mapped annual costs 631.34→637.80 / 622.31→632.15, also worsening gamma-.995 block cost; retain deterministic reference.
- [x] Inspect mapped PPO on-policy curves with weather-window fixed effects and update diagnostics. Late declines exceed 2% per100k in 3/5 seeds; last50k declines in 4/5. Strongest progress is seed124; no intermediate checkpoints or convergence claim. See [execution and progress](docs/SIMULATOR.md#frozen-ppo-execution-and-mapped-training-progress).
- [x] Implement/test the mapped-PPO duration runner: reproduce and verify each complete 200k prefix, continue in memory to 400k, then evaluate only the declared endpoint. Fail closed on prefix mismatch and withhold incomplete cohort means. Runner validation used reduced budgets; the user subsequently completed the full study. See [runner command](docs/SIMULATOR.md#mapped-ppo-duration-runner).
- [x] Complete the user-run duration-only mapped-PPO extension, with all five exact 200k prefixes and 25 complete 400k validation episodes. Cost falls 622.31 → 617.91 (0.71%); low-water days stay near 101 and severity worsens 2.8%. Preserve the 400k experimental endpoint without default promotion.
- [x] Inspect saved 200k–400k logs with whole-episode weather-adjusted fits and end-time sensitivity. Late trends are small/noisy and updates remain finite; record results and current understanding in RESULTS.md. Stop the duration sweep at 400k; no 800k run planned. See [inspection](docs/SIMULATOR.md#mapped-ppo-400k-result-and-curve-inspection).

### Phase 5 — hands-on online RL: implement and understand SAC

**Status:** [SAC walkthrough](docs/SAC.md), CPU PyTorch agent, correctness checks,
20-episode training-only smoke, and a separate Pendulum check are complete. The
five-seed 200k-step study scores −21,064 plain / −20,871 shared-filter on validation;
PPO remains the better reference at its larger 1M-step budget. All seeds and both
tracks are retained. PNGs and local reports remain ignored; test years are reserved.

**Learning goal:** understand and implement replay, bootstrapped value learning,
policy gradients, exploration, and target networks on a real control problem.
PPO already learns directly from interaction; FQI already learns from a fixed
batch. SAC adds an online, off-policy continuous actor–critic: collect experience,
reuse it in replay, learn Q-functions, and learn a stochastic actor. “Model-free”
means the plain agent does not call reservoir physics to plan actions or form its updates;
the simulator still supplies physical transitions during data collection.

**Primary experiment:** compare mean full validation reward at a predeclared
interaction budget against the frozen references. Secondary outcomes are learning
curves, seed variability, water-supply/severity trade-offs, and compute. A useful
learning phase does not depend on achieving a new best score.

- [ ] Read and annotate a small primary-source set: Wang et al. (2020), already in `/papers`, for a reservoir MDP and negative SAC result; Tabas & Samadi (2024), *Fill-and-Spill*, for continuous-action reservoir SAC/TD3; and Haarnoja et al. for SAC's actual update equations. Extract observations/timing, action units, reward, training data, episode boundaries, hyperparameters, comparators, and limitations. The initial screening and links are in [the focused review](docs/LITERATURE_REVIEW.md#hands-on-online-rl-phase-sac).
- [x] Write the reservoir MDP and each SAC update in our own words before coding: what is observed, what is hidden, what enters replay, what the actor/critics predict, what is detached, and why target networks are needed. Connect SAC's bootstrapped target to FQI and its actor update to PPO. Derive the modern twin-Q/automatic-temperature variant explicitly; do not mix incompatible equations from different SAC versions.
- [x] Implement a small transparent PyTorch agent: replay buffer, bounded stochastic actor, two Q-critics, target critics, and an explicit training loop. A trusted implementation may be a test reference, but keep the learning exercise's updates visible in our code. Reuse the environment adapter and evaluator; do not rewrite DamEnv or change its reward for an apparent win.
- [x] Add components incrementally: (1) transition storage/sampling and terminal masks, (2) critic Bellman regression and target updates, (3) reparameterized actor gradients, (4) entropy temperature. Check target detachment, gradient flow, bounded actions, tanh log-probability correction, and consistency of normalized action units/entropy targets. Explain each component and one failure it prevents.
- [x] Verify on a small known-value control task and short training-only DamEnv runs before expensive experiments. Confirm reset/terminal handling, no future-inflow leakage, finite targets/gradients, and checkpoint round trips. Debug correctness before changing reward weights, adding memory, or running a broad hyperparameter sweep.
- [x] Predeclare the first reservoir study: same five-feature past-inflow observation, original reward, action limits, chronological splits, annual resets, and no bootstrap across episode ends. Fit normalization using training data only. Record any reward scaling and its interaction with the entropy temperature; evaluate with original reward units. Keep the finite-scenario safety filter outside initial training and evaluate both frozen plain/shared-filter tracks afterward.
- [x] Choose and log warmup, replay size, minibatch size, update-to-data ratio, target-update rate, network capacity, evaluation policy, checkpoints, and a bounded interaction budget before validation. Match PPO's five learner seeds for the full study. Count warmup interactions and report gradient updates and runtime as well as environment steps: equal interaction budgets are not equal computation, and FQI's shared-data pilot is a separate budget/design.
- [x] Run one bounded SAC study and explain behavior: mean/weakest-seed reward, learning curves, action distribution, entropy/temperature, twin-Q disagreement, and wet/dry supply/severity outcomes. These diagnostics are not calibrated uncertainty or safety guarantees. Do not silently select the best seed or best validation checkpoint, and do not change the frozen PPO/MPC/FQI results.
- [x] Produce a short walkthrough and PNGs in ignored `background/`, showing replay → critic target → actor update → release. End with what was learned, implementation checks, empirical limits, and whether SAC merits the final comparison. A losing SAC result can still be a strong learning outcome.

- [x] Diagnose SAC's release timing and critics without tuning: extra unmet demand decomposes into 82.53 million m³ surplus plus 15.34 million m³ terminal storage; 28/60 selected training anchors have a cheaper identical-state request under realized inflow, and min-Q ranks it lower in 20. Conditional branch returns do not establish population calibration. See [diagnosis](docs/SAC.md#release-timing-and-critic-diagnosis).
- [x] Test frozen-actor critic fitting on paired clipping transitions: 210 fit pairs / 235 separate checking pairs from training years, 1,000 fixed steps per seed/arm. Checking cheaper-request ranking improves 28.5% → 59.1% with ordinary Bellman regression and 53.6% with added pair loss. General value error does not improve; actor and all controller scores remain unchanged. See [critic fitting](docs/SAC.md#frozen-actor-critic-fitting-experiment).
- [x] Separate focused sampling from extra updates on one fixed pool: at 100 / 1,000 updates, uniform checking-pair ranking is 30.6% / 34.5%, versus focused 43.4% / 51.9% (baseline 28.5%). Focus wins at 1,000 for all five seeds, but broader checking target error worsens in both arms. No actor or reference changes. See [sampling study](docs/SAC.md#sampling-emphasis-versus-update-count).
- [x] Check broad error against fresh ordinary-state returns and test 10% focus: 420 training-year anchors × eight continuations; conditional return MSE is 43.868 unchanged / 43.860 uniform / 43.905 with 10% focus at 1,000 updates. Cached-target error overstates the apparent deterioration, but 10% focus still fails the declared broad-preservation rule despite improving pair ranking to 41.7%. No actor or original score changes. See [fresh-reference check](docs/SAC.md#fresh-return-checks-and-10-focused-replay).
- [x] Compare frozen bootstrap versus fresh rollout-return labels on identical fitting states/batches: at 1,000 critic updates, checking return MSE is 44.388 bootstrap / 28.554 rollout (43.868 unchanged), improving all five seeds. Single-action rollout fitting still has poor action differences (36.6% pair ranking versus bootstrap 55.7%) and overfits from 100 to 1,000. No actor or original result changes. See [target comparison](docs/SAC.md#bootstrap-versus-rollout-return-targets).
- [x] Complete the final bounded iterative-TD diagnostic: frozen actor/alpha, matched data/batches/action noise, target tau 0 versus .005. At 1,000 updates, checking return MSE is 44.145 frozen / 44.493 moving (43.868 unchanged). Target updates alone do not resolve this gap; no convergence or policy improvement is claimed. See [iterative TD](docs/SAC.md#iterative-td-with-the-actor-frozen).
- [x] Test action-diverse rollout labels with the actor frozen: adding ±50 m³/s alternatives reduces checking gap MAE 0.2482 → 0.1877, but remains worse than unchanged critics (0.0363); directional agreement 44.5% versus 49.3% unchanged. Conditional return MSE still improves, without reliable release comparisons. All five seeds and 100/1,000 budgets retained; no actor or reference changed. See [action-diverse diagnosis](docs/SAC.md#action-diverse-rollout-supervision).

Start with **one SAC implementation**, not a SAC/TD3/DDPG/recurrent-agent sweep.
TD3 is a literature-supported alternative if the method review changes the choice,
not an automatic extra controller. Memory, multi-year episodes, synthetic inflows,
and learned dynamics are later extensions because each changes another part of
the problem. The standalone implementation and first bounded study are complete; inspect its
supply/low-water limitations before increasing the budget or adding extensions.

#### Environment tooling and related benchmarks

These tools serve different roles from learning algorithms such as PPO, FQI, or SAC:

- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** supplies the standard environment interface (`reset`, `step`, observation/action spaces), wrappers, and reference environments. We already use it through DamEnv and pin version 1.1.1. Keep this interface for SAC and later extensions.
- **[Gym-Anything](https://github.com/cmu-l3/gym-anything)** builds environments for agents operating real software, with task setup and verification. Our reservoir already has a numerical simulator and measurable outcomes, so no integration is planned. Revisit only if controlling external engineering software becomes an explicit research question.
- **[ExploitGym](https://github.com/sunblaze-ucb/exploitgym)** evaluates agents developing exploits against real-world software vulnerabilities. It is a separate cybersecurity benchmark, with no current reservoir-control experiment or integration planned.

- [x] As part of SAC verification, run Gymnasium's environment checker on the intended adapter/wrappers where supported; investigate warnings and retain explicit tests for causality, terminal observations, action units, and physical dynamics. Interface compliance alone does not establish physical correctness. Use a small continuous-action reference task for debugging before DamEnv; reference-task scores are implementation evidence, not the project's engineering result.
- [x] Include a short explanation in the SAC walkthrough of **agent versus environment versus evaluator**: the agent chooses a release, DamEnv advances reservoir physics through Gymnasium's interface, and the evaluator measures complete trajectories. Learn from Gym-Anything's explicit setup/verifier separation without introducing a software-use or cybersecurity benchmark into the main comparison.

### Phase 6 — literature exploration: build on the RL foundations

**Learning goal:** understand four research directions from Pierre-Luc Bacon and
collaborators through focused reading and small reservoir experiments. The first **MPC with residual RL** pilot is implemented and evaluated. Read the
relevant paper before each further branch; the remaining experiments
are optional follow-ups, not four additional controllers required for completion.
The remaining branches are proposed adaptations; the first 6A pilot is implemented
and does not establish general method superiority or final-test performance.

For each reading, write a short explanation of the problem, core equations,
assumptions, evidence, and what transfers to our reservoir. Before implementing an
extension, specify one hypothesis, a matched control, an interaction/compute budget,
and a stopping criterion. Use training years for fitting and debugging, validation
for development, and keep test years reserved until methods are frozen. Preserve
the existing references and distinguish changes to information, initialization,
architecture, and objective. Record results even when an extension does not help.

#### 6A — residual RL: learn a correction to MPC (first priority)

**Status:** short pilot and duration-only matched extension are complete.
The 3,650-step residual pilot scores −17,310 plain / −16,924 filtered. At
36,500 steps it worsens to −21,799 / −21,266 in all three seeds; the full-action
control improves to −21,353 plain. No reference is replaced. See
[implementation and results](docs/RESIDUAL_RL.md).

Read [*What Makes Value Learning Efficient in Residual Reinforcement Learning?*](https://arxiv.org/abs/2602.10539)
and inspect the authors' [DAWN implementation](https://github.com/Guozheng-Ma/DAWN).
The paper studies refinement of frozen pretrained policies; using our engineering
MPC as the base controller is a new adaptation.

**Question:** can a learned correction improve MPC under imperfect inflow forecasts?

- [x] Implement and explain `clip(MPC_release + delta, 0, 491.61)`, with delta bounded to ±50 m³/s before the common filter. Replay uses normalized delta, critic inputs use combined nominal request, and nominal saturation is included in actor gradients. Original hydraulic clipping/reward are unchanged.
- [x] Read DAWN's warmup/scale arguments and algorithm; adapt base-only implicit warmup plus critic LayerNorm, without explicit critic warmup or progressive exploration. Document the reservoir/base-controller/budget differences; this is not a paper reproduction or component ablation.
- [x] Compare frozen MPC, residual SAC, and a matched proposal-aware full-action SAC control. Both learned arms see the same six inputs, LayerNorm, warmup, and 3-seed/3,650-step budget. Preserve old five-feature SAC/PPO as unmatched contextual references. Count all MPC calls and time; evaluate every validation year with plain/shared-filter tracks.
- [x] Complete the first bounded residual pilot, with exact zero-correction MPC reproduction and both training smoke gates. Report corrections, seed variability, solver issues, supply/event/severity metrics, and resource cost. Mean correction +6.39 m³/s; both tracks improve the MPC prototype, but full-action control supplies more water at the cost of low-water exposure. Bounds/filter do not imply safety.
- [x] Extend both matched arms to 36,500 steps/seed, changing duration only. All six short prefixes reproduce exactly; final validation shows residual deterioration in every seed, while full-action improves. Report both, retain the short pilot, and do not promote the longer residual endpoint. Original references and test years remain reserved. Optional warmup/normalization ablations remain separate; their individual effects are not established.

- [x] Diagnose residual duration regression using saved trajectories and both frozen checkpoints on common states: extra low-water penalty explains 88% of reward loss; 32.03 million m³ additional surplus and 15.52 million m³ lower terminal storage reconcile 16.51 million m³ extra shortage. Actor corrections increase on both state sets; critic estimates are not calibrated counterfactual returns. No new training or validation rollouts.

#### 6B — learning from history: represent hidden hydrological conditions

Read [*Bridging State and History Representations: Understanding Self-Predictive RL*](https://arxiv.org/abs/2401.08898).

**Question:** does recent inflow/level history provide useful information beyond
our current five-feature observation?

- [ ] Explain partial observability and the paper's self-predictive latent objective, including target construction and stop-gradient treatment.
- [ ] Begin with current-observation SAC versus a fixed window of completed-day observations/actions. Define reset padding and causal normalization. If this motivates further work, compare a recurrent encoder, then its self-predictive auxiliary loss with architecture/history held fixed.
- [ ] Audit replay sequence sampling, episode boundaries, recurrent state resets, and evaluation timing. Report the extra information and capacity: this changes the observation experiment. Adding an LSTM alone is not a reproduction of the paper, and latent prediction accuracy alone does not establish better control.

#### 6C — offline-to-online RL: reuse experience before interacting

Read [*The Three Regimes of Offline-to-Online Reinforcement Learning*](https://arxiv.org/abs/2510.01460).

**Question:** when does our fixed FQI training dataset help SAC learn online?

- [ ] Explain the stability/plasticity trade-off and how dataset quality and pretrained-policy quality affect the starting point. Treat the paper as a framework to investigate, not a single drop-in algorithm.
- [ ] First compare SAC from scratch with SAC whose replay buffer is preloaded with the same training-only FQI transitions. Verify reward scaling, action units, terminal masks, and normalization compatibility. Keep new online interactions and update budgets matched; report offline collection cost separately and in total.
- [ ] Consider explicit neural critic pretraining or actor distillation only as a separate experiment. Our tree-based FQI weights cannot directly initialize SAC networks. Replay preloading alone does not reproduce the paper's full pretraining/fine-tuning setting.

#### 6D — risk-sensitive RL: focus on severe outcomes (advanced option)

Read [*Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity*](https://arxiv.org/abs/2602.03778).

**Question:** can explicit tail-risk optimization reduce severe reservoir losses
at an acceptable cost to average water supply?

- [ ] Explain CVaR, the distinction between trajectory-level and stage-wise risk, and why the paper introduces an augmented state and a different Bellman formulation. This is a substantive algorithm/objective change, not just a larger flood penalty.
- [ ] Define the loss and tail level before implementation: poor total return and physical flood severity are different targets. Compare expected-return and risk-sensitive controllers under common scenarios, reporting both mean performance and the chosen tail metric.
- [ ] Assess scenario adequacy first. Ten validation years provide weak evidence for rare-tail estimates; repeated learner seeds do not create independent hydrological years. Any additional scenario generator must be justified and fitted on training data only. CVaR optimization does not imply a hard no-flood guarantee.

**Phase outcome:** a concise reading note for each direction, one prioritized
implementation with a controlled result, and a clear explanation of what we
learned. Add intuitive PNG diagrams and walkthroughs to ignored `background/` as
methods are understood. Broader implementation remains optional; do not delay the
final comparison to exhaust every paper.

### Phase 7 — final analysis and presentation

- [ ] Evaluate the frozen controllers on held-out test years and report paired trade-offs, seed variability, and limitations.
- [ ] Add reward-weight or hydropower extensions only if time and defensible parameters support them.
- [ ] Prepare the report/poster and interactive results website; keep code, figures, and claims reproducible.

## Reading path and artifacts

Read [PPO findings](docs/PPO.md), [evaluation](docs/EVALUATION.md), and
[literature review](docs/LITERATURE_REVIEW.md) as needed. Start with
[environment walkthrough script](scripts/lake_como/environment_and_data.py) for the system or
[controller comparison script](scripts/lake_como/controller_comparison.py) for current comparisons;
intermediate analysis scripts are optional detailed evidence.

`configs/ppo.json` is the current selected baseline; `configs/experiments/ppo_studies.json`
consolidates exact historical study plans. Keep experiment code in `src/reservoir_control/systems/lake_como/`
and keep the pinned fork at `external/gym4real/` stable for frozen benchmarks.
Prototype operational extensions in `src/reservoir_control/systems/lake_como/simulation/`; upstreamable fork
changes can follow validation. Generated `results/`,
`reports/`, `background/`, and local PDFs remain ignored. Nothing is committed
without review and approval.

</details>
