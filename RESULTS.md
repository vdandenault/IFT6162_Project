# Results and current understanding

## Ontario–St. Lawrence

### Shared recovery after feasibility rejection

Adding gradual Ontario release profiles to the common recovery search
completed both the 2017 and 2019 annual cases for all four controllers.
Reserve release alone reopened the original stopped decisions, but all
controllers stopped again. The wider recovery also completed all 18
previously evaluated reserved annual cases, versus 15 for the original
filter. The peak-protected variant also completed all 18 without
recorded violations. These annual replays reuse the same years.

Broader search alone admitted a release that left too little Ottawa
storage room in one new sustained-rise case. A separate peak-protected
variant checks that the first action fits if the last observed high
inflow persists for one period after a sharp rise. It retains the hard
storage and downstream limits. The check was designed after that first
holdout, then evaluated on different synthetic inflow shapes and noise
seeds.

| Confirmation stress controller | Completed with original filter | Completed with peak protection and wider recovery |
|---|---:|---:|
| Filtered Plan 2014 | 4/36 | 15/36 |
| External-ranked MPC | 4/36 | 15/36 |
| Stackelberg MPC | 4/36 | 15/36 |
| PPO, five seeds | 20/180 | 69/180 |

On confirmation stress, the original filter recorded eight realized
storage violations and the peak-protected variant recorded zero.
External-ranked MPC lowered operating cost relative to filtered Plan
2014 in all 14 confirmation cases that both completed, averaging
**−0.0327** in cost and **−0.0162 m·periods** of Ontario shoreline
exposure. Pointe-Claire exposure was unchanged on those pairs.
Stackelberg and external ranking had identical costs on all 15 common
completions.

With peak protection, both MPC arms also beat filtered Plan 2014 on
the completed 2017 and 2019 annual cases. Costs were **7.5941 versus
7.8465** in 2017 and **0.8359 versus 0.8547** in 2019. Retraining all
five PPO seeds with the wider execution filter did not beat external
MPC on either flood year or on mean operating cost across the 18 annual
replays. The wider, peak-protected filter remains experimental because
the Ottawa inflows, capacities and delay are synthetic.

[Recovery comparison report](results/loslr/final_shared_recovery_report_verified/report.md)

### Ottawa storage and routing sensitivity

External-ranked MPC with peak-protected recovery is the research reference.
At 20 archived decisions from 2017 and 2019, we varied assumed Ottawa usable
storage from 3 to 5 billion m³ and routing delay from zero to two periods.
Ottawa's planned first release varied across candidate schedules in 45 of
150 ranked comparisons. Anticipation changed neither the candidate scores
nor the winning schedule in any of the 150. The other 30 comparisons
started above the smallest variant's storage ceiling. The closed-loop
follow-up was conditional on finding a different winning schedule.

External MPC reduced Ontario shoreline exposure from **13.4814 to 13.2766
m·periods** in 2017 and from **3.2334 to 3.1980 m·periods** in 2019 against
filtered Plan 2014. Both controllers had zero Lake St. Louis exposure above
the modeled targets in these runs. Across 18 annual replays, external MPC's
mean operating cost was 0.0045 lower and shoreline exposure was 0.0190
m·periods higher. Ottawa inflow, storage and routing are synthetic in all
of these comparisons.

[Sensitivity report](results/loslr/ottawa_ranking_sensitivity_verified_v1/report.md) · [Flood-year comparison](results/loslr/ottawa_ranking_sensitivity_verified_v1/flood_years.png) · [Interactive replay](background/ontario_peak_protected.html)

### Anticipating Ottawa near the downstream limit

We reproduced 56 external MPC rollouts and audited 545 decisions near a
downstream limit or where the controller stopped. Of the 524 decisions with
eligible plans, 85 had both tight forecast headroom and at least 100 m³/s
of routed Ottawa variation across eligible schedules at the same lead.
Using the follower response changed at least one predicted Pointe-Claire
level at 393 decisions. All 2,307 common eligible candidate evaluations
had zero forecast exposure above the target under both Ottawa flow
assumptions, so this cost gave both rankings the same scores and winners.
The conditional new closed-loop trial did not trigger. The archived
closed-loop MPC arms executed identical Ontario and Ottawa releases in
both flood years, all 18 annual replays and all 15 joint synthetic
confirmation completions.

[Diagnostic and selection criteria](results/loslr/downstream_anticipation_verified_v1/diagnostic.md) · [Final four-controller comparison](results/loslr/downstream_anticipation_verified_v1/report.md)

### Observed Ottawa benchmark and PPO filter audit

HYDAT provides 2,922 overlapping daily Britannia and Carillon discharge
values for 1987–1994. Carillon, Cornwall discharge, and the Pointe-Calumet
and Pointe-Claire stages overlap on 2,423 dates. This supports a historical
forecast benchmark and an empirical check of downstream stage timing.
HYDAT has no Carillon daily values for 2017–2019. The pinned LOSLR SLON
series can benchmark quarter-month aggregate non-Ontario flow in those
years. Branch flow and datum evidence are still needed for interpreting
modern flows and absolute flood elevations.
[Source audit](results/ottawa/flow_source_audit_v1/report.md)

The common filter changed PPO's executed release by at least 100 m³/s on
474/480 flood-year decisions, 3,126/4,320 decisions in the 18 annual
replays, and 887/1,137 executed confirmation-stress decisions. At 29 of
90 fixed flood-year states, nine distinct PPO requests selected a single
first release. The median state offered two distinct first releases in
the probe. This measures the range of control available through the
existing action interface on those trajectories.
[PPO filter audit](results/loslr/ppo_filter_audit_v1/report.md)

### Held-out Carillon forecasts and PPO development menus

On 1992–1994 daily Carillon data, persistence MAE was 114.8, 196.5 and
317.3 m³/s at one, three and seven days. A model using Carillon history
reduced those errors to 112.8, 191.7 and 299.6 m³/s. Adding Britannia
improved overall one-day MAE to 110.2 m³/s but increased one-day high-flow
MAE from 260.5 to 297.5 m³/s relative to Carillon history alone. Britannia
reduced high-flow MAE at three and seven days, with substantial
underprediction still present. These forecasts assume daily means are
available after the labeled day ends. They provide an older daily reference
for Carillon flow, separate from the modern quarter-month SLON series.
[Forecast report](results/ottawa/carillon_forecast_benchmark_final_v1/report.md)

All 100 frozen PPO development episodes completed with peak protection.
At the 900 sampled states, the full eligible menu had a median of five
first releases. Only 35 states had a single eligible release. The current
continuous proposal range could reach every eligible release at 890 states.
The nine-point proposal grid sampled a median of three releases, so
it understated the choices available through the actual interface.
[Development menu audit](results/loslr/ppo_development_menu_v1/report.md)

### Causal quarter-month SLON forecasts

On the pinned LOSLR aggregate non-Ontario flow series, seasonal history
had lower MAE than both persistence and a seasonal mean in every tested
cohort, lead and latency setting, for overall and high-flow periods.
On 18 reserved years, lead-one MAE fell from 270 to 262 m³/s, and from
744 to 697 m³/s across 79 high-flow periods. With a one-period data
delay, lead-one MAE fell from 389 to 359 m³/s, and from 980 to 851 m³/s
at high flow. The high-flow error still worsened by 154 m³/s when the
latest completed observation was withheld. At high flow with no delay,
lead-one bias shifted from −339 m³/s for persistence to −423 m³/s for
seasonal history, despite lower MAE. Both latency settings are assumptions
about input availability. These scores measure forecast use on archived
SLON, which includes several tributaries. The controller inputs remain
unchanged.
[SLON report](results/loslr/slon_forecasts_verified_v1/report.md) · [Reserved-year error plot](results/loslr/slon_forecasts_verified_v1/reserved_errors.png)

### Ontario MPC replay with historical SLON

All 160 Ontario-only replays completed. Seasonal history changed 360 of
864 executed releases across the 18 reserved years with the latest value
available and 396 with a one-period delay. It reduced the number of
infeasible predicted plans from 34 to 30 and from 35 to 27, respectively.
Realized Lake St. Louis exposure increased in every cohort under both
availability settings, including on high-flow periods. Better predicted
feasibility and lower forecast MAE did not translate to lower downstream
exposure.

| Cohort and SLON availability | Ontario exposure, persistence → seasonal history | Lake St. Louis exposure, persistence → seasonal history |
|---|---:|---:|
| 18 reserved years, timely | 1.554 → 1.695 | 3.80 → 4.16 |
| 18 reserved years, delayed | 1.280 → 1.066 | 5.10 → 5.63 |
| 2017, timely | 7.151 → 6.355 | 2.95 → 3.21 |
| 2017, delayed | 7.613 → 5.378 | 3.07 → 3.70 |
| 2019, timely | 3.883 → 2.574 | 1.99 → 2.82 |
| 2019, delayed | 3.963 → 2.160 | 2.02 → 3.10 |

Exposure is measured in metre-periods above the fixed replay thresholds.
Worst downstream headroom improved across the reserved years, even though
total exposure increased. The Ontario-only planner uses aggregate SLON as
fixed downstream forcing and continues with its least-violating candidate
when it cannot find a feasible plan. All existing controller defaults remain
unchanged.
[Replay report](results/loslr/slon_mpc_replay_verified_v1/report.md)

### Fixed-state diagnosis of the SLON replay

We re-ran both forecast paths from both archived controller states at
the 75 decisions where seasonal history increased realized Pointe-Claire
exposure. Of the 6.390 metre-periods of increase across these selected
decisions, 5.565 came from changing the forecast at the same inherited
state and 0.825 came from changing the inherited state. The decomposition
averages both swap orders. These are gross increases over the selected
decisions.

On development cases alone, 46 fixed-state forecast swaps increased
exposure. Seasonal history gave a lower first-lead SLON estimate than
persistence in all 46. At 44, the seasonal estimate was below realized
SLON, and at 44 the native F limit was the final regime for at least one
forecast. Swapping lead one alone reproduced the seasonal first release
at 43 of those 46 states. Swapping only leads two to four reproduced it at
zero. This motivates a targeted lead-one upper forecast bound, calibrated
before any new reserved-year replay. Persistence remains the reference.
[Fixed-state report](results/loslr/slon_state_diagnostic_verified_v1/report.md)

### One-sided SLON margin experiments

Three lead-one upper forecast variants reduced development downstream
exposure. Each added more than the declared 0.10 metre-periods of
upstream exposure at one or both data delays. A second screen tried a
smaller margin and a floor at persistence. The half-margin variant
passed the development gate at both delays and was frozen for the
reserved replay.

Across the 18 reserved years, half-margin SLON reduced Lake St. Louis
exposure from 3.800 to 3.170 metre-periods without data delay, and from
5.100 to 4.300 with a one-period delay. Ontario exposure increased from
1.554 to 2.394 and from 1.280 to 1.878, respectively. Both increases
exceeded the declared allowance. Persistence remains the reference.
The 2020 replay accounts for nearly all of the upstream increase at
both delays. In that year, the bound lowered downstream exposure by
0.44 and 0.52 metre-periods while adding 0.820 and 0.596 metre-periods
upstream. This is a year-level trade-off observed on the reserved data.
[Initial screen](results/loslr/slon_upper_bound_v1/report.md) · [Smaller-margin screen and reserved replay](results/loslr/slon_upper_bound_followup_v1/report.md)

### Observed downstream routing check

We fitted daily flow-to-stage proxies on 1987–1990 and checked them on
1,035 matched dates in 1991–1993. Adding same-day Carillon flow to Cornwall
flow reduced Pointe-Claire stage MAE from 0.224 to 0.100 m and high-stage
MAE from 0.398 to 0.143 m. It did not improve daily stage-change MAE.
Seven- and eight-day Carillon proxies had higher stage MAE than the
same-day proxy. Calibration-selected short lags did not improve on the
same-day two-flow proxy in the holdout. The data support including the
Ottawa contribution in a stage model. Physical travel time, flow
partition and the historical Pointe-Claire datum tie remain unresolved.

The agreed research rule for replacing the reference requires lower Lake St. Louis
exposure with zero added Ontario exposure in each matched year and no
loss of completion or physical feasibility. The prior half-margin
variant needed +0.841 and +0.598 Ontario metre-periods across the reserved
cohort at the two delays. It fails the rule at both delays. Positive
allowances are reported as sensitivity cases.
[Routing and agreed-rule report](results/ottawa/routing_validation_agreed_rule_v1/report.md)

### Earlier Ontario releases under the agreed rule

The finite hindsight search found a release shift that reduced Lake
St. Louis exposure in 13 of 14 timely-data cases and 12 of 14 delayed
cases with downstream exceedance. The shifts added no Ontario shoreline
exposure and kept each year-end level within 2 mm of persistence MPC.
Total downstream reductions were 1.55 and 1.61 metre-periods,
respectively. The search used historical outcomes to select the shifts.

The causal four-period planner requested one shift across 40 development
runs and reduced neither exposure. Of the 25 hindsight shifts, 12 placed
the later cut inside the four-period planning window. At the archived
release, the causal forecast warned of exceedance in zero of those 12.
The reserved replay did not trigger. The existing reference stays in place.
[Release timing report](results/loslr/release_timing_final_v1/report.md)

### Earlier high-SLON warning with Britannia flow

On 30 reserved-period high-SLON onsets, the archived seasonal forecast
warned of seven at either delay. A model trained to predict onset from
SLON history warned of 18 at both delays. Adding observed Britannia flow
reduced that count to 15 with timely data and 12 with delayed data. It
reduced false alarms relative to SLON-only, but did not pass the gate
for another release-search experiment.

Only three of the 12 development hindsight shifts within the planning
window coincided with a high-SLON onset. This motivates an alert for
Pointe-Claire headroom at a specified Ontario release before revisiting
the release search.
[Earlier warning report](results/loslr/britannia_slon_warning_verified_v1/report.md)

### Pointe-Claire headroom under the issued Ontario release

We evaluated alerts at the releases in each issue-time Ontario MPC plan.
A local opportunity required room for an additional 200 m³/s at issue
time and a later threshold exceedance at the planned release. On the
18 reserved years, the seasonal alert warned of seven of nine relevant
target periods under either data delay, with 12.6% and 13.4% false alarm
rates. The fitted headroom alert warned of five and six, with 8.7% and
8.6% false alarm rates. It did not pass the predeclared gate.

These labels use archived SLON and roughness at the specified releases.
They show local opportunities for an early increase. An actual release
shift still needs a feasible later cut and the annual shoreline check.
[Headroom alert report](results/loslr/pointe_claire_headroom_alert_verified_v1/report.md)

### Observed downstream headroom benchmark

LaSalle discharge, Pointe-Claire stage and Cornwall discharge give us
197 complete quarter-months across 2021–2025, beyond the legacy forcing
archive. Three had mean stage above the 22.10 m study threshold, in
May 2022 and May 2023. A held-out check that inputs observed LaSalle flow
had 0.050 m stage MAE and a 0.091 m 95th-percentile absolute error.
The modeled stage increase for 200 m³/s near the threshold was 0.057 m.
These errors are large enough to affect whether a proposed increase appears
feasible.
[Downstream gauge report](results/ottawa/downstream_gauges_verified_v1/report.md)

### Reach routing and headroom at handoff

Older records favored same-day Cornwall and Britannia inputs. This is
a daily alignment result. On 363 matched spring days from 2021–2024,
adding Britannia reduced Pointe-Claire stage MAE from 26.4 cm to
10.4 cm. Near the threshold the error was still 11.0 cm. The model
found room for an extra 200 m³/s of local throughflow on 12 days
when observed stage already reached or exceeded the threshold.
Adding the Prairies bypass gauge provided no improvement on matched
training and evaluation dates. For the handoff, keep filtered Plan 2014
as the operator baseline and retain the frozen controller comparison.
Calibrating the Ottawa split and testing release-dependent headroom
would require matched flow measurements at Carillon and the
Sainte-Anne and Vaudreuil outlets, or a validated hydraulic model.
[Reach validation report](results/ottawa/reach_validation_verified_v1/report.md)

### PPO probability and frozen execution

Across 860 development states with multiple reachable releases, the
saved policy assigned at least 5% probability to a median of four
releases. The most probable executed release differed from the current
deterministic choice at 379 states. This is a property of the policy
distribution after the common filter.

Choosing the most probable executed release increased mean annual
development cost from **0.13853 to 0.20548** across 20 complete years and
five seeds. Both policies had the same reported flood exposure. The
release-change cost rose from 0.07495 to 0.14165 per year and accounts for
nearly all of the total cost increase. External MPC cost 0.12119 on the
same years. Keep deterministic PPO.
[Probability audit](results/loslr/ppo_release_probabilities_v1/report.md) · [Frozen execution study](results/loslr/ppo_modal_release_v1/report.md)

### Longer PPO training under peak protection

Five fresh runs used the same peak-protected contract at 16,384 and
32,768 decisions. Each earlier checkpoint is the uninterrupted prefix
of the longer run. On the 19 jointly completed development years, mean
cost rose from **0.14178 to 0.14847**. Four of five seeds worsened.
External MPC cost **0.12622** on those same years. Mean Ontario exposure
was 0.27083 at the earlier checkpoint and 0.26952 at the later one,
with zero modeled downstream exposure at either checkpoint on this
matched subset.

The earlier checkpoints completed 100/100 full development runs. The
later checkpoints completed 99/100, with seed 125 stopping at period
13 in 1999. The fixed-state check found an eligible release from the
earlier checkpoint's inherited state for either policy proposal. The
later inherited state had an empty menu under both proposals. The
later endpoint failed the development gate. Keep the frozen comparison
and close the budget extension under this contract.
[Duration study](results/loslr/ppo_duration_peak_v1/report.md) · [Training curves](results/loslr/ppo_duration_peak_v1/ppo_learning_curves.png)

### Paired comparison with the hard Ottawa follower

Plan 2014 requests, external-ranked MPC, Stackelberg MPC, and PPO ran
through the same hard Ottawa follower, native Ontario limit checks and
one-period routing. PPO requested a residual around Plan 2014 through
the common feasible schedule filter. Five seeds trained for 16,384
decisions each on 1901–1980, and the final checkpoints were evaluated.
The Ontario windows are historical and the Ottawa inflows are synthetic.

| Controller | Development years | Reserved years | Synthetic stress runs |
|---|---:|---:|---:|
| Filtered Plan 2014 | 16/22 | 15/18 | 0/36 |
| External-ranked MPC | 16/22 | 15/18 | 0/36 |
| Stackelberg MPC | 16/22 | 15/18 | 0/36 |
| PPO, five seeds | 80/110 | 75/90 | 0/180 |

There were no realized storage-bound or downstream flow-limit violations
on executed periods. The annual controllers and PPO seeds completed the
same years. On the 15 completed reserved years, mean operating cost
relative to Plan 2014 was **−0.006 for both MPC arms** and **+0.016 for
PPO**. Stackelberg ranking did not improve on external ranking under the
shared candidate family. PPO did not reopen any of the common stops.

In the 2017 and 2019 full-year cases, every arm stopped after 17 and 23
executed periods, respectively. The report gives shoreline and
Pointe-Claire exposure over those common prefixes. It leaves full-window
cost blank. All stress runs also stopped, including 24 cases where the
shared filter rejected every controller before the first release. These
stress runs show the limitations of the common schedule filter.

The separate nine-state diagnostic accepted **9/9** realized-input stops
with physical bounds alone, and **1/9** with the forecast reserve, at
both fifth-lead values tested. In a separate two-path stress sensitivity,
retrying after dropping only the forecast reserve raised completion
from **7/36 to 18/36**, with zero recorded realized violations. This
sensitivity keeps the hard physical constraints and does not change the
scored comparison or the default controllers.

[Paired report](results/loslr/final_coupled_report_v1/report.md) ·
[PPO learning curves](results/loslr/final_coupled_report_v1/ppo_learning_curves.png) ·
[2017 and 2019 visual replay](background/ontario_coupled_final.html) ·
[Reproduction commands](README.md#paired-hard-follower-comparison)

### Standalone Ottawa hard-follower study (synthetic)

**Matched horizon study (120 rollouts):** excluding the deliberately infeasible
scenario, perfect completion improves from 4/5 at H4 to 5/5 at H5/6/8/12. Noisy
completion stays at 8/15 for every horizon, with the same eight completed cases.
On those eight matched noisy cases, H5 has the lowest mean cost (8.132066);
longer horizons do not consistently help. All remaining failures are
infeasibility, not numerical solver failure. H5 is a provisional working horizon
for the next uncertainty diagnostic, with H8 a longer-lookahead deterministic
reference; the H4 default is unchanged. Three seeds do not establish robustness.
The frozen forecast bank differs from the old sequential RNG allocation.
Report, matched sample definitions and plots:
`results/ottawa/horizon_sensitivity_v2/`.


**Delayed-case diagnosis:** the original 12-period experiment is feasible.
H=4 loses completion feasibility after its second action and detects failure
before period 4. At that point the downstream caps require 2,365.200 million m³
of retention, but only 2,224.135 million m³ of room remains. An independent LP
confirms the 141.065-million-m³ incompatibility. H=5, 6, 8 and full remaining
H=12 all complete with unchanged constraints and weights. The short window also
omits the downstream arrival of its last planned release; that water stays
queued, rather than disappearing. This is consistent with the implemented
finite-horizon formulation, but does not assure a feasible continuation.
Full diagnosis and plots: `results/ottawa/delay_diagnostic_verified_v2/`.
No controller defaults were changed.


A new, separate follower solves a normalized convex QP with hard storage,
outlet and downstream-capacity constraints for a supplied executable Ontario
schedule. No flood slack, clipping, leader solve or calibrated Ottawa physics.

- **Perfect forecasts:** 4/6 MPC rollouts complete: constant inflow, spring pulse,
  tight zero-delay capacity, and a feasible one-period delay. The impossible
  capacity case refuses an action. The tighter delayed case stops after three
  actions: four-period feasibility is insufficient for recursive feasibility.
- **Past-only noisy forecasts (seeds 11/22/33):** 8/18 complete; 10 stop as
  infeasible (including the deliberately impossible cases). No emergency action
  is substituted. These outcomes do not establish robustness under uncertainty.
- Executed MPC downstream excess is below **3.3e-10 m³/s**; cumulative
  reservoir-plus-transit balance error is below **2e-6 m³**. Water remaining in
  transit is preserved. The fixed-release reference exceeds storage bounds in
  the pulse cases and stops without clipping; its violations remain visible.

Full costs are comparable only for paired complete rollouts; common-prefix stage
costs are reported separately. Reports, PNGs, plans, residuals and timing:
`results/ottawa/hard_follower_v1_perfect_verified/` and
`results/ottawa/hard_follower_v1_noisy_verified/`. Commands are in the
[Ottawa study guide](configs/studies/ottawa/README.md). Earlier results below are
unchanged and use their original formulations.

**The coupled Ontario–Ottawa pilot now runs all four approaches:** the rule
curve, external-flow MPC, nested bilevel MPC and PPO. The earlier uncoupled
fixed-cap benchmark is retained below with its original score definition.

### Baskatong: approximate physical release-response scenarios

Saved under `results/ottawa/baskatong_linear_scenario_final_v2/`: three assumed
linear-storage scales × two separate discharge interpretations, each with replay
and retain-then-release trajectories. This is a **research scenario model**, not
calibrated Baskatong operations. It does not replace the strict reconstruction.

For the primary 02LG006 proxy, reducing April–May release by 40% retains
**247.271 million m³**, subsequently repaid under the assumed outlet limits.
Maximum modeled level increases are **2.198 / 1.099 / 0.549 m** for smaller,
inventory-order and larger assumed volume scales. The smallest reaches
**225.548 m**, exceeding the independently assumed 225 m bound by **0.548 m**:
that prescribed experiment is **infeasible**. The other two satisfy the declared
bounds. No storage clipping repairs the violation.

The 02LG007 sensitivity retains/re-releases **250.530 million m³**; maximum
increases are **2.227 / 1.113 / 0.557 m**, with the same small-scale infeasibility.
Both interpretations recover their replay after repayment, and modeled total
releases match within the declared 0.001 m³ accounting tolerance. Agreement is a
structural consequence of the assumed balance, not independent physical validation.

The area scales assume volume changes of **1.5245 / 3.049 / 6.098 billion m³**
over the observed level range. The listed inventory is merely a scale anchor.
Daily means are placed at model-day centers; interval flow is the adjacent-day
average. Each case constructs **effective net input** once and holds it fixed
across releases. Negative-input counts for 02LG006 are **1 / 29 / 129 out of 365**,
with minima **−33.0 / −332.5 / −931.5 m³/s**. All remain negative after excluding
source-flag-affected intervals. This strong scale sensitivity cautions against
interpreting effective input as measured/natural inflow.

29 intervals are affected by the 28 Partial Day dates under the primary proxy;
reported values remain uncertain inputs. The observed-record alternative is not
independent validation. The original observations, provenance and strict workflow
are preserved. No Lake St. Louis prediction or controller changes occur.
See [the assumptions and offline command](configs/studies/ottawa/README.md).

### Baskatong: actual one-year observation pilot

Saved under `results/ottawa/baskatong_1970_observed_v1/`: raw official ECCC API
responses, date-labelled `observations.csv`, pipeline audit, plots and hashes.
Baskatong 02LG007 level and downstream Mercier 02LG006 discharge overlap on all
**365 dates in 1970**, with no missing dates/values. **28 discharge values carry
Partial Day symbols**, leaving 337 unflagged paired dates. Source date coverage
is not proof of complete within-day measurement coverage.

Reported daily mean level spans **209.800–223.350 m in the station's arbitrary
datum**; downstream discharge spans **7.36–702 m³/s**. The plot shows winter
drawdown followed by spring refill. Normalized **water-level position** is a
rescaling of this observed annual level range, not a storage fraction. No modern
geodetic operating bounds are superimposed on that unaligned datum.

02LG007's own discharge differs from the downstream Mercier series on **231
dates**, up to **92.7 m³/s**. Both remain separate. No volumetric inflow is
computed: mean levels cannot stand in for midnight boundaries, daily clock and
all-outlet equivalence need confirmation, and no storage relationship is supplied.
The additional January 1, 1971 mean is context, not a closing midnight observation.
The earlier assessment and all controller experiments are preserved. See the
[reproduction commands](configs/studies/ottawa/README.md).

### 1970 discharge comparison: outlet interpretation

The read-only comparison in `results/ottawa/baskatong_discharge_comparison_v2/`
provides both original series, daily differences, source flags, an April–June
view and a sourced station-description comparison. Mean absolute difference is
**10.049 m³/s**, maximum **92.7 m³/s on September 26**; the signed annual mean
of −0.264 m³/s conceals cancellation. All **28 Partial Day pairs agree exactly**;
all **231 discrepancies occur on unflagged dates**. Excluding flags does not
resolve the disagreement.

The official station description supports **02LG006 as a Mercier downstream-flow
proxy**. Neither record has verified completeness for all Baskatong outlets.
02LG007's reservoir name does not establish how its discharge was derived or
whether it is outlet flow. No transfer, natural-inflow or corrected-average
interpretation is inferred. Current official Partial Day guidance concerns
within-day gaps; precise 1970 flag provenance and averaging clocks remain open.
The original pilot files are verified unchanged. Reproduce via
`python -m reservoir_control.systems.ottawa.discharge_comparison`; see
[the guide](configs/studies/ottawa/README.md).

### Baskatong historical reconstruction: assessment only

The daily source/measurement assessment is saved in
`results/ottawa/baskatong_assessment_v2/report.md`. It found no suitably licensed,
aligned 2016–2019 level/total-outlet pair: the inspected ECCC Baskatong archive
ends before the target, owner open products describe rolling recent coverage,
and ORRPB's observation service restricts reproduction. Other leads have unresolved
access, coverage or measurement definitions; their absence is not claimed.

Baskatong's listed inventory volume is not assumed to be usable storage between
its operating elevations. Datum realization, historical day boundaries, a storage
curve and total outlet coverage (including the separate Castor structure) remain
unresolved. A tested ingestion/accounting interface is implemented; **historical
reconstruction and operational calibration are not completed**. No Ontario
controller, synthetic coupling parameter or previous experiment was changed.
See [the evidence and input guide](configs/studies/ottawa/README.md).

### Coupled research pilot: current overview

Saved run: `results/loslr/coupled_v1_information_matched/`. Each controller covers complete
1995, 2017 and 2019 annual episodes from matched Ontario and Ottawa starts.
Five PPO seeds each trained for **4,096 decisions on 1901–1980 only** and were
then evaluated using their saved final endpoint. There are **24 annual records**
and **1,152 quarter-month trajectory rows**. Reserved test years were not run.

Mean annual operating cost, **lower is better**:

- **Rule curve: 1.694.** The common Plan 2014 release policy in the responsive-follower simulator.
- **External-flow MPC: 1.995.** Plans against a neutral-announcement Ottawa forecast.
- **Bilevel MPC: 1.838.** Recomputes Ottawa's best response for each candidate request schedule.
- **PPO: 1.690**, averaged across five seeds. Individual seed means span **1.591–1.760**.

Anticipating the follower lowers cost by **7.9% relative to external-flow MPC**
in these three cases. Both MPC variants still cost more than the rule curve.
PPO's mean is close to the rule curve; this short pipeline pilot does not establish
convergence, statistical superiority or generalization. No controller was promoted.

The flood tradeoff remains visible. In **2017**, external/bilevel MPC Ontario
exceedance totals are **7.284 / 7.120 m·periods**, and downstream totals are
**0.40 / 0.38 m·periods**. In **2019**, upstream totals improve from **5.074 to
4.514 m·periods**, while downstream totals increase from **0.45 to 0.54 m·periods**.
Both MPC variants use forecast-cap recovery on **20 periods in 2017** and
**17 in 2019**. Forecast feasibility is not a realized safety guarantee.

This study changes the formulation deliberately. It uses the sourced Plan 2014
**tiered F-limit targets**, squared upstream/downstream exposure, low-water and
release-change costs, plus annual terminal storage value. PPO reward is exactly
negative operating cost. These costs cannot be compared numerically with the
older fixed-cap flood-exposure score. All controllers use the same native Bv7
execution projection; MPC additionally searches for forecast stage-cap feasibility.
PPO has no separate hard stage-safety certificate. Physical severities and follower
service/storage outcomes are retained separately from the scalar objective.

The follower is a strictly convex storage/release QP responding to Ontario's
announced **requests**, which can differ from Bv7-executed releases. Nested finite
leader search is a reference bilevel controller, not a global optimum. Independent
KKT residuals remained below **7.49 × 10⁻⁵**; maximum Ottawa mass-balance error was
below **7.2 × 10⁻⁷ m³**. All physical follower safety-adjustment counts were zero.
The training logs contain finite actor/critic gradients and **425 annual task ends**;
local KKT response derivatives also pass a finite-difference check. The full project
suite passes **330 tests, with one skip**.

**Physical calibration remains open.** The controlled share of historical SLON,
6 billion m³ useful capacity, half-full initial storage, outlet bounds and zero-lag
routing are declared research assumptions. Historical SLON is already regulated;
this synthetic decomposition is not a reconstruction of natural Ottawa inflows.
The IJC source establishes the reference IGLD 1985 F-tier convention, but it does
not validate every historical gauge record or make 75.50 m an absolute Ontario
operating limit. See the [protocol and evidence](configs/studies/loslr/README.md).

Next validation: capacity/flow-split/routing sensitivity, stronger leader search,
longer predeclared PPO training and continuous multi-year evaluation. Native
performance indicators remain available for the original reference; extending
all seven to coupled trajectories is still open.

### Earlier uncoupled simulation and benchmark checks

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


The paired `comparison.csv` contains requests, executed releases, levels, forecast
inputs, every predicted stage margin and fallback status. `summary.json` aggregates
metrics and records the study settings and source hashes.

## Lake Como

In the improved Como simulator, history MPC has annual cost **517.51**, compared
with **617.91** for mapped PPO trained for 400k decisions per seed. PPO costs
approximately **19.4% more**, with slightly lower flood exposure and substantially
more low-water exposure. These are development results over **1988–1997**, using
training/fitting years **1946–1987**. Final **1998–2010** testing remains reserved.

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
