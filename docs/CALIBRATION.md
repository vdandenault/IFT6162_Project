# Calibration evidence and identification limits

The source audit and daily balance fit are complete. **Full operational calibration
is not complete and cannot be identified from the available daily records.** The
current evidence profile explicitly rejects that promotion. No original simulator
parameter, benchmark, checkpoint or test outcome has been changed.

## What is now supported

**Dated daily flows are available.** Ranzi et al. (2020) publish reconstructed daily
inflows and outflows on PANGAEA. We downloaded and hashed the original files, with
attribution, then restricted numerical crossmatching/fitting to **1946–1987**.
Same-date matches are very strong: benchmark inflow versus published inflow has
correlation **0.999626** and MAE **2.022 m³/s**. The series are close, not identical;
we do not claim exact file provenance from correlation alone.

**The rating curve uses Malgrate levels.** Ranzi et al. (2021), journal pp. 15–16,
explicitly give the fully-open curve `Q = 33.37 × (2.50 + h_M)^2.015`, with `h_M`
from Malgrate. The historical official Malgrate zero is 197.37 m above sea level;
Fortilizio has a different reference and location. Fortilizio operating limits must
not simply be transplanted to Malgrate as a 2 cm offset. The paper notes revised
survey elevations too; the historical reference is not a claim about a modern survey.

**Training-date level anchors match Malgrate.** The operator's 2013 manual reports
2.64 m on 21 July 1987 and a minimum rounded to −0.71 m on 13 February 1984. The CSV
has **2.64** and **−0.705 m** on those ordinal-as-January-1 dates. These corroborate
the date anchor and gauge interpretation; they do not authenticate every row's clock.
The methods paper describes Malgrate readings as normally 08:00 after 1946. The
precise averaging interval/time standard of each Gym4ReaL CSV row remains unverified.

## Important target-semantics finding

The column labeled `demand` is exceptionally close to independently published
historical outlet flow: **correlation 0.999368**, MAE **2.057 m³/s** over 15,340
training-year daily values. Two documented outlet-flow extremes match the target
exactly: **18 m³/s on 4 April 1953** and **917.7 m³/s on 20 July 1987**.

In addition, all **10,927 within-year transitions in 1946–1975** satisfy, to floating
point precision:

```text
benchmark_inflow[t] − benchmark_target[t]
    = 145,152,000 × (level[t+1] − level[t]) / 86,400
```

This strongly supports an **outflow-related historical target** and a reconstructed
water-balance origin. It does not establish independently measured irrigation
consumption. Keep the CSV and the published benchmark reward unchanged, but describe
shortages as deficits against the benchmark target until demand construction is
verified. A future irrigation task needs its own supported allocation/demand model.

The external PANGAEA series are also reconstructions: the methods paper describes
post-1946 outlet reconstruction using Lavello and the Garlate control volume, and
inflow reconstruction through continuity. Agreement with those records is therefore
not independent validation of the lake geometry or controller actions.

## What was fitted and checked

The primary estimate is the median per-transition balance coefficient, using changes
of at least 5 mm. Fit years: **1946–1975**. Checking years: **1976–1987**, still inside
the project training split and inspected during source exploration. This is not an
untouched final test or a preregistered accuracy claim.

The fitted effective area is **145,152,000 m²**. It represents the historical dataset's
balance coefficient, not a new bathymetric survey. Compare it with the pinned
**145,900,000 m²** and the methods paper's **145,033,000 m²** at elevation 199 m.
Do not silently replace one with another as if they were identical estimands.

On the later checking years, one-step level MAE using the benchmark target as a
release proxy is:

- Pinned coefficient: **0.1823 mm**.
- Published geometry reference: **0.0658 mm**.
- Fitted data-balance coefficient: **0.0446 mm**.

The perfect early-period fit is a sign of a data-generation identity, **not**
extraordinary independent calibration accuracy. Prescribed-proxy-release annual
replay and all residuals are saved; the checking replay RMSE averages **3.64 mm**
with the fitted coefficient versus **5.86 mm** pinned. No fitted coefficient is
promoted into frozen controller experiments.

## A structural validity limit

There are **112 training transitions touching levels below −0.5 m**, all with a
positive benchmark outlet-flow proxy, down to −0.705 m. The operator documents
experimental minima below its normal operating reference. The pinned model instead
forces release to zero at/below −0.5 m. An operating lower reference is not the
geological bottom or a physical zero-flow elevation; the published fully-open curve
uses −2.5 m as its zero-flow parameter.

Consequently, surface-area fitting cannot make the pinned model a historically
valid discharge model for all low-water operating regimes. Its piecewise operating
rules and true hydraulic capacity need to be separated in a future justified
hydraulic-model revision. The current experiment is still a bounded research model.

## What cannot be calibrated from these records

The user confirmed that no additional operator records are available. The reviewed
sources do not identify the numerical hourly ramp, river-routing time or downstream
threshold used in our synthetic examples. Daily average discharge cannot reveal
all intraday command changes; many hourly trajectories share the same daily mean.
A reconstructed single-reach daily outflow is not a synchronized pair of downstream
gauge hydrographs suitable for identifying a routing response.

The evidence profile therefore leaves these parameters **unidentified/null**:

- Hourly gate/flow-command ramp: need timed gate openings or commands, discharge,
  head/tailwater, gate availability and the applicable control rules.
- River routing: need synchronized upstream/downstream gauge series at named
  reaches, gauge/datums, diversion/return flows and suitable event records.
- Downstream reference: need a documented threshold/capacity for a named reach and
  operating regime; 300 m³/s is not established by the reviewed sources.
- Actual irrigation demand: need dated allocation/request or consumption records,
  rather than assuming historical outlet flow measures crop demand.

The **5 m³/s/hour, 6-hour, 300 m³/s** settings remain explicitly labeled sensitivity
experiments. They are not enabled by this evidence profile or renamed as fitted
parameters. `require_operational_calibration(profile)` rejects the incomplete
profile; it is an internal readiness check, not an independent certification service.
No messages or data requests have been sent to anyone.

## Public subdaily measurements: separate 2021 audit

Public measurements were located after the daily-source audit. ARPA Lombardia's
open-data archive supplies ten-minute records for **Valmadrera Malgrate level
(sensor 8386), Olginate S. Maria Lavello level (8383), and Lavello discharge
(10805)**. A bounded 2021 acquisition is saved separately in
`results/simulator/modern_measurements_2021_v2/`; it does not fit simulator parameters
or evaluate controllers. It uses no reserved benchmark years.

There are 52,465 Malgrate observations (95 missing ten-minute slots), and 52,560
observations for each Lavello series. Their timestamps yield 52,465 synchronized
triplets before quality filtering. Original responses, metadata, station registry,
quality-code attachment, URLs and SHA-256 hashes are retained. No gaps are filled.

The [ARPA documentation](https://www.dati.lombardia.it/stories/s/auv9-c2sj/)
specifies **fixed UTC+1, end-of-measurement-interval** timestamps. This supports
alignment of these modern series; it does not authenticate the older Gym4ReaL rows.
Metadata describe levels in cm relative to a conventional gauge zero, and discharge
in m³/s, measured or calculated. Station `quota` is not a verified gauge-zero
elevation. The dataset titled "level from 2021" has contradictory older wording in
its description; the downloaded rows themselves are explicitly restricted to 2021.
Registry start dates are later than the archive observations, so they cannot be
treated as authenticated installation dates for this historical sample.

The publisher's attached quality legend distinguishes code **1** (good recorded
data), **2** (good processed or reconstructed data), **70** (estimated) and **82**
(linear interpolation). The audit retains all codes and exports a separate subset
requiring code 1 at all three sensors (**50,310 synchronized triplets**). A good recorded discharge code alone does
not prove direct velocity/discharge measurement: the dataset permits calculated
discharge. Of 258 distinct Lavello levels in that subset, 257 have a single
discharge value. This almost single-valued relationship is
consistent with a rating-derived product, not an independent hydraulic experiment.

These records enable modern observational checks, but **do not close operational
calibration**. Malgrate is a lake-level sensor, not an independent upstream discharge
series. Lavello level and discharge at the same station are not a pair of river
gauges. Fitting a lag between lake level and downstream flow would mix storage,
gate decisions and routing. Observed discharge increments do not identify the
maximum gate-command ramp. No independent inflow, gate openings, verified rating
validity periods, diversions or actual irrigation demand accompany this download.
The earlier evidence profile remains valid and is not promoted.

Reproduce this modern acquisition into a fresh directory:

```bash
uv run python scripts/analysis/operational_measurements.py \
  --output results/simulator/new_modern_measurements
```

The report records the common code-1 sample count and remaining identification
gaps. Public source: [station registry](https://www.dati.lombardia.it/d/nf78-nj6b),
[levels](https://www.dati.lombardia.it/d/3e8b-w7ay), and
[discharge](https://www.dati.lombardia.it/d/ujnt-2rtf).
Attribution: ARPA Lombardia — Servizio IdroNivoMeteoClima, via Regione Lombardia
Open Data. No data-request forms or messages were submitted.

## Outlet ablation: implemented findings

The opt-in implementation is `reservoir_control.systems.lake_como.simulation.outlet`. It wraps the existing
operational model without changing the fork, benchmark or previously hashed
modules. `OutletSpec(capacity='legacy')` reproduces a full annual benchmark rollout
exactly. The default candidate changes only maximum outlet capacity, keeping the
same area, observations, nominal actions, objective and operating minimum rules.
Storage reference is now an independent parameter; changing it changes relative
volume bookkeeping, not the discharge at a given level.

The first bounded study is saved in `results/simulator/outlet_ablation_v2/`.
The earlier v1 attempt stopped on an upstream-file versus chronological-split
lookup mismatch; its runner snapshot is retained. V2 filters both upstream files
to project training years before analysis. No project final test years are scored.

- Across 15,298 within-year training transitions, the number with the historical
  outlet proxy above point-level capacity falls from **336 to 144**. Among the
  **112 transitions touching levels below −0.5 m**, it falls from **112 to 1**.
  These are conditional consistency diagnostics with reconstructed flows and
  unresolved timing, not independent discharge validation.
- Integration tests conserve water (maximum logged hourly residual below
  **3 × 10⁻⁸ m³**) and finer integration reduces the sampled end-level errors.
  At the low-water test point −0.6 m, candidate one-hour integration differs from
  the 1536-substep reference by **0.0863 mm** after a day. The retained high-water
  forced-opening policy still has a threshold transition; it is not calibrated by
  this work. Area sensitivity is reported separately.
- A causal feedback rule completes **1946–1947** identically under both variants:
  its trajectory stays above −0.4 m, outside the changed capacity regime.
- Constant request **100 m³/s** completes both years under the legacy rule but
  the candidate crosses the declared lower evaluation limit during **day 51**.
  The legacy capacity taper had provided implicit low-water protection. Removing
  it exposes the need to specify protection as an operating/controller decision.
  Partial returns are not compared with complete-run returns as performance scores.
- Fresh SAC's declared 96-step smoke stops at that limit on its **24th attempted
  agent transition**, after 30 spin-up days and 23 completed warmup transitions.
  With a 32-transition warmup it performs **zero learner updates**; no trained
  policy is produced. A separate integration test performs four finite-gradient
  SAC updates on 16 actual prescribed-action candidate transitions. That checks
  learner plumbing, not successful free-action training.

The numerical evaluation envelope **[−0.705, 2.64] m** is declared for this study
using historical extrema. It is not a source-certified hydraulic validity range
or a physical lake-bottom definition. The candidate raises `OutletDomainError`
outside it; an invalid model-domain event is not passed to replay as an ordinary
terminal transition. Keep this variant experimental until low-water operations
and its allowable use range are defined. Do not widen the range just to complete
training or fit the capacity curve to every regulated outflow observation.

```python
from reservoir_control.systems.lake_como.simulation.outlet import make_outlet_env, OutletSpec

env = make_outlet_env(years=(1946,), outlet=OutletSpec())
```

Reproduce into a fresh output directory:

```bash
uv run --group reproduction python scripts/analysis/outlet_hydraulics.py \
  --output results/simulator/new_outlet_ablation
```

[Outlet PNG](../background/images/41_outlet_hydraulics.png) shows capacity, the
constant-release trajectory and historical low-water consistency. Background
assets and run artifacts remain ignored.

## Explicit low-water operating protection

`reservoir_control.systems.lake_como.simulation.low_water` adds an optional operating layer to the published
capacity candidate. It leaves that capacity curve unchanged. The rule recomputes
an operating release ceiling each hour:

```text
reserve_volume = max(0, area × (current_level − operating_floor))
operating_ceiling = reserve_volume / (reserve_hours × 3600)
actual_release = min(unprotected_actual_release, operating_ceiling)
```

The first declared settings are **floor −0.5 m and reserve horizon 24 hours**.
They are experimental design choices, not inferred operator parameters. The rule
uses current level only; it does not read current/future inflow or clip storage
after the transition. Relative storage below its datum remains meaningful.
Below the operating floor, the rule closes discretionary discharge; the lake
does not become physically incapable of discharge. Recomputing the ceiling hourly
produces gradual restriction, not a once-per-day allocation of all reserve water.

For an initial level at/above the floor, hourly integration, nonnegative inflow
and **zero evaporation**, the cap keeps the next level at/above the floor. The
implementation rejects supplied evaporation in this experiment. This conditional
property does not ensure flood safety, initial-state recovery, environmental
service or safety under unmodeled losses. It also permits levels between −0.5 and
the existing low-water penalty threshold −0.45 m.

Priority is explicit: reserve protection can override the existing minimum-flow
release rule. Environmental shortages, operating-minimum shortfalls, interventions
and curtailed volume are logged. The physical objective is unchanged and does not
add a new environmental-shortage weight; these metrics must therefore be examined
separately from return. There is no claim that this priority matches Lake Como's
actual operating obligations.

The predeclared study in `results/simulator/low_water_protection_v1/` compares
legacy, unprotected candidate and protected candidate under constant 100 m³/s and
causal feedback requests, in **1946–1947, 1966–1967 and 1986–1987**. These windows
are training-only chronological samples, not a full-split evaluation. All three
variants also receive the same **three 96-step fresh SAC smokes**, seeds 901–903,
with 32 warmup steps and 30 spin-up days. Nothing was tuned after seeing results.

Findings:

- The protected candidate avoids the low-water domain exits seen in unprotected
  constant-release runs. Its level remains above −0.5 m in the recorded runs.
  In 1986–1987 it nevertheless leaves the **upper** 2.64 m evaluation limit during
  day 565, at a trial level of **2.641576 m**. The guard is retained; no extrapolation
  is introduced. Five of six protected control windows complete, including all
  three feedback windows.
- All feedback trajectories and metrics remain identical across variants; the
  rule never intervenes for that controller in these windows.
- Against the unprotected candidate over the common first 50 days of 1946,
  protection reduces low-water severity by **25.351 m·hours**, but increases unmet
  benchmark target by **30.002 million m³**, worsening physical return by **4.650**.
  This is a matched-prefix comparison, not a complete two-year ranking.
- Against legacy over the complete 1946–1947 constant-release run, return changes
  from **−2684.497 to −2693.343** (worse). Unmet target rises by **0.840 million m³**,
  low-water severity by **7.762 m·hours**, and flood severity by **0.243 m·hours**.
  In 1966–1967 the protected return improves by only **0.347** versus legacy.
  These mixed results do not establish a general performance gain. The incomplete
  1986–1987 case is compared only over matched prefixes.
- Environmental shortages and operating-minimum shortfalls are **zero in these
  recorded controller windows**, but tests explicitly verify that the rule can
  create them in a sufficiently dry state. Zero observed shortfall is not a general
  environmental-flow guarantee.
- All **three protected SAC smokes finish with 64 finite learner updates each**;
  all three legacy smokes also finish. Unprotected seed 901 stops during warmup
  after 23 completed agent transitions; unprotected 902/903 finish. These are
  training-integration results, not learned-policy performance evaluations.

This improves separation of hydraulic capacity and operating decisions, and
removes the tested low-water failure of the new candidate. It has not demonstrated
better control than legacy or calibrated real operations. Before longer training,
review the operating-floor/environmental-flow priority and the retained high-water
rule/domain limit. Do not tune these choices to eliminate observed failures and
then describe that same sample as independent validation.

```python
from reservoir_control.systems.lake_como.simulation.low_water import make_protected_outlet_env, LowWaterRule

env = make_protected_outlet_env(years=(1946,), rule=LowWaterRule())
```

```bash
uv run --group reproduction python scripts/analysis/low_water_protection.py \
  --output results/simulator/new_low_water_protection
```

[Protection PNG](../background/images/42_low_water_protection.png) illustrates the
operating ceiling, low-water trajectories and short SAC integration checks.

## High-water diagnosis: release timing and model limits

The fixed-model replay in `results/simulator/high_water_diagnosis_v1/` reproduces
the protected constant-release exit exactly: **1987 model day 200, hour 24**.
The step begins at **2.622830 m**; with inflow **1657.2 m³/s** and discharge
**897.468 m³/s**, its rejected trial level is **2.641576 m**. Discharge already
equals the full-open capacity. Trial records are flagged and stopped; the curve
is never evaluated outside the declared envelope.

The model has three different thresholds: **1.24 m** starts the flood penalty,
**1.25 m** triggers the inherited forced-full-open operating rule, and **2.64 m**
is the declared research evaluation limit. None is interchangeable. The published
source supports a full-open curve; the reviewed paragraph/figure does not establish
2.64 m as a physical no-flow or maximum-lake-level boundary. The figure's displayed
range alone does not certify extrapolation or a modern high-flow calibration.
See Ranzi et al. (2021), printed pp. 15–16, Figure 2 and accompanying text.

**Action authority at the peak.** Above 1.25 m the inherited minimum and maximum
release are both the full-open curve. Requests 0, 100 and 491.61 m³/s therefore
produce identical discharge there, including actual flow above the nominal action
maximum. Below that threshold the full-open capacity is at most **478.662 m³/s**,
already reachable by the nominal maximum request. Increasing the action bound
would not unlock additional capacity in this model. Across the final 48 hours
from the two-day-before-exit snapshot, the rule remains forced open throughout;
all tested requests have exactly the same trajectory to the exit. From that same
snapshot, changing only nominal requests cannot alter the outcome under these
fixed rules. This is a conditional model statement, not a real-operator claim.

**Earlier intervention gives feasible witnesses.** Branches start from identical
constant-controller snapshots, preserving storage and demand history. Their start
times are chosen in hindsight; policies use only current level/history or a fixed
maximum request, with unchanged constraints and costs.

- Switching to maximum admissible release **four days before the rejected trial**
  keeps the replay in range through seven days after the baseline exit. Maximum
  level is **2.620791 m**. Switching two days before still exits.
- Switching to causal feedback **eight days before the rejected trial** stays in
  range, peaking at **2.522531 m**. A four-day feedback switch still exits, later.
- From the original 1986 initial state, feedback and maximum-release policies
  complete both years, peaking at **2.276578 and 2.203561 m**, respectively. Both
  still flood above 1.24 m. Maximum release has more target deficit and low-water
  severity than feedback, and worse total physical cost. It is a feasibility
  diagnostic, not a recommended operating policy or a proven optimal controller.

Branch filenames use days before the **start of the failure day**, while the
failure is at that day's end. Thus `branch_3d_*` has a four-day lead to the rejected
trial, and `branch_7d_*` has an eight-day lead. Reported leads above use the actual
elapsed times. The optimistic volume/capacity bound is inconclusive for the tested
anchors; it is not used to claim global unavoidable flooding.

**Numerical resolution affects timing, not the existence of this exit.** From the
same seven-day-before-failure-day snapshot, 1, 2, 4, 8, 16 and 32 integration
substeps/hour all exit above 2.64 m. At 32 substeps/hour the exit occurs about
**3.156 hours later** than at one step/hour. Daily requests and hourly reserve
updates are held fixed; only the state-dependent hydraulic integration is refined.
The operating minimum jumps at 1.25 m for a small requested release, so this is
also a threshold-sensitive numerical model. No costs, coefficients, operating
priorities or range limits were adjusted.

The evidence points to **earlier release timing** as a way to avoid this particular
range exit, while peak-stage authority is limited by the existing full-open curve
and forced-opening rule. It does not establish elimination of flooding, physical
calibration, or that operational forecasters could have supplied the needed warning.
The subsequent [causal protected-MPC screen](SIMULATOR.md#protected-mpc-with-causal-forecasts)
tests anticipation at the same costs: all MPC arms stay in range, but feedback
has lower total cost and the forecasts underpredict the surge. Operating priorities and
the curve's justified high-level use range remain open before broader training.

```bash
uv run --group reproduction python scripts/analysis/high_water_diagnosis.py \
  --output results/simulator/new_high_water_diagnosis
```

[Flood diagnosis PNG](../background/images/43_high_water_diagnosis.png) shows the
counterfactual timing, outlet saturation and numerical refinement. All replay
years are in the training split; no learner is updated or final test year scored.

## Next implementation: lake balance and outlet hydraulics

The outlet variant, explicit low-water operating experiment and short free-action
smokes, high-water diagnosis and causal MPC screen are implemented above. Next
review operating priorities and curve use range, and diagnose target forecasts
and schedule search before a longer matched training study. Use the sequence below
as acceptance criteria for subsequent revisions.
The immediate goal is to remove a demonstrated structural inconsistency and expose
assumptions, not to claim full operational calibration. Implement outside the fork
and preserve source-hashed studies and the original benchmark.

1. **Separate definitions.** Represent the storage reference elevation independently
   of preferred operating levels, reward thresholds and the hydraulic curve's
   zero-flow parameter. Relative storage may be negative below its reference;
   this must not imply an empty lake or force zero release. Document a supported
   level range for each relationship and explicitly flag excursions.
2. **Expose the outlet-capacity model.** Keep the legacy piecewise curve as a
   comparison. Add an explicit candidate using the documented Malgrate fully-open
   curve, within its documented applicability; do not silently extrapolate it or
   treat every observed regulated flow as a fully-open sample. Distinguish this
   capacity from the controller's request range. Minimum environmental release and
   any forced opening above an operating level are separate operating assumptions,
   not consequences of the capacity formula. Gate-position hydraulics remain open.
3. **Keep balance assumptions visible.** Use the existing surface coefficient as
   the reference and compare the published geometry coefficient separately from
   the fitted dataset coefficient. Use consistent SI units, interval-mean forcings
   and actual time increments. Omitted precipitation/evaporation/other exchanges
   remain explicit assumptions; balance-derived inflow cannot independently
   validate the same level/outflow observations.
4. **Run a small physical validation study.** First isolate the outlet change with
   area and operating rules held fixed, then test area sensitivity separately.
   Check finite nonnegative capacities, continuity and behavior near −0.5/−0.4 m,
   capacity/request distinction, environmental-flow feasibility, conservation and
   integration-step convergence. Recheck the 112 historical low-level transitions
   and count outflow proxies above the proposed capacity; daily mean outlet flow
   versus a single level sample is only a consistency diagnostic, not a definitive
   hydraulic-capacity violation. Report changed versus unchanged regimes and
   remaining high-level forced-opening assumptions.
5. **Then reconnect controllers.** Run constant-release and feedback rollouts, then
   a short fresh RL integration smoke with ramping disabled, routing bypassed and
   no downstream threshold. Retain the same observation/action/reward conventions
   for the first outlet ablation. Any later change to target semantics, allocation
   or rewards gets its own experiment. Only after physical checks pass should a
   new matched training study be declared; final test years remain untouched.

The first deliverable is the outlet/balance variant plus an old-versus-candidate
diagnostic, with no learned-controller ranking. The existing historical data and
published curve support this work. Independent flow measurements, gauge/rating
metadata and gate records remain requirements for stronger physical validation.
The modern 2021 records support observational diagnostics but cannot supply an
independent lake inflow or authenticate gate opening by themselves.

## Reproduce and inspect

```bash
uv run python -m reservoir_control.systems.lake_como.simulation.calibration --fetch \
  --output results/simulator/new_calibration_evidence
```

Existing source downloads are reused and hashed; existing output directories are
protected. The active run is `results/simulator/calibration_evidence_v2/`:
`profile.json`, `readiness.json`, dated crosswalks, balance fits/checks and annual
proxy-release replay. The initial source audit is preserved with its code snapshot.
A draft operator-data request is stored locally with the results for review only.

[Calibration PNG](../background/images/40_calibration_evidence.png) separates source
crossmatching, the fitted balance identity and intraday non-identifiability.

Primary sources and attribution:

- Ranzi, Roberto; Michailidi, Eleni Maria; Tomirotti, Massimo; Crespi, Alice;
  Brunetti, Michele; Maugeri, Maurizio (2020), [daily inflows](https://doi.org/10.1594/PANGAEA.919886)
  and [daily outflows](https://doi.org/10.1594/PANGAEA.919889), PANGAEA, CC BY 4.0.
- Ranzi et al. (2021), [Hydrometry of the Adda River at Lecco from 1845 to 2016](https://iris.unibs.it/retrieve/13f3dac1-dd91-411a-9473-b16b4e69f0dc/2021_Ranzi-et-al%40L_Acqua_Adda_HR%2BFrontespizio_compressed.pdf),
  *L'Acqua* 1/2021, printed pp. 13–20, especially pp. 15–16.
- Consorzio dell'Adda (2013), [regulation manual](https://www.addaconsorzio.it/wp-content/uploads/2021/07/dispensa-consorzio-adda.pdf),
  p. 6 for historical gauge/flow anchors. These are historical documentation, not a
  verification of current legal limits.
