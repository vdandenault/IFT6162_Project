## Standalone Ottawa follower with hard constraints

The separately selectable `systems/ottawa/hard_follower.py` implements the Step 2
QP for a **supplied executable Ontario schedule**. It leaves the legacy soft
follower and coupled Ontario experiments unchanged. The pure entry point is
`solve_hard_follower(HardFollowerProblem(...))`; successful results include the
first action, full plan, exact normalized objective and SI/KKT residuals.
Invalid inputs raise `InvalidFollowerInput`; infeasible or failed solves return
no action. Storage variables are eliminated using the exact water balance.

Synthetic assumptions are frozen in
`configs/studies/ottawa/hard_follower_v1.json`. No parameters are calibrated to
Baskatong or the entire Ottawa system. Demonstrations use 657,000-second periods,
a four-period horizon, explicit transit queues and first-action-only execution.

```sh
uv run --group reproduction python -m reservoir_control.systems.ottawa.hard_follower_study --forecast perfect --output results/ottawa/hard_follower_perfect_rerun
uv run --group reproduction python -m reservoir_control.systems.ottawa.hard_follower_study --forecast noisy --output results/ottawa/hard_follower_noisy_rerun
```

Use fresh output directories. Each contains `report.md`, `summary.csv`,
`paired_comparison.csv`, per-rollout traces and PNGs, full horizon `plans.json`,
configuration and source hashes. Noisy forecasts use past-only persistence plus
seeded perturbations; realized forcing is separate. Predicted constraints do not
guarantee realized safety or recursive feasibility. Terminal transit is retained.

# Baskatong historical water-balance pilot

## Actual one-year observation pilot: 1970

Actual records are now available for Baskatong (level station **02LG007**) and
Gatineau downstream of Mercier (discharge station **02LG006**) from ECCC's
supported historical GeoMet OGC API. The fixed selection is in
`baskatong_1970.json`; the older 2016–2019 assessment/configuration is preserved.

```bash
uv run python -m reservoir_control.systems.ottawa.historical_pilot \
  --output results/ottawa/baskatong_1970_review
```

This acquires the raw daily records and station/collection metadata, runs the
existing processing functions, and writes `observations.csv`, `pipeline_daily.csv`,
PNG plots, source symbols, metadata, a report and provenance hashes. ECCC permits
copying/distribution with attribution under its linked Wateroffice terms; no
restricted ORRPB observations are downloaded. For offline source-identical replay:

```bash
uv run python -m reservoir_control.systems.ottawa.historical_pilot \
  --raw-dir results/ottawa/baskatong_1970_observed_v1/raw \
  --output results/ottawa/baskatong_1970_reproduced
```

The saved run has **365 paired dates**, no missing values, but **28 downstream
flow values are flagged Partial Day**: only 337 paired dates are unflagged.
The level datum is `ARBITRAIRE`. Published levels are daily means, and the DATE
field does not establish midnight instants or the historical clock. Normalized
**water-level position** uses the observed 1970 level range (209.800–223.350 m),
not storage capacity or geodetic operating bounds. `observations.csv` is the
primary date-labelled dataset. UTC anchors/86400-second scaffolding in the
pipeline audit are explicitly unverified, **not measured physical intervals**.

The reservoir station's own flow and the downstream Mercier record differ on
231 dates, up to 92.7 m³/s. Both are preserved, not substituted or interpreted
as a known transfer. The downstream gauge's coverage of all Baskatong outlets
remains unverified. No volumetric storage or inflow is estimated. The following
assessment remains valid for the original 2016–2019 target.

## Separate approximate reservoir scenario model

```bash
uv run python -m reservoir_control.systems.ottawa.baskatong_scenario \
  --pilot results/ottawa/baskatong_1970_observed_v1 \
  --output results/ottawa/baskatong_scenario_review
```

This is **not the strict reconstruction command**. It explicitly assumes daily
mean levels occur at abstract day centers, nominal 86,400-second center-to-center
intervals and the average of adjacent daily mean flow proxies. The following-date
context mean is used; missing endpoints remain missing, with no interpolation or
reset of unknown simulated states. It does not claim real historical clock alignment.

Configuration: `baskatong_1970_scenario.json`. Three positive effective-area scales
are derived from assumed changes of **1.5245, 3.049 and 6.098 billion m³** across
the observed 1970 mean-level range. These give approximately **112.509, 225.018
and 450.037 million m²**. The 3.049-billion-m³ inventory is only an order-of-magnitude
scenario anchor, not verified usable volume. Set `effective_area_m2` explicitly
in a copied config to use a different area. Relative storage uses an assumed
200 m origin, not total reservoir volume. Experimental **208–225 m** levels and
**0–800 m³/s** outlet bounds are independent assumptions in the historical datum,
not actual operating limits. No modern geodetic elevations are applied.

Each area/proxy pair reconstructs its own **effective net input**, then holds it
fixed across historical-proxy replay and a prescribed retain-then-release schedule.
Inputs from 02LG006 and 02LG007 are separate interpretations, never blended or
considered independent validation. The full trajectories include source-flagged
reported values as uncertain inputs. Adjacent flags propagate; negative inputs
remain negative. Additional interval statistics exclude flagged intervals, without
pretending that later simulated states are independent of earlier uncertain data.

Default perturbation: reduce April–May release by 40%, then release the retained
volume during August–September at no more than 100 m³/s additional flow, respecting
outlet bounds. Timing/rates/bounds are configurable. It is an offline prescribed
experiment, **not a forecast or optimized controller**. Storage is never clipped;
bounds violations are saved and make the experiment infeasible under its assumptions.
Historical replay is an accounting identity, not physical model validation.

Saved final run: `results/ottawa/baskatong_linear_scenario_final_v2/`, containing
configuration/provenance, six resolved cases, interval inputs, twelve trajectories,
release-volume ledgers, negative-input/quality diagnostics and PNGs. No Lake St.
Louis prediction, Ontario change, network model or training is performed. Use a
fresh output directory for another run, optionally `--config /path/to/new.json`.
The original observations and strict reconstruction workflow remain unchanged.

## Compare the two 1970 discharge records

```bash
uv run python -m reservoir_control.systems.ottawa.discharge_comparison \
  --pilot results/ottawa/baskatong_1970_observed_v1 \
  --output results/ottawa/baskatong_discharge_comparison_review
```

The read-only comparison verifies the saved pilot checksums and writes an annual
overlay, signed differences with source flags, April–June focus, station metadata
table and a sourced report. The original dataset is checked unchanged after the run.
The saved comparison is `results/ottawa/baskatong_discharge_comparison_v2/`.

All 28 source-flagged pairs are identical; all 231 discrepancies occur on
unflagged pairs. Mean absolute difference is 10.049 m³/s, maximum 92.7 m³/s.
The current official Partial Day definition concerns incomplete within-day
coverage, not a missing daily value or a known correction factor. Its precise
1970 coding history is not established. 02LG006 is the better-supported
**Mercier downstream-flow proxy**; neither series is verified as the complete
Baskatong outlet balance. Differences are not evidence of physical transfers,
and the series are never averaged or shifted to force agreement.

## Original assessment and generic daily ingestion

This milestone does not change the synthetic coupled study, Ontario controllers
or RL training. Metadata in `baskatong.json` records the source assessment as of
2026-11-15. Inventory volume is **not** silently used as active storage.

Reproduce the source/coverage assessment, without downloading observations:

```bash
uv run python -m reservoir_control.systems.ottawa.reconstruction \
  --output results/ottawa/baskatong_assessment_review
```

No suitably licensed, aligned 2016–2019 level/outlet pair was obtained. This
command writes a short `report.md`, `reservoir_metadata.json`, `source_coverage.csv`,
`summary.json` with hashes, and header-only input templates. It does not fabricate
a cleaned historical series or claim that the owner has no records. ORRPB
observations were not downloaded because its service restricts reproduction.
Official ECCC availability tables, owner open-product descriptions, public
structure inventories and an indexed disclosure lead are distinguished in metadata.
A 403 or unverified source is recorded rather than bypassed.

## Accepted inputs

Use a fresh output folder for every run. Keep owner-provided files outside version
control, and retain their original bytes separately from these normalized inputs.
The command copies permitted normalized inputs and their contract into `raw/` and
records SHA-256 hashes. It does not redistribute records or grant permission to
use a source. All generated outputs remain under ignored `results/`.

**Boundary levels CSV:**

```text
timestamp,level,unit,aggregation,statistic,quality_flag,source_id
```

- `timestamp`: offset-aware ISO 8601 instant. No naive dates. Convert a source's
  “D at 24:00” label to “D+1 at 00:00” only after confirming its clock and label
  convention. Never silently substitute a daily mean for an instantaneous boundary.
- `level`, `unit`: numeric observation or blank; accepted units `m`, `cm`, `ft`.
- `aggregation`: `instantaneous_boundary` for eligible levels; other values are
  retained as reported levels, flagged and excluded from storage/balance estimation.
- `statistic`: `observed`; medians, percentiles and extrema must be labelled
  differently. They are exported separately and never used as observations.
- `quality_flag`: preserved source flag, with usable flags explicitly declared
  in the contract. Unknown/bad flags remain visible and are not used in estimates.
- `source_id`: registered source in the contract.

**Mean total outlet discharge CSV:**

```text
interval_start,interval_end,mean_discharge,unit,aggregation,statistic,quality_flag,source_id
```

- Both instants must describe exactly one declared calendar day. Units are `m3/s`,
  `L/s` or `ft3/s`; aggregation must be `interval_mean` for balance estimation.
- `mean_discharge` is the mean of **all boundary outlet components**. Do not add
  spill twice if a total already includes it. Turbine-only data must be marked as
  incomplete via the contract; they cannot establish net reservoir inflow.
- The source's original units, flags and records remain in `raw/`.

Copy `baskatong_input_contract.example.json` to a local input directory and resolve
its deliberately unset fields. Set the source processing/copy permission only
when supported by the named terms or authorization. A licence for one Hydro-Québec
open product is not presumed to cover another disclosure or ORRPB data. Multiple
sources must each have permission, URLs and provenance. The contract must specify:

- an IANA daily timezone (for example, only if documented, `America/Toronto` or a
  fixed standard-time zone), day-boundary evidence, and timing verification;
- datum alignment evidence and datum verification;
- all-outlet definition and completeness verification;
- actual owner quality flags accepted for calculations.

The example is intentionally **not a runnable historical-data assertion**. If
clock, datum or outlet completeness remains unverified, explicitly supplied
records can be inspected, but the pipeline suppresses net-inflow reconstruction.
The original records must document all boundary levels from 2016-01-01 00:00
through 2020-01-01 00:00: **1,462 values for 1,461 intervals**. End-of-day labels
could span 2015-12-31 through 2019-12-31; confirm with the owner. Leap days are
retained. DST days use 23/25 hours only if the declared historical clock uses DST.
No conversion to the Ontario 48-period calendar occurs.

## Storage modes and commands

Default normalized water-level position (no assumed volume, no volumetric inflow):

```bash
uv run python -m reservoir_control.systems.ottawa.reconstruction \
  --levels /path/to/levels.csv --outflows /path/to/outflows.csv \
  --contract /path/to/input_contract.json \
  --output results/ottawa/baskatong_normalized_v1
```

A linear volume **scenario** requires an explicit K, scenario ID and rationale:

```bash
uv run python -m reservoir_control.systems.ottawa.reconstruction \
  --levels /path/to/levels.csv --outflows /path/to/outflows.csv \
  --contract /path/to/input_contract.json \
  --usable-volume-m3 YOUR_JUSTIFIED_SCENARIO_K \
  --scenario-id YOUR_SCENARIO --volume-rationale 'State the evidence or assumption' \
  --output results/ottawa/baskatong_linear_scenario_v1
```

The literal placeholders above must be replaced. There is no default K.
The approximation is `S=K*(h-h_min)/(h_max-h_min)`, labelled scenario storage,
not measured volume. Normalized water-level positions outside [0,1] remain flagged and
unclipped. A sourced curve instead uses `--storage-curve /path/to/curve.csv`,
with `elevation_m,volume_m3` columns and `storage_curve` provenance in the contract:

```json
{"storage_curve": {"source_id": "registered_curve_source", "datum": "same as level_datum", "provenance": "Owner curve version, validity and datum evidence"}}
```

The curve must be strictly increasing, cover the operating elevations and have
separate source permission. Storage is `V(h)-V(h_min)` with piecewise-linear
interpolation within the supplied curve; no extrapolation. Curve-based,
approximate and normalized outputs go into separate named subdirectories.
A storage curve remains a sourced model of volume, not direct measurement of it.

## Quality and interpretation

Conflicting duplicates are excluded without selecting a winner. Exact daily
alignment is required; overlapping/partial/multiday means cannot be relabelled as
daily data. Expected missing days remain NaN. No interpolation or zero fill occurs.
The default 0.75 m/day jump flag is a **screening assumption**, not an established
Baskatong physical limit; change it explicitly with `--jump-m-per-day`.
`--maximum-inflow-m3-s` sets an optional, explicit plausibility flag. Negative
reconstructed inflows are retained and flagged, never clipped.

For eligible intervals, `i_net=(S_end-S_start)/duration_seconds+q_out` includes
upstream regulated releases and net other gains/losses. A curve or scenario K,
verified boundary levels, timing and total outlet are all needed. Rows with
jumps or negative inflow remain labelled suspect; raw quality failures and
unverified definitions do not produce estimates. See `quality_flags`, not just
`reconstruction_available`, when deciding suitability for later modeling.

PNG diagnostics cover level/bounds, reported outlet flow, storage/water-level position, net
inflow and coverage flags. Focused plots cover December 2016–June 2017 and
December 2018–June 2019. NaN gaps are visible. Cumulative closure is checked only
within contiguous computable segments. It is a dependent accounting identity;
it does not independently validate reservoir physics or inflow estimates.

Next data requirement: owner-permitted instantaneous boundary levels, matching
mean **total** outflow, all-outlet inventory, clock/datum/quality definitions and
a storage–elevation curve (or an explicitly justified usable-volume scenario).
Only then can this method be evaluated historically and extended reservoir by
reservoir without double counting transfers. Do not replace the existing
synthetic Ottawa parameters with these inventory figures.

## Diagnose the delayed perfect-forecast failure

```sh
uv run --group reproduction python -m reservoir_control.systems.ottawa.delay_diagnostic --output results/ottawa/delay_diagnostic_rerun
```

The standalone diagnosis uses an independent explicit-storage feasibility LP,
checks full-remaining feasibility after each original action, and repeats H=4,
5, 6, 8 and 12 without changing the controller or scenario. Outputs include
`report.md`, `diagnostic.png`, feasible witness/continuation trajectories,
constraint certificates, endpoint queues and implementation consistency checks.
The saved run is `results/ottawa/delay_diagnostic_verified_v2/`.
