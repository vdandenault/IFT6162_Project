# Current project state

## 1. What we built

A quarter-month simulator for Lake Ontario–St. Lawrence releases and an
aggregate Ottawa reservoir. It combines historical Ontario inputs, Plan
2014 release limits, and a hard-constrained Ottawa follower. Ottawa inflow,
storage, and routing in the coupled experiments are synthetic.

Four controllers share the same execution and recovery logic. Filtered
Plan 2014 uses the operator rule curve. External-ranked MPC plans with a
fixed Ottawa release forecast. Stackelberg MPC ranks the same candidates
using Ottawa's optimized response. PPO learns a release adjustment to the
rule curve.

## 2. How we measure it

The main metrics are **completed episodes**, **operating cost on matched
complete years**, **flood exposure at each lake**, and **realized storage
and downstream flow violations**. Flood exposure sums metres above the
threshold over all periods. The Ontario threshold is 75.50 m. The
Lake St. Louis threshold at Pointe-Claire follows Plan 2014's F-limit tier.

**PPO reward is negative operating cost**, with an extra −5 for an early
stop. Each period's cost adds squared flood exceedance at both lakes,
0.25 × squared Ontario low-water shortfall below 74.00 m, and
0.01 × squared release change in units of 1,000 m³/s. At a completed year's
end, it also adds 0.10 × squared Ontario level deviation from 74.80 m.

## 3. Baseline and results

**Filtered Plan 2014** is the operator baseline. **External-ranked MPC
with peak-protected recovery** is our current research reference.

| Controller          | Annual cost | Ontario exposure |
| ------------------- | ----------: | ---------------: |
| Filtered Plan 2014  |      0.4625 |           0.8144 |
| External-ranked MPC |  **0.4581** |           0.8334 |
| Stackelberg MPC     |  **0.4581** |           0.8334 |
| PPO, five-seed mean |      0.4768 |       **0.8105** |

Values are annual means. Exposure is in metre-periods. All four
completed the 18 annual replays above with zero modeled Lake
St. Louis exposure. External MPC lowered both cost and Ontario exposure
in 2017 and 2019, but increased mean Ontario exposure across the 18 replays.
Stackelberg matched external MPC's releases. The later forecast and PPO
training experiments left the frozen comparison in place.

To replace the reference, a candidate must reduce total Lake St. Louis
exposure, add zero Ontario exposure in every matched year, preserve
completion, and add no physical violations.

**Handoff:** The observed routing check improved stage estimates with
Britannia flow, but error near the Pointe-Claire threshold remained
about 11 cm. The records support a historical benchmark. Validating
release-dependent headroom requires reach and branch flow evidence.
Keep the frozen comparison for the project.

[Full results](RESULTS.md) · [Project plan](PROJECT_PLAN.md)
