# Ontario–Ottawa results

## Frozen controller comparison

| Controller          | Completed | Annual cost | Ontario exposure |
| ------------------- | --------: | ----------: | ---------------: |
| Filtered Plan 2014  |     18/18 |      0.4625 |           0.8144 |
| External-ranked MPC |     18/18 |      0.4581 |           0.8334 |
| Stackelberg MPC     |     18/18 |      0.4581 |           0.8334 |
| PPO, five-seed mean |     90/90 |      0.4768 |           0.8105 |

Values are means across 18 matched years. PPO averages the five seeds
within each year. Exposure is in metre-periods. All runs had zero modeled
Lake St. Louis exposure. The two MPC variants executed identical Ontario
and Ottawa releases. They lowered mean cost by about 1% relative to
filtered Plan 2014 and slightly increased Ontario shoreline exposure.

The cost sums squared Ontario and Lake St. Louis exceedance, a weighted
Ontario low-water shortfall and a weighted release-change penalty. Complete
years include a terminal level penalty. PPO receives negative cost and a
penalty for early termination. See [current state](CURRENT_STATE.MD) for
threshold and weighting details.

### Flood-year walkthrough

| Year | Plan 2014 cost | MPC cost | Plan 2014 Ontario exposure | MPC Ontario exposure |
| ---: | -------------: | -------: | --------------------------: | -------------------: |
| 2017 |         7.8465 |   7.5941 |                     13.4814 |              13.2766 |
| 2019 |         0.8547 |   0.8359 |                      3.2334 |               3.1980 |

These are separate annual replays with historical Ontario inputs and
synthetic Ottawa inflow, storage and routing. Both MPC variants completed
both years. The candidate audit found that anticipating Ottawa changed some
predicted Pointe-Claire levels, but all 2,307 common eligible schedules
had zero modeled downstream exposure under both flow assumptions.

## What the gauge check established

The observed benchmark adds 197 complete quarter-months from 2021–2025.
Three were above the fixed 22.10 m study threshold. Development favored
same-day Cornwall and Britannia inputs. On 363 matched spring days,
Britannia improved stage MAE from 26.4 cm to 10.4 cm. Near the threshold,
error remained about 11 cm. The model found room for an extra 200 m³/s of
local throughflow on 12 days when observed stage was already at or above
the threshold. The Prairies proxy added no improvement on matched dates.

Keep filtered Plan 2014 as the operator baseline and external-ranked MPC
as the research reference. Matched Carillon and outlet measurements or a
validated hydraulic routing model would support testing release-specific
headroom. [Reproduction commands](README.md#observed-downstream-validation)
