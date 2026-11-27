# Ottawa study inputs

`hard_follower_v1.json` declares six synthetic aggregate scenarios on a
48-period calendar. The follower balances storage in m³ using release and
inflow in m³/s and a period duration of 657,000 seconds. Storage, outlet and
downstream capacity bounds are hard. The study includes deliberate
infeasibility and delayed arrivals. Water in transit remains in the queue
at the experiment endpoint.

Standalone checks write to fresh directories:

```sh
uv run --locked python -m reservoir_control.systems.ottawa.hard_follower_study \
  --forecast noisy --lookahead 5 --output results/ottawa/follower_noisy
uv run --locked python -m reservoir_control.systems.ottawa.horizon_study \
  --output results/ottawa/horizon_rerun
```

`shared_recovery_final_check_v1.json` defines the final synthetic stress cases.
`downstream_anticipation_v1.json` and `ranking_sensitivity_v1.json` define
the last candidate and routing sensitivity audits.

## Baskatong source evidence

The Baskatong files document a separate daily pilot. The 1970 level archive
uses HYDAT station 02LG007. The approximate scenario tests 02LG006 as a
Mercier downstream-flow proxy and 02LG007 discharge as a separate
sensitivity. The measurement boundary and storage relation are unresolved.
The pilot demonstrates storage response to changing releases under a
declared scenario and does not calibrate the Ottawa aggregate.

```sh
uv run --locked python -m reservoir_control.systems.ottawa.historical_pilot \
  --output results/ottawa/baskatong_1970_rerun
uv run --locked python -m reservoir_control.systems.ottawa.baskatong_scenario \
  --pilot results/ottawa/baskatong_1970_rerun --output results/ottawa/baskatong_scenario_rerun
```
