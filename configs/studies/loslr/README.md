# Ontario study inputs

- `native_baseline.json` configures the Plan 2014 parity replay.
- `scoring.json` describes the standalone Ontario controller score.
- `coupled_v1.json` defines the split, cost weights and Plan 2014 F-tier
  assumptions shared by the final runner.
- `final_coupled_v1.json` defines the synthetic Ottawa annual shapes and the
  frozen PPO budget of 16,384 decisions per seed.

The research model uses 48 quarter-months each year. The F-limit tiers depend
on the Ontario beginning level. Ottawa's hard follower acts on executable
Ontario releases and forecasts other downstream contributions. Ontario
water supply is historical and Ottawa inflows, capacity and delay are
synthetic. See the [main README](../../../README.md) for the final runner.
[Plan 2014 compendium](https://www.ijc.org/sites/default/files/Plan2014_CompendiumReport.pdf)
