# Current results: PPO and MPC

Best current settings on **1988–1997 validation years**, using Gym4ReaL's Lake Como
reservoir simulator. Both choose a daily release to balance water demand, flooding,
and low water. Training/fitting uses 1946–1987; the final 1998–2010 test comparison
has not been run.

- **PPO:** learns a release policy through simulated experience, using lake level,
  season, and averaged past demand. The selected setting trains for one million
  daily steps; evaluation freezes the policy and uses its mean action.
- **MPC:** uses the reservoir physics to optimize seven daily releases, executes
  only the first, then replans. The best current variant uses yesterday's inflow
  across the forecast horizon and retains bounded, cost-improving iterates when
  the optimizer reaches its iteration limit. Failure flags remain recorded.

All values below are **annual averages**. Higher reward is better; lower is better
for the physical penalties.

- **Overall reward:** PPO **−17,232** · MPC **−26,361**
- **Days outside preferred water levels:**
  PPO **4.4 flood days** and **5.8 low-water days**;
  MPC **12.3 flood days** and **10.4 low-water days**.
- **Unmet water demand:** PPO **514.9 million m³** · MPC **565.6 million m³**
- **Cumulative level exceedance:**
  Flood: PPO **2.609 m·days** · MPC **2.750 m·days**.
  Low water: PPO **0.0411 m·days** · MPC **0.0676 m·days**.
  This measures how far and how long daily water levels lie outside the thresholds;
  for example, 0.1 m beyond a threshold for 10 days contributes 1 m·day.

PPO averages five training seeds over ten years; MPC is deterministic over the
same ten years. PPO's reward SD across seed means is 1,823. Episodes reset annually
and use 365 daily actions, including the current leap-year approximation.
Full reward sums DamEnv's demand, flood, low-water, wasted-release, and clipping
penalties. Exceedance sums daily endpoint distances beyond level thresholds;
it is not flood damage or volume.

**Comparison limits:** both settings were developed on these validation years.
MPC additionally reconstructs past inflow from exact storage/release telemetry,
which PPO does not receive. MPC plans with a smooth surrogate but is scored using
the same environment reward as PPO. Its retention variant is experimental; the
original MPC remains unchanged. These results do not establish performance against
actual Lake Como operators or constitute a final matched-information comparison.

A matched **1/3/7-day MPC horizon test** retained seven days as the best mean-reward
setting; shorter horizons were faster and left less demand unmet. See the
[horizon findings](docs/EVALUATION.md#matched-mpc-horizon-experiment).

Details: [PPO setup](docs/PPO.md), [MPC retention experiment](docs/EVALUATION.md#cost-checked-retention-of-iteration-limit-iterates).
Saved metrics: `results/comparison/validation_v1/` and
`results/mpc/iterate_retention_v1/` (generated outputs are git-ignored).
