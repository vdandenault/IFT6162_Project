# Current results: PPO, MPC, and FQI

Development results on **1988–1997 validation years**. Training/fitting uses
1946–1987; the final 1998–2010 test comparison remains reserved.

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

Higher reward is better; lower physical penalties are better. The filter helps
MPC's event count much more than severity: flood exceedance falls only **2.4%**,
while low-water severity worsens slightly. Remaining floods and forecast-range
breaches mean this is **not a hard safety guarantee**. Keep the two tracks separate.

The original four-feature PPO scored **−17,232**, with seed SD **1,823**; past inflow
scores **−16,505**, with SD **238**. Three seeds improve and two worsen; most mean
gain comes from the previously weakest seed, while unmet demand rises about 2%.

These are tuned validation results, not proof of general or operator-level
superiority. Episodes reset annually and retain the 365-step leap-year approximation.
PPO figures average five seeds and ten years; MPC is deterministic over ten years.
PPO/MPC references and both tracks are frozen while FQI remains exploratory; existing
simulator/default configuration choices remain unchanged.

Details: [PPO observation study](docs/PPO.md#past-inflow-observation-ablation),
[common filter experiment](docs/EVALUATION.md#common-finite-scenario-safety-filter),
[FQI pilot](docs/EVALUATION.md#first-tree-based-fqi-pilot),
[frozen protocol](configs/experiments/control_protocol_v1.json).
