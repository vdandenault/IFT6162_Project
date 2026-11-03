# Current controller results

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
