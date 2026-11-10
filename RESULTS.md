# Results and current understanding

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
