# PPO: current baseline and findings

PPO is ready for the MPC comparison. Use the selected **1M-step, five-seed**
configuration in [configs/ppo.json](../configs/ppo.json). Further PPO tuning is
paused while we build the engineering controller.

## Current settings

Actor and critic each have two 16-unit hidden layers. Training uses 2,048-step
rollouts, 10 epochs, 32 mini-batches, discount 0.995, GAE lambda 0.95, policy clip
0.2, and initial log standard deviation -0.5. The KL-adaptive learning rate starts
at 8e-6 and stays within 1e-6–1e-4 (KL threshold 0.008). Entropy coefficient is zero.
CPU mixed precision uses bfloat16 autocast; evaluation uses FP32 deterministic
mean actions and frozen observation normalization. Each run uses one CPU thread.

Keep all five learner seeds, 123–127. The environment year-sampling seed is fixed
at 123. Train on 1946–1987 and select settings using validation years 1988–1997.
The chronological 1998–2010 test set has not been used for this selection.
Annual episodes reset level to 0.35 m and demand average to zero. The terminal
level observation is corrected; `time_limit_bootstrap=False` is unchanged.
Leap years retain the upstream 365-step approximation. There is no cross-year
storage continuity, inflow forecast, hydropower model, or learned reservoir physics.

## What the experiments established

Each ablation changes one setting. The conservative rule requires non-worse
mean full validation return, sample SD across five seed means, and weakest-seed
return. This is a practical selection rule, not a significance test. Five seeds
share the same ten validation years; they are not 50 independent observations.
Full return includes five penalties; the paper's three-term objective excludes
waste and clipping. Physical metrics are reported separately.

### Learning rate

At 200k steps, capping the scheduler improved mean full return from -32,357 to
-23,515 and seed SD from 6,465 to 5,075. Fixed 8e-6 was worse (-37,240), despite
lower SD. Select the cap. It reaches 1e-4 early and stays there, so this does not
establish that adaptation is necessary; fixed 1e-4 was not tested.

### Precision and exploration

FP32 training scored -23,610 (SD 5,054), with a slightly worse weakest seed than
mixed precision. An entropy coefficient of 0.01 widened exploration but scored
-24,153 (SD 4,638), worsening the mean. Both failed the rule; retain mixed precision
and zero entropy. These results concern the tested settings, not all numerical
precision or exploration strategies.

### Training duration

Late training reward improved for four seeds at 200k, motivating bounded duration
studies. Sampled training reward and critic loss alone did not establish convergence.
Continuous reruns saved fixed endpoints and exactly matched earlier actor, critic,
optimizer, and normalizer states. Earlier validation metrics matched too. Evaluate
only after each continuous run ends so evaluation cannot perturb later training.

| Steps | Mean full validation return | Seed SD | Weakest seed |
|---|---:|---:|---:|
| 200k | -23,515 | 5,075 | -31,753 |
| 400k | -19,301 | 5,289 | -28,756 |
| 600k | -17,882 | 2,940 | -23,076 |
| 800k | -17,338 | 2,023 | -20,787 |
| 1M | -17,232 | 1,823 | -20,295 |

400k failed the original rule because SD increased. 600k qualified against 400k.
The final study compared 800k and 1M independently against 600k; both qualified,
and the predeclared higher-mean rule selected 1M. These successive validation
investigations are exploratory, not independent confirmations of an optimal budget.

The final 800k-to-1M gain was only 107 reward on average, about 92% from seed 127.
Seed 125 worsened slightly versus 600k. Low-water days fell from 8.16 to 5.76/year,
but unmet demand rose from 512.89 to 514.91 million m³/year. Stop duration tuning
at 1M and move to MPC; the improved reward still involves physical trade-offs.

## Relationship to the paper

The supplied checkpoint reproduced Figure 1's qualitative ordering and approximate
scale on the original 13 test years. An exact numerical match is not established:
the figure has no numerical reference table. Fresh 200k training was seed-sensitive;
the favorable seed-123 objective (-12,394) became -18,654 averaged over five seeds
(SD 3,633). The supplied checkpoint scored -15,372 on the three-term objective.
Those runs use upstream membership and must remain separate from our chronological
protocol. The current -17,232 full return (-15,202 three-term objective) is on
validation, with different training years, budget, scheduler, and terminal handling.
It does not demonstrate that we beat the published PPO on a matched held-out test.
The paper's year-level uncertainty is distinct from our training-seed variation.

## Reproduce or inspect details

Most readers need only [the current controller comparison](../scripts/analysis/controller_comparison.py).
For the experimental evidence, read the relevant script rather than reading
all studies in sequence:

- [benchmark and original seed study](../scripts/analysis/benchmark_reproduction.py).
- [learning rate](../scripts/analysis/ppo_stabilization.py), [precision](../scripts/analysis/ppo_precision.py), [entropy](../scripts/analysis/ppo_entropy.py).
- [learning curves](../scripts/analysis/ppo_learning_curves.py).
- [400k](../scripts/analysis/ppo_duration.py), [600k](../scripts/analysis/ppo_duration_600k.py), [800k/1M](../scripts/analysis/ppo_duration_1m.py).

Commands all use `uv run --group reproduction`:

| Command suffix | Purpose |
|---|---|
| `lake-como-reproduce` | Supplied checkpoint and rules on original membership |
| `lake-como-train-reference` / `lake-como-seed-study --workers 2` | Original 200k retraining / five seeds |
| `lake-como-stabilize --workers 3` | Learning-rate ablation |
| `lake-como-precision-study --workers 3` | FP32 ablation |
| `lake-como-entropy-study --workers 3` | Entropy ablation |
| `lake-como-inspect-curves` | Read saved logs without training |
| `lake-como-duration-study --max-steps 1000000 --workers 3` | Bounded duration study; 400000 and 600000 reproduce earlier stages |

Exact predeclared plans are consolidated without changing their payloads in
[configs/experiments/ppo_studies.json](../configs/experiments/ppo_studies.json):
`learning_rate`, `precision`, `entropy`, `duration_400k`, `duration_600k`, `duration_1m`.
Runners reject modifications to an existing named plan. These are historical study
records; changing `configs/ppo.json` does not change their training settings.
Logs, checkpoints, and detailed reports remain ignored under `results/` and
`reports/`. The former individual notes are preserved locally under
`reports/archive/ppo_notes/`. Historical provenance retains the original filenames;
compare each old config hash with the corresponding manifest payload, not the
combined file's hash. The original baseline snapshot is unchanged.

## Past-inflow observation ablation

```bash
uv run --group reproduction python scripts/analysis/compare_ppo_inflow.py
```

This adds one feature to the selected PPO setting: the last completed day's inflow,
reconstructed from exact storage/release telemetry with evaporation disabled.
The reset feature is the first-day training climatology (67.27619 m³/s), matching
causal MPC's fallback. After step t, inflow t accompanies the next observation;
it is never supplied before choosing action t. Annual resets discard previous-year
history, and the corrected terminal observation includes the final completed inflow.

Five new policies (seeds 123–127) each train for 1M steps, with unchanged hidden
layers, capped scheduler, rollout/update settings, BF16 training, environment seed,
reward, split, and truncation handling. The input and observation scaler grow from
four to five dimensions; the actor and critic together gain 32 input weights.
Identical seed labels do not imply identical initialization or exploration streams
once input dimensions change. No four-feature checkpoint is resumed or expanded.

Controls reuse the existing five verified 1M checkpoints, rather than retraining
unchanged models. Checkpoint hashes, training membership, settings, library versions,
and exact daily validation replay are checked. Candidate evaluation uses deterministic
FP32 mean actions and frozen normalization, only after training ends. Final 1M
checkpoints are evaluated without intermediate/best-seed selection. Report five-seed
means, seed SD, weakest seed, paired annual metrics, and physical severity; fifty
seed/year rows are not fifty independent training replicates. No automatic baseline
promotion is performed.

The wrapper changes observations only; the fork and base environment stay unchanged.
It closes the past-inflow information gap with causal MPC, but their internal
models, forecasts, and demand representations still differ. Exact telemetry and
annual reset assumptions remain part of the experimental scope.

Ignored outputs live in `results/ppo/past_inflow_v1/`, including per-seed progress,
checkpoints, trajectories, provenance, and aggregate metrics when complete. The
predeclared plan is also recorded in `configs/experiments/ppo_studies.json` under
`past_inflow_observations_v1`. This experiment does not use chronological test years.

All five new 1M-step runs completed. Annual validation averages:

- **Full reward:** four features **−17,232** → past inflow **−16,505** (+726).
- **Reward SD across seed means:** **1,823 → 238**; weakest seed **−20,295 → −16,705**.
- **Flood / low-water days:** **4.40 / 5.76 → 4.30 / 3.50** per year.
- **Unmet demand:** **514.91 → 525.24 million m³/year** (+2.0%).
- **Flood / low-water exceedance:** **2.60909 / 0.04108 → 2.58976 / 0.01847 m·days/year**.

The main result is greater consistency in these five runs. Three paired seed means
improve and two worsen; the previously weakest seed contributes more than the net
aggregate reward gain. The other four seeds together have a slightly worse mean
(−11.67 points). Yearly seed-averaged reward improves in nine of ten years. This
is not a uniform improvement or proof of general seed robustness. The input change
also changes parameter count and initialization draws, as declared above.

Lower low-water penalties account for +565 reward points/year, flooding for +100,
and request adjustment for +267; seasonally weighted deficit and surplus penalties
worsen by about 172 and 34 points. Thus improved reward does not mean improved water
supply. Low-water severity falls about 55%, while flood severity improves less than
1%. All five controls reproduced their saved daily evaluation trajectories exactly.

The result is a promising experimental PPO variant; `configs/ppo.json` and the HTML
simulator remain on the existing four-feature reference. No automatic promotion or
extra tuning occurred. Both this variant and persistence MPC now have past-inflow
information, although their internal models and forecast/demand representations
differ. Review the supply/stability trade-off before changing the selected baseline.

### What changes in wet and dry years?

```bash
uv run --group reproduction python scripts/analysis/analyze_ppo_inflow.py
```

This descriptive follow-up freezes all trained policies. The selection rule is
lowest/highest total validation-year inflow, with ties resolved toward the earlier
year; no reward or violation metric chooses the examples. The driest year is
**1990** and the wettest **1993**. “Wettest” means annual water volume, not largest
storm or worst flood year. These are cases within ten validation years, not climate
quantiles or a held-out test.

- **Dry 1990:** reward improves −12,338 → −11,791. Low-water days fall **4.4 → 2.4**,
  with no flood days in either variant. Unmet demand rises **485.17 → 493.99 million
  m³**. Low-water exceedance falls **0.03892 → 0.01498 m·days**.
- **Wet 1993:** reward improves −45,650 → −44,691. Flood days barely change
  (**31.2 → 31.0**); low-water days fall **11.4 → 7.6**. Unmet demand rises
  **493.86 → 513.19 million m³**. Flood exceedance falls only **20.17594 → 20.05613
  m·days**. A wet year can still contain low-water periods.

The seed-averaged requested release is lower in every validation year, by about
4.5–14.3 m³/s in annual mean. On 169–228 days per year, the two seed-averaged requests
differ by more than 10 m³/s. Seed-averaged unmet demand rises in every year; low-water
days fall in nine years, with 1989 the exception. This demonstrates changed decisions
and physical trade-offs, not merely a better scalar score. It does not prove a
particular learned response to inflow: policies visit different states and their
parameter initialization also changed. Annual mean request is not a storage target
or a direct accounting of water saved.

The wet-year flood burden remains substantial. This motivates an explicit
flood-safety feasibility audit before imposing a hard requirement; adding inflow
information alone does not demonstrate flood prevention. Raw inputs, declared
selection, paired per-year metrics, reward contributions, and a PNG of levels,
releases, and cumulative shortages stay ignored in
`results/ppo/past_inflow_regimes_v1/`. Shading shows five-seed ranges, not confidence
intervals. There is no new training, policy promotion, or chronological test run.
