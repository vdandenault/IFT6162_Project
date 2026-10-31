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
