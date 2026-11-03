# SAC: implementation and learning walkthrough

**Status:** the implementation, training-only smoke checks, and first bounded
five-seed study are complete. At 200k steps per seed, plain SAC scores **−21,064**
on validation, trailing the frozen 1M-step PPO reference (**−16,505**). Test years
remain reserved; no reference or default controller was replaced.

The illustrated walkthrough is local and ignored:
[background/SAC.md](../background/SAC.md). Regenerate its PNGs with
`uv run python background/generate_sac_diagrams.py`.

## What SAC learns on the lake

Every day the actor receives five numbers: lake level, seasonal sine/cosine,
exponentially averaged demand, and the last completed day's inflow. These are
observations, not the complete physical state; hidden future inflows remain unknown.
The actor chooses a requested release. DamEnv applies hydraulic limits and advances
24 hourly water balances. The resulting observation, reward, and episode flags
form one transition. Learning never calls the reservoir model to compare future
action sequences. This distinguishes plain SAC from MPC.

SAC learns **one stochastic actor and two Q-critics**. Each critic receives an
observation AND a candidate requested action, and estimates its discounted soft
return under the policy. A soft return includes future exploration bonuses, so Q
is not measured water, predicted lake level, electricity, or the original annual
reward. Twin critics limit optimistic value estimates by taking their minimum;
their disagreement is not calibrated uncertainty.

Two slowly updated copies of the critics produce targets. They are not additional
policies. There is no separate V-network in this modern SAC variant.

The underlying simulator state includes storage, calendar/profile position, and
demand-average memory; the historical forcing profile determines subsequent
disturbances. The policy sees only the five-feature observation, not the profile
identity or future forcing. Thus the feedforward agent works with an observation
of a partially observed process, not a proven Markov state. Its training objective
is discounted reward plus policy entropy over each annual episode. Reported
evaluation uses original environmental rewards without adding an entropy bonus.

## From network output to requested and actual release

The actor predicts Gaussian mean and log standard deviation. During training:

```text
epsilon ~ Normal(0, 1)
z = mean(observation) + exp(log_std(observation)) * epsilon
a = tanh(z)                           # normalized request, [-1, 1]
requested_release = low + (a + 1)/2 * (high - low)
actual_release = DamEnv's hourly hydraulic clipping(requested_release, level)
```

For our scalar action, low = 0 and high = 491.61 m³/s. A normalized action of zero
requests 245.805 m³/s. Store the **normalized request**, not hydraulic actual release,
in replay: that is the decision the policy controls. Actual release and lake level
are outcomes from the environment. Different requests can produce the same actual
release near a hydraulic boundary.

Log probabilities include the tanh change-of-variables correction. The entropy
convention is the density of the normalized action; it excludes the physical affine
scaling. The target entropy uses those same units. Differential entropy can be
negative; it is not an ordinary probability or a physical safety measure.

During deterministic evaluation use `tanh(mean)`, then map to release. This is a
common evaluation convention, not exactly the mean of the squashed distribution.
The evaluator freezes weights/statistics and performs no learning updates.

## Replay: reuse experience across policies

Replay stores `(raw_observation, normalized_request, original_reward,
raw_next_observation, ended)` for transitions collected on training years only.
A uniform ring buffer overwrites its oldest data when full. Minibatches can contain
transitions collected by older versions of the actor: SAC is **off-policy**.
Unlike PPO, it does not use an old/new policy probability ratio to limit updates.
Unlike our FQI pilot, new experience can be added continually.

Observation normalization is an immutable affine transform supplied when creating
the agent: `(observation - training_mean) / training_scale`. Replay keeps raw values
so collection and sampled updates use the same transform. Identity is supported for
tests, but a reservoir study must specify and save training-only statistics before
running. The implementation neither fits statistics on evaluation years nor silently
updates them during inference.

## The four learning operations

Let `r` be original reward, `c` its positive learning scale, `d` the episode-end
indicator, `gamma` the discount, `alpha` the entropy temperature, and `logp` the
normalized-action log density. Bars denote slow target critics.

**1. Form a detached target and regress both critics.**

```text
a_next ~ actor(observation_next)
y = c*r + gamma*(1-d) * (
      min(Q1_target(observation_next, a_next), Q2_target(...))
      - alpha*logp(a_next | observation_next))
critic_loss = mean((Q1(observation,a)-y)^2 + (Q2(observation,a)-y)^2)
```

The entire target is detached: its construction cannot update actor, temperature,
or target critics. This is FQI's Bellman regression idea with neural twin critics,
sampled continuous actions, and an entropy term.

Our annual-reset protocol sets `d = terminated OR truncated`. A corrected terminal
observation is saved before reset, but no value is propagated across the annual
boundary. Other tasks can bootstrap time limits; changing that convention here
would change the experiment and must be explicit. Pass an ordinary environment,
not a vector/autoreset wrapper that substitutes the next episode's reset observation.

**2. Improve the actor through the critics.**

```text
a_new = tanh(mean(observation) + std(observation)*epsilon)
actor_loss = mean(alpha*logp(a_new | observation) - min(Q1(...), Q2(...)))
```

Reparameterization lets gradients pass through the action into the actor. Critic
weights are frozen for this operation, but the derivative of Q with respect to
the action stays available. Alpha is detached. The actor balances estimated return
against exploration; exploration does not itself guarantee safety or improvement.

**3. Adjust exploration temperature.**

```text
alpha = exp(log_alpha)
temperature_loss = -mean(log_alpha * stop_gradient(logp + target_entropy))
```

This standard log-temperature surrogate increases alpha when sampled entropy falls
below the target and decreases it when above. Only log-alpha is updated. Default
target entropy is minus action dimension (-1 here). Automatic temperature does not
eliminate reward-scale sensitivity: the starting alpha, reward scale, and learning
dynamics still interact. The initial learning scale is 0.001; reported environment
returns must retain their original units.

**4. Slowly update target critics.**

```text
target_parameters = (1-tau)*target_parameters + tau*critic_parameters
```

Targets follow rather than instantly copying the rapidly changing critics. Initial
target weights equal online weights. Updates use full float32 on CPU; device or
mixed-precision experiments are outside this first implementation.

## Code and checks

- [sac.py](../src/lake_como/sac.py): configuration, replay, actor, critics,
  detached Bellman target, temperature, Polyak updates, and learner checkpoints.
- [sac_loop.py](../src/lake_como/sac_loop.py): explicit collection → replay →
  updates → episode reset loop. It starts no work on import and has no study CLI.
- [test_sac.py](../tests/test_sac.py): numerical targets, annual masks, ring-buffer
  behavior, transformed density including saturation, gradient direction through
  frozen critics, parameter updates, temperature direction, known terminal-value
  fitting, physical units, checkpoint restoration, and reset ordering.

```bash
uv run --group reproduction python -m unittest discover -s tests -p 'test_sac.py' -v
```

Checkpoints preserve the observation transform, action bounds, networks, optimizer
states, sampling RNG, and update count. Replay and simulator state are **not** saved:
this is a learner checkpoint, not an exact interrupted-training resume. Loading
expects a trusted local file. Deterministic inference does not consume sampling RNG.

The first study supplies fixed training-only statistics and a passing DamEnv
smoke experiment. A separate 10k-step Pendulum check exercises end-to-end continuous
control. Neither smoke check establishes reservoir performance; that requires the
validation study.

## Sources

The update follows Haarnoja et al., [*Soft Actor-Critic Algorithms and Applications*](https://arxiv.org/abs/1812.05905),
using the twin-Q/automatic-temperature variant, rather than mixing it with the
earlier separate-V formulation. [Spinning Up's SAC equations](https://spinningup.openai.com/en/latest/algorithms/sac.html)
are a supporting derivation for twin-Q targets and reparameterized policy updates;
its example implementation uses fixed temperature, whereas ours learns it.


## First bounded reservoir study

[configs/sac.json](../configs/sac.json) is the predeclared study configuration.
[sac_study.py](../src/lake_como/sac_study.py) runs the preparation, smoke gate,
five independent learner processes, and final validation stages. Generated artifacts
are in ignored `results/sac/baseline_200k_v1/`; its `plan.json` freezes configuration,
source/data hashes, reference-result hashes, dependency versions, and normalization.

- **Budget:** seeds 123–127 each collect 200,000 training interactions, including
  5,000 random warmup steps, then make one update per interaction: 195,000 updates
  per seed. Each update changes both critics, the actor, and temperature. Replay
  capacity is 200,000 and minibatch size 128; networks have two 64-unit hidden layers.
- **Information and objective:** the frozen five-feature observation, original
  DamEnv reward, 0.001 learning reward multiplier, gamma 0.995, annual end masks,
  action bounds, and physical model are unchanged. Adam learning rate is 0.0003,
  target averaging tau 0.005, initial alpha 0.2, normalized target entropy -1.
- **Normalization:** one shared 15,330-transition sample (one episode per training
  year, seed 90210) mixes uniform and demand-following requests equally. Fixed
  feature means/stds are fitted from pre-action observations only; none of those
  samples preload replay or become critic-training targets. Collection still costs
  simulator interactions and is counted separately. Each learner uses environment
  seed 123 for the same profile sequence, with its own network/action/replay RNGs.
- **Smoke gate:** seed 901 collects 7,300 transitions with 512 warmup steps and
  6,788 updates. All 20 complete episodes have exactly 365 steps. Every transition
  is checked for bounds, finite values, reward-component sum, physical water balance,
  causal next-day inflow, and corrected terminal level. Nonfinite losses/gradient
  norms stop the run before the corresponding optimizer step. Checkpoint inference
  is checked after reloading. Smoke data/weights are not reused by the five seeds.
- **Evaluation:** only final 200k checkpoints; no validation during training or
  checkpoint selection. All ten validation years, 1988–1997, are evaluated under
  both plain and frozen common-filter tracks. Weights/statistics remain fixed.
  Save daily trajectories, annual physical metrics/severity, per-seed means and SD,
  paired differences against saved PPO/MPC years, and decision time. Exact forcing
  and reset parity are checked against reference trajectories. No test evaluation.

The frozen PPO reference used 1M interactions per seed and smaller networks.
This is an exploratory SAC comparison with a different interaction/compute budget,
not a matched-budget algorithm ranking or numerical paper reproduction. Training
returns also mix years and stochastic actions; their curves are not validation
learning curves. Initial study checkpoint snapshots are retained for provenance,
not selected by validation performance.

Gymnasium's environment checker flags an existing upstream limitation: DamEnv does
not initialize Gymnasium's RNG through `super().reset(seed=seed)`; it uses its own
`random.Random` instead. The study preserves the simulator and records that checker
failure explicitly. Separate tests confirm repeatable seeded training trajectories,
training/evaluation observation parity for a full training year, and annual resets.
Unbounded observation-space and physical-action-scale warnings are also recorded;
the agent itself learns in normalized action units. API-checker failure is not
silently reported as a pass.

To reproduce in a **new** output directory, execute these stages in order (existing
outputs are protected):

```bash
uv run --group reproduction python -m lake_como.sac_study --stage prepare --output results/sac/new_run
uv run --group reproduction python -m lake_como.sac_study --stage smoke --output results/sac/new_run
uv run --group reproduction python -m lake_como.sac_study --stage train --output results/sac/new_run
uv run --group reproduction python -m lake_como.sac_study --stage evaluate --output results/sac/new_run
```


### Independent continuous-control smoke check

A separate seed-801, 10,000-step run on Gymnasium `Pendulum-v1` completed 50 episodes
and 9,000 SAC updates. With reset seeds 10–14 held fixed, the mean deterministic
return changed from -1,460.7 to -600.9.
This uses Pendulum-specific bounds/scaling and a 0.99 discount; it is one
implementation check, not a tuned benchmark, convergence result, or reservoir
training. Its script and record are in the ignored study directory as
`continuous_control_smoke.py` and `continuous_control_smoke.json`. These 10,000
interactions are separate from Lake Como's collection budget and do not initialize
the reservoir learners.


### First validation results

Annual means over all five seeds and ten validation years:

- **Plain SAC:** reward **−21,064**, seed SD **759**; **4.84 flood days**,
  **8.90 low-water days**, and **623.12 million m³** unmet demand.
- **Shared-filter SAC:** reward **−20,871**, seed SD **720**; **4.70 flood days**,
  **8.90 low-water days**, and **620.34 million m³** unmet demand. Only 23 of
  18,250 decisions change.
- Plain mean rewards for seeds 123–127: **−20,921 / −20,164 / −21,601 / −20,592 /
  −22,044**. These are all retained, not a best-seed selection.

Plain SAC trails PPO on reward, unmet demand, event counts, and exceedance severity;
its seed-averaged reward is lower in each validation year. It beats the plain MPC
prototype on reward and flood days while leaving **57.52 million m³** more demand
unmet. With the common filter, MPC scores **−18,486**, better than SAC. This result
is a first configuration at a smaller budget than PPO, not an algorithm ranking.

Plain flood/low-water severity is **2.6603 / 0.06884 m·days**, versus filtered
**2.6382 / 0.06884**. Maximum evaluation water-balance error is **2.20e−7 m³**;
exact forcing/reset parity and frozen evaluation weights/statistics checks pass.
Each learner completed 547 annual episodes plus 345 steps, with identical year
exposure across seeds. Five concurrent single-thread runs took about 498–511 seconds
each. Accounted study collection was **1,022,630** reservoir interactions including shared
normalization and discarded smoke data. Unit-test simulator calls are additional
diagnostics outside this budget: the new adapter/repeatability checks use 385
training-only transitions per pass, and were run both separately and in the suite.

[Learning curves](../results/sac/baseline_200k_v1/learning_curves.png) show rapid
initial improvement followed by fluctuations with changing training years, rather
than clear sustained late improvement. Alpha finishes around 0.0047–0.0068 and
entropy near the -1 target; actor gradients remain finite and nonzero. These are
update diagnostics, not evidence of critic calibration or convergence. Inspect
supply timing and critic/policy behavior before deciding whether more steps are useful.

See the [local comparison graphic](../background/images/18_sac_first_study.png)
and [full local report](../results/sac/baseline_200k_v1/report.md). The full test suite
passes **109 checks**, with one optional FQI integration check skipped. All generated
results/checkpoints/graphics stay ignored; the study config and code are reviewable.


## Release timing and critic diagnosis

A read-only follow-up explains the gap without changing or retraining any controller.
It uses saved validation trajectories for timing/accounting; new branch experiments
use only training years selected by annual inflow, with settings declared before
execution. [Full local diagnostic](../results/sac/diagnostics_200k_v1/report.md).

- SAC's **97.87 million m³** extra annual unmet demand decomposes into **82.53 million
  m³** extra surplus after environmental flow/demand plus **15.34 million m³** extra
  terminal storage. This is a volume identity, not proof that every surplus could
  have been avoided. January contributes the largest shortage gap; September and
  October contribute **4.16 of the extra 5.40 low-water days**.
- Mean absolute request-versus-actual-release difference is **56.84 m³/s**, versus
  PPO's **29.94**. Extra clipping costs **1,134 reward units/year**, alongside **1,350**
  extra low-water penalty and **1,334** extra demand penalty. Requesting more does not
  necessarily deliver more when hydraulic limits bind.
- At **28/60 training anchors**, a smaller request produces exactly the same physical
  next state under the day's historical inflow and a lower immediate clipping cost.
  The critics prefer the costlier request in **20/28**. These are conditional local
  comparisons; the smaller request need not dominate under other possible inflows.
- Four-noise-seed continuations use the frozen stochastic actor, final alpha, original
  reward scale, and no terminal bootstrap. Soft-Q excludes entropy for the fixed first
  action. Historical-year return errors also reflect hidden forcing and sampling;
  this is not held-out calibration or proof of a particular optimization bug.

See [release timing](../background/images/19_sac_release_timing.png),
[critic/action plateau](../background/images/20_sac_critic_action_surface.png), and
[conditional returns and penalty costs](../background/images/21_sac_critic_and_penalties.png).
The extra 150,400 continuation transitions and other training-only branch/prefix
calls are diagnostics, not learning data. Test years and frozen references remain
untouched. Next: a bounded critic-fit check around hydraulic plateaus with the actor
frozen, before changing optimization settings or extending training.

```bash
uv run --group reproduction python -m lake_como.sac_diagnostics --output results/sac/new_diagnostic
```

Diagnostic checks cover the volume identity, training-only selection, exact state
identity, snapshot isolation, and soft-return entropy/terminal conventions. Full
suite after this addition: **113 passed, one optional check skipped**. Frozen source,
checkpoint, normalization, and reference-result hashes still match.


## Frozen-actor critic-fitting experiment

The bounded follow-up is complete: 1,000 critic-only steps per seed and arm,
with actor, alpha, target critics, normalization, and actor RNG fixed. Paired
requests share exactly the same transition under their realized training-day
inflow; only clipping cost differs. Twenty-one training years supply **210 fit
pairs**; the other 21 supply **235 checking pairs**. Checking years were also in
the original SAC training, but are excluded from this new fit. No validation or
test years are evaluated.

Correct min-Q cheaper-request rankings on checking pairs:

- **Unchanged critics:** 67/235 (**28.5%**).
- **Ordinary Bellman regression:** 139/235 (**59.1%**).
- **Bellman + explicit pair loss:** 126/235 (**53.6%**).

Pair loss fits training examples better (83.8% correct versus 70.0% for ordinary
regression) and slightly improves checking difference MAE, but does not generalize
better on ranking accuracy. Ordinary regression improves checking ranking in all
five seeds. Broader checking point-target error does not improve: pooled MSE is
1.553 unchanged, 1.674 ordinary, and 1.598 with pair loss. Critic fitting therefore
does not establish better overall values, a superior policy, or a fixed root cause.

The actor remains identical, so **controller behavior and existing scores remain
unchanged**. Same-transition comparisons are conditional on historical inflow;
they are not dominance guarantees across unknown inflows. Only critic weights are
saved in the experimental outputs, separate from baseline SAC checkpoints.

See [the PNG](../background/images/22_sac_frozen_actor_critic_fit.png) and
[full local report](../results/sac/critic_fit_v1/report.md). The predeclared settings
and code are in [sac_critic_fit.py](../src/lake_como/sac_critic_fit.py); no best-step
selection is made. Cost: 95,970 training-only transitions and 10,000 critic-only
updates across both arms and five seeds. Full suite: **117 passed, one optional
check skipped**. Frozen source/reference hashes and actor-state checks pass.

Next, separate targeted data coverage from extra update count using a fixed
training-only dataset and equal fitting budgets before unfreezing any actor.

```bash
uv run --group reproduction python -m lake_como.sac_critic_fit --output results/sac/new_critic_fit
```


## Sampling emphasis versus update count

The fixed-pool follow-up is complete. Both arms use the same transitions, ordinary
Bellman MSE, batch size 128, initial critics, frozen actor/alpha/targets, and optimizer.
Uniform sampling is compared with half-uniform, half-paired sampling at predeclared
100 and 1,000 updates. No pair loss is added. Checking years are excluded from this
new fit but were present in original SAC training; validation/test remain reserved.

On the same 235 checking pairs, correct cheaper-request rankings are:

- **Baseline:** 28.5%.
- **100 updates:** uniform **30.6%**, focused **43.4%**.
- **1,000 updates:** uniform **34.5%**, focused **51.9%**.

Focus improves the final ranking in all five seeds, adding **17.4 percentage points**
over uniform at matched compute. Focus at 100 updates also exceeds uniform at 1,000.
Checking difference MAE at 1,000 is **0.10322 uniform / 0.08414 focused**, versus
0.10294 before fitting. Sampling emphasis contributes beyond merely adding updates.

However, broader ordinary-state checking error against frozen targets worsens:
**0.81457 → 0.90953 uniform / 0.90184 focused** at 1,000. These are approximate target
residuals, not true-return calibration. The result supports a local coverage/sampling
hypothesis, not improved overall values or control. The actor and all original
controller scores remain unchanged.

See [the PNG](../background/images/23_sac_sampling_vs_updates.png),
[full local report](../results/sac/critic_sampling_v1/report.md), and
[implementation](../src/lake_como/sac_critic_sampling.py). The common pool consists of
fresh frozen-deterministic-actor rollouts plus the earlier cheaper branch examples;
it is not the original stochastic replay buffer. This isolates sampling emphasis,
not collection strategy or pair co-occurrence versus weighting separately.

Cost: 76,650 training-only rollout steps, 445 reused counterfactual rows, 10,000
critic updates across all seeds/arms, about 17.5 seconds. Full suite: **120 passed,
one optional check skipped**. Noncritic-state, source-hash, and sampling-isolation
checks pass. Next investigate maintaining broad value fit alongside focused coverage
before declaring any actor/control experiment.

```bash
uv run --group reproduction python -m lake_como.sac_critic_sampling --output results/sac/new_sampling_study
```


## Fresh-return checks and 10% focused replay

Both follow-ups are complete. Fresh references use **420 ordinary states** (fixed
days 45/135/225/315 in all 21 checking training years, five seeds), each with eight
stochastic continuations of the frozen actor to year-end. The reference includes
future entropy, excludes first-action entropy, and has no terminal critic bootstrap.
It is conditional on each historical inflow path, not an exact population value.

At 1,000 critic updates, fresh conditional-return MSE is:

- **Unchanged:** 43.868.
- **Uniform:** 43.860.
- **10% focus:** 43.905.
- **Prior 50% focus:** 44.150.

On those same states, min-Q cached-target MSE is 2.492 unchanged, 2.689 uniform,
2.698 with 10% focus, and 2.655 with 50%. Thus worsening cached-target fit does
**not** establish a comparable deterioration in return prediction or control. The
10% conditional-return MSE change is only +0.084%; learner-seed effects are mixed.
Substantial conditional error persists in all variants and is much larger than
estimated action-noise sampling error. Hidden forcing and target approximation
remain unresolved; no root cause is established.

The 10% fit reuses the previous pool/targets, matches update counts, and reproduces
uniform weights exactly at 100 and 1,000 updates. Checking-pair rankings rise from
28.5% before fitting to **32.3% at 100** and **41.7% at 1,000**, versus uniform's
30.6% and 34.5%. Ordinary Bellman loss is unchanged. Alternating 6/6/7/6/7 complete
pairs in 128-row batches gives exactly 10% focused rows across both budgets.

But whole-ordinary checking target MSE still rises **0.81457 → 0.91119**, and fresh
return MSE/MAE at the primary 1,000-step endpoint do not improve. Only the local
ranking condition passes the predeclared preservation rule. No actor is unfrozen
and all original controller scores stay unchanged. Do not select the earlier
endpoint after seeing its slightly better return error.

[Annotated PNG](../background/images/24_sac_return_reference.png),
[full local report](../results/sac/critic_reference_v1/report.md), and
[code](../src/lake_como/sac_critic_reference.py). The 658,035 new simulator transitions
are training-year reference diagnostics, not fit data; both fitting arms reuse the
cached pool. Full suite: **123 passed, one optional check skipped**. Baseline hashes,
noncritic state, exact uniform reproduction, and checking/fitting separation pass.

Next hypothesis: a frozen-actor comparison of bootstrapped versus fresh rollout
value targets, fitted only on fitting years and checked against untouched checking
returns. Sampling-fraction tuning alone has not resolved broad value estimation.

```bash
uv run --group reproduction python -m lake_como.sac_critic_reference --output results/sac/new_reference_check
```


## Bootstrap versus rollout-return targets

Implemented and completed the target-only comparison: both arms fit the same
420 fitting state/action anchors across five seeds, with identical initial critics,
batches, loss, optimizer, and 100/1,000 budgets. Only labels differ: original cached
soft-Bellman targets versus eight fresh fitting-year continuations of the frozen
actor to year-end. Separate checking-year returns never enter fitting; validation
and test years remain reserved.

At the predeclared 1,000-update endpoint, checking conditional-return MSE / MAE is:

- **Unchanged:** 43.868 / 5.511.
- **Bootstrap targets:** 44.388 / 5.503.
- **Rollout targets:** **28.554 / 3.014**.

Rollout-target MSE is **35.7% lower than bootstrap**, with improvement in all five
seeds. But continued fitting overfits: rollout fit MSE improves 13.071 → 5.914 from
100 to 1,000 updates while checking MSE worsens 22.391 → 28.554. Both snapshots are
reported; the earlier one is not selected after the fact.

Action comparisons remain a gap. Only one first request per fitting state is
supervised. On the separate 235 clipping pairs, correct cheaper-request rankings
are **55.7% bootstrap / 36.6% rollout**, versus 28.5% unchanged. Rollout fitting
also worsens difference MAE (**0.56852 versus 0.10258 bootstrap**). Better values at
the actor's actions do not establish a reliable action-value surface or gradients.
The actor stays frozen, and all original controller behavior/scores stay unchanged.

This compares **frozen one-shot bootstrap labels**, not iterative TD with updated
target networks or live SAC. Fresh returns are conditional on historical inflows,
not unknown-inflow population truth. The much larger distance to old cached labels
after rollout fitting is not itself a performance failure: fresh-return accuracy
and action comparisons must be measured separately.

[PNG](../background/images/25_sac_bootstrap_vs_rollout_targets.png),
[full local report](../results/sac/critic_targets_v1/report.md), and
[implementation](../src/lake_como/sac_critic_targets.py). Cost: 658,035 fitting-year
simulator calls for fresh labels, checking references reused, 10,000 critic updates;
about 72.2 seconds. Full suite: **126 passed, one optional check skipped**. All
noncritic state and baseline/source hashes remain unchanged.

Before an actor update, test iterative critic-only target updates to separate
stale labels from bootstrapping, and assess action-diverse rollout targets rather
than assuming single-action value fits teach release comparisons.

```bash
uv run --group reproduction python -m lake_como.sac_critic_targets --output results/sac/new_target_comparison
```


## Iterative TD with the actor frozen

The final bounded SAC diagnostic is complete. Both arms use the same uniform
fitting pool, minibatches, fresh next-action noise, optimizer, and update budget;
only target-network updates differ (tau=0 versus 0.005). Actor, alpha, normalization,
and private actor RNG remain fixed. At 1,000 updates, checking return MSE is
**44.145 frozen / 44.493 moving**, versus **43.868 before fitting**. Checking clipping
rankings are 33.2% / 37.4%, versus 28.5% before fitting.

Moving targets alone do not resolve the gap in this small experiment. This is not
TD convergence or general inferiority evidence, and does not change any actor or
controller score. No new simulator calls or validation/test evaluation were needed.
See [PNG](../background/images/26_sac_iterative_td.png),
[local report](../results/sac/iterative_td_v1/report.md), and
[code](../src/lake_como/sac_iterative_td.py). All five seeds are retained.

The standalone SAC learning phase has a verified implementation, a frozen first
study, and bounded follow-up evidence, including the action-diverse diagnostic
below. Phase 6A implements a separate
[MPC plus residual-RL controller](RESIDUAL_RL.md); it does not silently modify SAC.

## Action-diverse rollout supervision

This completed diagnostic asks whether a critic learns useful **release comparisons**
when supervision covers several requests at each state. It uses the same ordinary
training-year anchors as the prior target study: days 45, 135, 225, and 315, with
21 fitting and 21 separate checking years per seed. The actor stays frozen.

At each anchor, label its usual request and requests **50 m³/s lower and higher**,
clipped to nominal bounds. Follow each first request with eight stochastic frozen
actor continuations, using the same action noise across alternatives. These are
returns conditional on that historical year's inflows, not population value truth.
Identical physical next states share their continuation and differ only in
immediate reward. Duplicate requests at nominal bounds remain in the dataset but
are excluded from action-gap scores.

Compare four arms: single-action/bootstrap, single-action/rollout,
multi-action/bootstrap, and multi-action/rollout. Reuse the completed single-action
controls exactly. The new arms use the same sampled states and 128 labels per
update, randomly choosing one of three requests for each sampled state; keep
100 and 1,000 update budgets, with 1,000 the declared primary endpoint.

Report both absolute conditional-return error and errors in alternative-minus-actor
value differences on checking years. Also report directional agreement for gaps
larger than twice their paired Monte Carlo standard error (and 0.0001); this is
a descriptive noise screen, not a statistical significance test. Checking years
were part of original SAC training and prior diagnostics. Neither these labels nor
these diagnostic critics train an actor or establish better reservoir control.

Run `uv run --group reproduction python -m lake_como.sac_action_diverse`.
The active declared output is `results/sac/action_diverse_v2/`; v1 was stopped
before fitting to tighten physical-state cache identity. Source hashes, cached
single-action controls, all seeds, and both budgets remain auditable locally.

**Completed result:** at the declared 1,000-update endpoint, adding alternative
requests reduces rollout-trained checking action-gap MAE from **0.2482 to 0.1877**
(all five seeds improve). It remains much worse than the unchanged critics'
**0.0363**. Directional agreement barely changes: **44.1% → 44.5%** on 787
noise-screened pairs, versus **49.3%** unchanged. Multi-action rollout return MSE
is **28.691**, versus single-action **28.592**, bootstrap multi-action **44.406**,
and unchanged **43.902**. Thus better absolute return fitting still fails to
produce reliable release comparisons.

Multi-action bootstrap fitting gives smaller gap MAE (**0.0329**) than either
single-action bootstrap (**0.0448**) or unchanged critics, but directional agreement
is **61.1%**, below single-action bootstrap's **65.2%**. Magnitude and sign metrics
answer different questions. These ±50 comparisons differ from the earlier selected
clipping-pair test; their percentages are not directly interchangeable.

All 100-update results are retained; return error again worsens by 1,000. No budget
or seed is promoted. The experiment required **1,480,830 new simulator transitions**,
including prefix reconstruction and one-step probes, plus **10,000 new critic
updates**; another 10,000 single-action control updates were reused. Original actors,
alpha, target critics, and checkpoint files remain unchanged. This closes the
bounded diagnostic, with no justification yet for using these refitted critics to
update the policy. See the [PNG](../background/images/30_action_diverse_critic_targets.png)
and [local scores](../results/sac/action_diverse_v2/summary.csv).
