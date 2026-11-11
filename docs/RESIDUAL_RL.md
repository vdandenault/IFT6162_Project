# MPC with learned residual corrections

**Status:** the controller, matched full-action control, smoke tests, and first
three-seed pilot are complete, including all ten validation years in both tracks.
The original PPO/MPC/FQI/SAC references and default simulator choices remain fixed.

## What the controller does

MPC forecasts inflow by persistence, plans seven days, and proposes today's release.
The residual actor sees the original five past-inflow features plus that causal
proposal. It requests a bounded adjustment:

```text
normalized_residual = tanh(Gaussian sample)
delta = 50 * normalized_residual                  # m³/s
nominal_request = clip(MPC_request + delta, 0, 491.61)
actual_release = DamEnv's level-dependent hourly clipping(nominal_request)
```

The MPC algorithm, forecast, horizon, surrogate cost, SLSQP cap, and retention rule
remain unchanged. Its proposal responds to the actual lake trajectory produced by
learning. MPC keeps its own planning history, so this adds model-derived information;
we do not claim the six-feature observation is a complete Markov state.

A correction of 50 m³/s corresponds to 4.32 million m³ over a full day if physically
feasible. The actor cannot guarantee that its request is delivered. The existing
shared flood filter is a separate evaluation track applied afterward, and may make
a correction larger than ±50 m³/s. No flood safety guarantee follows.

## Why this paper is relevant

Ma et al., [*What Makes Value Learning Efficient in Residual Reinforcement Learning?*](https://arxiv.org/abs/2602.10539),
and the [official DAWN code](https://github.com/Guozheng-Ma/DAWN) motivate two choices:
base-policy-only experience before updates, and LayerNorm in the critics. The paper
studies refinement of pretrained robotics policies, highlighting critic cold starts
and difficulty detecting small residual effects. It omits explicit critic-only
warmup and progressive exploration. We adopt those design ideas, not its reported
performance or safety properties. Our base is engineering MPC, not a pretrained
robot policy; actions, reward, warmup length, episode boundaries, and data differ.
We do not isolate the individual benefit of warmup or normalization in this pilot.

## Replay and learning semantics

Replay stores six raw observations, normalized learnable actions, original rewards,
next observations, and `terminated OR truncated`. The sixth feature is the MPC
proposal in m³/s. Its normalization uses fixed mean/scale = 491.61/2; the first five
statistics are reused from training-only SAC normalization. No evaluation statistics
are fitted and no new normalization-data collection is hidden.

For the residual arm, replay action is normalized delta. The critic maps it and
the normalized base proposal into the combined nominal-request coordinate before
its LayerNorm MLP. The policy gradient passes through this mapping; saturation at
nominal bounds has zero action-value gradient. Replay never substitutes actual
hydraulic delivery for the requested decision.

Entropy/log probability is defined in the normalized **learnable-action** coordinate,
not the clipped nominal-request or actual-release density. The tanh correction is
included; the later noninvertible physical clipping is not added as a Jacobian.
Both matched arms use the same convention and target entropy (-1).

The terminal observation preserves the correct lake level and completed-day inflow,
with sixth feature zero. The wrapper never calls MPC on a terminal observation, and
no bootstrap crosses the annual reset. Each new episode rebuilds the base controller.

## Matched full-action control

A separate full-action SAC actor receives the **same six features**, LayerNorm critic
architecture, fixed normalization, base-only warmup, seeds, years, update budget,
and original reward. It chooses the entire nominal request rather than a bounded
correction. Its critic sees that combined request directly. Its normalized-action
support and physical exploration differ by design from bounded residual control.

This is a **proposal-aware full-action control**, not the existing five-feature
200k SAC baseline. It also computes MPC every day to obtain its sixth feature;
therefore both pilot arms incur planning cost. The old SAC/PPO runs have different
information/capacity/budgets and provide context only, not matched controls here.

## Predeclared first pilot

- Seeds **123–125**, two learned arms, **3,650 steps each** (ten annual episodes).
- The first **365 steps** execute the exact MPC request, with no gradient updates.
  Residual replay action is zero; full-action replay stores the normalized base
  request. Both arms have exactly the same warmup trajectory and profile sequence.
- Then **3,285 SAC updates** per job, replay capacity 3,650 and batch size 128.
  Both use two 64-unit hidden layers, critic-only LayerNorm, Adam 0.0003,
  gamma 0.995, tau 0.005, reward multiplier 0.001, initial alpha 0.2. Actor mean
  starts at zero with log standard deviation -2; full mean corresponds to midrange
  request, residual mean to zero correction. There is no separate critic warmup.
- Original training years 1946–1987, annual resets, 365-day leap approximation,
  no evaporation, original reward/physics, no training safety filter.
- Final checkpoint only; all ten validation years, plain and shared-filter tracks;
  all seeds retained. No test evaluation or hyperparameter selection from validation.
- This is a short implementation pilot, not convergence, sample-efficiency evidence,
  a matched-budget comparison with old PPO/SAC, or evidence of operator deployment.

`results/residual/pilot_v1/plan.json` freezes settings and source hashes. The smoke
stage reproduces all 365 MPC actions exactly under zero correction on training year
1946, checks no terminal planning, then trains both arms for two training-only annual
episodes. Tests also cover nominal saturation, critic mapping/gradients, terminal
handling, zero-mean residual initialization, and custom checkpoint restoration.

## Code and reproduction

- [residual_mpc.py](../src/reservoir_control/systems/lake_como/residual_mpc.py): proposal wrapper, bounded
  composition, combined-action LayerNorm critics, and proposal-aware SAC checkpoints.
- [residual_study.py](../src/reservoir_control/systems/lake_como/residual_study.py): smoke gate, training,
  validation, exact forcing/reset checks, paired results, and compute accounting.

Use a new directory; existing outputs are protected:

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.residual_study --stage prepare --output results/residual/new_pilot
uv run --group reproduction python -m reservoir_control.systems.lake_como.residual_study --stage smoke --output results/residual/new_pilot
uv run --group reproduction python -m reservoir_control.systems.lake_como.residual_study --stage train --output results/residual/new_pilot
uv run --group reproduction python -m reservoir_control.systems.lake_como.residual_study --stage evaluate --output results/residual/new_pilot
```

[Training-loop PNG](../background/images/27_mpc_residual_training_loop.png).
Generated policies, trajectories, reports, and PNGs stay ignored. Nothing is committed.


## First pilot results

Annual validation means across three seeds:

- **MPC + residual, plain:** reward **−17,310** (seed SD **968**), **4.70 flood days**,
  **5.80 low-water days**, **533.42 million m³** unmet demand.
- **MPC + residual, shared filter:** reward **−16,924** (SD **834**), **4.40 flood
  days**, **5.87 low-water days**, **528.11 million m³** unmet demand.
- **Frozen MPC:** plain **−26,361**, filtered **−18,486**. Residual improves all six
  reported mean reward/physical metrics versus that prototype in each track.
- **Matched proposal-aware full-action control:** plain **−26,973**, with **43.03
  low-water days** and **499.13 million m³** unmet demand. It delivers more water
  than residual but pays much larger low-water penalties; there is no dominance
  across all objectives. Filtered full control scores **−26,610**.

Plain residual improves MPC reward in 29/30 seed-year cases and all ten year means.
Its seed means are **−16,194 / −17,913 / −17,823**. These are a first short pilot,
not convergence or a general algorithm ranking. Current five-seed PPO still has a
better mean (−16,505) at a larger budget and without daily MPC calls. No selected
reference/default changes.

Plain corrections average **+6.39 m³/s**, range **−28.42 to +47.15**. The filter may
exceed the prefilter residual bound; 137 decisions do so, as allowed by the protocol.
Remaining floods and finite forecast scenarios prevent a safety guarantee.

The study uses 21,900 training interactions/MPC calls and 19,710 SAC update rounds,
plus 120 validation episodes (43,800 decisions). Measured residual decision time
is about **13.38 ms plain / 16.88 ms filtered**, including planning. All six jobs
share exactly the same warmup/profile sequence, and all evaluation weights and
statistics stay frozen. Full suite: **131 passed, one optional check skipped**.

See [comparison PNG](../background/images/28_mpc_residual_pilot.png) and
[full local report](../results/residual/pilot_v1/report.md). The duration-only
extension below preserves these original results.

## Duration-only extension

The next declared study extends **both learned arms to 36,500 steps per seed**
(100 annual episodes), retaining seeds 123–125, the same MPC, six observations,
±50 m³/s residual bound, LayerNorm, optimizer, and 365-step MPC-only warmup.
Replay capacity stays **3,650**. Thus older warmup data can leave the buffer;
this study tests duration under the existing replay rule, not a larger buffer.

Runs restart from the original initial seeds because the short checkpoints do
not contain replay/environment state. Their first ten completed training episodes
must match the short pilot exactly. Only final checkpoints are evaluated, on all
ten validation years, with plain and shared-filter tracks. Test years stay reserved.

Run with `uv run --group reproduction python -m reservoir_control.systems.lake_como.residual_duration
--stage prepare`, followed by `--stage train` and `--stage evaluate`.
The declared plan and outputs live in ignored
`results/residual/duration_36500_v1/`; the short pilot remains intact.

**Completed result — longer does not help the residual arm.** All six jobs
reproduced the short pilot's first ten training episodes exactly. Only duration
changed, and all final checkpoints were evaluated on all ten validation years:

- **Residual, plain:** reward **−21,799** (seed SD **2,843**), **4.87 flood days**,
  **21.67 low-water days**, unmet demand **549.93 million m³**.
- **Residual, shared filter:** reward **−21,266**, **4.43 flood days**, **21.77
  low-water days**, unmet demand **543.55 million m³**.
- **Matched full action, plain:** reward **−21,353** (seed SD **1,282**), **4.27 flood
  days**, **18.80 low-water days**, unmet demand **548.79 million m³**.
- **Matched full action, shared filter:** reward **−21,343**, **4.27 flood days**,
  **18.80 low-water days**, unmet demand **548.40 million m³**.

Residual reward worsens in **all three seeds and both tracks**, while full-action
reward improves in all three. Residual low-water severity grows from **0.0309 to
0.1617 m·days** plain; flood severity changes **2.6672 → 2.6989 m·days**. The short
residual pilot's advantage over full-action control does not persist at this budget.
The longer residual track also trails filtered MPC on reward.

The longer residual's mean plain correction is **+20.09 m³/s**, versus **+6.39**
in the short pilot; **40.4%** of its requests add more than 45 m³/s. This documents
changed release behavior; it does not isolate whether critic fitting, exploration,
or replay composition caused the deterioration. No intermediate checkpoint sweep,
new hyperparameter tuning, or test-year evaluation was performed. Keep both studies;
do not promote the 36,500-step residual endpoint or claim convergence.

Total new controller training: **219,000 environment steps and MPC calls**, with
**216,810 SAC update rounds**. The 120 annual validation episodes add **43,800**
controller decisions. Summed training MPC time is **2,794.7 seconds** (jobs run
concurrently). Water-balance, bounded corrections, matched exposure, finite-update,
frozen evaluation-state, and source-hash checks pass.
[PNG comparison](../background/images/29_residual_duration.png) ·
[per-seed results](../results/residual/duration_36500_v1/seed_means.csv) ·
[paired duration changes](../results/residual/duration_36500_v1/paired_duration.csv).

## Diagnosing the duration regression

Read-only analysis of all saved plain-track validation trajectories and both
frozen residual checkpoints confirms that the longer actor changes its release
rule, rather than merely encountering a different state distribution.

On the **short policy's same recorded states**, the later actor requests a mean
**+23.99 m³/s** correction versus the early actor's **+6.72**. On the later policy's
states the corresponding means are **+20.61 versus +5.10**. These are raw actor
corrections before nominal saturation; the previously reported +20.09 and +6.39
are effective combined-request corrections on each actor's own trajectories.
Single-state inference reproduces all recorded actor requests within CSV
serialization tolerance; batched float32 forwards can differ slightly.

Extra low-water exposure concentrates in **February–April**. The longer policy
adds about **3,966.7 low-water penalty points/year**, roughly **88%** of its
**4,489.5-point** mean deterioration. Other contributions are +166.7 flood,
+186.6 seasonal shortage, +46.0 surplus, and +123.6 request-adjustment points.

The annual physical accounting reconciles exactly under matched forcing:
**16.51 million m³ extra unmet demand = 32.03 million m³ extra surplus allocation
− 15.52 million m³ terminal storage**. Environmental allocation is unchanged.
The longer controller releases more total water but supplies demand less effectively.
This surplus allocation subtracts environmental flow first, unlike the reward's
surplus term. Neither quantity is a hydropower estimate.

On common short-policy states, min-Q predicts **+0.309** scaled soft-return units
for the later actor's correction versus zero correction; the early critic predicts
**+0.031** for its own correction. Twin-Q disagreement also rises. These are learned
estimates without new counterfactual return labels. The pattern is consistent with
an increasingly aggressive, critic-supported release rule, but does not identify
critic bias, entropy tuning, or replay turnover as the cause. No stabilization
change is justified as proven from these correlations alone.

All training traces are retained in the diagnostic output; changing training-year
profiles prevent interpreting raw episode returns as matched hydrological learning
curves. No extra training, simulation, checkpoint selection or test evaluation was
performed. Prior sources/checkpoints remain hash-verified. Two incomplete diagnostic
attempts are retained locally with explanations for float32/CSV replay-check fixes.

Reproduce into a new directory:

```bash
uv run --group reproduction python -m reservoir_control.systems.lake_como.residual_diagnosis --output results/residual/new_diagnosis
```

[Physical diagnosis PNG](../background/images/34_residual_release_deterioration.png) ·
[common-state critic PNG](../background/images/35_residual_common_state_critics.png) ·
[local accounting](../results/residual/diagnosis_v3/run.json).

A future residual stabilization study should first declare one change and require
it to preserve low-water exposure, not merely increase total release or Q-values.
