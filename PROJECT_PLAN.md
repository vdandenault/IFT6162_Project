# Lake Como reservoir control: project plan

## Research question and current status

Can a learned daily release policy improve the trade-off between downstream water
supply and flood/low-water risk on held-out Lake Como years?

The core comparison is **literature-based MPC versus learning-based control** in
DamEnv, with matched decision-time information, physical constraints, and evaluation
metrics. PPO/MPC references are frozen, the common filter experiment is complete,
and FQI/SAC pilots are implemented. A first MPC-residual pilot improves the MPC
prototype on validation, with a matched proposal-aware full-action control. See [current results](RESULTS.md). The
chronological test split remains reserved while experimental choices are settled.
Actual operator use of the proposed MPC remains unverified.

PPO and FQI are already model-free RL methods. The **hands-on SAC learning phase**
implemented an online, replay-based continuous actor–critic in PyTorch and completed
bounded diagnostics. Its purpose is understanding and a controlled comparison;
the original SAC pilot did not beat PPO. Existing frozen
references remain intact; FQI refinement and SDP are not prerequisites for learning
SAC. A subsequent literature exploration phase connects SAC to residual control,
history representations, offline-to-online learning, and risk-sensitive objectives.
Hydropower still requires justified turbine, head, efficiency, and price
assumptions; stored water or surplus discharge is not measured generation.

## The simulated control problem

Each day, the controller observes lake level, seasonal sine/cosine, and exponentially
averaged past demand, then requests a release in m³/s. No inflow forecast is included
by default. The simulator holds the day's historical inflow through 24 hourly
water-balance updates, clipping releases to level-dependent hydraulic limits.
An action of 100 m³/s over one day corresponds to 8.64 million m³.

```text
next_storage = storage + seconds × (inflow - actual_release - evaporation)
level = storage / surface_area + minimum_level
```

Surface area is 145.9 km². Storage is relative to the model's minimum-level datum,
not total lake volume. Evaporation is disabled. Simulated levels follow our actions;
observed historical levels do not overwrite the trajectory.

The nominal action range is 0–491.61 m³/s. Hydraulic level boundaries are -0.5 and
1.25 m; reward penalties begin below -0.45 m or above 1.24 m. Hydraulic release
limits do not guarantee flood avoidance under extreme inflows. The reward penalizes
unmet demand, flooding, low water, surplus release, and action clipping. Demand
penalties vary seasonally; report physical metrics alongside weighted return.

An episode has 365 decisions and 8,760 hourly updates. Reset restores level 0.35 m
and demand average zero; training randomly selects a year, not an initial level.
Stop rollouts on `terminated or truncated`. A wrapper corrects the final observed
level. Annual resets still prevent storage continuity between years; default PPO
uses `time_limit_bootstrap=False`. Any different terminal treatment needs an explicit
comparison protocol. MPC does not automatically repair resets or observations.

## Data and evaluation protocol

Daily Lake Como inflow, downstream demand, and observed levels cover 1946–2010.
Files live in `external/gym4real/gym4real/data/dam/`: `inflow/`, `demand/`, `level/`,
`MEF.csv` (environmental flow), and `metadata_dam.json` (provenance/license metadata).
The audit records hashes and unresolved provenance/license discrepancies locally.

[configs/splits.json](configs/splits.json) defines train 1946–1987 (42 years),
validation 1988–1997 (10), and test 1998–2010 (13). Leap years retain the upstream
365-step approximation: the final sample is unused and seasonal alignment is
approximate. Excluding leap years remains available for sensitivity checks.
The original paper's 52/13 membership is a separate reproduction protocol.

Select settings on validation, retain every learner seed, and reserve chronological
test years for the final comparison. Future historical inputs are simulator forcing,
not policy information. Any perfect-inflow controller must be labeled as an oracle.

The current PPO baseline is 1M steps with a capped learning-rate scheduler, mixed
precision, zero entropy bonus, and seeds 123–127. See [PPO findings](docs/PPO.md)
and [current settings](configs/ppo.json). Its improved reward does not imply every
physical outcome improved. [Evaluation](docs/EVALUATION.md) documents the common
runner, metrics, pairing, provenance, runtime, and validation results.

## Remaining work

### Phase 1 — environment and data: complete

- [x] Pin the simulator fork and uv environment; audit years, units, checksums, and licenses.
- [x] Run the environment/rule walkthrough; check water balance, clipping, timing, and observation boundaries.

### Phase 2 — benchmark and evaluation: complete

- [x] Implement chronological membership and retain separate upstream reproduction.
- [x] Reproduce rule/PPO benchmark qualitatively and investigate five-seed sensitivity.
- [x] Complete controlled PPO ablations and stop duration tuning at 1M.
- [x] Evaluate every year of a selected split; save paired metrics, reward components, provenance, terminal storage, and decision runtime.
- [x] Compare four rules and five PPO seeds on all validation years; produce tables and paired plots without adding objectives.
- [x] Extract the deterministic MPC specification from the [focused literature review](docs/LITERATURE_REVIEW.md#implemented-deterministic-mpc-prototype).

### Phase 3 — engineering versus learning

- [x] Specify the initial MPC observations, training-only climatology, hourly planning dynamics, local solver, seven-day horizon, surrogate objective, constraints, and zero terminal value.
- [x] Fit training-only seasonal inflow/demand climatology with no held-out future inputs.
- [x] Compare seasonal, persistence, and training-fitted seasonal-anomaly forecasts under matched one-day search and both costs. Persistence improves return in all validation years; prediction-error ranking differs from control ranking, and unmet demand increases. See [forecast study](docs/EVALUATION.md#causal-past-inflow-forecast-comparison).
- [x] Extend causal forecasts to seven-day MPC with horizon-specific training fits and fixed costs/solver: both improve every validation year. Persistence returns -28,931 versus seasonal -50,274, but raises unmet demand and solver failures. Preserve severity/supply trade-offs and the extra-history label; see [seven-day study](docs/EVALUATION.md#causal-forecasts-in-seven-day-mpc).
- [x] Audit additional solver failures via exact replay of all 10,950 decisions: all hit the iteration cap; 148/149 discarded persistence iterates have lower predicted cost within request bounds. Fixed-tail first-action slices improve 141/149 failed states. See [solver audit](docs/EVALUATION.md#audit-of-additional-causal-mpc-solver-failures).
- [x] Test cost-checked iteration-limit retention under persistence with fixed forecast/cost/budget: return improves in all validation years (-26,361 versus -28,931), low-water severity falls 39%, unmet demand rises 0.16%, and all 131 retained iterates remain flagged as failures. Experimental only; see [retention study](docs/EVALUATION.md#cost-checked-retention-of-iteration-limit-iterates).
- [ ] Resolve the supply-versus-level-risk criterion before promoting a baseline or tuning safety margins. Match PPO information before a final comparison.
- [x] Implement deterministic MPC and test physics, information timing, optimization, fallbacks, and resets; apply only the first action before replanning.
- [x] Audit boundary failures and short-horizon optimization without changing the deployed controller: exact historical replay, selected grid/refinement checks, and seven-day cost slices. Forecast, objective, and solver issues remain distinct; see [diagnosis](docs/EVALUATION.md#boundary-and-optimizer-diagnosis).
- [x] Run a clearly labeled perfect-inflow counterfactual with fixed demand forecast, surrogate, solver, and horizon: validation return improves from -50,274 to -28,962 across all ten years, but violations and optimization issues remain.
- [x] Isolate objective preferences on common hindsight grids: the surrogate has material binary-cost regret in 19/36 selected short-horizon problems; 15 change the first release. See [objective audit](docs/EVALUATION.md#binary-versus-surrogate-objective-on-common-grids).
- [x] Compare objectives in closed loop under common one-day exhaustive hindsight search: binary improves validation reward in all ten years (-15,976 versus -50,066), but unmet demand and flood-exceedance severity increase slightly. This is not the deployed seven-day MPC.
- [x] Test training-only seasonal inflows under the same one-day search, retaining severity: binary versus surrogate returns are -50,463 versus -50,716; the large hindsight gain disappears. See [forecast comparison](docs/EVALUATION.md#non-oracle-forecasts-under-the-same-search).
- [ ] Establish the engineering-risk criterion using supply, event counts, and exceedance severity; investigate forecast uncertainty and safety margins before promoting a controller or extending the horizon. Preserve the initial baseline.
- [ ] Validate engineering-baseline quality: the initial seven-day prototype performs poorly. Separate forecast, surrogate-objective, horizon/terminal, and optimizer effects; retain the first result and avoid claiming PPO superiority from this prototype alone.
- [ ] Compare full annual trajectories, physical metrics, both reward scores, terminal storage, training interactions, and decision time. Inspect failures and wet/dry years using a predeclared selection rule.
- [ ] Run one focused ablation (horizon or forecast availability), then freeze the protocol before final test evaluation.
- [ ] If useful and feasible, add one optional method: tree-based FQI (preferred), scenario MPC, or a reduced-state SDP with grid-refinement checks.

Cestari et al. (2023) uses hourly actions, a 24-hour horizon, and a 90-day simulation
starting January 1, 2000; reported planning uses 380 scenarios and 50 Monte Carlo
inflow realizations. Its costs differ from DamEnv. Our MPC will be an adaptation,
not a numerical reproduction. Disclose any planning-model or reward surrogate,
validate resulting actions in unchanged DamEnv physics, and do not transfer scenario
guarantees without checking assumptions. Deterministic/scenario comparisons need
the same forecast generator and horizon. An approximate oracle is not automatically
a performance bound.

### Phase 4 — understand and extend the control problem

Start from [DamEnv's existing equations](https://github.com/Daveonwave/gym4ReaL/blob/main/gym4real/envs/dam/env.py)
(use our pinned local source for experiments) and investigate **information and safety**
before expanding the algorithm list. Preserve the current PPO/MPC results as the
reference; these are new experiments, not retroactive benchmark changes.

- [x] Document hourly storage balance, hydraulic clipping/forced spill, all reward terms, information timing, no evaporation, resets, and endpoint versus hourly safety. See [equations and safety definitions](docs/EVALUATION.md#reservoir-equations-and-safety-definitions).
- [x] Compare MPC horizons **1, 3, and 7 days** with persistence, surrogate, retention, initialization recipe, and solver cap fixed: validation returns -36,015 / -28,077 / -26,361. Seven days beats one day in all years and three days in nine; shorter horizons use less CPU and leave less demand unmet. Keep seven days as the current best mean-reward setting, not a universal optimum. See [matched horizon study](docs/EVALUATION.md#matched-mpc-horizon-experiment).
- [x] Add **last completed-day inflow** through `PastInflowObservation`, with training-only January 1 fallback, reset isolation, terminal handling, and causality tests. Dynamics/reward and the first four observation values are unchanged; no current/future forcing is supplied before decisions.
- [x] Train five past-inflow PPO seeds for 1M steps each, reusing the verified five 1M four-feature controls with exact daily evaluation replay. Mean reward improves -17,232 → -16,505 and seed SD 1,823 → 238, while unmet demand rises 2.0%. Three seeds improve; two worsen. No automatic promotion. See [observation ablation](docs/PPO.md#past-inflow-observation-ablation).
- [x] Analyze changed decisions and wet/dry outcomes with years selected solely by total validation inflow: dry 1990 and wet 1993. Past-inflow PPO reduces low-water events in both, but unmet demand rises in every validation year and wet-year flood days barely change. See [decision analysis](docs/PPO.md#what-changes-in-wet-and-dry-years).
- [ ] Confirm project expectations with the instructor; this controlled engineering analysis is stronger than adding a feature alone, but no particular grade or acceptance is assumed.
- [x] Audit one-day flood feasibility from 40,150 saved states under known-inflow and training-calibrated stress diagnostics. Past-inflow PPO has 208/215 flood endpoints certified infeasible from that day’s state; MPC has daily-safe grid alternatives for 79/123. Distinguish endpoint/hourly safety and grid witnesses from certificates; this does not establish avoidability from the annual reset. See [feasibility audit](docs/EVALUATION.md#flood-safety-feasibility-audit).
- [x] Specify a bounded common filter experiment: float32 observed level, exact past storage/release telemetry, five training-residual interval scenarios plus persistence, hourly flood priority, and explicit recovery. These finite scenarios are not a robust inflow bound or unconditional guarantee.
- [x] Complete one common-filter experiment (120 validation episodes). MPC reward improves -26,361 → -18,486, flood days 12.3 → 4.5, and unmet demand falls; low-water severity worsens slightly. PPO barely changes (8 interventions). Actual forecast-range breaches prevent a safety guarantee. See [filter findings](docs/EVALUATION.md#common-finite-scenario-safety-filter).
- [x] Freeze the past-inflow PPO and seven-day persistence/retention MPC references with plain and shared-filter evaluation tracks in `configs/experiments/control_protocol_v1.json`. Preserve four-feature PPO as an ablation, stop PPO/MPC tuning, and reserve the test split while FQI is developed.
- [x] Implement and run a first **tree-based FQI pilot** using the frozen five-feature observation, 30,660 training-only transitions, 21 actions, and 60 Bellman fits. Three forest seeds share one dataset. Validation rewards: plain -20,062; shared-filter -19,219. This is an Ernst et al. adaptation, not a reservoir-paper reproduction or mature benchmark. See [FQI pilot](docs/EVALUATION.md#first-tree-based-fqi-pilot).
- [ ] Audit FQI conditional state/action coverage, target/fit behavior, and discretization before selecting a larger dataset or backup budget. Preserve pilot results and the frozen PPO/MPC controls; no test evaluation until FQI is settled.
- [ ] Use stochastic DP only as an optional reduced-state, coarse-grid reference or FQI sanity check. Fit inflow transition probabilities from training years, account for season, and check grid refinement. Do not make both SDP and FQI prerequisites: PPO plus a credible MPC already covers the main comparison, and extra methods must answer a concrete question.

### Phase 5 — hands-on online RL: implement and understand SAC

**Status:** [SAC walkthrough](docs/SAC.md), CPU PyTorch agent, correctness checks,
20-episode training-only smoke, and a separate Pendulum check are complete. The
five-seed 200k-step study scores −21,064 plain / −20,871 shared-filter on validation;
PPO remains the better reference at its larger 1M-step budget. All seeds and both
tracks are retained. PNGs and local reports remain ignored; test years are reserved.

**Learning goal:** understand and implement replay, bootstrapped value learning,
policy gradients, exploration, and target networks on a real control problem.
PPO already learns directly from interaction; FQI already learns from a fixed
batch. SAC adds an online, off-policy continuous actor–critic: collect experience,
reuse it in replay, learn Q-functions, and learn a stochastic actor. “Model-free”
means the plain agent does not call reservoir physics to plan actions or form its updates;
the simulator still supplies physical transitions during data collection.

**Primary experiment:** compare mean full validation reward at a predeclared
interaction budget against the frozen references. Secondary outcomes are learning
curves, seed variability, water-supply/severity trade-offs, and compute. A useful
learning phase does not depend on achieving a new best score.

- [ ] Read and annotate a small primary-source set: Wang et al. (2020), already in `/papers`, for a reservoir MDP and negative SAC result; Tabas & Samadi (2024), *Fill-and-Spill*, for continuous-action reservoir SAC/TD3; and Haarnoja et al. for SAC's actual update equations. Extract observations/timing, action units, reward, training data, episode boundaries, hyperparameters, comparators, and limitations. The initial screening and links are in [the focused review](docs/LITERATURE_REVIEW.md#hands-on-online-rl-phase-sac).
- [x] Write the reservoir MDP and each SAC update in our own words before coding: what is observed, what is hidden, what enters replay, what the actor/critics predict, what is detached, and why target networks are needed. Connect SAC's bootstrapped target to FQI and its actor update to PPO. Derive the modern twin-Q/automatic-temperature variant explicitly; do not mix incompatible equations from different SAC versions.
- [x] Implement a small transparent PyTorch agent: replay buffer, bounded stochastic actor, two Q-critics, target critics, and an explicit training loop. A trusted implementation may be a test reference, but keep the learning exercise's updates visible in our code. Reuse the environment adapter and evaluator; do not rewrite DamEnv or change its reward for an apparent win.
- [x] Add components incrementally: (1) transition storage/sampling and terminal masks, (2) critic Bellman regression and target updates, (3) reparameterized actor gradients, (4) entropy temperature. Check target detachment, gradient flow, bounded actions, tanh log-probability correction, and consistency of normalized action units/entropy targets. Explain each component and one failure it prevents.
- [x] Verify on a small known-value control task and short training-only DamEnv runs before expensive experiments. Confirm reset/terminal handling, no future-inflow leakage, finite targets/gradients, and checkpoint round trips. Debug correctness before changing reward weights, adding memory, or running a broad hyperparameter sweep.
- [x] Predeclare the first reservoir study: same five-feature past-inflow observation, original reward, action limits, chronological splits, annual resets, and no bootstrap across episode ends. Fit normalization using training data only. Record any reward scaling and its interaction with the entropy temperature; evaluate with original reward units. Keep the finite-scenario safety filter outside initial training and evaluate both frozen plain/shared-filter tracks afterward.
- [x] Choose and log warmup, replay size, minibatch size, update-to-data ratio, target-update rate, network capacity, evaluation policy, checkpoints, and a bounded interaction budget before validation. Match PPO's five learner seeds for the full study. Count warmup interactions and report gradient updates and runtime as well as environment steps: equal interaction budgets are not equal computation, and FQI's shared-data pilot is a separate budget/design.
- [x] Run one bounded SAC study and explain behavior: mean/weakest-seed reward, learning curves, action distribution, entropy/temperature, twin-Q disagreement, and wet/dry supply/severity outcomes. These diagnostics are not calibrated uncertainty or safety guarantees. Do not silently select the best seed or best validation checkpoint, and do not change the frozen PPO/MPC/FQI results.
- [x] Produce a short walkthrough and PNGs in ignored `background/`, showing replay → critic target → actor update → release. End with what was learned, implementation checks, empirical limits, and whether SAC merits the final comparison. A losing SAC result can still be a strong learning outcome.

- [x] Diagnose SAC's release timing and critics without tuning: extra unmet demand decomposes into 82.53 million m³ surplus plus 15.34 million m³ terminal storage; 28/60 selected training anchors have a cheaper identical-state request under realized inflow, and min-Q ranks it lower in 20. Conditional branch returns do not establish population calibration. See [diagnosis](docs/SAC.md#release-timing-and-critic-diagnosis).
- [x] Test frozen-actor critic fitting on paired clipping transitions: 210 fit pairs / 235 separate checking pairs from training years, 1,000 fixed steps per seed/arm. Checking cheaper-request ranking improves 28.5% → 59.1% with ordinary Bellman regression and 53.6% with added pair loss. General value error does not improve; actor and all controller scores remain unchanged. See [critic fitting](docs/SAC.md#frozen-actor-critic-fitting-experiment).
- [x] Separate focused sampling from extra updates on one fixed pool: at 100 / 1,000 updates, uniform checking-pair ranking is 30.6% / 34.5%, versus focused 43.4% / 51.9% (baseline 28.5%). Focus wins at 1,000 for all five seeds, but broader checking target error worsens in both arms. No actor or reference changes. See [sampling study](docs/SAC.md#sampling-emphasis-versus-update-count).
- [x] Check broad error against fresh ordinary-state returns and test 10% focus: 420 training-year anchors × eight continuations; conditional return MSE is 43.868 unchanged / 43.860 uniform / 43.905 with 10% focus at 1,000 updates. Cached-target error overstates the apparent deterioration, but 10% focus still fails the declared broad-preservation rule despite improving pair ranking to 41.7%. No actor or original score changes. See [fresh-reference check](docs/SAC.md#fresh-return-checks-and-10-focused-replay).
- [x] Compare frozen bootstrap versus fresh rollout-return labels on identical fitting states/batches: at 1,000 critic updates, checking return MSE is 44.388 bootstrap / 28.554 rollout (43.868 unchanged), improving all five seeds. Single-action rollout fitting still has poor action differences (36.6% pair ranking versus bootstrap 55.7%) and overfits from 100 to 1,000. No actor or original result changes. See [target comparison](docs/SAC.md#bootstrap-versus-rollout-return-targets).
- [x] Complete the final bounded iterative-TD diagnostic: frozen actor/alpha, matched data/batches/action noise, target tau 0 versus .005. At 1,000 updates, checking return MSE is 44.145 frozen / 44.493 moving (43.868 unchanged). Target updates alone do not resolve this gap; no convergence or policy improvement is claimed. See [iterative TD](docs/SAC.md#iterative-td-with-the-actor-frozen).
- [x] Test action-diverse rollout labels with the actor frozen: adding ±50 m³/s alternatives reduces checking gap MAE 0.2482 → 0.1877, but remains worse than unchanged critics (0.0363); directional agreement 44.5% versus 49.3% unchanged. Conditional return MSE still improves, without reliable release comparisons. All five seeds and 100/1,000 budgets retained; no actor or reference changed. See [action-diverse diagnosis](docs/SAC.md#action-diverse-rollout-supervision).

Start with **one SAC implementation**, not a SAC/TD3/DDPG/recurrent-agent sweep.
TD3 is a literature-supported alternative if the method review changes the choice,
not an automatic extra controller. Memory, multi-year episodes, synthetic inflows,
and learned dynamics are later extensions because each changes another part of
the problem. The standalone implementation and first bounded study are complete; inspect its
supply/low-water limitations before increasing the budget or adding extensions.

#### Environment tooling and related benchmarks

These tools serve different roles from learning algorithms such as PPO, FQI, or SAC:

- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** supplies the standard environment interface (`reset`, `step`, observation/action spaces), wrappers, and reference environments. We already use it through DamEnv and pin version 1.1.1. Keep this interface for SAC and later extensions.
- **[Gym-Anything](https://github.com/cmu-l3/gym-anything)** builds environments for agents operating real software, with task setup and verification. Our reservoir already has a numerical simulator and measurable outcomes, so no integration is planned. Revisit only if controlling external engineering software becomes an explicit research question.
- **[ExploitGym](https://github.com/sunblaze-ucb/exploitgym)** evaluates agents developing exploits against real-world software vulnerabilities. It is a separate cybersecurity benchmark, with no current reservoir-control experiment or integration planned.

- [x] As part of SAC verification, run Gymnasium's environment checker on the intended adapter/wrappers where supported; investigate warnings and retain explicit tests for causality, terminal observations, action units, and physical dynamics. Interface compliance alone does not establish physical correctness. Use a small continuous-action reference task for debugging before DamEnv; reference-task scores are implementation evidence, not the project's engineering result.
- [x] Include a short explanation in the SAC walkthrough of **agent versus environment versus evaluator**: the agent chooses a release, DamEnv advances reservoir physics through Gymnasium's interface, and the evaluator measures complete trajectories. Learn from Gym-Anything's explicit setup/verifier separation without introducing a software-use or cybersecurity benchmark into the main comparison.

### Phase 6 — literature exploration: build on the RL foundations

**Learning goal:** understand four research directions from Pierre-Luc Bacon and
collaborators through focused reading and small reservoir experiments. The first **MPC with residual RL** pilot is implemented and evaluated. Read the
relevant paper before each further branch; the remaining experiments
are optional follow-ups, not four additional controllers required for completion.
The remaining branches are proposed adaptations; the first 6A pilot is implemented
and does not establish general method superiority or final-test performance.

For each reading, write a short explanation of the problem, core equations,
assumptions, evidence, and what transfers to our reservoir. Before implementing an
extension, specify one hypothesis, a matched control, an interaction/compute budget,
and a stopping criterion. Use training years for fitting and debugging, validation
for development, and keep test years reserved until methods are frozen. Preserve
the existing references and distinguish changes to information, initialization,
architecture, and objective. Record results even when an extension does not help.

#### 6A — residual RL: learn a correction to MPC (first priority)

**Status:** short pilot and duration-only matched extension are complete.
The 3,650-step residual pilot scores −17,310 plain / −16,924 filtered. At
36,500 steps it worsens to −21,799 / −21,266 in all three seeds; the full-action
control improves to −21,353 plain. No reference is replaced. See
[implementation and results](docs/RESIDUAL_RL.md).

Read [*What Makes Value Learning Efficient in Residual Reinforcement Learning?*](https://arxiv.org/abs/2602.10539)
and inspect the authors' [DAWN implementation](https://github.com/Guozheng-Ma/DAWN).
The paper studies refinement of frozen pretrained policies; using our engineering
MPC as the base controller is a new adaptation.

**Question:** can a learned correction improve MPC under imperfect inflow forecasts?

- [x] Implement and explain `clip(MPC_release + delta, 0, 491.61)`, with delta bounded to ±50 m³/s before the common filter. Replay uses normalized delta, critic inputs use combined nominal request, and nominal saturation is included in actor gradients. Original hydraulic clipping/reward are unchanged.
- [x] Read DAWN's warmup/scale arguments and algorithm; adapt base-only implicit warmup plus critic LayerNorm, without explicit critic warmup or progressive exploration. Document the reservoir/base-controller/budget differences; this is not a paper reproduction or component ablation.
- [x] Compare frozen MPC, residual SAC, and a matched proposal-aware full-action SAC control. Both learned arms see the same six inputs, LayerNorm, warmup, and 3-seed/3,650-step budget. Preserve old five-feature SAC/PPO as unmatched contextual references. Count all MPC calls and time; evaluate every validation year with plain/shared-filter tracks.
- [x] Complete the first bounded residual pilot, with exact zero-correction MPC reproduction and both training smoke gates. Report corrections, seed variability, solver issues, supply/event/severity metrics, and resource cost. Mean correction +6.39 m³/s; both tracks improve the MPC prototype, but full-action control supplies more water at the cost of low-water exposure. Bounds/filter do not imply safety.
- [x] Extend both matched arms to 36,500 steps/seed, changing duration only. All six short prefixes reproduce exactly; final validation shows residual deterioration in every seed, while full-action improves. Report both, retain the short pilot, and do not promote the longer residual endpoint. Original references and test years remain reserved. Optional warmup/normalization ablations remain separate; their individual effects are not established.

- [x] Diagnose residual duration regression using saved trajectories and both frozen checkpoints on common states: extra low-water penalty explains 88% of reward loss; 32.03 million m³ additional surplus and 15.52 million m³ lower terminal storage reconcile 16.51 million m³ extra shortage. Actor corrections increase on both state sets; critic estimates are not calibrated counterfactual returns. No new training or validation rollouts.

#### 6B — learning from history: represent hidden hydrological conditions

Read [*Bridging State and History Representations: Understanding Self-Predictive RL*](https://arxiv.org/abs/2401.08898).

**Question:** does recent inflow/level history provide useful information beyond
our current five-feature observation?

- [ ] Explain partial observability and the paper's self-predictive latent objective, including target construction and stop-gradient treatment.
- [ ] Begin with current-observation SAC versus a fixed window of completed-day observations/actions. Define reset padding and causal normalization. If this motivates further work, compare a recurrent encoder, then its self-predictive auxiliary loss with architecture/history held fixed.
- [ ] Audit replay sequence sampling, episode boundaries, recurrent state resets, and evaluation timing. Report the extra information and capacity: this changes the observation experiment. Adding an LSTM alone is not a reproduction of the paper, and latent prediction accuracy alone does not establish better control.

#### 6C — offline-to-online RL: reuse experience before interacting

Read [*The Three Regimes of Offline-to-Online Reinforcement Learning*](https://arxiv.org/abs/2510.01460).

**Question:** when does our fixed FQI training dataset help SAC learn online?

- [ ] Explain the stability/plasticity trade-off and how dataset quality and pretrained-policy quality affect the starting point. Treat the paper as a framework to investigate, not a single drop-in algorithm.
- [ ] First compare SAC from scratch with SAC whose replay buffer is preloaded with the same training-only FQI transitions. Verify reward scaling, action units, terminal masks, and normalization compatibility. Keep new online interactions and update budgets matched; report offline collection cost separately and in total.
- [ ] Consider explicit neural critic pretraining or actor distillation only as a separate experiment. Our tree-based FQI weights cannot directly initialize SAC networks. Replay preloading alone does not reproduce the paper's full pretraining/fine-tuning setting.

#### 6D — risk-sensitive RL: focus on severe outcomes (advanced option)

Read [*Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity*](https://arxiv.org/abs/2602.03778).

**Question:** can explicit tail-risk optimization reduce severe reservoir losses
at an acceptable cost to average water supply?

- [ ] Explain CVaR, the distinction between trajectory-level and stage-wise risk, and why the paper introduces an augmented state and a different Bellman formulation. This is a substantive algorithm/objective change, not just a larger flood penalty.
- [ ] Define the loss and tail level before implementation: poor total return and physical flood severity are different targets. Compare expected-return and risk-sensitive controllers under common scenarios, reporting both mean performance and the chosen tail metric.
- [ ] Assess scenario adequacy first. Ten validation years provide weak evidence for rare-tail estimates; repeated learner seeds do not create independent hydrological years. Any additional scenario generator must be justified and fitted on training data only. CVaR optimization does not imply a hard no-flood guarantee.

**Phase outcome:** a concise reading note for each direction, one prioritized
implementation with a controlled result, and a clear explanation of what we
learned. Add intuitive PNG diagrams and walkthroughs to ignored `background/` as
methods are understood. Broader implementation remains optional; do not delay the
final comparison to exhaust every paper.

### Phase 7 — final analysis and presentation

- [ ] Evaluate the frozen controllers on held-out test years and report paired trade-offs, seed variability, and limitations.
- [ ] Add reward-weight or hydropower extensions only if time and defensible parameters support them.
- [ ] Prepare the report/poster and interactive results website; keep code, figures, and claims reproducible.

## Reading path and artifacts

Read [PPO findings](docs/PPO.md), [evaluation](docs/EVALUATION.md), and
[literature review](docs/LITERATURE_REVIEW.md) as needed. Start with
[environment walkthrough script](scripts/analysis/environment_and_data.py) for the system or
[controller comparison script](scripts/analysis/controller_comparison.py) for current comparisons;
intermediate analysis scripts are optional detailed evidence.

`configs/ppo.json` is the current selected baseline; `configs/experiments/ppo_studies.json`
consolidates exact historical study plans. Keep experiment code in `src/lake_como/`
and simulator changes in the fork at `external/gym4real/`. Generated `results/`,
`reports/`, `background/`, and local PDFs remain ignored. Nothing is committed
without review and approval.
