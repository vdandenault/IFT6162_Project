# Lake Como reservoir control: project plan

## Proposed Research Question

Can a learned daily reservoir release policy improve the trade-off between downstream water supply and flood/low-water risk on held-out Lake Como years?

Our core comparison is **literature-based deterministic MPC versus PPO** on the Lake Como system, using DamEnv as the shared simulator. MPC is the engineering baseline and a required part of the project; PPO is the learning-based controller and published RL reference. Identify held-out conditions where learning improves decisions, and report failures and trade-offs as well as improvements.

Use the same decision-time information, physical constraints, and evaluation objective for the main comparison. Reproducing Gym4ReaL PPO with its original inputs is a separate reference experiment. We can call MPC a literature-based engineering controller; actual use by Lake Como operators remains unverified and requires separate evidence.

Aim for **two well-validated substantial controllers**, with **FQI as the preferred third method** after the core comparison works. Scenario MPC and stochastic DP are optional alternatives/extensions, selected through a focused literature review. Simple rule baselines remain sanity checks. There is no established course requirement here for a particular number of algorithms.

Keep the initial scope on demand, flood, and low-water control. Hydropower remains optional and requires turbine specifications, hydraulic head, and efficiency assumptions; revenue modeling also requires electricity prices.

## How the reservoir simulation works

We will use `gym4real/dam-v0`, the reservoir environment in Gym4ReaL. It combines a water-balance model with historical Lake Como inflow and downstream demand. Our controller decides how much water to release, and the simulator calculates how that decision changes the lake level.

### One simulated day

Each day follows the same sequence:

1. **Observe the lake.** The controller receives the current water level, the time of year, and a running estimate of recent demand.
2. **Choose a release.** It requests a flow rate in m³/s to apply over the coming day. For example, 100 m³/s sustained for a day corresponds to 8.64 million m³ of water.
3. **Simulate the next 24 hours.** The simulator uses that day's historical inflow and updates storage once per hour. At each hourly update, it adjusts the requested release to the flow the dam can physically deliver at the current level.
4. **Evaluate the result.** The environment calculates the day's reward and returns the updated observation. The controller then makes the next day's decision.

The underlying calculation is a water balance:

```text
change in stored water = water entering - water released - evaporation
```

Flow rates are multiplied by the duration of each update to obtain water volumes. The model converts storage to level using a constant lake surface area of 145.9 km². Its storage is measured relative to the model's minimum lake level. Evaporation is disabled in the default configuration.

The simulated level follows our controller's decisions throughout the episode. Historical measured levels are available separately for analysis; they do not replace the simulated level each day.

### What the controller knows

The default observation contains four numbers: lake level, two values encoding the time of year (sine and cosine), and an exponentially weighted average of past demand. The controller has no inflow forecast by default.

The controller acts before the simulator processes that day's inflow and demand. After the transition, the demand average includes the day that just elapsed. Any forecasts we add must use information available when the decision is made. Using the actual future inflows would be a separate, explicitly labeled perfect-information comparison.

### What makes a good decision

The controller must balance keeping water available for future demand against leaving room for incoming water. Releasing too little can leave demand unmet or allow flooding; releasing too much can waste water and leave the lake too low later.

The default reward penalizes unmet demand, flooding, low water, surplus release, and requests that require adjustment to feasible releases. The demand penalty also varies seasonally. We will preserve these settings for the initial reproduction and report each outcome separately so we can explain what a higher total reward means physically.

The default action range is 0–491.61 m³/s, but the actual release also depends on the lake level and hydraulic constraints. Those constraints limit what the dam can release; extreme inflows can still cause flooding. The hydraulic model uses level boundaries of -0.5 and 1.25 m, while reward penalties start below -0.45 m or above 1.24 m.

### Starting and ending a simulation

A default episode lasts one year: **365 daily decisions**, each containing 24 hourly water-balance updates. It starts at a lake level of 0.35 m with the demand average set to zero. During training, `random_init` selects a historical year; the initial lake level stays fixed.

The environment follows the Gymnasium interface: `reset()` starts an episode and `step(action)` advances one day. Reaching 365 days sets `truncated=True`; rollout code should stop when either `terminated` or `truncated` is true.

## Data: source, availability, and unresolved provenance

The local main paper (Sections C.1, D.1, E.1) describes Lake Como, northern Italy, daily water demand, inflow, and level records for 1946–2010: 65 annual profiles, split into 52 training and 13 test years.

The immediate downloadable source is the [upstream data directory](https://github.com/Daveonwave/gym4ReaL/tree/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/gym4real/data/dam):

| Files | Role |
|---|---|
| `demand/water_demand_{train,test}.csv` | Daily downstream demand profiles |
| `inflow/water_inflow_{train,test}.csv` | Daily inflow forcing; train/test files inspected directly |
| `level/water_level_{train,test}.csv` | Observed level for diagnostics or possible expert-action reconstruction |
| `MEF.csv` | Minimum environmental flow schedule |
| `metadata_dam.json` | File descriptions, hashes, and dataset license metadata |



## Simulation timeframes and computation

| Time concept | Initial setting | Meaning |
|---|---|---|
| Control interval | 1 day | Policy chooses once per simulated day |
| Internal integration | 1 hour | 24 physics updates per action |
| Training episode | 365 days | 365 agent steps / 8,760 internal updates |
| Historical coverage | 65 annual profiles | Distinct forcing years, not 65 years of live interaction |
| Published training budget | 200,000 agent steps | About 548 annual rollouts with repeated training years |
| Published test suite | 13 annual episodes | 4,745 agent steps per policy/seed |
| MPC planning window | Proposed 7, 14, 30 days | Separate from episode length; execute only the first action, then replan |
| Seasonal diagnostics | Proposed 90-day summaries | Derive from full annual runs, preserving prior storage history |
| Multiyear extension | Deferred | Requires explicit calendar stitching, continuous storage, and boundary handling |

The paper reports 0.045 seconds per DamEnv episode in Table 2. Treat this as a published simulator timing, not a measured local speed or a training runtime guarantee. Training includes network updates, evaluation, logging, and possibly optimization. Benchmark locally before budgeting full runs; MPC solver cost may dominate.

Measure elapsed time for 100 annual baseline rollouts and a 10,000-step learner pilot. Extrapolate training time from measured seconds/step, then add evaluation and weight/seed sweeps. Report hardware and software versions. Changing integration from 24 to 1 for speed changes the simulator and should not be hidden.

## Evaluation design

1. **Reproduction track:** retain upstream train/test membership and reward settings. Label results as reproduction of that split. Create a validation subset from training years only for tuning.

2. **Generalization track:** proposed chronological split: train 1946–1987 (42 years), validate 1988–1997 (10), test 1998–2010 (13). Reassemble signals by year only after alignment checks. This is our protocol, not the published one.
3. Fit scalers, forecasters, discretization choices, and policy hyperparameters using training/validation only. Lock choices before final test evaluation. Do not select reward weights by looking at test performance.
4. Compare every controller on identical forcing years and initial conditions. Within an episode, carry storage forward from the controller's own previous actions. Never replay historical levels as the next state of a new controller.
5. Report annual deficit volume (convert daily mean rates using 86,400 seconds), flood days, low-water days, maximum level, clipping frequency/magnitude, surplus-release volume, terminal storage, and weighted return. Inspect within-day extremes if claiming hourly safety; daily endpoint metrics alone miss them.
6. Use paired annual differences, per-year plots, and uncertainty across years. For stochastic learners, start with 3 seeds and expand to 5 if budget allows; distinguish training-seed variability from hydrological variability.
7. Plot flood versus deficit trade-offs across a small predeclared reward-weight grid. Identify nondominated policies among those tested; this is an empirical frontier, not proof of the global Pareto frontier.
8. Annual resets can encourage terminal depletion. Report terminal storage; test a terminal-value/reserve treatment as an explicit extension rather than silently altering the reproduction.

Observed operator releases are not directly available as an executable counterfactual policy in the inspected setup. Reconstruction from levels and inflows needs unit/calendar alignment and water-balance assumptions (including evaporation and unmodeled flows). Treat reconstructed actions as uncertain historical evidence. Simulator rollouts permit policy comparison under model assumptions; held-out next-level prediction alone does not validate policy quality or real-world causality.

## Controller comparison: required MPC/PPO core and optional extensions

The new methodological reference is [Scenario-based model predictive control of water reservoir systems](papers/Scenario-based%20model%20predictive%20control%20of%20water%20reservoir%20systems.pdf), by Raffaele G. Cestari, Andrea Castelletti, and Simone Formentin (2023; local copy arXiv:2309.00373v1).

### What this paper contributes

The paper implements scenario-based MPC for Lake Como. At each decision, it generates plausible future inflow trajectories, optimizes one release sequence against the average cost across those scenarios, applies the first release, and replans after receiving new observations. Scenarios come from Prophet with an expanding training window. Its comparisons include deterministic MPC using climatology, deterministic MPC using the nominal Prophet forecast, and oracle MPC with perfect future inflow information.

It does **not** implement stochastic DP or fitted Q-iteration. The `SDPT3` named in Section 4 is a numerical optimization solver used with CVX, not an SDP reservoir controller. FQI and stochastic DP have a separate reservoir-specific reference: Castelletti et al. (2010), linked in the focused literature review below.

### Planned methods and their roles

- **Random, Mean, Max, EAD:** reproduce the simple Gym4ReaL baselines before comparing more complex controllers.
- **Required PPO learning controller:** reproduce or evaluate the Gym4ReaL learning baseline. It learns a policy through simulator interactions.
- **Preferred third method, FQI:** learn an action-value function from a fixed batch of transitions `(observation, action, reward, next observation, episode boundary)`. Generate this batch from training-year simulations with documented action/state coverage. Repeatedly fit Bellman targets, then choose actions maximizing the fitted Q-function over a documented candidate-action grid. Implement after the MPC/PPO core, subject to the literature review and compute budget.
- **Required deterministic MPC engineering baseline:** plan using one inflow forecast, starting with training-derived seasonal climatology. Add a nominal forecast from the same generator used by scenario MPC to isolate the effect of scenarios.
- **Optional scenario MPC:** adapt the paper's scenario-averaged planning idea to daily DamEnv decisions. Start with a modest validation-selected number of trajectories, then study performance versus scenario count and decision time. Prophet is an optional generator to reproduce the paper's design more closely; begin with a simpler training-derived generator if needed.
- **Optional stochastic DP:** discretize storage, release, and seasonal state, and estimate inflow distributions from training data. Document discretization/model error and refine the grid to check sensitivity. This is a model-based dynamic-programming reference, distinct from scenario MPC and FQI.
- **Optional oracle MPC:** supply perfect future inflow only as an information-advantaged reference. It is not deployable and is not a guaranteed bound on realized performance under an approximate solver or surrogate objective.

FQI replaces the earlier proposed FVI implementation to avoid building both fitted methods. FVI approximates state values `V(s)`; FQI approximates action values `Q(s,a)`. Neither requires a historical operator-action dataset here: FQI can use transitions collected from the training simulator. That makes this a batch-learning comparison within a simulator, not a claim about learning safely from real-world observational logs.

### How we will make the comparison meaningful

The MPC paper uses **hourly decisions**, a **24-hour prediction horizon**, and a **2,160-hour (90-day) simulation starting January 1, 2000**. It uses 380 planning scenarios in its reported configuration and 50 generated inflow realizations for Monte Carlo validation. These are different from DamEnv's daily decisions, 365-day episodes, and 13 historical test years. Its final MPC objective in Section 3.3/Equation (12) uses unsquared norm costs, while Section 4 also evaluates a separate nonlinear cost; neither is the default DamEnv reward.

We will therefore describe our work as an **adaptation of the scenario MPC method**, not a numerical reproduction of that paper. In particular:

1. Evaluate every controller in the same DamEnv physics on the same years, initial conditions, and daily action schedule. A 24-hour horizon in that paper is not equivalent to a 24-day horizon here; our proposed 7/14/30-day planning windows are new choices.
2. Report both the full five-term DamEnv return and the three-term objective score used in Gym4ReaL's main baseline figure, alongside physical metrics. Do not compare the papers' original scores directly.
3. Aim to optimize the same reward across methods. DamEnv has threshold penalties and level-dependent release limits, so the paper's convex optimization formulation cannot simply be reused with the exact reward. Specify and validate a suitable solver; if using a smooth surrogate or simplified planning model, disclose that mismatch and evaluate the resulting controller with the unchanged DamEnv reward.
4. Make information availability explicit. Retain default observations for the PPO reproduction. For the main comparison, require matched decision-time history/forecast inputs across controllers. Any unmatched-information run must be a separately labeled experiment and cannot support the main algorithm comparison. Fit forecasters and scenarios using training data and only observations already available at decision time. Never initialize a forecaster from held-out future data.
5. Give deterministic and scenario MPC the same forecast generator and horizon when testing whether uncertainty scenarios help. Distinguish the planning scenario count from the number of evaluation years.
6. Log simulator transition counts, FQI batch size and fitting iterations, PPO training steps, MPC forecast/optimization time, and total compute. Start from the published PPO budget but do not call methods sample- or compute-matched without measuring their actual usage.
7. Choose discounting, terminal storage/value treatment, and truncation handling explicitly. Use a separate reproduction configuration when these differ from Gym4ReaL PPO. Resolve the known terminal-observation issue before bootstrapped learning.
8. Do not transfer the paper's probabilistic scenario guarantees automatically: our scenario generator, objective, constraints, and solver may differ. Any guarantee needs its assumptions checked independently.

Required comparison: deterministic MPC versus PPO, retaining the simple rules. Add FQI as the preferred third method only after validating the core; scenario MPC or SDP can instead be chosen if they better address the research question. Keep an exact Prophet setup and hydropower optional.

### Focused literature review and method selection

Use [the working literature review](docs/LITERATURE_REVIEW.md) to resolve algorithm choices before implementing optional methods. Start with Gym4ReaL, Cestari et al. (2023), Castelletti et al. (2010) on reservoir FQI/SDP, and Ernst et al. (2005) on FQI. Record state/information, dynamics, objective, training data, evaluation protocol, and implementation cost. Investigate operational-practice evidence separately.

FQI and SDP both make sense as candidates. Our working preference is FQI as a third method, not an assumption that it will beat PPO. For SDP, estimate the state-grid/model burden and check refinement sensitivity before committing. For FQI, investigate seasonal representation, batch coverage, regression settings, and action search. These are review/pilot questions, not additional mandatory algorithms.

Freeze the core MPC/PPO protocol first. Select at most one optional substantial method initially. Consider learned forecasts inside MPC, learned terminal values, or MPC policy imitation as alternative learning contributions rather than adding every family to the experiment roster.

## Work breakdown and completion criteria

### Phase 1: runnable environment and audited data (week 1)

- [x] Record upstream commit, data checksums, license discrepancies, and a minimal Python environment. Upstream docs suggest Python 3.12; verify dependencies rather than trusting placeholder packaging metadata.
- [x] Install only DamEnv requirements plus the chosen learning tools. Confirm registration, config paths, reset, step, and 365-step termination.
- [x] Write `scripts/audit_data.py`: years, dimensions, units, missing/nonfinite values, alignment across signals, leap-day handling, and train/test overlap.
- [x] Create `notebooks/01_environment_and_data.ipynb`: seasonal inflow/demand, observed levels, one simulated trajectory, and annotated units.
- [x] Run constant, random, and exponential-average-demand (EAD) policies; save tidy per-day trajectories and timing.
- [x] Check water conservation and hydraulic release clipping with meaningful numerical cases.


Implementation and measured findings: [Phase 1 findings](reports/PHASE1_FINDINGS.md). The walkthrough notebook has been executed successfully; open provenance/license and calendar questions are documented there.

### Phase 2: benchmark reproduction and evaluation setup (week 2)

- [ ] Begin the focused literature review and extract the deterministic MPC baseline specification.
- [ ] Implement fixed train/validation/test manifests and a shared evaluation runner.
- [ ] Reproduce Random, Mean, Max, and EAD baselines from upstream notebooks.
- [ ] Run the published PPO configuration as a reference: 200k steps, gamma 0.995, learning rate 8e-6, [16,16] policy network, rollout length 2048; verify remaining skrl settings against source before claiming reproduction.
- [ ] Save reward components, physical metrics, configs, seeds, and upstream version alongside each run. Report both the full five-term return and the three-term Gym4ReaL objective score.
- [ ] Produce a baseline table and paired annual plots; review scope before adding objectives.


### Phase 3: core engineering-versus-learning comparison (weeks 3–4)

- [ ] Complete the focused literature comparison sheet and verify the engineering controller's assumptions; distinguish literature proposals from documented operator practice.
- [ ] Build a simple training-only forecast (seasonal climatology, then persistence/history). Validate forecast quality and specify identical decision-time inputs for the main MPC/PPO comparison.
- [ ] **Required:** implement deterministic MPC, adapting the reservoir engineering literature to daily DamEnv. Document planning model, solver, horizon, terminal treatment, constraints, and any reward surrogate. Validate the controller before claiming learning improvements.
- [ ] **Required:** train/evaluate PPO with the matched information setting, separate from the original Gym4ReaL reproduction. Resolve terminal observations and truncation/bootstrapping before training.
- [ ] **Required:** compare full annual closed-loop trajectories on held-out years with paired physical metrics, reward components, terminal storage, training budgets, and decision runtime. Investigate where each controller succeeds or fails.
- [ ] Run a focused ablation such as MPC horizon or matched forecast availability, chosen on validation data, before adding more algorithms.
- [ ] **Preferred optional third method:** pilot tree-based FQI using the linked primary references. Specify training-only batch collection, coverage, regressor, action grid, Bellman/terminal targets, and stopping criteria. Implement instead of FVI, not alongside it.
- [ ] **Alternative optional extension:** scenario MPC following Cestari et al. (2023), using the same forecast generator and horizon as deterministic MPC. Study scenario count versus runtime. Prophet is optional; hourly settings do not transfer automatically.
- [ ] **Alternative optional reference:** a small seasonal/storage stochastic DP controller following the reservoir literature, with training-derived inflow distributions and grid-refinement checks. Explicitly disclose reduced state/information assumptions.
- [ ] Label perfect-inflow MPC as an information-advantaged reference, not a deployable controller. Keep all method/hyperparameter selection separate from final test evaluation.

### Phase 4: trade-offs and report (weeks 5–6)

- [ ] Run a small reward-weight sweep, held-out-year evaluation, and multiple learner seeds.
- [ ] Inspect wet/dry years selected using a predeclared rule; analyze failures, clipping, and terminal storage.
- [ ] Add hydropower only if parameters and time support a defensible model. Otherwise clearly delimit the project to water management.
- [ ] Write results, reproducibility instructions, limitations, and slides.


## Suggested repository structure after plan review

```text
PROJECT_PLAN.md
configs/                  # versioned environment, split, and experiment settings
src/lake_como/            # wrappers, policies, FQI, forecasts, MPC, metrics
scripts/                  # audit, rollout, train, evaluate
notebooks/                # exploratory analysis and presentation of saved results
data/                     # downloaded inputs + provenance manifest
results/                  # small summaries; ignore large trajectories/checkpoints
papers/                   # local PDFs, already ignored by the existing .gitignore
```

The simulator is included as a pinned Git submodule at `external/gym4real/`, pointing to [our fork](https://github.com/vdandenault/gym4ReaL). Keep experiment code in this repository and simulator changes in the fork. Keep a small provenance manifest under version control even when raw data remains external. The Phase 1 adapter, audit, baseline runner, tests, and walkthrough notebook are implemented; the MPC/PPO core comparison remains to be built.

## Reading map and evidence

- **Main:** [Gym4ReaL local PDF](papers/Gym4ReaL-%20A%20Suite%20for%20Benchmarking%20Real-World%20Reinforcement%20Learning.pdf): Section 2.1 (task), C.1 (data), D.1/Table 3 (physics), E.1/Table 9 (PPO), Table 2 (timing). Local copy is arXiv:2507.00257v1, dated June 2025.
- **Scenario MPC comparison:** [Scenario-based model predictive control of water reservoir systems](papers/Scenario-based%20model%20predictive%20control%20of%20water%20reservoir%20systems.pdf), Cestari, Castelletti, and Formentin (2023): Sections 2–3 (deterministic/scenario MPC, Prophet, optimization), Section 4 (hourly/90-day experiments and Monte Carlo comparisons). This paper supplies the MPC reference, not FQI or stochastic DP.
- **FQI/SDP investigation:** [Focused literature review](docs/LITERATURE_REVIEW.md), with Castelletti et al. (2010), DOI 10.1029/2009WR008898, and [Ernst et al. (2005)](https://jmlr.org/papers/v6/ernst05a.html) as starting primary references.
- **Reservoir modeling:** [Efficient Reservoir Management through Deep Reinforcement Learning](papers/Efficient%20Reservoir%20Management%20through%20Deep%20Reinforcement%20Learning.pdf): inflow models and simulator-based policy evaluation in a different reservoir setting; methodological context, not Lake Como parameters.
- **Optional hydropower extension:** [Deep Reinforcement Learning for Intraday Multireservoir Hydropower Management](papers/Deep%20Reinforcement%20Learning%20for%20Intraday%20Multireservoir%20Hydropower%20Management.pdf): finer-grained hydropower scheduling and optimization comparisons; a different problem from annual daily single-reservoir control.
- **Interface background:** local Gymnasium and OpenAI Gym papers. Read Gymnasium first for the modern API; Gym-Anything is optional broader context.
- **Verified implementation:** [DamEnv documentation](https://github.com/Daveonwave/gym4ReaL/blob/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/docs/dam.md), [environment](https://github.com/Daveonwave/gym4ReaL/blob/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/gym4real/envs/dam/env.py), [lake model](https://github.com/Daveonwave/gym4ReaL/blob/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/gym4real/envs/dam/lake.py), and adjacent YAML configs/data metadata. Sources inspected on 2026-10-30; raw files were fetched from `main` while the API reported this commit, so pin and recheck hashes during setup.
