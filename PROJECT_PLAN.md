# Lake Como reservoir control: project plan

## Proposed Eesearch Question 

Can a learned daily reservoir release policy improve the trade-off between downstream water supply and flood/low-water risk on held-out Lake Como years?

Start by reproducing Gym4ReaL DamEnv with its existing physics and rewards. Then implement fitted value iteration (FVI) and compare it with a forecast-based model predictive controller (MPC), if time permits. PPO provides a published reference. Sweep reward weights to examine trade-offs rather than claiming one policy wins on every objective.

We will extend DamEnv with inflow forecasts and an FVI-versus-MPC comparison, with stochastic dynamic programming (SDP) and hydropower generation as possible further additions. We will first establish demand, flood, and low-water control, then add hydropower if time and available data permit. Modeling electricity generation will require turbine specifications, hydraulic head, and efficiency assumptions; modeling revenue would also require electricity prices.

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

## Work breakdown and completion criteria

### Phase 1: runnable environment and audited data (week 1)

- [ ] Record upstream commit, data checksums, licenses, and a minimal Python environment. Upstream docs suggest Python 3.12; verify dependencies rather than trusting placeholder packaging metadata.
- [ ] Install only DamEnv requirements plus the chosen learning tools. Confirm registration, config paths, reset, step, and 365-step termination.
- [ ] Write `scripts/audit_data.py`: years, dimensions, units, missing/nonfinite values, alignment across signals, leap-day handling, and train/test overlap.
- [ ] Create `notebooks/01_environment_and_data.ipynb`: seasonal inflow/demand, observed levels, one simulated trajectory, and annotated units.
- [ ] Run constant, random, and exponential-average-demand (EAD) policies; save tidy per-day trajectories and timing.
- [ ] Check water conservation and hydraulic release clipping with meaningful numerical cases.


### Phase 2: credible baselines and evaluation (week 2)

- [ ] Implement fixed train/validation/test manifests and a shared evaluation runner.
- [ ] Reproduce Random, Mean, Max, and EAD baselines from upstream notebooks.
- [ ] Run the published PPO configuration as a reference: 200k steps, gamma 0.995, learning rate 8e-6, [16,16] policy network, rollout length 2048; verify remaining skrl settings against source before claiming reproduction.
- [ ] Save reward components, physical metrics, configs, seeds, and upstream version alongside each run.
- [ ] Produce a baseline table and paired annual plots; review scope before adding objectives.


### Phase 3: main methods (weeks 3–4)

- [ ] Implement FVI with a documented sampled-state/action design and fitted value approximator. Include season and sufficient inflow/demand history or a forecast state; default observations alone need not be Markov.
- [ ] Define how the Bellman backup models uncertain inflow/demand using training data only. Compare with a small discretized DP/SDP reference if feasible.
- [ ] Build a simple forecast baseline (seasonal climatology, then persistence/history). Evaluate forecast quality on validation years.
- [ ] Implement MPC using the same water-balance model, constraints, available information, and reward metrics. Try 7/14/30-day windows with a terminal storage/value treatment.
- [ ] If adding ensembles, document scenario generation, scenario count, and computational cost. Perfect future inflow is an explicitly labeled oracle reference.
- [ ] Separate the effect of forecasts from controller choice: compare methods with matched information where practical, and disclose differences.


### Phase 4: trade-offs and report (weeks 5–6)

- [ ] Run a small reward-weight sweep, held-out-year evaluation, and multiple learner seeds.
- [ ] Inspect wet/dry years selected using a predeclared rule; analyze failures, clipping, and terminal storage.
- [ ] Add hydropower only if parameters and time support a defensible model. Otherwise clearly delimit the project to water management.
- [ ] Write results, reproducibility instructions, limitations, and slides.


## Suggested repository structure after plan review

```text
PROJECT_PLAN.md
configs/                  # versioned environment, split, and experiment settings
src/lake_como/            # wrappers, policies, FVI, forecasts, MPC, metrics
scripts/                  # audit, rollout, train, evaluate
notebooks/                # exploratory analysis and presentation of saved results
data/                     # downloaded inputs + provenance manifest
results/                  # small summaries; ignore large trajectories/checkpoints
papers/                   # local PDFs, already ignored by the existing .gitignore
```

The simulator is included as a pinned Git submodule at `external/gym4real/`, pointing to [our fork](https://github.com/vdandenault/gym4ReaL). Keep experiment code in this repository and simulator changes in the fork. Keep a small provenance manifest under version control even when raw data remains external. The experiment implementation layout has not been created yet.

## Reading map and evidence

- **Main:** [Gym4ReaL local PDF](papers/Gym4ReaL-%20A%20Suite%20for%20Benchmarking%20Real-World%20Reinforcement%20Learning.pdf): Section 2.1 (task), C.1 (data), D.1/Table 3 (physics), E.1/Table 9 (PPO), Table 2 (timing). Local copy is arXiv:2507.00257v1, dated June 2025.
- **Reservoir modeling:** [Efficient Reservoir Management through Deep Reinforcement Learning](papers/Efficient%20Reservoir%20Management%20through%20Deep%20Reinforcement%20Learning.pdf): inflow models and simulator-based policy evaluation in a different reservoir setting; methodological context, not Lake Como parameters.
- **Optional hydropower extension:** [Deep Reinforcement Learning for Intraday Multireservoir Hydropower Management](papers/Deep%20Reinforcement%20Learning%20for%20Intraday%20Multireservoir%20Hydropower%20Management.pdf): finer-grained hydropower scheduling and optimization comparisons; a different problem from annual daily single-reservoir control.
- **Interface background:** local Gymnasium and OpenAI Gym papers. Read Gymnasium first for the modern API; Gym-Anything is optional broader context.
- **Verified implementation:** [DamEnv documentation](https://github.com/Daveonwave/gym4ReaL/blob/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/docs/dam.md), [environment](https://github.com/Daveonwave/gym4ReaL/blob/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/gym4real/envs/dam/env.py), [lake model](https://github.com/Daveonwave/gym4ReaL/blob/774d5ceb1c218e09e5c6458ead7d4b0b10e2d8ac/gym4real/envs/dam/lake.py), and adjacent YAML configs/data metadata. Sources inspected on 2026-10-30; raw files were fetched from `main` while the API reported this commit, so pin and recheck hashes during setup.
