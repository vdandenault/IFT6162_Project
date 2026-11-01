# Focused literature review: Lake Como control

Working notes, not a completed systematic review. The purpose is to select a defensible engineering baseline, a learning contribution, and a manageable experimental comparison.

## Current recommendation

Required: literature-based deterministic MPC versus PPO, with identical decision-time information, daily release constraints, and evaluation metrics. Use EAD and the other simple rules as sanity checks.

Preferred third method: tree-based FQI, if resources permit. Optional alternatives: scenario MPC to study forecast uncertainty, or stochastic DP to study discretized long-horizon planning. Do not implement every method before establishing the core comparison.

Two substantial controllers can answer the project question; three provide a useful additional comparison. This is a scope recommendation, not an algorithm-count requirement from the course. A horizon sweep, scenario-count sweep, or forecast-input ablation is an experiment within a method, not a new independent algorithm.

## Starting references

### Engineering baseline: Cestari, Castelletti, Formentin (2023)

[Scenario-based model predictive control of water reservoir systems](../papers/Scenario-based%20model%20predictive%20control%20of%20water%20reservoir%20systems.pdf).

The local paper supplies deterministic and scenario MPC formulations for Lake Como. Our planned daily DamEnv controller is an adaptation: its action interval, hydraulics, reward, and evaluation years differ. Extract the controller equations, forecast assumptions, and constraints before implementation. A paper proposing a controller does not establish actual deployment by the reservoir operator; that claim remains unverified.

### Learning benchmark: Salaorni et al. (2025)

[Gym4ReaL](../papers/Gym4ReaL-%20A%20Suite%20for%20Benchmarking%20Real-World%20Reinforcement%20Learning.pdf).

Source for DamEnv and its PPO benchmark. Separate reproducing the paper's default observation/reward setup from the main comparison with matched MPC information. Verify both the three-term plotted objective and the full environment return.

### Reservoir FQI and SDP: Castelletti, Galelli, Restelli, Soncini-Sessa (2010)

[Tree-based reinforcement learning for optimal water reservoir operation](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009WR008898), Water Resources Research, DOI: 10.1029/2009WR008898.

This study applies seasonal, tree-based FQI to daily Lake Como operation and compares it with stochastic DP. It provides a directly relevant methodological reference for both approaches. Prioritize Sections 2–4 and 6 for state/inflow modeling, batch generation, seasonal policies, discretization, and comparison design. Its results do not establish superiority over PPO or transfer automatically to DamEnv.

### FQI algorithm: Ernst, Geurts, Wehenkel (2005)

[Tree-Based Batch Mode Reinforcement Learning](https://jmlr.org/papers/v6/ernst05a.html), JMLR 6:503–556.

Foundational reference for fitting Q-functions from transition batches through repeated supervised regression. Use it to specify Bellman targets, regression choices, action maximization, and stopping conditions. Combining this with the reservoir-specific study is preferable to implementing FQI from an informal description.

## What each approach would teach us

- **MPC versus PPO:** can a learned policy improve on receding-horizon engineering control under matched information? Investigate seasonal storage decisions, wet/dry years, terminal handling, and action latency. Do not assume learning wins.
- **Add FQI:** does learning from a fixed, deliberately collected transition batch offer a useful alternative to PPO's repeated policy-driven interaction? Measure sample usage and training compute separately.
- **Add scenario MPC:** does representing forecast uncertainty improve decisions over using one nominal forecast? Match the forecast generator, objective, and horizon.
- **Add stochastic DP:** how does explicit stochastic planning perform in a deliberately small seasonal/storage state model? Grid refinement and inflow-model assumptions are essential; its result is not automatically ground truth for richer observations or different dynamics.

FQI is promising enough to investigate. Our proposed first implementation would use an Extra-Trees Q-function regressor, a documented release candidate grid, seasonal inputs, and a training-only batch with coverage checks. Validate the batch and Bellman/terminal handling before increasing model size. This is a proposed implementation choice, not a claim of proven superiority.

For SDP, single-reservoir storage with a small seasonal state may be tractable. Adding recent demand, inflow history, or forecasts increases the state/model burden. Estimate grid size and runtime first, and disclose any reduced information relative to PPO/MPC. Do not present a coarse or misspecified SDP result as an exact optimal benchmark.

## Other directions to consider, not commitments

The project could learn a forecast inside MPC, learn an MPC terminal value, or imitate an MPC policy to reduce online computation. Each would isolate a different contribution from learning. Alternative continuous-action RL algorithms such as SAC or TD3 are also candidates, but adding another RL implementation is lower priority than a credible engineering baseline and controlled evaluation. Select further primary references before pursuing any of these branches.

## Review questions and completion criteria

Build a comparison sheet for the four starting papers; expand to roughly 5–8 directly relevant studies only when needed to resolve a decision. For each record:

- System, data provenance, time resolution, and evidence of operational use.
- Observed state/history, action, forecast availability, and information timing.
- Physical constraints, training objective, reported metric, discounting, and terminal treatment.
- How learning data or inflow scenarios are generated and how test data is held out.
- Comparators, reproduced quantities, runtime/sample budgets, and limitations.
- Which parts can be reused in DamEnv and which require a new adaptation.

Finish with an explicit decision: retain MPC/PPO as the core; select FQI, scenario MPC, SDP, or no third method based on relevance and estimated implementation cost. Prefer one controlled ablation and several held-out trajectories over a larger set of weakly verified controllers.

## Implemented deterministic MPC prototype

The first implementation is `src/lake_como/mpc.py`, configured by
[configs/mpc.json](../configs/mpc.json). Cestari et al. Section 2 explicitly includes
a deterministic climatology baseline; that is the starting reference. Their hourly
convex formulation is adapted to daily requests and nonlinear DamEnv hydraulics.
This is not a reproduction of their scenario-MPC scores or guarantees.

The prototype plans seven daily requests, executes the first, then replans. Its
inflow and demand forecasts are ordinal-day means over 1946–1987 only. At runtime
it consumes observed level and seasonal sine/cosine, not live year profiles or
future inflow/demand; the observed demand average is available but unused. Forecasts
are deterministic functions of season and training data, not privileged future
observations. This preserves the default PPO's external information setting.

The planner calls the upstream 24-hour hydraulic integration, including minimum
flows, release limits, and forced overflow releases. Requests are bounded to
0–491.61 m³/s; level limits are soft costs, not guaranteed safety constraints.
Its discounted objective retains forecast deficit, surplus, and clipping terms,
but replaces binary flood/low-water penalties with squared level violations,
scaled by 0.05 m. This surrogate changes the planning objective. Evaluation always
uses unchanged DamEnv physics and both original reward scores.

SciPy SLSQP optimizes normalized requests, starting from the best of a demand
schedule, all-low/all-high requests, and the shifted previous plan. This is local
optimization, not a global optimality claim. On failure or a worse-than-initial
solution, apply the best initial plan and log the fallback. Default discount is
0.995; there is no terminal penalty/value. Planning shrinks at the annual boundary,
and storage still resets between years. These are prototype choices, not tuned
or established operational settings. See the [solver documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-slsqp.html).

Tests cover transitions against DamEnv, one-day optimization against a dense grid,
anticipation of a forecast flood, training-only forecasts, hidden-input independence,
solver failure, and annual reset/horizon handling. The initial ten-year validation
run is complete, with poor performance: full return about -50,274 versus PPO's
-17,232 and EAD's -33,858. There were 20 iteration-limit failures and 151 fallback
days over 3,650 decisions; most fallbacks retained a better starting plan despite
solver success. This is a working prototype, not yet a validated competitive
engineering baseline. Investigate forecast errors, objective mismatch, horizon
and terminal treatment, and optimizer sensitivity separately before drawing a
learning-versus-MPC conclusion. See [evaluation](EVALUATION.md#mpc-prototype).
