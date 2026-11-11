# Ontario–Ottawa coupled study

`coupled_v1.json` freezes the new research formulation. Existing `mpc_v2` results
and `scoring.json` retain their earlier fixed-cap definition.

## Evidence and conventions

- [IJC Plan 2014, Annex B](https://www.ijc.org/sites/default/files/Plan2014_CompendiumReport.pdf):
  Ontario and Pointe-Claire levels use metres IGLD 1985. The F-limit table sets
  Pointe-Claire targets of 22.10, 22.20, 22.33, 22.40 and 22.48 m for Ontario
  beginning levels below 75.30, 75.37, 75.50, 75.60 m, and at least 75.60 m,
  respectively. These are flow-regulation targets, not guaranteed flood-free levels.
- [IJC operational description](https://www.ijc.org/en/loslrb/watershed/outflow-changes):
  actual releases are weekly averages, with operational adjustments and deviations.
  This experiment retains the pinned simulator's 48-period year and fixed Ontario
  balance conversion of 2970. Ottawa's SI balance uses 365 days / 48 per period;
  no claim is made that it resolves real weekly commands or travel times.
- [Ottawa River Regulation Planning Board inventory](https://www.ottawariver.ca/about-us/what-we-do/the-ottawa-river-basin-and-its-reservoirs/):
  thirteen principal reservoir capacities are listed, totalling 12.515 billion m³.
  Listed capacities do not establish synchronously available active storage.
  Our 6 billion m³ useful capacity, half-full initialization, outlet limits and
  half-full target are sensitivity assumptions, not fitted operator parameters.
- [Board description of partial regulation](https://www.ottawariver.ca/about-us/what-we-do/collaborative-management/):
  about 40% of the basin has partial control. Applying a 0.4 fraction to historical
  SLON is a synthetic scenario construction, **not** an inference that 40% of its
  observed flow is controllable or that SLON is natural Ottawa runoff. SLON already
  includes historical regulation and other non-Ontario contributions.

## Decision sequence and Adapter pattern

At each period the Ontario leader announces a four-period **request** schedule.
The follower observes it and solves a strictly convex quadratic program with
storage/outlet bounds, service and storage targets, and a soft downstream-flow
exceedance cost. Only the first follower action is executed. Uncontrolled runoff
is then realized; combined flow is routed using the native Ontario simulator.
The Ontario request is projected through native Bv7 using decision-time estimates
of the follower's response. The announced request can differ from executed flow;
the follower responds to the request, not an assumed executable commitment.
Future schedules are advisory and reoptimized next period.

External-flow MPC predicts a follower that responds to a neutral, rule-curve
announcement. Bilevel MPC recomputes the follower best response for each candidate
announcement. **Both execute against the same responsive follower.** Nested QP
solves provide the reference bilevel implementation; KKT residuals and local
active-set derivatives check the follower. Finite schedule search does not certify
a global leader optimum. This is an idealized announcement-response game.

Physics, observations, reward and coupling live in system adapters. The generic
PPO agent and generic convex-response solver have no Ontario/Ottawa imports.

## Objectives and hard constraints

Native Bv7 release projection always applies. Candidate leader schedules that
satisfy both predicted stage caps (within native 0.01 m routing rounding) rank
first. If none do, the declared emergency ordering minimizes maximum metre
exceedance, then total metre exceedance, then operating cost. This fallback is
reported; infeasible caps are never described as enforced. Forecast feasibility
cannot guarantee realized safety under forecast errors. The 75.50 m Ontario cap
and 74.00 m low-water threshold are research thresholds, not verified universal
operational prohibitions. F tiers apply to the predicted beginning level.

All four controllers receive identical causal information. PPO's 64 inputs include
all 48 stored mean levels and the September-rule state, as well as current level,
calendar, previous release, past-only supply/SLON forecasts, Ottawa storage,
roughness and lagged ice. No future realized supply or runoff is observed. Period cost is squared
Ontario and downstream exceedance, plus 0.25 times squared Ontario low-water
shortfall and 0.01 times squared release change in units of 1000 m³/s. At a finite
annual endpoint add 0.1 times squared deviation from 74.80 m. PPO reward is exactly
negative cost. Raw severities, counts, peaks, follower service, storage, residuals,
release changes and runtime remain separate. Costs are research preferences,
not calibrated stakeholder damages or hydropower production.

## Evaluation and limitations

Training: 1901–1980. Development: 1981–2000 plus 2017/2019 events. Remaining
2001–2020 years are reserved for the declared final comparison. These memberships
are disjoint; previously published full-record baseline replays remain historical
evidence. Annual cases start at a common causal rule-curve Ontario state and the
same assumed Ottawa storage; within-year state is continuous. Multi-year carryover
is a future evaluation extension. Past-only persistence forecasts are shared.

The initial five-seed PPO budget is **4096 decisions per seed**, a bounded pipeline
pilot, not a convergence study. No checkpoint selection or test evaluation occurs.
Training logs include action/reward ranges, actor/critic gradient norms, KL and
finite-horizon terminal boundaries. Study plans and source hashes are written
before execution. Changing budgets or physical assumptions requires a new output
folder and plan. Capacity, controlled fraction and routing lag remain sensitivity
parameters; zero-lag results alone cannot validate real operator coordination.
