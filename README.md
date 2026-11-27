# IFT6162 Project
Course projects for IFT6162 from 
[pierrelucbacon.com/rlbook/interactive/projects.html](https://pierrelucbacon.com/rlbook/interactive/projects.html).

## Team Members: 
- Daniel Martineau
- Nadine Mohamed
- Vincent Dandenault


# Lake Ontario–Ottawa reservoir control

## Project brief from Pierre-Luc Bacon

- **Simulator and baseline.** Use the Plan 2014 rule curve as the operator baseline. Start from LOSLR’s 1900–2020 quarter-monthly supplies, release limits, and performance indicators. Use the Ottawa River Regulation Planning Board as a reference for Ottawa flows. [LOSLR repository](https://github.com/ksemmendinger/loslr_regulation_optimization) · [Ottawa River Regulation Planning Board](https://ottawariver.ca/)
- **Model predictive control.** Use supply forecasts to choose Lake Ontario releases, with hard flood constraints for the Ontario shoreline and Lake St. Louis.
- **Stackelberg control.** Treat Ontario as the leader and the aggregated Ottawa reservoirs as a follower with its own storage model and objective. Compare MPC that takes Ottawa flows as an external forecast with MPC that anticipates the follower’s optimized response. Include PPO as a supplementary comparison.
- **Evaluation.** Report flood exposure at both locations alongside the other operating objectives, with a close look at spring 2017 and 2019. Focus on a careful comparison of Plan 2014, MPC, Stackelberg MPC, and PPO.

## Simulator

Lake Ontario releases and Ottawa River flows meet at Lake St. Louis.
Plan 2014’s F limit uses the Pointe-Claire level to cap Ontario outflow.
We model decisions at a quarter-month step with historical Ontario
forcing and synthetic Ottawa inflow, storage, and routing. Every
controller uses the same Ontario limits, Ottawa follower with hard
constraints, and feasibility and recovery checks.

## Project layout

| Folder | What’s there |
| --- | --- |
| [`src/reservoir_control/systems/`](src/reservoir_control/systems/) | Ontario simulator and Plan 2014 rules in `loslr/`, Ottawa storage and the hard-constrained follower in `ottawa/` |
| [`src/reservoir_control/controllers/`](src/reservoir_control/controllers/) | Shared MPC and PPO algorithms |
| [`src/reservoir_control/experiments/`](src/reservoir_control/experiments/) | Coupled training and evaluation, Gymnasium environment, and forecast and replay code |
| [`configs/`](configs/) | Settings for coupled and Ontario-only runs |
| [`data/`](data/) and [`external/`](external/) | Local baseline settings and pinned LOSLR reference data |
| [`tests/`](tests/) and `results/` | Automated checks and locally generated outputs |

Use CSV for new local traces. The pinned LOSLR submodule supplies the
original tab-separated inputs and policy table.

## Controllers

Both MPC variants plan over the next few quarter-months, execute the first
release, and replan from the next state. They share a finite search over
Ontario release schedules and the same Ottawa feasibility checks.
[Course book: Receding-Horizon Control](https://pierrelucbacon.com/rlbook/receding-horizon-control/)

- **Filtered Plan 2014** is the operator baseline. It proposes Ontario releases using the rule curve.
- **External-ranked MPC** treats Ottawa releases as a fixed forecast when
  scoring Ontario schedules. It takes the highest-ranked schedule whose
  optimized Ottawa response passes the shared checks.
- **Stackelberg MPC** models Ontario as the leader and Ottawa as the follower.
  For each eligible Ontario schedule, it solves Ottawa’s constrained
  release problem and uses that best response in Ontario’s score. The
  leader’s search is limited to the schedules generated at each decision.
  [Stackelberg commitment and best response](https://users.cs.duke.edu/~conitzer/commitEC06.pdf)
- **PPO** learns a release adjustment to the Plan 2014 proposal. Its reward
  is negative operating cost, with an extra penalty for stopping early.
  [Course book: Policy Gradients and Actor-Critic](https://pierrelucbacon.com/rlbook/policy-gradients/)

[Shared MPC implementation](src/reservoir_control/experiments/coupled_planning.py)

## Results

| Controller          | Annual cost | Ontario flood exposure (m·periods) |
| ------------------- | ----------: | ---------------------------------: |
| Filtered Plan 2014  |      0.4625 |                             0.8144 |
| External-ranked MPC |      0.4581 |                             0.8334 |
| Stackelberg MPC     |      0.4581 |                             0.8334 |
| PPO, five-seed mean |      0.4768 |                             0.8105 |

External-ranked and Stackelberg MPC executed identical Ontario and Ottawa
releases across all 18 matched years. Anticipating Ottawa changed predicted
Pointe-Claire levels, and all 2,307 eligible schedules in the audit stayed
at or below the modeled flood threshold under both forecasts, so the
downstream penalty was zero and the ranking stayed the same.

Across 18 matched annual replays, all four controllers completed every
run with zero modeled Lake St. Louis exposure. Both MPC variants lowered
average cost by about 1% and increased average Ontario exposure slightly.
In the separate 2017 and 2019
replays, both cost and Ontario exposure fell relative to Plan 2014.
Cost combines flood exposure, Ontario low water, release changes, and
end-of-year level. Gauge validation found about 11 cm of stage error
near the Pointe-Claire study threshold, enough to affect whether an
additional release appears feasible. External-ranked MPC is the
research reference.

## Runbook

From the repository root with [uv](https://docs.astral.sh/uv/) installed.
Use a fresh output directory for each run.

```sh
git submodule update --init --recursive
export UV_PROJECT_ENVIRONMENT=.venv-ontario
uv sync --locked --group ontario
```

**Plan 2014, external-ranked MPC, and Stackelberg MPC.** Run the two flood
years for all three, or keep just the controller you want in the loop.

```sh
for controller in rule_curve external_mpc stackelberg_mpc; do
  uv run --locked --group ontario python -m reservoir_control.experiments.final_coupled_study \
    --mode development --event-only --recovery peak_protected \
    --controller "$controller" --output "results/loslr/${controller}_events"
done
```

**PPO.** Train five checkpoints once, or point `training_root` to an
existing directory containing the five saved seed runs and skip the loop.

```sh
training_root=results/loslr/ppo_training
for seed in 123 124 125 126 127; do
  uv run --locked --group ontario python -m reservoir_control.experiments.final_coupled_study \
    --mode train --seed "$seed" --output "$training_root/seed_$seed"
done

uv run --locked --group ontario python -m reservoir_control.experiments.final_coupled_study \
  --mode development --event-only --recovery peak_protected --controller ppo \
  --training-root "$training_root" --output results/loslr/ppo_events
```

Use `--mode reserved` in place of `--mode development --event-only`
for the 18-year evaluation. Each run writes `runs.csv` and
`trajectories.csv`. With a training root, omit `--controller` to evaluate
all four together.
