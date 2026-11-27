# Lake Ontario–Ottawa reservoir control

We compare Lake Ontario release policies when an aggregate Ottawa reservoir
also affects Lake St. Louis. Lake Ontario is the leader and the Ottawa
reservoir is the follower. The simulator uses historical Ontario forcing
and a synthetic Ottawa aggregate. See [results](RESULTS.md) for the frozen
comparison and the observed routing check.

## Controllers

- **Filtered Plan 2014** is the operator baseline. It proposes a rule-curve
  release and uses the shared execution filter.
- **External-ranked MPC** is the research reference. It plans against a fixed
  Ottawa release forecast.
- **Stackelberg MPC** ranks the same Ontario candidates using the optimized
  Ottawa response to each executable schedule.
- **PPO** learns a release adjustment around the Plan 2014 proposal.

The same native Ontario limits, hard-constrained Ottawa follower and
peak-protected recovery apply to all four controllers. The folder
[layouts and assumptions](PROJECT_PLAN.md) describe the adapters and model scope.

## Setup and smoke checks

From the repository root with [uv](https://docs.astral.sh/uv/) installed:

```sh
git submodule update --init --recursive
export UV_PROJECT_ENVIRONMENT=.venv-ontario
uv sync --locked --group ontario --group dev
uv run --locked --group ontario --group dev pytest -q
```

Run a short Ontario-only or Ottawa-only controller demo:

```sh
uv run --locked --group ontario python -m reservoir_control.systems.loslr.mpc \
  --years 2017 --output results/loslr/ontario_mpc_demo

uv run --locked --group ontario python -m reservoir_control.systems.ottawa.hard_follower_study \
  --forecast perfect --lookahead 5 --output results/ottawa/follower_demo
```

Use a fresh output directory for each run. These commands keep the existing
controller horizons and data unchanged.

## Reproduce the annual comparison

Train the declared five PPO seeds, then evaluate the frozen endpoints with
the shared peak-protected filter. The optional `--event-only` flag restricts
development evaluation to the 2017 and 2019 cases.

```sh
study_root=results/loslr/final_annual_rerun
for seed in 123 124 125 126 127; do
  uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
    --mode train --seed "$seed" --output "$study_root/training/seed_$seed"
done

uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
  --mode development --event-only --recovery peak_protected \
  --training-root "$study_root/training" --output "$study_root/events"

uv run --locked --group ontario python -m bilevel_reservoirs.solvers.final_coupled_study \
  --mode reserved --recovery peak_protected \
  --training-root "$study_root/training" --output "$study_root/reserved"
```

Each evaluation writes `runs.csv`, `trajectories.csv` and its plan with
configuration and source hashes. The separate stress and candidate-audit
commands use additional locally saved historical replays and synthetic
stress inputs.

## Observed downstream validation

```sh
gauges=results/ottawa/downstream_gauges_rerun
uv run --locked --group ontario python -m reservoir_control.systems.ottawa.downstream_observations \
  --output "$gauges"
uv run --locked --group ontario python -m reservoir_control.systems.ottawa.reach_validation \
  --downstream "$gauges" --output results/ottawa/reach_validation_rerun
```

The commands fetch official daily gauge data with source and quality flags.
Pass `--raw` with the respective saved raw-response directory to rerun
offline. Keep the Canadian Hydrographic Service responses local under their
data terms. See the [Ottawa study guide](configs/studies/ottawa/README.md) for
the standalone follower and Baskatong evidence workflow.

## Data and research archive

- The pinned [LOSLR submodule](external/loslr_regulation_optimization) supplies
  Ontario reference code, forcing and performance indicators.
- [Tracked fixtures](data/input/README.md) support the standalone Ontario
  adapter and reference parity tests.
- Gauges, checkpoints, plots and replay outputs live in ignored local
  directories. Their absence from a fresh checkout is expected.

The Lake Como project, exploratory Ontario/Ottawa experiments and their longer
documentation are in the complete [pre-cleanup snapshot](https://github.com/vdandenault/IFT6162_Project/tree/2f08eab30d89f994abd6ebd18cc311b9005770a2).
To work with that version alongside this checkout:

```sh
git worktree add --detach ../IFT6162_Project_full 2f08eab30d89f994abd6ebd18cc311b9005770a2
```

Follow the snapshot's README for its submodules and dependencies. Existing
local results remain in the original checkout.

## Team

Nadine, Daniel and Vincent · IFT6162
