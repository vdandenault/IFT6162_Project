# IFT6162 Project

Course projects for IFT6162 from 
[pierrelucbacon.com/rlbook/interactive/projects.html](https://pierrelucbacon.com/rlbook/interactive/projects.html).

## Lake Como reservoir control

See [the project plan](PROJECT_PLAN.md) for the simulation model, data sources,
evaluation protocol, and implementation milestones.

## Simulator source

Our [Gym4ReaL fork](https://github.com/vdandenault/gym4ReaL) is included as a
pinned Git submodule at `external/gym4real/`.

Clone this project with its simulator:

```bash
git clone --recurse-submodules https://github.com/vdandenault/IFT6162_Project.git
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

Keep experiment code in this repository and simulator changes in the fork.
After reviewing and committing simulator changes in the fork, update this
repository's submodule reference to the chosen commit. Dependencies are not
installed by the submodule commands; see `external/gym4real/docs/dam.md` for
the upstream DamEnv setup instructions.

## Phase 1: run and explore

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (on macOS: `brew install uv`), initialize the submodule, then run from the project root:

```bash
uv sync --locked
uv run lake-como-audit
uv run python -m unittest discover -s tests -v
uv run lake-como-rollout --benchmark 100
```

`uv sync` creates `.venv`, uses Python 3.12, and installs both workspace
packages in editable mode, including the notebook tools in the default `dev`
group. `uv.lock` records the resolved dependencies; `--locked` prevents an
unexpected lockfile update. No environment activation is needed for `uv run`.

Normal imports such as `from lake_como.baselines import rollout` work in scripts
and notebooks regardless of the current working directory. Select this project's
`.venv` as the notebook kernel. The simulator fork includes a packaging-only
change to its `pyproject.toml`; review it separately from the parent changes.
The submodule Git commit pins simulator source; `uv.lock` pins Python packages.

To add a dependency, use `uv add <package>` (or `uv add --dev <package>` for
notebook/development tools). Review both `pyproject.toml` and `uv.lock` afterward.
The workspace layout follows [uv's workspace documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/).

The default evaluation protocol is chronological, with leap years included:

- Training: 1946–1987 (42 years).
- Validation: 1988–1997 (10 years).
- Testing: 1998–2010 (13 years).

Year membership is recorded in `configs/splits.json`. Use validation to select
settings and reserve testing for the final comparison. Example commands:

```bash
uv run lake-como-rollout --split train --years 1984
uv run lake-como-rollout --split validation --years 1990
uv run lake-como-rollout --no-include-leap-years
# Separate reference protocol with the authors' original mixed-year membership:
uv run lake-como-rollout --protocol upstream --split train --years 1985
```

Leap-year profiles retain upstream 365-step behavior; their final sample is
unused. This is an accepted temporary approximation, not full calendar support.
The Python API exposes `protocol` and `include_leap_years` too. A project wrapper
corrects the returned final level without changing physics, rewards, or resets.
Annual episodes still start at 0.35 m; continuous year-to-year storage is not yet
implemented. `make_env(..., correct_terminal_level=False)` exposes the unmodified
observation for source reproduction checks.

Start with [the executed walkthrough](notebooks/01_environment_and_data.ipynb)
and [Phase 1 findings](reports/PHASE1_FINDINGS.md) for the data audit, initial
results, timing scope, and known simulator issues. Select `.venv` as the notebook
kernel in your editor. Generated trajectories and plots go to ignored `results/`.

## Team Members: 

- Nadine Mohamed
- Daniel Martineau
- Vincent Dandenault
