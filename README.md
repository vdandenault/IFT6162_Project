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

## Team Members: 

- Nadine Mohamed
- Daniel Martineau
- Vincent Dandenault
