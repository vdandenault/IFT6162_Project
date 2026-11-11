# Scripts

Training, evaluation and controller implementations live in `src/reservoir_control/`.
Run those through the installed `lake-como-*` commands or `python -m reservoir_control.…`.
Scripts here provide walkthroughs, analysis of saved experiments and visualization.

- `visualization/`: builds the Ontario HTML playback and controller diagrams,
  plus the overview of both systems. The HTML template stays beside its builder.
- `lake_como/`: Como walkthroughs, historical PPO result analyses and physical-model
  diagnostics. Retained to explain and reproduce earlier investigations. Some
  diagnostics run simulations; inspect their arguments before running them.

For the current Ontario visualization:

```bash
uv run python scripts/visualization/ontario_visualization.py
```

For the Como walkthrough and saved controller comparison:

```bash
uv run --group reproduction python scripts/lake_como/environment_and_data.py
uv run --group reproduction python scripts/lake_como/controller_comparison.py
```

Redundant wrappers were removed. For example, use `lake-como-audit` instead of
`scripts/audit_data.py`, and `python -m reservoir_control.systems.lake_como.mpc_diagnostics`
instead of `scripts/analysis/diagnose_mpc.py`. Documentation commands use the
corresponding module directly. Generated outputs remain in ignored directories.
