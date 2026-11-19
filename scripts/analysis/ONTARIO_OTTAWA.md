# Ontario–Ottawa coupling experiment

From the project root run:

```sh
uv run --group reproduction python scripts/analysis/ontario_ottawa_bridge.py --ontario ~/Downloads/IFT6162_Project-feature-ontario-simulator --output results/ottawa/ontario_bridge_rerun
```

Use a fresh output directory. This compares two finite-candidate planners, not the archived Powell shooting optimizer. The download lacks the MetaMARL submodule, and its shooting solver writes a flow action whereas its environment reads an adjustment. To avoid changing the download, the pilot loads only the reviewed NumPy lake transition, total-flow decoder, routing and Bv7 wrapper methods from source, guarded by file hashes (archive commit 25e7339). It uses the main project's LOSLR checkout at the matching gitlink 8b8de9e. A shared, importable physics module would replace this extraction in the integrated package.

The candidate is executable **total discharge in m³/s**. Invert the volume balance to obtain the adjustment, run the downloaded cubic transition, and verify the decoded flow. Solve the hard follower for each constant candidate in the anticipating arm. In the external arm, use a 900 m³/s Ottawa release forecast, routed with the same current queue, and solve the same actual follower after Ontario selects. Reject Bv7 corrections; do not execute a corrected release against the old response. Replace both downstream non-Ontario inputs with synthetic background plus routed arrival. Use actual coupled flows to check the first action in both arms.

This is a hybrid perfect-information smoke test. Both arms use Ontario driver rows from 1995 quarter-months 17–28, the synthetic Ottawa one-period-delay case, initial lake level 74.8 m, previous Ontario release 7,000 m³/s, H=5 shrinking to the endpoint, and five constant candidate schedules (6,400–7,600 m³/s). Only the first action is executed. The original shooting solver's four-lead forecast interface is bypassed, not extended; future archive rows here are an oracle. The initial states are not historical estimates. The 75.5 m Ontario end-level ceiling, unrounded 22.1 m Pointe-Claire ceiling and synthetic flow capacities are experimental limits layered on Bv7. Terminal water in transit is retained.

The illustrative leader score sums the same four legacy indicator components with raw unit weights (hydropower enters negatively), not the team's baseline-normalized weights. It should not be interpreted as a balanced welfare metric. Inspect executed.csv and candidates.csv for the applied flows, rejected commitments and scores. A rejected finite candidate set would not establish infeasibility of the full continuous leader problem.
