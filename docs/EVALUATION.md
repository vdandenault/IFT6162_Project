# Evaluation: commands, metrics, and results

Run the shared chronological comparison:

```bash
uv run --group reproduction lake-como-compare
```

The default evaluates Random, Mean, Max, EAD, and all five selected PPO checkpoints
from [configs/ppo.json](../configs/ppo.json) on 1988–1997 validation years. It does
no training or chronological test evaluation. See [controller comparison script](../scripts/analysis/controller_comparison.py)
for the table, reward components, terminal storage, decision timings, and plots.

## Results and their limits

The completed comparison covers 90 annual episodes. PPO has the highest full
return averaged across seeds in every validation year, and its scores match the
saved 1M study. EAD beats one PPO seed in one year; PPO wins the other 49 seed/year
comparisons against EAD. These are paired outcomes, not 50 independent trials.

| Controller | Full return | Flood days/year | Low-water days/year | Unmet demand (million m³/year) |
|---|---:|---:|---:|---:|
| Random | -48,353 | 12.30 | 45.20 | 936.16 |
| Mean | -40,469 | 9.90 | 63.80 | 562.77 |
| Max | -53,370 | 4.20 | 64.30 | 520.27 |
| EAD | -33,858 | 23.80 | 11.20 | 356.14 |
| PPO | -17,232 | 4.40 | 5.76 | 514.91 |

PPO improves the weighted objective, not every physical metric. EAD leaves less
unmet demand; Max has slightly fewer flood days but many more low-water days.
The PPO setting was tuned on these validation years. This is a development
comparison against simple rules, not evidence of superiority over MPC or operators.

## Run another comparison

```bash
uv run --group reproduction lake-como-compare --split validation \
  --controllers ead max ppo --output results/comparison/validation_subset
```

`--split` accepts `train`, `validation`, or `test`; reserve test for the final frozen
comparison. `--controllers` also accepts `constant` (100 m³/s). Repeat
`--ppo-checkpoint SEED=PATH` for explicit checkpoint choices. Adjacent training
`run.json` must match the seed, chronological training membership, protocol, and
checkpoint hash. Nonempty output directories are rejected; choose a new `--output`.

Every controller visits the same years in order, with initial level 0.35 m and
demand average zero. The runner verifies identical forcing and resets, complete
365-day episodes, reward accounting, and water balance. Leap years retain the
365-step approximation; the terminal observation is corrected, but storage does
not carry between years. Random has one action RNG seeded at 123 across the split,
not multiple random-rule seeds. PPO uses frozen normalization, deterministic mean
actions, CPU FP32, and one PyTorch thread.

## Outputs and metric definitions

Results remain ignored under `results/comparison/validation_v1/`:

- `trajectories.csv` / `annual.csv`: daily and annual results, all weighted reward terms, terminal storage, and decision timing; each PPO seed is a separate controller.
- `controller_means.csv`, `family_years.csv`, `summary.csv`: controller means and equal-weight PPO averages. Seed SD uses five seed means; year SD uses ten yearly seed averages. Rules have no learner-seed SD.
- `paired_annual.csv` / `paired_family_annual.csv`: candidate-minus-reference differences joined by year, before/after PPO seed averaging. Higher reward and lower physical penalties are better.
- `annual_comparison.png` / `paired_ppo_vs_rules.png`: year-by-year PNG plots. Shading is the observed PPO seed range, not a confidence interval.
- `timings.csv` / `run.json`: runtimes, checkpoint/input/source hashes, versions, split, reset/calendar/inference settings, and reward coefficients.

Full return includes five penalties; the three-term objective excludes waste and
clipping. Flood/low-water days use daily endpoints, not hourly exceedance counts.
Unmet demand is unweighted physical volume. Terminal storage is relative to the
model's minimum-level datum; no new terminal penalty is imposed.

CPU decision medians were about 36–37 µs for PPO and below 2 µs for deterministic
rules. Timings exclude simulation, action rescaling in `env.step`, loading, CSV,
and per-year PPO warmup. Python/timer overhead and local noise are material.

A future controller can reuse `evaluate(name, ..., controller_factory=factory,
sequential_years=True, measure_decisions=True)`. The factory receives the environment
and returns `(observation, previous_info) -> physical_release_request`. Respect
decision-time information; future historical inputs must not leak into the policy.
Adding the MPC CLI adapter belongs to implementing MPC, our next task.
