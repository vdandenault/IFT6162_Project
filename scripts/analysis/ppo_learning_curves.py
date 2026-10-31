"""PPO learning curves.

Run: uv run --group reproduction python scripts/analysis/ppo_learning_curves.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""
import json
import pandas as pd
from lake_como.paths import PROJECT_ROOT


def main():
    # Should capped PPO train for longer?
    #
    # This is a read-only inspection of saved logs for capped mixed-precision PPO with
    # entropy coefficient 0. No new training, validation rollout, or chronological test
    # evaluation is performed.
    #
    # The question is whether a bounded longer-training comparison is warranted—not
    # whether noisy training logs prove policy convergence.

    output = PROJECT_ROOT / "results/stabilization/curve_inspection_v1"
    assert (
        output / "inspection.json"
    ).exists(), "Run: uv run --group reproduction lake-como-inspect-curves"
    metadata = json.loads((output / "inspection.json").read_text())
    assert (
        not metadata["new_training"]
        and (not metadata["new_validation_evaluations"])
        and (not metadata["test_evaluated"])
    )
    print(metadata["setting"])
    changes = pd.read_csv(output / "late_changes.csv")
    print(
        changes[
            [
                "training_seed",
                "episode_reward_previous_50k",
                "episode_reward_last_50k",
                "episode_reward_change",
            ]
        ]
        .round(2)
        .to_string()
    )

    # Training reward has not clearly plateaued for every seed
    #
    # Four seeds improve on average between the 100–150k and 150–200k windows. Seed 127
    # does not show consistent progress across those windows, although it rebounds in
    # the last 25k. Higher full reward is better; these are descriptive changes, not
    # slope significance tests.

    print("Plot:", str(output / "learning_curves.png"))

    # Why this is not a validation curve
    #
    # skrl's episode-return tracking is cleared at each 500-step logging write. The
    # displayed reward scalar averages tracked episode-return means within that
    # interval. It describes sampled actions under a changing policy, not the
    # deterministic policy evaluated on every year. Historical years/seasonal conditions
    # also differ across training windows.
    #
    # Exploration is shrinking, so the training curve can improve simply because fewer
    # noisy actions are sampled. Smoothing adjacent logs helps visualization but does
    # not create independent observations or demonstrate convergence. Only final
    # checkpoints were saved, so intermediate deterministic validation performance
    # cannot be recovered.

    print("Plot:", str(output / "training_diagnostics.png"))

    # Normalized critic loss generally declines late in training, but its normalization
    # statistics change too. A small PPO policy loss is not a reliable stopping
    # criterion on its own. The learning rate remains at the cap and action spread is
    # still changing.
    #
    # Decision
    #
    # A bounded 200k-versus-400k experiment is worth testing, with the same
    # configuration and five seeds. This is not evidence that longer training will fix
    # the weak seed or improve deterministic validation performance. It could improve
    # strong runs while leaving a weak strategy behind.
    #
    # Prefer a continuous run saving checkpoints at both 200k and 400k, checking its
    # 200k prefix against existing controls. Existing model/optimizer/scaler files do
    # not guarantee exact resume of RNG, on-policy buffer, and trainer/scheduler state.
    # Evaluate fixed checkpoints on the same ten validation years, with a rule decided
    # before training. Keep the chronological test years out of tuning.
    #
    # No longer training has been launched in this inspection. If a single bounded
    # extension does not justify its cost, retain 200k and move to the MPC comparison
    # rather than extending indefinitely.


if __name__ == "__main__":
    main()
