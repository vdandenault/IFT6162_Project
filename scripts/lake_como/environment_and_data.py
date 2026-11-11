"""Environment and data.

Run: uv run --group reproduction python scripts/lake_como/environment_and_data.py
Tables print to the terminal; PNG plots stay in ignored results directories.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reservoir_control.systems.lake_como.environment import make_env, settings
from reservoir_control.systems.lake_como.paths import DAM_DATA, PROJECT_ROOT
from reservoir_control.systems.lake_como.audit import audit
from reservoir_control.systems.lake_como.baselines import rollout


def main():
    output = PROJECT_ROOT / "results/analysis/environment_and_data"
    output.mkdir(parents=True, exist_ok=True)

    # Lake Como: first look at the simulator
    #
    # This walkthrough uses training data only. Start with the inputs, follow one daily
    # decision, then compare three simple controllers on the same year. These are setup
    # diagnostics, not evidence of generalization.
    #

    plt.rcParams.update(
        {
            "figure.figsize": (11, 4),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Where the inputs come from
    #
    # Gym4ReaL bundles annual inflow, demand, and observed-level CSVs plus a minimum
    # environmental flow schedule. The audit checks all seven files and records their
    # hashes. Original measurement provenance and the conflicting paper/metadata data
    # licenses still need resolution.
    #
    # Leap years are included by default. As in upstream DamEnv, episodes use only the
    # first 365 samples; seasonal alignment remains an accepted temporary approximation.
    # The main chronological protocol uses 1946–1987 for training (42 years), 1988–1997
    # for validation (10), and 1998–2010 for testing (13). The audit still checks all
    # original data. The horizontal axes below use simulation days.

    report = audit()
    assert not report["errors"], report["errors"]
    print("Pinned simulator:", report["upstream_commit"])
    print(report["calendar_policy"])
    params = settings("train")
    year = 1985
    print("Training years:", len(params["inflow"]), "| walkthrough year:", year)
    print("Observation order: level, sin(year phase), cos(year phase), recent demand")

    (fig, ax) = plt.subplots()
    for (signal, label) in [("inflow", "Inflow"), ("demand", "Downstream demand")]:
        annual = np.array([v[:365] for v in params[signal].values()])
        ax.plot(np.arange(1, 366), annual.mean(axis=0), label=label)
        ax.fill_between(
            np.arange(1, 366), *np.quantile(annual, [0.1, 0.9], axis=0), alpha=0.15
        )
    ax.set(
        xlabel="Simulation day (ordinal sample)",
        ylabel="Flow (m³/s)",
        title="Training years: mean and 10–90% range",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "seasonal_flows.png", dpi=170)
    print("Plot:", output / "seasonal_flows.png")
    plt.close(fig)

    # One decision, one day
    #
    # A release request is a rate, in m³/s. Over the day the simulator integrates 24
    # hourly balances and adjusts the actual release to hydraulic limits. The
    # observation initially contains zero estimated demand; after the first step it
    # includes the first elapsed day's demand.
    #
    # The water balance uses actual release. Reward combines several penalties; raw
    # physical deficit should be calculated separately because the reward deficit is
    # seasonally weighted.

    env = make_env("train", year)
    (obs, info) = env.reset(seed=123)
    before = info["storage"]
    print("Before decision:", obs)
    (next_obs, reward, terminated, truncated, info) = env.step(
        np.array([100.0], dtype=np.float32)
    )
    print("After one day:", next_obs)
    print(
        "Requested release: 100 m³/s; actual mean release:",
        round(info["release"], 3),
        "m³/s",
    )
    print("New level:", round(info["level"], 3), "m; reward:", round(reward, 3))
    residual = (
        info["storage"] - before - 86400 * (env.unwrapped.inflow[0] - info["release"])
    )
    print("Water balance residual:", residual, "m³")
    assert abs(residual) < 0.0001
    env.close()

    # Three controllers on the same training year
    #
    # - Constant: request 100 m³/s every day (a diagnostic choice).
    # - Random: sample within nominal action bounds with seed 123.
    # - EAD: request the exponentially averaged past demand.
    #
    # Each starts at 0.35 m and experiences the same inflow/demand. Their resulting
    # storage trajectories differ. Flood and low-water counts use daily endpoints and do
    # not establish hourly safety.

    runs = {policy: rollout(policy, year) for policy in ["constant", "random", "ead"]}
    summary = pd.DataFrame([value[1] for value in runs.values()])
    print(
        summary[
            [
                "policy",
                "return_",
                "deficit_m3",
                "flood_days",
                "low_water_days",
                "terminal_storage_m3",
            ]
        ]
        .round(2)
        .to_string()
    )
    assert (summary.steps == 365).all()
    assert (summary.max_mass_balance_error_m3 < 0.0001).all()

    (fig, axes) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for (policy, (frame, _)) in runs.items():
        axes[0].plot(frame.day, frame.level_after_m, label=policy)
        axes[1].plot(frame.day, frame.release_m3_s, label=policy, alpha=0.8)
        axes[2].plot(frame.day, frame.reward.cumsum(), label=policy)
    axes[0].axhline(
        1.24, color="firebrick", linestyle="--", label="Flood penalty threshold"
    )
    axes[0].axhline(
        -0.45, color="gray", linestyle="--", label="Low-water penalty threshold"
    )
    axes[0].set(
        ylabel="Lake level (m)", title=f"Annual simulation: training year {year}"
    )
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set(ylabel="Actual release (m³/s)")
    axes[2].set(ylabel="Cumulative reward", xlabel="Simulation day")
    fig.tight_layout()
    fig.savefig(output / "controller_trajectories.png", dpi=170)
    print("Plot:", output / "controller_trajectories.png")
    plt.close(fig)

    # Observed levels are a separate record
    #
    # The observed historical level reflects actual historical management and
    # conditions. Our simulation uses our selected controller and a fixed initial level,
    # so it is not expected to reproduce the historical level trace. This overlay
    # illustrates that distinction; it is not a calibrated model-validation result.

    levels = pd.read_csv(DAM_DATA / "level/water_level_train.csv", index_col="year")
    observed = levels.loc[year].dropna().to_numpy()[:365]
    (fig, ax) = plt.subplots()
    ax.plot(np.arange(1, 366), observed, label="Historical observed level")
    ax.plot(
        runs["ead"][0].day,
        runs["ead"][0].level_after_m,
        label="Simulated EAD daily endpoint",
    )
    ax.set(
        xlabel="Simulation day (ordinal sample)",
        ylabel="Lake level (m)",
        title="Different management, different trajectories",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "observed_vs_simulated.png", dpi=170)
    print("Plot:", output / "observed_vs_simulated.png")
    plt.close(fig)

    # What to inspect together
    #
    # 1. When does a release request differ from the actual release? Look at high/low
    # lake levels.
    # 2. How does a controller trade current demand against future storage?
    # 3. Does better weighted return also improve the physical metrics we care about?
    # 4. What should happen to stored water at the end of an annual episode?
    #
    # Corrected observation: the project wrapper returns the actual final lake level
    # after day 365, instead of the upstream lookup of the initial level. Rewards and
    # physical trajectories are unchanged. This does not connect years: reset() still
    # starts the next episode at 0.35 m. Learning code must separately choose truncation
    # and terminal-value bootstrapping semantics.
    #
    # Continuous storage across successive years is a separate extension. MPC cannot
    # repair an incorrect returned observation or a reset in the environment. No learner
    # is trained in this script.


if __name__ == "__main__":
    main()
