"""Draw the physical coupling and Adapter architecture as local PNG figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


INK = "#18354a"
BLUE = "#2784ae"
GREEN = "#24896c"
PURPLE = "#8464a7"


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor("#f5f8fc")
    ax.set(xlim=(0, 15), ylim=(0, 9))
    ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    ax.text(.3, 8.45, title, fontsize=24, weight="bold", color=INK)
    ax.text(.3, 7.95, subtitle, fontsize=13, color="#566d80")
    return fig, ax


def box(ax, x, y, width, height, title, body, color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.12", linewidth=2, edgecolor=color, facecolor="white"))
    ax.text(x+.18, y+height-.4, title, color=color, fontsize=16, weight="bold", va="top")
    ax.text(x+.18, y+height-.95, body, color=INK, fontsize=13, va="top", linespacing=1.6)


def arrow(ax, start, end, color=BLUE):
    ax.annotate("", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=19))


def generate(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = canvas("Two operators, one downstream flood risk",
        "Proposed Ontario–Ottawa experiment · quarter-month decisions · schematic, not a geographic map")
    box(ax, .5, 4.7, 4.2, 2.3, "LAKE ONTARIO",
        "Water supply → lake storage / level\nLeader chooses Ontario outflow\nMore release relieves Ontario", BLUE)
    box(ax, 9.6, 4.7, 4.7, 2.3, "OTTAWA RESERVOIRS",
        "Reservoir inflow → aggregate storage\nFollower chooses its own release\nIts storage and service needs matter", GREEN)
    box(ax, 4.95, 1.5, 5.1, 2.2, "LAKE ST. LOUIS / POINTE-CLAIRE",
        "Ontario flow + routed Ottawa contribution\n+ other uncontrolled tributary inflows\nCombined flow determines modeled stage", PURPLE)
    arrow(ax, (3.7, 4.6), (5.7, 3.8))
    arrow(ax, (11.4, 4.6), (9.3, 3.8), GREEN)
    ax.text(.6, 3.55, "Holding water upstream can reduce\ndownstream flow now, but uses storage.",
        fontsize=12, color=INK, linespacing=1.6)
    ax.text(10.55, 3.5, "Uncontrolled runoff is\nnot a follower action.",
        fontsize=12, color=INK, linespacing=1.6)
    ax.text(.5, .55,
        "External-flow MPC forecasts Ottawa flow. Stackelberg MPC anticipates how Ottawa responds to its plan.\n"
        "Both must be compared in the same responsive physical model; neither can guarantee flood avoidance in every event.",
        fontsize=13, color=INK, linespacing=1.7)
    fig.savefig(output / "ontario_ottawa_physical_problem.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = canvas("One physical engine, independent controllers",
        "Adapter pattern · MPC and PPO share physical definitions, not controller implementations")
    box(ax, .5, 5.0, 3.1, 2.0, "MPC", "Forecast + optimize\nExecute first action\nReplan", BLUE)
    box(ax, 5.0, 5.0, 3.7, 2.0, "PLANNING ADAPTER", "Independent state copies\nForecast-driven rollout\nCosts + constraints", BLUE)
    box(ax, .5, 1.65, 3.1, 2.0, "PPO", "Actor / critic\nCollect experience\nPolicy updates", GREEN)
    box(ax, 5.0, 1.65, 3.7, 2.0, "GYMNASIUM ADAPTER", "reset / step\nObservations + actions\nRewards + episode endings", GREEN)
    box(ax, 10.1, 3.35, 4.15, 2.6, "PHYSICAL ENGINE", "State + water balance\nRouting + operating rules\nComo OR Ontario–Ottawa\nNo MPC / PPO dependencies", PURPLE)
    arrow(ax, (3.75, 6.0), (4.8, 6.0))
    arrow(ax, (3.75, 2.65), (4.8, 2.65), GREEN)
    arrow(ax, (8.85, 6.0), (9.9, 5.3))
    arrow(ax, (8.85, 2.65), (9.9, 3.9), GREEN)
    ax.text(.5, .55,
        "The experiment runner composes the system, adapters and selected controller, then records executed trajectories.\n"
        "Planning never advances the live environment. System-specific observations, objectives and constraints remain explicit.",
        fontsize=13, color=INK, linespacing=1.7)
    fig.savefig(output / "multisystem_adapter_architecture.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def baseline_events(trace: Path, output: Path):
    """Plot native model trajectories; no unverified flood threshold is added."""
    frame = pd.read_csv(trace)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=150, sharex=True)
    fig.patch.set_facecolor("#f5f8fc")
    for column, year in enumerate((2017, 2019)):
        event = frame.loc[frame.Year.eq(year)]
        if len(event) != 48:
            raise ValueError(f"Expected 48 baseline periods for {year}")
        axes[0, column].set_title(str(year), color=INK, fontsize=17, weight="bold")
        axes[0, column].plot(event.QM, event.ontLevel, color=BLUE, lw=2)
        axes[1, column].plot(event.QM, event.ptclaireLevel, color=PURPLE, lw=2)
        axes[2, column].plot(event.QM, event.ontFlow * 10, color=BLUE, lw=2, label="Ontario outflow")
        axes[2, column].plot(event.QM, event.stlouisontOut, color=GREEN, lw=2, label="Non-Ontario contribution (SLON)")
        for ax in axes[:, column]:
            for period in event.loc[event.flowRegime.eq("F"), "QM"]:
                ax.axvspan(period-.5, period+.5, color=BLUE, alpha=.10, lw=0)
            ax.grid(alpha=.2)
            ax.set_xlim(.5, 48.5)
            ax.set_xticks((1, 9, 17, 25, 33, 41), ("Jan", "Mar", "May", "Jul", "Sep", "Nov"))
            ax.spines[["top", "right"]].set_visible(False)
        axes[2, column].set_xlabel("Quarter-monthly model timeline")
    axes[0, 0].set_ylabel("Ontario period-mean level (m)")
    axes[1, 0].set_ylabel("Pointe-Claire modeled level (m)")
    axes[2, 0].set_ylabel("Period-mean flow (m³/s)")
    axes[2, 0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Ontario–St. Lawrence: native Python reference replay", fontsize=21, color=INK, weight="bold")
    fig.text(.07, .025,
        "Blue shading: native F limit governs release. These are simulated trajectories, not observed water levels.\n"
        "Supplied forecast / realized-SLON limit behavior is preserved. Agreement with archived GLRRM remains unresolved.",
        fontsize=11, color=INK, linespacing=1.6)
    fig.tight_layout(rect=(.02, .09, .99, .95))
    fig.savefig(output / "ontario_2017_2019_native_baseline.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output = root / "background/images"
    generate(output)
    trace = root / "results/loslr/native_baseline_v1/native_trace.csv"
    if trace.exists():
        baseline_events(trace, output)
