"""Check historical downstream stage response and summarize exposure trade-offs."""

from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .reconstruction import digest

CALIBRATION = (1987, 1990)
HOLDOUT = (1991, 1993)
LAGS = tuple(range(4))
BUDGETS = (0., .10, .25, .50, 1.)
TOL = 1e-9
STATIONS = {
    "carillon": ("02LB024_discharge", "02LB024_flag"),
    "cornwall": ("02MC002_discharge", "02MC002_flag"),
    "pointe_calumet": ("02OA105_level", "02OA105_flag"),
    "pointe_claire": ("02OA039_level", "02OA039_flag"),
}


def years(index, bounds):
    return (index.year >= bounds[0]) & (index.year <= bounds[1])


def aligned_data(path):
    source = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if not source.index.equals(pd.date_range(source.index.min(), source.index.max())):
        raise ValueError("Expected an ordered daily calendar with explicit missing values")
    observed = pd.DataFrame({name: source[value].where(source[flag].isna())
                             for name, (value, flag) in STATIONS.items()})
    features = observed.copy()
    for field, levels in (("cornwall", LAGS), ("carillon", (*LAGS, 7, 8))):
        if observed[field].lt(0).any():
            raise ValueError(f"Expected a nonnegative discharge at {field}")
        for lag in levels:
            features[f"{field}_{lag}"] = observed[field].shift(lag)
    return features.dropna()


def rating(flow):
    return (np.asarray(flow) / 10000.) ** .58


def calibration(frame, train, cols):
    design = np.column_stack([np.ones(len(frame)), *(rating(col) for col in cols)])
    fit = lsq_linear(design[train], frame.pointe_claire.to_numpy()[train],
                     bounds=([-np.inf] + [0.] * len(cols), np.inf))
    if not fit.success:
        raise ArithmeticError(fit.message)
    fitted = design @ fit.x
    return fitted, fit.x.tolist(), np.abs(fitted[train] - frame.pointe_claire.to_numpy()[train]).mean()


def compare_routing(frame):
    train = years(frame.index, CALIBRATION)
    holdout = years(frame.index, HOLDOUT)
    if not train.any() or not holdout.any():
        raise ValueError("Calibration and holdout require overlapping gauge values")
    models = {
        "cornwall_only": [frame.cornwall_0],
        "full_carillon_same_day": [frame.cornwall_0 + frame.carillon_0],
        "full_carillon_7_days": [frame.cornwall_0 + frame.carillon_7],
        "full_carillon_8_days": [frame.cornwall_0 + frame.carillon_8],
    }
    lag_rows, lag_fits = [], {}
    for cornwall, carillon in product(LAGS, repeat=2):
        result = calibration(frame, train, [frame[f"cornwall_{cornwall}"], frame[f"carillon_{carillon}"]])
        lag_fits[cornwall, carillon] = result
        lag_rows.append({"cornwall_days": cornwall, "carillon_days": carillon, "train_mae_m": result[2]})
    chosen = min(lag_fits, key=lambda lag: (lag_fits[lag][2], sum(lag), lag))
    fitted = {name: calibration(frame, train, cols) for name, cols in models.items()}
    fitted["two_flow_same_day"] = lag_fits[0, 0]
    fitted["two_flow_selected_lag"] = lag_fits[chosen]
    observed = frame.pointe_claire
    threshold = float(observed[train].quantile(.9))
    output = frame[["carillon", "cornwall", "pointe_claire", "pointe_calumet"]].copy()
    output["split"] = np.where(train, "calibration", "holdout")
    output = output.loc[train | holdout]
    for name, (values, _, _) in fitted.items():
        output[name] = pd.Series(values, index=frame.index)
    scores, peaks = [], []
    held = output.loc[output.split.eq("holdout")]
    spring = held[held.index.month.isin((4, 5, 6))]
    daily = held[["pointe_claire", *fitted]].reindex(
        pd.date_range(held.index.min(), held.index.max()),
    ).diff()
    for name in fitted:
        error = held[name] - held.pointe_claire
        score = {"model": name, "calibration_days": int(train.sum()), "holdout_days": len(held),
                 "mae_m": error.abs().mean(), "bias_m": error.mean(),
                 "daily_change_mae_m": (daily[name] - daily.pointe_claire).abs().mean(),
                 "high_stage_days": int(held.pointe_claire.ge(threshold).sum()),
                 "high_stage_mae_m": error.loc[held.pointe_claire.ge(threshold)].abs().mean()}
        scores.append(score)
        for year, group in spring.groupby(spring.index.year):
            if len(group) < 80:
                continue
            a, b = group.pointe_claire.idxmax(), group[name].idxmax()
            peaks.append({"model": name, "year": year, "matched_spring_dates": len(group),
                          "observed_peak_date": a.date(), "predicted_peak_date": b.date(),
                          "peak_offset_days": (b - a).days})
    metadata = {
        "selected_lag_days": {"cornwall": chosen[0], "carillon": chosen[1]},
        "calibration": CALIBRATION, "holdout": HOLDOUT, "calibration_high_stage_m": threshold,
        "coefficients": {name: fit[1] for name, fit in fitted.items()},
    }
    return output, pd.DataFrame(scores), pd.DataFrame(lag_rows), pd.DataFrame(peaks), metadata


def compare_tradeoff(path):
    frame = pd.read_csv(path)
    pair = frame[frame.method.isin(["persistence", "seasonal_half_upper"])]
    values = ["ontario_exposure_m", "downstream_exposure_m"]
    keys = ["delay", "year", "method"]
    if (pair.duplicated([*keys, "period"]).any() or not pair.period.between(1, 48).all()
            or not pair.groupby(keys).size().eq(48).all()):
        raise ValueError("Trade-off must use matched complete years")
    annual = pair.groupby(["delay", "year", "method"])[values].sum().unstack("method")
    candidate = annual.xs("seasonal_half_upper", axis=1, level="method")
    reference = annual.xs("persistence", axis=1, level="method")
    delta = (candidate - reference).rename(columns={
        "ontario_exposure_m": "added_ontario_m_periods", "downstream_exposure_m": "added_downstream_m_periods",
    }).reset_index()
    if delta.isna().any().any():
        raise ValueError("Each comparison needs both methods")
    budgets = []
    for delay, group in delta.groupby("delay"):
        total = group.added_ontario_m_periods.sum()
        worst = group.added_ontario_m_periods.max()
        benefit = -group.added_downstream_m_periods.sum()
        for limit in BUDGETS:
            budgets.append({"delay": delay, "allowed_added_ontario_m_periods": limit,
                            "added_ontario_m_periods": total, "worst_year_added_ontario_m_periods": worst,
                            "avoided_downstream_m_periods": benefit,
                            "passes_total_budget": total <= limit + TOL and benefit > TOL,
                            "passes_every_year_budget": worst <= limit + TOL and benefit > TOL})
    return delta, pd.DataFrame(budgets)


def render_report(scores, peaks, metadata, deltas, budgets):
    chosen = metadata["selected_lag_days"]
    n, high = (scores.iloc[0][field] for field in ("holdout_days", "high_stage_days"))
    lines = [
        "# Ottawa routing validation and shoreline trade-off", "",
        "## Observed downstream stage response", "",
        "Daily HYDAT records are matched by station date. Carillon and Cornwall provide upstream "
        "discharges on the Ottawa and St. Lawrence respectively. Pointe-Claire stage is the target. "
        "Pointe-Calumet stage is retained as context upstream of the branch outlets. Only unflagged "
        "values are used. All comparisons use the same dates with every lag available. "
        "The rating exponent follows the native LOSLR relation. Offsets and nonnegative rating "
        "coefficients are fitted only on 1987–1990, preserving the reported station's datum. "
        "High stages use the calibration 90th percentile of Pointe-Claire stage.", "",
        f"The 1991–1993 holdout has {int(n):,} common dates, {int(high)} of them at high stage. "
        f"Calibration chose effective lags of {chosen['cornwall']} day(s) from Cornwall and "
        f"{chosen['carillon']} day(s) from Carillon in the two-flow proxy.", "",
        "| Flow-to-stage proxy | Holdout MAE (m) | High-stage MAE (m) | Daily stage change MAE (m) |",
        "|---|---:|---:|---:|",
    ]
    for row in scores.itertuples(index=False):
        lines.append(f"| {row.model.replace('_', ' ')} | {row.mae_m:.3f} | "
                     f"{row.high_stage_mae_m:.3f} | {row.daily_change_mae_m:.3f} |")
    lines += ["", "Adding same-day Carillon improved held-out level estimates and high-stage estimates "
              "relative to Cornwall alone. Daily stage-change MAE stayed about the same. "
              "The calibration-selected lags did not improve held-out level MAE over the same-day two-flow fit.", "",
              "The full-Carillon variants deliberately put all Carillon discharge into a single "
              "rating proxy to test this simplifying assumption. The two-flow model fits separate "
              "stage contributions. Both approaches evaluate station-stage response. They cannot "
              "identify an outlet flow split, a physical travel time or the response to an Ottawa "
              "control action. The 7- and 8-day cases bracket one nominal quarter-month, but "
              "whole-river Carillon flow and a synthetic controlled-reservoir release have different boundaries.", ""]
    if not peaks.empty:
        pp = peaks[peaks.model.eq("two_flow_selected_lag")]
        peak_text = ", ".join(f"{row.year} {row.peak_offset_days:+d} d" for row in pp.itertuples())
        lines += [f"Spring peak-date offsets for the selected-lag model were {peak_text}. "
                  "Each peak is the maximum on the common available April–June dates. "
                  "A gap could hide the actual daily peak.", ""]
    lines += [
        "Pointe-Claire's historical datum still needs a verified tie to IGLD 1985 to test absolute "
        "F-limit compliance. Reproducing stage changes on these data does not validate the synthetic "
        "Ottawa storage, release response or routing queue.", "",
        "[ECCC Water Survey daily data and flags](https://wateroffice.ec.gc.ca/contactus/faq_e.html) · "
        "[Québec branch hydraulics study](https://www.cehq.gouv.qc.ca/mille-iles/etude-solutions-soutien-etiages-critiques.pdf) · "
        "[Plan 2014 compendium](https://ijc.org/sites/default/files/Plan2014_CompendiumReport.pdf)", "",
        "## How much extra Ontario exposure would the candidate require?", "",
        "The archived half-margin replay is compared with standalone persistence MPC on "
        "the same 18 complete years. Exposure is in metre-periods above the respective replay "
        "thresholds. Differences below are sums over the cohort.", "",
        "| Data delay | Extra Ontario exposure | Avoided Lake St. Louis exposure | Extra Ontario exposure in worst year |",
        "|---:|---:|---:|---:|",
    ]
    for delay, group in deltas.groupby("delay"):
        total = group.added_ontario_m_periods.sum()
        worst = group.loc[group.added_ontario_m_periods.idxmax()]
        lines.append(f"| {delay} | {total:.3f} | {-group.added_downstream_m_periods.sum():.3f} | "
                     f"{worst.added_ontario_m_periods:.3f} ({int(worst.year)}) |")
    common = budgets.groupby("allowed_added_ontario_m_periods").passes_total_budget.all()
    passing = common[common].index.tolist()
    zero = budgets[budgets.allowed_added_ontario_m_periods.eq(0.)]
    passed_delays = int(zero.passes_every_year_budget.sum())
    lines += [
        "", f"The smallest tested cohort allowance that accepts the historical candidate at both "
        f"delays is {passing[0]:.2f} metre-periods." if passing else
        "The tested allowances did not accept the candidate at both delays.", "",
        "**Agreed research rule.** To replace the reference, require lower total Lake St. Louis "
        "exposure, zero added Ontario exposure in every matched year, preserved completion and "
        "no added physical violations. Report cost and worst-year trade-offs alongside the gate. "
        "Positive Ontario allowances are sensitivity cases. The earlier 0.10 metre-period "
        "allowance was an experimental screen.", "",
        f"The archived half-margin candidate passed the exposure criteria at {passed_delays}/{len(zero)} "
        "data delays.", "",
    ]
    return "\n".join(lines)


def plot_stage(predictions, peaks, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = sorted(peaks.year.unique()) if not peaks.empty else []
    if not matched:
        return
    styles = {
        "pointe_claire": ("Observed", "#162c3a", "-"),
        "full_carillon_same_day": ("Same-day Carillon proxy", "#007b75", "-"),
        "full_carillon_7_days": ("Seven-day Carillon proxy", "#ac542d", "--"),
        "two_flow_selected_lag": ("Lag selected on calibration", "#6252a0", ":"),
    }
    fig, axes = plt.subplots(len(matched), 1, figsize=(8.6, 2.65 * len(matched)), squeeze=False,
                             layout="constrained", sharey=True)
    for year, ax in zip(matched, axes[:, 0], strict=True):
        daily = predictions.reindex(pd.date_range(f"{year}-04-01", f"{year}-06-30"))
        for field, (label, color, style) in styles.items():
            ax.plot(daily.index, daily[field], style, color=color, label=label, linewidth=1.5)
        ax.set_title(str(year), loc="left")
        ax.set_ylabel("Stage in gauge datum (m)")
        ax.grid(alpha=.20)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Held-out Pointe-Claire spring stage")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=Path("results/ottawa/flow_source_audit_v1/historical_daily.csv"))
    parser.add_argument("--tradeoff", type=Path,
                        default=Path("results/loslr/slon_upper_bound_followup_v1/reserved_trajectories.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_hashes = {str(path): digest(path) for path in (args.observations, args.tradeoff)}
    predictions, scores, lags, peaks, metadata = compare_routing(aligned_data(args.observations))
    deltas, budgets = compare_tradeoff(args.tradeoff)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in (("predictions", predictions.reset_index()), ("routing_scores", scores),
                          ("calibration_lags", lags), ("spring_peaks", peaks),
                          ("annual_tradeoffs", deltas), ("allowance_sensitivity", budgets)):
        records.to_csv(args.output / f"{name}.csv", index=False)
    assert input_hashes == {name: digest(Path(name)) for name in input_hashes}
    (args.output / "protocol.json").write_text(json.dumps({
        **metadata, "stations": STATIONS, "allowances_m_periods": BUDGETS, "input_sha256": input_hashes,
    }, indent=2) + "\n")
    text = render_report(scores, peaks, metadata, deltas, budgets)
    (args.output / "report.md").write_text(text)
    plot_stage(predictions, peaks, args.output / "spring_stages.png")
    print(text)


if __name__ == "__main__":
    main()
