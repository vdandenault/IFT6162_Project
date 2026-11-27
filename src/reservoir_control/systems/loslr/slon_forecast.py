"""Benchmark past-only quarter-month SLON forecasts off line."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coupled import CONFIG, load_protocol
from .data import load_inputs
from .native import INPUT_FILE, NativeReference, sha256

LEADS = (1, 2, 3, 4)
DELAYS = (0, 1)
MODELS = ("persistence", "seasonal_mean", "seasonal_history")


def aligned_pairs(frame, lead, delay):
    """Issue at open of k, predict k+lead-1, use SLON through k-1-delay."""
    flow = frame.stlouisontOut
    last = flow.shift(1 + delay)
    return pd.DataFrame({
        "issue_index": np.arange(len(frame)),
        "issue_year": frame.Year,
        "issue_qm": frame.QM,
        "last_available_year": frame.Year.shift(1 + delay),
        "last_available_qm": frame.QM.shift(1 + delay),
        "target_year": frame.Year.shift(1 - lead),
        "target_qm": frame.QM.shift(1 - lead),
        "target_m3_s": flow.shift(1 - lead),
        "persistence": last,
        "recent_change": last - flow.shift(2 + delay),
        "recent_mean": (last + flow.shift(2 + delay) + flow.shift(3 + delay)) / 3.,
    })


def fit_forecasts(pairs, training_years, climatology):
    data = pairs.dropna(subset=pairs.columns.difference(["target_m3_s"])).copy()
    phase = 2 * np.pi * (data.target_qm - 1) / 48
    data["phase_sin"] = np.sin(phase)
    data["phase_cos"] = np.cos(phase)
    data["seasonal_mean"] = data.target_qm.map(climatology)
    features = ["persistence", "recent_change", "recent_mean", "seasonal_mean", "phase_sin", "phase_cos"]
    design = np.column_stack([np.ones(len(data)), data[features].to_numpy()])
    start, end = training_years
    fit = data.target_year.between(start, end) & data.issue_year.between(start, end) & data.target_m3_s.notna()
    coefficients, *_ = np.linalg.lstsq(
        design[fit], (data.target_m3_s - data.persistence).loc[fit].to_numpy(), rcond=None,
    )
    data["seasonal_history"] = data.persistence + design @ coefficients
    return data.loc[~fit], pd.Series(coefficients, index=["intercept", *features]).to_dict()


def scores(sample, threshold):
    for model in MODELS:
        actual = sample.target_m3_s
        forecast = sample[model]
        error = forecast - actual
        yield {
            "model": model, "periods": len(sample), "mae_m3_s": error.abs().mean(),
            "rmse_m3_s": np.sqrt(error.pow(2).mean()), "bias_m3_s": error.mean(),
            "mean_underprediction_m3_s": (-error).clip(lower=0).mean(),
            "high_flow_misses": int((actual.ge(threshold) & forecast.lt(threshold)).sum()),
            "high_flow_false_alarms": int((actual.lt(threshold) & forecast.ge(threshold)).sum()),
        }


def benchmark(frame, protocol):
    training = protocol["training_years"]
    first, last = protocol["development_years"]
    cohorts = {
        "development": range(first, last + 1),
        "reserved": protocol["reserved_test_years"],
        **{str(year): [year] for year in protocol["development_events"]},
    }
    observed_training = frame.loc[frame.Year.between(*training)].copy()
    threshold = float(observed_training.stlouisontOut.quantile(.9))
    seasonal = observed_training.groupby("QM").stlouisontOut.mean()
    forecasts, metrics, fits = [], [], {}
    for lead in LEADS:
        # Use the same decision and target periods under both delays.
        issued = {delay: aligned_pairs(frame, lead, delay) for delay in DELAYS}
        eligible = issued[DELAYS[-1]].notna().all(axis=1)
        for delay in DELAYS:
            predicted, coefficients = fit_forecasts(issued[delay].loc[eligible], training, seasonal)
            fits[f"lead_{lead}_delay_{delay}"] = coefficients
            for cohort, years in cohorts.items():
                sample = predicted[predicted.target_year.isin(years)].copy()
                sample["high_flow"] = sample.target_m3_s.ge(threshold)
                forecasts.append(sample.assign(cohort=cohort, lead=lead, delay=delay))
                for subset, rows in (("all", sample), ("high", sample[sample.high_flow])):
                    metrics.extend({"cohort": cohort, "lead": lead, "delay": delay, "subset": subset, **row}
                                   for row in scores(rows, threshold))
    predictions = pd.concat(forecasts, ignore_index=True)
    keys = ["cohort", "lead", "issue_index", "target_m3_s"]
    assert predictions[predictions.delay.eq(0)][keys].reset_index(drop=True).equals(
        predictions[predictions.delay.eq(1)][keys].reset_index(drop=True))
    return predictions, pd.DataFrame(metrics), threshold, fits


def report(metrics, protocol, threshold):
    lines = [
        "# Past-only SLON forecasts on the Ontario decision clock", "",
        "SLON is Lake St. Louis flow less Lake Ontario outflow in the pinned LOSLR forcing, "
        "an aggregate contribution that includes the Ottawa River and other tributaries. "
        "This is a benchmark of that forcing series. The controller and synthetic Ottawa experiment remain unchanged.", "",
        "At the start of period k, the no-delay mode uses values through k−1 and the one-period-delay mode "
        "uses values through k−2. Lead 1 targets k. Lead 4 targets k+3. Values on the issue period and "
        "later serve only as targets. Availability is a scenario assumption since the inputs carry no publication timestamps.", "",
        f"Fits and the seasonal climatology use {protocol['training_years'][0]}–{protocol['training_years'][1]}. "
        "The 1981–2000 development years, 18 reserved years and 2017 and 2019 flood years are reported "
        f"separately. The high-flow threshold is the training 90th percentile, {threshold:,.0f} m³/s. "
        "Forecasts retain signed values because the aggregate series can be negative.", "",
        "Persistence repeats the latest available value. Seasonal mean uses the historical mean for the "
        "target quarter-month. Seasonal history adds a fitted correction to persistence using recent "
        "SLON levels and changes and the target season. The two latency cases use separate fits on matched rows.", "",
        "| Sample | Delay | Lead | Overall n | Overall MAE, persistence / seasonal / history | High-flow n | High-flow MAE, persistence / seasonal / history | History bias at high flow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (cohort, delay, lead), group in metrics.groupby(["cohort", "delay", "lead"], sort=False):
        all_rows = group[group.subset.eq("all")].set_index("model").loc[list(MODELS)]
        high = group[group.subset.eq("high")].set_index("model").loc[list(MODELS)]
        overall_mae = " / ".join(f"{value:,.0f}" for value in all_rows.mae_m3_s)
        high_mae = " / ".join(f"{value:,.0f}" for value in high.mae_m3_s)
        lines.append(f"| {cohort} | {delay} | {lead} | {all_rows.periods.iloc[0]} | {overall_mae} | "
                     f"{high.periods.iloc[0]} | {high_mae} | {high.bias_m3_s.iloc[-1]:+,.0f} |")
    lines += ["", "Errors are in m³/s. Negative bias means underprediction. Full metrics include RMSE, "
              "mean underprediction and high-flow miss and false-alarm counts.", ""]
    return "\n".join(lines)


def plot_reserved(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("Persistence", "Seasonal mean", "Seasonal history")
    colors = ("#646d78", "#936041", "#157879")
    reserved = metrics[metrics.cohort.eq("reserved")]
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.3), layout="constrained", sharex=True, sharey="row")
    for row, subset in enumerate(("all", "high")):
        for delay in DELAYS:
            ax = axes[row, delay]
            sample = reserved[reserved.subset.eq(subset) & reserved.delay.eq(delay)]
            for model, label, color in zip(MODELS, labels, colors, strict=True):
                curve = sample[sample.model.eq(model)].sort_values("lead")
                ax.plot(curve.lead, curve.mae_m3_s, marker="o", color=color, label=label)
            ax.grid(alpha=.24)
            ax.set_xticks(LEADS)
            if row == 0:
                ax.set_title(("Latest completed period", "One-period data delay")[delay])
            else:
                ax.set_xlabel("Forecast lead (quarter-month periods)")
            if delay == 0:
                ax.set_ylabel(("All periods MAE (m³/s)", "High-flow MAE (m³/s)")[row])
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("SLON forecast error on 18 reserved years")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args()
    reference = NativeReference(args.checkout)
    # Keep observed periods only. The first 48 rows are initialization.
    frame = load_inputs(reference).iloc[48:].reset_index(drop=True)
    protocol = load_protocol()
    predictions, metrics, threshold, fits = benchmark(frame, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "scores.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "source": str(reference.root / INPUT_FILE),
        "source_sha256": sha256(reference.root / INPUT_FILE),
        "chronology_sha256": sha256(CONFIG),
        "training_years": protocol["training_years"],
        "development_years": protocol["development_years"],
        "reserved_years": protocol["reserved_test_years"],
        "flood_years": protocol["development_events"],
        "leads": LEADS, "delays": DELAYS, "high_flow_threshold_m3_s": threshold, "coefficients": fits,
    }, indent=2) + "\n")
    (args.output / "report.md").write_text(report(metrics, protocol, threshold))
    plot_reserved(metrics, args.output / "reserved_errors.png")


if __name__ == "__main__":
    main()
