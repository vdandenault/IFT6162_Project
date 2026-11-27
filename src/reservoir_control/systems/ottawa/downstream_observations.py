"""Acquire and check daily Cornwall–LaSalle flows and Pointe-Claire levels."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import sleep
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.data import quarter_start
from .flow_source_audit import LICENCE, address, check_records, fetch
from .reconstruction import digest

FIRST, LAST = 2000, 2025
FIT = (2000, 2016)
HOLDOUT = (2021, 2025)
CAP = 22.10
SHIFT = 200.
CHS_ID = "5cebf1e43d0f4a073c4bc480"
CHS_URL = f"https://api-iwls.dfo-mpo.gc.ca/api/v1/stations/{CHS_ID}/stats/calculate-daily-means-igld85"
CHS_LICENCE = "https://www.tides.gc.ca/en/licence-agreement"
USGS_URL = "https://waterservices.usgs.gov/nwis/dv/?" + urlencode({
    "format": "json", "sites": "04264331", "startDT": f"{FIRST}-01-01", "endDT": f"{LAST}-12-31",
    "parameterCd": "00060", "statCd": "00003",
})
STATIONS = {"lasalle": ("02OA016", "DISCHARGE"), "pointe_claire": ("02OA039", "LEVEL")}
NOTICE = (
    "This product is not to be used for navigation.\n\n"
    "This product was made by or for Vincent Dandenault and contains intellectual property (Data) "
    "of the Canadian Hydrographic Service of the Department of Fisheries and Oceans.\n\n"
    "The copyrights in the Data are and remain the property of His Majesty the King in Right of Canada "
    "and shall not be sold, licensed, leased, assigned or given to a third party.\n\n"
    "The incorporation of the Data in this product does not constitute an endorsement or an approval "
    "of this product by the Canadian Hydrographic Service, the Department of Fisheries and Oceans or "
    "His Majesty the King in Right of Canada."
)


def requests():
    queries = {"cornwall_usgs.json": USGS_URL}
    windows = [(f"{a}-01-01", f"{min(a + 4, LAST)}-12-31") for a in range(FIRST, LAST + 1, 5)]
    for station, _ in STATIONS.values():
        queries[f"{station}_station.json"] = address("hydrometric-stations", station, limit=2)
        for window in windows:
            queries[f"{station}_{window[0]}.json"] = address(
                "hydrometric-daily-mean", station, datetime="/".join(window), limit=2200, sortby="DATE",
            )
    for year in range(HOLDOUT[0], HOLDOUT[1] + 1):
        for month in (4, 5, 6):
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month + 1:02d}-01"
            queries[f"chs_{start}.json"] = CHS_URL + "?" + urlencode({"from": start, "to": end})
    return queries, windows


def read_cornwall(payload):
    [series] = payload["value"]["timeSeries"]
    if series["variable"]["unit"]["unitCode"] != "ft3/s":
        raise ValueError("Unexpected Cornwall discharge unit")
    [values] = series["values"]
    rows = values["value"]
    index = pd.to_datetime([row["dateTime"][:10] for row in rows])
    if index.has_duplicates:
        raise ValueError("Duplicate Cornwall dates")
    return pd.DataFrame({
        "cornwall": [float(row["value"]) * .3048 ** 3 if float(row["value"]) >= 0 else np.nan for row in rows],
        "cornwall_flags": ["|".join(row["qualifiers"]) for row in rows],
    }, index=index)


def load_records(raw, windows, queries):
    load = lambda name: json.loads((raw / name).read_bytes())
    calendar = pd.date_range(f"{FIRST}-01-01", f"{LAST}-12-31", name="date")
    frame = read_cornwall(load("cornwall_usgs.json")).reindex(calendar)
    metadata = {}
    for name, (station, measure) in STATIONS.items():
        [record] = load(f"{station}_station.json")["features"]
        assert record["properties"]["STATION_NUMBER"] == station
        metadata[name] = record
        values = pd.concat([
            check_records(load(f"{station}_{window[0]}.json"), station, measure, window) for window in windows
        ])
        values.index = pd.to_datetime(values.index)
        if values.index.has_duplicates:
            raise ValueError(f"Duplicate dates at {station}")
        frame[name], frame[f"{name}_flag"] = values.value, values.flag
    rows = [row for filename in queries if filename.startswith("chs_") for row in load(filename)]
    chs = pd.DataFrame(rows).set_index("date")
    chs.index = pd.to_datetime(chs.index)
    if chs.index.has_duplicates:
        raise ValueError("Duplicate CHS stage dates")
    return frame, chs.dailyMean_IGLD85, metadata


def check_datum(frame, chs):
    pair = frame.pointe_claire.where(frame.pointe_claire_flag.isna()).rename("hydat").to_frame().join(
        chs.rename("chs_igld85"), how="inner",
    ).dropna()
    error = pair.hydat - pair.chs_igld85
    return {"matched_days": len(pair), "median_difference_m": float(error.median()),
            "mae_m": float(error.abs().mean()), "p95_absolute_difference_m": float(error.abs().quantile(.95)),
            "datum_check_passed": bool(len(pair) >= 30 and error.abs().quantile(.95) <= .05)}


def validation(frame):
    data = frame.copy()
    data["local_quality"] = (data.lasalle.notna() & data.pointe_claire.notna()
                             & data.lasalle_flag.isna() & data.pointe_claire_flag.isna())
    data["three_station_quality"] = data.local_quality & data.cornwall.notna() & data.cornwall_flags.eq("A")
    data["lasalle_minus_cornwall_m3_s"] = (data.lasalle - data.cornwall).where(data.three_station_quality)
    data["observed_stage_headroom_m"] = (CAP - data.pointe_claire).where(data.pointe_claire_flag.isna())
    fit = data[data.index.year.isin(range(FIT[0], FIT[1] + 1)) & data.local_quality
               & data.index.month.isin((4, 5, 6))]
    x = (fit.lasalle.to_numpy() / 604.) ** .58
    if not len(x) or not (x @ x > 0):
        raise ValueError("No usable calibration flows")
    factor = float(x @ (fit.pointe_claire.to_numpy() - 16.57) / (x @ x))
    data["modeled_stage_m"] = (16.57 + factor * (data.lasalle / 604.) ** .58).where(data.local_quality)
    flow_cap = 604. * ((CAP - 16.57) / factor) ** (1 / .58)
    shift_stage = factor * (((flow_cap + SHIFT) / 604.) ** .58 - (flow_cap / 604.) ** .58)
    return data, {"fit_years": FIT, "fit_months": [4, 5, 6], "fit_days": len(fit),
                  "rating_factor": factor, "modeled_200_m3_s_stage_increment_at_cap_m": shift_stage}


def summarize(data):
    annual = []
    for year, group in data.groupby(data.index.year):
        spring = group[group.index.month.isin((4, 5, 6)) & group.local_quality]
        actual_excess = spring.observed_stage_headroom_m.lt(0.)
        modeled_excess = spring.modeled_stage_m.gt(CAP)
        annual.append({"year": year, "days": len(group), "lasalle_flow_days": group.lasalle.notna().sum(),
                       "pointe_claire_stage_days": group.pointe_claire.notna().sum(),
                       "cornwall_flow_days": group.cornwall.notna().sum(), "paired_unflagged_days": group.local_quality.sum(),
                       "three_station_approved_days": group.three_station_quality.sum(),
                       "spring_days": len(spring), "observed_spring_exceedance_days": int(actual_excess.sum()),
                       "missed_spring_exceedance_days": int((actual_excess & ~modeled_excess).sum()),
                       "spring_stage_mae_m": (spring.modeled_stage_m - spring.pointe_claire).abs().mean()})
    dates = [(quarter_start(y, q), y, q) for y in range(FIRST, LAST + 1) for q in range(1, 49)]
    quarters = []
    for (begin, year, qm), (end, _, _) in zip(dates, [*dates[1:], (pd.Timestamp(LAST + 1, 1, 1), 0, 0)], strict=True):
        sample = data.loc[(data.index >= begin) & (data.index < end)]
        valid = sample.local_quality.all()
        quarters.append({
            "year": year, "qm": qm, "start": begin.date(), "end_exclusive": end.date(),
            "days": len(sample), "paired_unflagged_days": sample.local_quality.sum(),
            "pointe_claire_mean_m": sample.pointe_claire.mean() if valid else np.nan,
            "pointe_claire_max_m": sample.pointe_claire.max() if valid else np.nan,
            "lasalle_mean_m3_s": sample.lasalle.mean() if valid else np.nan,
            "cornwall_mean_m3_s": sample.cornwall.mean() if sample.three_station_quality.all() else np.nan,
        })
    return pd.DataFrame(annual), pd.DataFrame(quarters)


def heldout_spring(data):
    return data.loc[data.index.year.isin(range(HOLDOUT[0], HOLDOUT[1] + 1))
                    & data.index.month.isin((4, 5, 6)) & data.local_quality]


def plot(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spring = heldout_spring(data)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharey=True, layout="constrained")
    for ax, (year, sample) in zip(axes.flat, spring.groupby(spring.index.year)):
        sample = sample.reindex(pd.date_range(f"{year}-04-01", f"{year}-06-30"))
        ax.plot(sample.index, sample.pointe_claire, color="#136b73", label="Observed stage")
        ax.plot(sample.index, sample.modeled_stage_m, color="#a34e20", label="Stage estimated from LaSalle flow")
        ax.axhline(CAP, color="#555555", ls="--", lw=1, label="22.10 m study threshold")
        ax.set_title(str(year), loc="left")
        ax.set_xticks(pd.to_datetime([f"{year}-04-01", f"{year}-05-01", f"{year}-06-01"]),
                      labels=["Apr 1", "May 1", "Jun 1"])
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[-1, -1].axis("off")
    axes[-1, -1].legend(handles, labels, loc="center left", frameon=False)
    fig.suptitle("Pointe-Claire stage under observed downstream flows · held-out springs")
    fig.supylabel("Stage (m, IGLD 1985)")
    fig.savefig(output / "heldout_spring_stage.png", dpi=170)
    plt.close(fig)


def report(data, annual, quarters, datum, fit):
    new = annual[annual.year.between(*HOLDOUT)]
    eligible = quarters[quarters.year.between(*HOLDOUT)].dropna(subset=["pointe_claire_mean_m"])
    spring = heldout_spring(data)
    error = spring.modeled_stage_m - spring.pointe_claire
    high = spring.pointe_claire.gt(CAP)
    modeled_high = spring.modeled_stage_m.gt(CAP)
    high_periods = eligible[eligible.pointe_claire_mean_m.gt(CAP)]
    period_dates = ", ".join(pd.to_datetime(high_periods.start).dt.strftime("%Y-%m-%d"))
    lines = [
        "# Independent downstream gauge records for Ontario headroom", "",
        "## Acquisition and boundary", "",
        "The date-aligned daily benchmark includes LaSalle total river discharge (HYDAT 02OA016), "
        "Pointe-Claire water level (HYDAT 02OA039), and Cornwall–Massena discharge (USGS 04264331). "
        "LaSalle is about 15 km downstream of Pointe-Claire. The published LaSalle discharge is "
        "derived from stage–discharge methods. It provides a separate downstream station record "
        "from the LOSLR archive, with uncertainty from the rating and ice. Cornwall is a sum of "
        "upstream dam, diversion and canal flows. LaSalle minus Cornwall is retained as a reach "
        "residual and can reflect branch inputs, transit, storage and measurement effects.", "",
        "[NRC station and rating description](https://publications.gc.ca/collections/collection_2023/cnrc-nrc/NR16-420-2023-eng.pdf) · "
        "[USGS Cornwall measurement definition](https://water.usgs.gov/nwc/NWC/sw/man/S04264331.html) · "
        "[ECCC HYDAT licence](https://eccc-msc.github.io/open-data/licence/readme_en/)", "",
        "## Pointe-Claire datum", "",
        f"On {datum['matched_days']:,} overlapping unflagged spring dates in 2021–2025, HYDAT minus "
        f"the CHS IGLD85 daily product had a median difference of {datum['median_difference_m']:+.3f} m "
        f"and a 95th-percentile absolute difference of {datum['p95_absolute_difference_m']:.3f} m. "
        "CHS and HYDAT use the same physical gauge. This verifies agreement of datum and paired daily "
        "levels over the modern overlap. The older archive remains a comparison with the same gauge's "
        "published values. The CHS check is not a second independent stage measurement.", "",
        "[CHS datum notice](https://cartes.gc.ca/help-aide/announcements-annonces/old/2020-03-23-eng.html) · "
        "[CHS IGLD daily mean definition](https://www.tides.gc.ca/en/definitions-content-tides-and-currents#igld-means)", "",
        "## New evaluation cases and headroom check", "",
        f"The post-2020 sample adds {int(new.paired_unflagged_days.sum()):,} unflagged LaSalle–Pointe-Claire "
        f"pairs and {int(new.three_station_approved_days.sum()):,} with approved Cornwall discharge. "
        f"All daily values were present and eligible in {len(eligible)}/240 quarter-month periods, "
        f"of which {int(eligible.cornwall_mean_m3_s.notna().sum())} also have approved Cornwall flow. "
        f"{int(eligible.pointe_claire_mean_m.gt(CAP).sum())} complete periods exceeded the fixed {CAP:.2f} m "
        "study threshold on the period mean. Quarter-month alignment follows the legacy IJC calendar "
        "as a date convention. These are new validation cases outside the existing LOSLR forcing archive.", "",
        f"Above-threshold period start dates: {period_dates or 'none'}. "
        "Daily records are joined by source date labels. The benchmark does not establish matched hourly windows "
        "or routing lag between stations. Quarter-month averages use available complete calendar bins.", "",
        "A first offline check uses the native rating shape at LaSalle discharge with one scale fitted "
        "on unflagged April–June 2000–2016 pairs. The 2021–2025 spring evaluation uses concurrent "
        "discharge as input, so it measures stage agreement under observed flow conditions.", "",
        f"On {len(spring)} held-out spring days, stage MAE was {error.abs().mean():.3f} m and "
        f"the 95th-percentile absolute error was {error.abs().quantile(.95):.3f} m, versus an estimated "
        f"{fit['modeled_200_m3_s_stage_increment_at_cap_m']:.3f} m change for 200 m³/s near the study "
        f"threshold. The curve overestimated stage by {error.mean():.3f} m on average. "
        f"It missed {int((high & ~modeled_high).sum())} of {int(high.sum())} spring exceedance days "
        f"and marked {int((~high & modeled_high).sum())} of {int((~high).sum())} other days above the threshold. "
        "The gauge pair can validate stage agreement and event timing. Reconstructing a counterfactual "
        "release or F-limit compliance requires matched forcing, travel and branch partition. "
        "A fixed 22.10 m comparison does not apply the changing F-limit tier.", "",
        "![Observed and modeled stage on the held-out springs](heldout_spring_stage.png)", "",
        "Next, use the paired gauge records to check travel time and the conversion between Cornwall releases "
        "and Lake St. Louis throughflow on development years. Reserve the 2021–2025 observations to check stage "
        "error near the threshold. The three new high-stage periods provide event checks and a limited sample "
        "for alert comparisons.", "",
        "Source data: Environment and Climate Change Canada / Water Survey of Canada, USGS, and CHS. "
        "CHS responses stay local under the [CHS data licence](https://www.tides.gc.ca/en/licence-agreement).", "",
        NOTICE, "",
    ]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Use the original local source responses")
    args = parser.parse_args()
    urls, windows = requests()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.raw or args.output / "raw"
    if args.raw is None:
        raw.mkdir()
        regular = {name: url for name, url in urls.items() if not name.startswith("chs_")}
        with ThreadPoolExecutor(max_workers=3) as executor:
            for name, body in zip(regular, executor.map(fetch, regular.values()), strict=True):
                (raw / name).write_bytes(body)
        for name, url in urls.items():
            if name.startswith("chs_"):
                (raw / name).write_bytes(fetch(url))
                sleep(0.4)
    frame, chs, metadata = load_records(raw, windows, urls)
    datum = check_datum(frame, chs)
    (args.output / "protocol.json").write_text(json.dumps({
        "gauge_metadata": metadata, "datum": datum, "study_threshold_m": CAP, "shift_m3_s": SHIFT,
        "ec_licence": LICENCE, "chs_licence": CHS_LICENCE,
        "sources": {name: {"url": url, "sha256": digest(raw / name)} for name, url in urls.items()},
    }, indent=2) + "\n")
    if not datum["datum_check_passed"]:
        raise ValueError(f"IGLD85 comparison needs investigation before absolute stage scoring: {datum}")
    data, fit = validation(frame)
    annual, quarters = summarize(data)
    data.to_csv(args.output / "daily.csv")
    annual.to_csv(args.output / "annual.csv", index=False)
    quarters.to_csv(args.output / "quarter_months.csv", index=False)
    (args.output / "rating_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    plot(data, args.output)
    text = report(data, annual, quarters, datum, fit)
    (args.output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
