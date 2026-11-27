from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .historical_pilot import extract
from .reconstruction import digest

ROOT = "https://api.weather.gc.ca/collections/"
LICENCE = "https://eccc-msc.github.io/open-data/licence/readme_en/"
STATIONS = {
    "02LB024": ("Carillon Dam", "DISCHARGE"),
    "02LB001": ("Grenville", "DISCHARGE"),
    "02KF005": ("Britannia", "DISCHARGE"),
    "02OA039": ("Pointe-Claire", "LEVEL"),
    "02OA105": ("Pointe-Calumet", "LEVEL"),
    "02OA013": ("Sainte-Anne-de-Bellevue", "LEVEL"),
    "02OA004": ("Rivière des Prairies", "DISCHARGE"),
    "02MC002": ("Cornwall", "DISCHARGE"),
}
WINDOWS = {"historical": ("1987-01-01", "1994-12-31"), "modern": ("2017-01-01", "2019-12-31")}


def address(collection, station, **params):
    return ROOT + collection + "/items?" + urlencode({"STATION_NUMBER": station, "f": "json", **params})


def fetch(url):
    with urlopen(url, timeout=45) as response:
        return response.read()


def check_records(payload, station, field, window):
    features = payload["features"]
    if any(link.get("rel") == "next" for link in payload.get("links", [])):
        raise ValueError(f"Paged response for {station}; refusing partial coverage")
    if payload.get("numberMatched", len(features)) != len(features):
        raise ValueError(f"Incomplete response for {station}")
    if not features:
        return pd.DataFrame(columns=["value", "flag"], index=pd.Index([], name="DATE"))
    data = extract(payload, station)
    if not data.DATE.between(*window).all():
        raise ValueError(f"Out-of-window values for {station}")
    return data.set_index("DATE")[[field, f"{field}_SYMBOL_EN"]].set_axis(["value", "flag"], axis=1)


def main():
    parser = ArgumentParser(
        description="Check actual HYDAT daily coverage for prospective Ottawa routing benchmarks."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    urls = {}
    for station in STATIONS:
        urls[f"{station}_station.json"] = address("hydrometric-stations", station, limit=2)
        for direction, order in (("first", "DATE"), ("last", "-DATE")):
            urls[f"{station}_{direction}.json"] = address("hydrometric-daily-mean", station, limit=1, sortby=order)
        for name, window in WINDOWS.items():
            urls[f"{station}_{name}.json"] = address("hydrometric-daily-mean", station, limit=3500,
                                                     datetime="/".join(window), sortby="DATE")

    with ThreadPoolExecutor(max_workers=3) as executor:
        for filename, data in zip(urls, executor.map(fetch, urls.values()), strict=True):
            (raw / filename).write_bytes(data)

    coverage, overlap = [], []
    for name, window in WINDOWS.items():
        calendar = pd.date_range(*window).strftime("%Y-%m-%d")
        data = pd.DataFrame(index=pd.Index(calendar, name="date"))
        usable = pd.DataFrame(index=data.index)
        for station, (label, field) in STATIONS.items():
            load = lambda suffix: json.loads((raw / f"{station}_{suffix}.json").read_bytes())
            metadata = load("station")["features"]
            assert len(metadata) == 1 and metadata[0]["properties"]["STATION_NUMBER"] == station
            bounds = [load(part)["features"] for part in ("first", "last")]
            points = check_records(load(name), station, field, window)
            data[f"{station}_{field.lower()}"] = points.value
            data[f"{station}_flag"] = points.flag
            usable[station] = points.value.notna() & points.flag.isna()
            coverage.append({
                "sample": name, "station": station, "site": label, "measure": field,
                "days_expected": len(calendar), "dates_returned": len(points),
                "non_null_values": int(points.value.notna().sum()),
                "flagged_values": int((points.value.notna() & points.flag.notna()).sum()),
                "flag_counts": json.dumps(points.loc[points.value.notna(), "flag"].value_counts().to_dict()),
                "first_dated_record_any_measure": bounds[0][0]["properties"]["DATE"] if bounds[0] else None,
                "last_dated_record_any_measure": bounds[1][0]["properties"]["DATE"] if bounds[1] else None,
                "datum": metadata[0]["properties"]["VERTICAL_DATUM"],
                "contributor": metadata[0]["properties"]["CONTRIBUTOR_EN"],
            })
        data.to_csv(output / f"{name}_daily.csv")
        groups = {
            "Ottawa flow forecast inputs and target": ["02KF005", "02LB024"],
            "Historical downstream check": ["02LB024", "02OA105", "02OA039", "02MC002"],
            "Modern proxy and downstream stage": ["02KF005", "02OA105", "02OA039"],
        }
        for label, stations in groups.items():
            nonnull = data[[f"{s}_{STATIONS[s][1].lower()}" for s in stations]].notna().all(axis=1)
            unflagged = usable[stations].fillna(False).astype(bool).all(axis=1)
            overlap.append({"sample": name, "group": label, "expected": len(calendar),
                            "overlap": int(nonnull.sum()), "overlap_unflagged": int(unflagged.sum())})
    for name, rows in (("coverage", coverage), ("overlap", overlap)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
        print(pd.DataFrame(rows).drop(columns=["datum", "contributor", "flag_counts"], errors="ignore").to_string(index=False))
    (output / "provenance.json").write_text(json.dumps({
        "source": "Environment and Climate Change Canada / HYDAT / MSC GeoMet",
        "terms": LICENCE, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "windows": WINDOWS, "requests": {name: {"url": url, "sha256": digest(raw / name)}
                                        for name, url in urls.items()},
        "daily_time": "Official station date labels and daily means. No intraday alignment inferred.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
