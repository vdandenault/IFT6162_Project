from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ROOT = "https://api.weather.gc.ca/collections/"
LICENCE = "https://eccc-msc.github.io/open-data/licence/readme_en/"


def extract(payload, station):
    if any(x.get('rel') == 'next' for x in payload.get('links', [])):
        raise ValueError('Incomplete API page; do not mistake a partial response for full coverage')
    features = payload.get('features', [])
    records = [f['properties'] for f in features]
    if not records or any(r['STATION_NUMBER'] != station for r in records):
        raise ValueError('Station filter did not return the requested station')
    dates = [r['DATE'] for r in records]
    if len(set(dates)) != len(dates): raise ValueError('Duplicate source daily records require investigation')
    return pd.DataFrame(records).sort_values('DATE').reset_index(drop=True)


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
