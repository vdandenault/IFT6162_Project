from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from reservoir_control.study_config import OTTAWA_EVIDENCE, load_settings
from .reconstruction import StorageMapping, digest, prepare, plots

CONFIG = OTTAWA_EVIDENCE


def requests_for(c):
    root = c['api_root']
    queries = {}
    for station in sorted({c['level_station'], c['discharge_station']}):
        queries[f'{station}_daily.json'] = root+c['dataset']+'/items?'+urlencode(dict(
            STATION_NUMBER=station, datetime=c['target']['start_date']+'/'+c['target']['end_date_exclusive'],
            limit=500, f='json'))
        queries[f'{station}_station.json'] = root+'hydrometric-stations/items?'+urlencode(dict(
            STATION_NUMBER=station, limit=2, f='json'))
    queries['collection.json'] = root+c['dataset']+'?f=json'
    return queries


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


def symbol(value):
    return 'HYDAT_unflagged' if pd.isna(value) or value == '' else 'HYDAT_symbol:'+str(value)


def process(c, raw, output):
    level = extract(json.loads((raw/f"{c['level_station']}_daily.json").read_text()), c['level_station'])
    discharge = extract(json.loads((raw/f"{c['discharge_station']}_daily.json").read_text()), c['discharge_station'])
    station = json.loads((raw/f"{c['level_station']}_station.json").read_text())['features']
    if len(station) != 1 or station[0]['properties']['STATION_NUMBER'] != c['level_station']:
        raise ValueError('One matching level-station metadata record required')
    station = station[0]['properties']
    collection = json.loads((raw/'collection.json').read_text())
    if 'daily mean' not in collection['title'].lower():
        raise ValueError('Source collection no longer declares daily means')
    start, end = c['target']['start_date'], c['target']['end_date_exclusive']
    annual = level[(level.DATE >= start) & (level.DATE < end)]
    observed = pd.to_numeric(annual.LEVEL, errors='raise').dropna()
    if len(observed) < 2 or observed.max() == observed.min(): raise ValueError('Nonconstant observed reference range required')
    # This reference is descriptive, not a level-to-storage model.
    lo, hi = float(observed.min()), float(observed.max())
    metadata = dict(c, assessment_date=json.loads((raw/'manifest.json').read_text())['retrieved_utc'][:10],
                    level_position_reference=dict(minimum_m=lo, maximum_m=hi,
                        basis='Observed 1970 daily mean range, same station datum; not operating bounds'),
                    source_vertical_datum=station['VERTICAL_DATUM'])
    contributors = {key: json.loads((raw/f'{key}_station.json').read_text())['features'][0]['properties']['CONTRIBUTOR_EN']
                    for key in sorted({c['level_station'], c['discharge_station']})}
    metadata['contributors_by_station'] = contributors
    sources = {s: dict(owner='Environment and Climate Change Canada; '+contributors[s],
            url=requests_for(c)[f'{s}_daily.json'], terms_url=c['terms_url'],
            permission_basis=c['reuse'], local_processing_and_copy_permitted=True,
            attribution=c['attribution']) for s in sorted({c['level_station'], c['discharge_station']})}
    contract = dict(schema_version=1, timezone='UTC', timing_verified=False,
        day_definition_evidence=c['measurement_timing'],
        level_datum=station['VERTICAL_DATUM'], datum_verified=True,
        datum_alignment_evidence='Only comparisons within the same archived station datum and observed-year range. No alignment to geodetic operating elevations asserted.',
        total_outlet_verified=False, outlet_definition=c['discharge_boundary'],
        usable_quality_flags=['HYDAT_unflagged'], sources=sources)
    levels = pd.DataFrame([dict(timestamp=r.DATE+'T00:00:00+00:00', level=r.LEVEL,
        unit='m', aggregation='daily_mean', statistic='observed',
        quality_flag=symbol(r.LEVEL_SYMBOL_EN), source_id=c['level_station']) for r in level.itertuples()])
    outflows = pd.DataFrame([dict(interval_start=r.DATE+'T00:00:00+00:00',
        interval_end=(pd.Timestamp(r.DATE)+pd.Timedelta(days=1)).date().isoformat()+'T00:00:00+00:00',
        mean_discharge=r.DISCHARGE, unit='m3/s', aggregation='interval_mean', statistic='observed',
        quality_flag=symbol(r.DISCHARGE_SYMBOL_EN), source_id=c['discharge_station'])
        for r in discharge.itertuples() if start <= r.DATE < end])
    # Reuse the existing pipeline, leaving its interval-alignment safety gates in place.
    frame, summary, _, _ = prepare(levels, outflows, metadata, contract, StorageMapping(lo, hi))
    assert frame.net_inflow_m3_s.isna().all() and frame.storage_start_m3.isna().all()
    frame['physical_interval_verified'] = False
    # Primary deliverable uses source dates rather than pretending UTC anchors are measured instants.
    dates = pd.DataFrame({'DATE': pd.date_range(start, pd.Timestamp(end)-pd.Timedelta(days=1), freq='D').strftime('%Y-%m-%d')})
    clean = dates.merge(level[['DATE', 'LEVEL', 'LEVEL_SYMBOL_EN', 'DISCHARGE']], on='DATE', how='left', validate='one_to_one')
    clean = clean.rename(columns={'LEVEL': 'daily_mean_level_m', 'LEVEL_SYMBOL_EN': 'level_symbol',
                                   'DISCHARGE': 'reservoir_station_discharge_m3_s'})
    clean = clean.merge(discharge[['DATE', 'DISCHARGE', 'DISCHARGE_SYMBOL_EN']].rename(columns={
        'DISCHARGE': 'downstream_mercier_daily_mean_discharge_m3_s',
        'DISCHARGE_SYMBOL_EN': 'discharge_symbol'}), on='DATE', how='left', validate='one_to_one')
    clean['normalized_water_level_position'] = (clean.daily_mean_level_m-lo)/(hi-lo)
    clean['level_source_id'], clean['discharge_source_id'] = c['level_station'], c['discharge_station']
    clean['definition_flags'] = 'daily_mean_not_boundary;date_clock_unverified;all_outlets_unverified;no_storage_mapping'
    missing = clean[['daily_mean_level_m', 'downstream_mercier_daily_mean_discharge_m3_s']].isna()
    both = ~missing.any(axis=1)
    common = clean[['reservoir_station_discharge_m3_s', 'downstream_mercier_daily_mean_discharge_m3_s']].dropna()
    difference = common.iloc[:, 0]-common.iloc[:, 1]
    summary.update(status='actual_observations_processed_no_volumetric_reconstruction',
        actual_overlap_days=int(both.sum()),
        unflagged_overlap_days=int((both & clean.level_symbol.isna() & clean.discharge_symbol.isna()).sum()),
        missing_level_days=int(missing.iloc[:, 0].sum()),
        missing_discharge_days=int(missing.iloc[:, 1].sum()),
        level_range_m=[lo, hi], discharge_range_m3_s=[float(clean.downstream_mercier_daily_mean_discharge_m3_s.min()), float(clean.downstream_mercier_daily_mean_discharge_m3_s.max())],
        discharge_series_differ_days=int((difference.abs()>1e-6).sum()),
        discharge_difference_max_m3_s=float(difference.abs().max()),
        endpoint_context='1971-01-01 daily mean level retained; it is not the 1971-01-01 midnight boundary level',
        date_axis_only=True, historical_interval_clock_verified=False,
        level_symbol_counts=dict(Counter(symbol(x) for x in annual.LEVEL_SYMBOL_EN)),
        discharge_symbol_counts=dict(Counter(symbol(x) for x in clean.discharge_symbol)),
        partial_day_discharge_values=int(clean.discharge_symbol.eq('Partial Day').sum()),
        independent_validation=False)
    output.mkdir(exist_ok=True)
    clean.to_csv(output/'observations.csv', index=False)
    frame.to_csv(output/'pipeline_daily.csv', index=False)
    levels.to_csv(output/'levels_input.csv', index=False); outflows.to_csv(output/'outflows_input.csv', index=False)
    (output/'reservoir_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    (output/'input_contract.json').write_text(json.dumps(contract, indent=2, ensure_ascii=False)+'\n')
    plots(frame, metadata, output)
    # A compact date-labelled plot is easier to interpret than the balance-only diagnostics.
    observation_plot(clean, c, output)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'report.md').write_text(pilot_report(c, summary, station))
    return summary


def observation_plot(clean, c, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout='constrained')
    dates = pd.to_datetime(clean.DATE)
    for ax, column, label in zip(axes[:3], ['daily_mean_level_m', 'downstream_mercier_daily_mean_discharge_m3_s',
            'normalized_water_level_position'], ['Daily mean level (m)\nstation arbitrary datum',
            'Mercier downstream\ndaily mean discharge (m³/s)', 'Normalized water-level\nposition (1970 range)']):
        ax.plot(dates, clean[column], color='#287ea0', linewidth=1.3); ax.set_ylabel(label); ax.grid(alpha=.2)
    flagged = clean.discharge_symbol.notna()
    axes[1].scatter(dates[flagged], clean.loc[flagged, 'downstream_mercier_daily_mean_discharge_m3_s'],
                    s=15, color='#c56d24', label='Source quality symbol present')
    axes[1].legend(loc='upper right')
    axes[2].set_ylim(-.05, 1.05)
    present = clean[['daily_mean_level_m', 'downstream_mercier_daily_mean_discharge_m3_s']].notna()
    axes[3].step(dates, present.iloc[:, 0].astype(int), where='post', label='Level present')
    axes[3].step(dates, present.iloc[:, 1].astype(int), where='post', linestyle='--', label='Discharge present')
    axes[3].step(dates, (present.iloc[:, 1] & ~flagged).astype(int), where='post', label='Unflagged discharge', alpha=.6)
    axes[3].set_ylim(-.1, 1.15); axes[3].set_ylabel('Observation coverage'); axes[3].legend()
    fig.suptitle('Baskatong / Mercier — actual 1970 HYDAT observations\nDaily means and source dates; no storage or inflow reconstruction')
    fig.savefig(output/'observed_1970.png', dpi=150); plt.close(fig)


def pilot_report(c, s, station):
    return f'''# Baskatong / Mercier: actual 1970 observations

**{s['actual_overlap_days']} / 365 dates have both a reported level and discharge.**
The ECCC official OGC API supplied raw daily records for stations 02LG007 and
02LG006, including one following-date context record. Original JSON and hashes
are preserved in `raw/`; `observations.csv` is date-labelled, not an invented
instantaneous boundary series. Source: ECCC HYDAT / MSC GeoMet, level contributor {station['CONTRIBUTOR_EN']};
discharge contributor is recorded separately in the station metadata. [Reuse terms]({c['terms_url']}).

## Measurement definitions and limits

- Levels: daily means at 02LG007, datum **{station['VERTICAL_DATUM']}**.
  Range: **{s['level_range_m'][0]:.3f}–{s['level_range_m'][1]:.3f} m**.
- Discharge: reported daily means at 02LG006, officially downstream of Mercier.
  Its all-outlets equivalence for Baskatong, including Castor, remains unverified.
- 02LG007's own discharge is preserved separately. The two discharge series differ
  on **{s['discharge_series_differ_days']} dates**, by up to **{s['discharge_difference_max_m3_s']:.3f} m³/s**.
  Do not silently substitute one for the other or interpret their difference as a known transfer.
- The API publishes calendar DATE and daily means. The station's historical day
  clock/offset was not established. UTC anchors in `pipeline_daily.csv` are only
  internal plotting/alignment scaffolding, explicitly flagged unverified. They
  are **not** evidence of UTC measurements or aligned physical intervals.
- 1971-01-01 is another daily mean, not a midnight closing observation.
- Modern 207.60–223.14 m operating elevations are not overlaid on the historical
  arbitrary datum. Normalization uses **this year's observed range**, not usable
  storage or an operating range. No inventory capacity is used as K.

## Outputs and checks

Missing levels: {s['missing_level_days']}; missing downstream discharges:
{s['missing_discharge_days']}. However, **{s['partial_day_discharge_values']} downstream discharge values carry the
source's Partial Day symbol**; only **{s['unflagged_overlap_days']} dates** have
unflagged values for both variables. Full date coverage does not mean complete
within-day measurement coverage. Flagged values remain visible and are marked
on the plot; they are excluded by the pipeline's strict usable-quality contract.
No interpolation, clipping or missing-to-zero fill.
Source symbols are retained; null symbols are labelled unflagged, not invented
approval grades. Level symbols: {s['level_symbol_counts']}; discharge symbols: {s['discharge_symbol_counts']}. `summary.json` records flags, gap counts and the source mismatch.
`observed_1970.png` shows the observed seasonal drawdown/refill, downstream
release pattern, normalized water-level position and coverage. The existing
pipeline additionally writes `daily_diagnostics.png` and `pipeline_daily.csv`.

**No volumetric inflow is reconstructed.** These mean levels are not boundary
levels, outlet completeness and daily timing are unverified, and no storage
curve or justified usable K was obtained. All computed storage/inflow columns
remain missing. An accounting replay would not independently validate a model.

## Reproduce

```bash
uv run python -m reservoir_control.systems.ottawa.historical_pilot \\
  --output results/ottawa/baskatong_1970_review
```

For byte-identical source reuse, add `--raw-dir /path/to/this/run/raw`; the saved
raw manifest is verified. A fresh network run may reflect later archive revisions.
The selected year and station definitions are frozen in
the `baskatong_1970` section of `configs/ottawa_evidence.json`.

Next specific requirement for a closed balance: instantaneous boundary levels,
an owner-confirmed mapping of 02LG006/02LG007 and all active outlets, historical
clock/datum linkage, and a storage–elevation curve or explicitly justified usable
volume scenario. This pilot establishes acquisition and seasonal observation
coverage, not calibrated inflow or suitability for a reservoir network.

[Official collection](https://api.weather.gc.ca/collections/hydrometric-daily-mean?f=json)
· [API documentation]({c['api_documentation']})
'''


def main():
    parser = ArgumentParser(
        description=(
            "Acquire actual Baskatong/Mercier daily records for the fixed 1970 pilot.\n"
            "\n"
            "Uses the official ECCC OGC API and the existing daily processing functions.\n"
            "Only date-labelled daily means are claimed; no fabricated boundary observations.\n"
        )
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, help='Reuse a verified local acquisition instead of network access')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh output directory')
    c = load_settings(CONFIG, "baskatong_1970"); urls = requests_for(c)
    args.output.mkdir(parents=True); raw = args.output/'raw'; raw.mkdir()
    if args.raw_dir:
        manifest = json.loads((args.raw_dir/'manifest.json').read_text())
        if set(manifest['files']) != set(urls): raise ValueError('Raw bundle does not match this fixed pilot')
        for name, info in manifest['files'].items():
            if digest(args.raw_dir/name) != info['sha256']: raise ValueError('Raw source checksum mismatch')
            shutil.copy2(args.raw_dir/name, raw/name)
    else:
        manifest = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                        attribution=c['attribution'], terms_url=c['terms_url'], files={})
        for name, url in urls.items():
            with urlopen(url, timeout=60) as response: content = response.read()
            json.loads(content)  # Fail on an error/HTML response, never parse it as records.
            (raw/name).write_bytes(content)
            manifest['files'][name] = dict(url=url, sha256=digest(raw/name))
    (raw/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result = process(c, raw, args.output)
    provenance = dict(source_manifest_sha256=digest(raw/'manifest.json'), config_sha256=digest(CONFIG),
        implementation_sha256=digest(__file__), pipeline_sha256=digest(Path(__file__).with_name('reconstruction.py')),
        output_sha256={p.name: digest(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('actual_overlap_days', 'missing_level_days', 'missing_discharge_days',
                     'level_range_m', 'discharge_series_differ_days', 'reconstruction_intervals')}, indent=2))
    print(f'Report: {args.output / "report.md"}')


if __name__ == '__main__': main()
