"""Read-only comparison of the saved 1970 Baskatong/Mercier discharge records."""
from argparse import ArgumentParser
import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import numpy as np
import pandas as pd
from .historical_pilot import extract
from .reconstruction import ROOT, digest

EVIDENCE = ROOT/'configs/studies/ottawa/discharge_comparison_evidence.json'
STATIONS = ('02LG007', '02LG006')


def snapshot(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def align(payloads):
    rows = pd.DataFrame({'DATE': pd.date_range('1970-01-01', '1970-12-31').strftime('%Y-%m-%d')})
    for station in STATIONS:
        source = extract(payloads[station], station)
        source = source[['DATE', 'DISCHARGE', 'DISCHARGE_SYMBOL_EN', 'DISCHARGE_SYMBOL_FR']].rename(columns={
            'DISCHARGE': f'q_{station}_m3_s', 'DISCHARGE_SYMBOL_EN': f'symbol_{station}_en',
            'DISCHARGE_SYMBOL_FR': f'symbol_{station}_fr'})
        rows = rows.merge(source, on='DATE', how='left', validate='one_to_one')
        key = f'symbol_{station}_en'
        rows[f'flagged_{station}'] = rows[key].notna() & rows[key].fillna('').astype(str).str.strip().ne('')
    rows['difference_007_minus_006_m3_s'] = rows.q_02LG007_m3_s-rows.q_02LG006_m3_s
    rows['any_source_flag'] = rows.flagged_02LG007 | rows.flagged_02LG006
    rows['both_values_present'] = rows[['q_02LG007_m3_s', 'q_02LG006_m3_s']].notna().all(axis=1)
    return rows


def statistics(rows, label):
    pairs = rows[rows.both_values_present]
    d = pairs.difference_007_minus_006_m3_s
    result = dict(subset=label, calendar_dates=len(rows), paired_dates=len(pairs),
                  missing_pairs=len(rows)-len(pairs), flagged_pairs=int(pairs.any_source_flag.sum()))
    if pairs.empty: return result
    biggest = pairs.loc[d.abs().idxmax()]
    result.update(mean_signed_difference_m3_s=float(d.mean()), mean_absolute_difference_m3_s=float(d.abs().mean()),
        root_mean_square_difference_m3_s=float(np.sqrt((d**2).mean())), max_absolute_difference_m3_s=float(d.abs().max()),
        max_difference_date=biggest.DATE, equal_dates=int((d.abs() <= 1e-6).sum()),
        positive_dates=int((d > 1e-6).sum()), negative_dates=int((d < -1e-6).sum()))
    return result


def location_distance(stations):
    lon1, lat1 = stations[STATIONS[0]]['geometry']['coordinates']
    lon2, lat2 = stations[STATIONS[1]]['geometry']['coordinates']
    a = sin(radians(lat2-lat1)/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(radians(lon2-lon1)/2)**2
    return 2*6371*asin(sqrt(a))


def station_table(stations):
    records = []
    for name in STATIONS:
        feature = stations[name]; p = feature['properties']
        if p['STATION_NUMBER'] != name: raise ValueError('Station metadata identity mismatch')
        longitude, latitude = feature['geometry']['coordinates']
        records.append(dict(station=name, description=p['STATION_NAME'], latitude=latitude, longitude=longitude,
            contributor=p['CONTRIBUTOR_EN'], status=p['STATUS_EN'], datum=p['VERTICAL_DATUM'],
            gross_drainage_area_km2=p['DRAINAGE_AREA_GROSS'], published_statistic='Daily mean discharge',
            historical_clock='Not established by saved metadata',
            derivation_method='Not documented in consulted station metadata',
            complete_baskatong_outlet='Not verified'))
    return pd.DataFrame(records)


def figures(rows, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    colors = {'02LG007': '#217aa5', '02LG006': '#32906f'}
    def overlay(ax, data):
        dates = pd.to_datetime(data.DATE)
        for st in STATIONS:
            ax.plot(dates, data[f'q_{st}_m3_s'], label=st, color=colors[st], linewidth=1.3)
            flagged = data[f'flagged_{st}']
            if flagged.any():
                ax.scatter(dates[flagged], data.loc[flagged, f'q_{st}_m3_s'],
                           s=23, facecolors='none', edgecolors='#c56a22', zorder=4,
                           label=f'{st} source flag (Partial Day in this record)')
        ax.set_ylabel('Published daily mean discharge (m³/s)'); ax.legend(loc='upper right', fontsize=9)
    def differences(ax, data):
        dates = pd.to_datetime(data.DATE)
        ax.plot(dates, data.difference_007_minus_006_m3_s, color='#7953a2', linewidth=1.,
                label='02LG007 − 02LG006')
        for st, marker in [('02LG007', '^'), ('02LG006', 'o')]:
            flagged = data[f'flagged_{st}']
            if flagged.any():
                ax.scatter(dates[flagged], data.loc[flagged, 'difference_007_minus_006_m3_s'],
                           marker=marker, s=25, facecolors='none', edgecolors='#c56a22', zorder=4,
                           label=f'{st} source flag')
        ax.axhline(0, color='gray', linewidth=.7); ax.set_ylabel('Signed difference (m³/s)')
        ax.legend(loc='upper right', fontsize=9)
    def finish(fig, axes, title, path):
        for ax in axes:
            ax.grid(alpha=.2)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.suptitle(title)
        fig.supxlabel('Source calendar date; historical daily clock not established')
        fig.savefig(folder/path, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    overlay(ax, rows); finish(fig, [ax], '1970 Baskatong / Mercier discharge — original series kept separate', 'annual_overlay.png')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout='constrained', gridspec_kw={'height_ratios': [3, 1]})
    differences(axes[0], rows)
    axes[1].step(pd.to_datetime(rows.DATE), rows.flagged_02LG007.astype(int), where='post', label='02LG007 source flag', color=colors['02LG007'])
    axes[1].step(pd.to_datetime(rows.DATE), rows.flagged_02LG006.astype(int), where='post', label='02LG006 source flag', color='#c56a22')
    axes[1].set_ylim(-.1, 1.15); axes[1].set_ylabel('Flag present'); axes[1].legend(fontsize=9)
    finish(fig, axes, 'Daily differences: positive means 02LG007 reports more discharge', 'daily_differences.png')
    spring = rows[(rows.DATE >= '1970-04-01') & (rows.DATE < '1970-07-01')]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout='constrained')
    overlay(axes[0], spring); differences(axes[1], spring)
    finish(fig, axes, 'April–June 1970 — overlay and signed differences with source flags', 'april_june.png')


def report(summary, stations, evidence):
    all_data, unflagged, flagged, spring = summary['subsets']
    text = '''# 1970 Baskatong/Mercier discharge comparison

**Working decision:** use 02LG006 only as a clearly labelled **Mercier downstream-flow proxy**
for the observational pilot. Neither series is verified as complete Baskatong outlet
discharge for a closed reservoir balance. Keep 02LG007 as a separate alternative record,
not as a replacement, average, natural inflow or inferred transfer.

## Documented station descriptions and locations

| Station | Official description | Latitude, longitude | Contributor |
|---|---|---|---|
'''
    for row in stations.itertuples():
        text += f'| {row.station} | {row.description} | {row.latitude:.6f}, {row.longitude:.6f} | {row.contributor} |\n'
    text += f'''
The saved API lists the same gross drainage area, 15,710 km², and `ARBITRAIRE`
vertical datum for both stations. Both station records are discontinued; this does
not mean the reservoir or dam is closed. Coordinates are about **{summary['straight_line_distance_km']:.3f} km
apart in a straight line**, not a routing/travel distance. Modern archive metadata
do not establish that locations or methods were unchanged in 1970.

The official collection publishes **daily means**. Its metadata do not specify
station-level calculation/gauging methods, the historical daily clock, or an
all-outlets equivalence. The reservoir name for 02LG007 does not define its flow
boundary. 02LG006's name places it downstream of Mercier, which supports the proxy
interpretation but does not prove inclusion of Castor or every other outlet.

## Derived comparison, preserving all observations

- Paired dates: **{all_data['paired_dates']} / 365**, missing pairs: **{all_data['missing_pairs']}**.
- Exactly equal within 10⁻⁶ m³/s: **{all_data['equal_dates']}** dates; different: **{all_data['paired_dates']-all_data['equal_dates']}**.
- Difference is always defined as **02LG007 minus 02LG006**.
- Mean signed difference: **{all_data['mean_signed_difference_m3_s']:.3f} m³/s**;
  mean absolute difference: **{all_data['mean_absolute_difference_m3_s']:.3f} m³/s**;
  RMSE: **{all_data['root_mean_square_difference_m3_s']:.3f} m³/s**.
- Largest absolute difference: **{all_data['max_absolute_difference_m3_s']:.3f} m³/s**
  on **{all_data['max_difference_date']}**. A small annual signed mean masks cancellation;
  it does not establish daily agreement or equivalent measurement boundaries.
- April–June: **{spring['paired_dates']}** paired dates,
  **{spring['mean_absolute_difference_m3_s']:.3f} m³/s** mean absolute difference;
  maximum **{spring['max_absolute_difference_m3_s']:.3f} m³/s**.

02LG007 has no source discharge symbol on these dates. 02LG006 has **{flagged['paired_dates']}**
`Partial Day` values, **{summary['flagged_first_date']} through {summary['flagged_last_date']}**.
**All {flagged['paired_dates']} flagged-date pairs are equal.** All {all_data['paired_dates']-all_data['equal_dates']} nonzero
differences occur among the {unflagged['paired_dates']} unflagged pairs, whose mean absolute
difference is **{unflagged['mean_absolute_difference_m3_s']:.3f} m³/s**. Removing flagged days
therefore does not resolve the discrepancy. Lack of a flag does not establish
independence, accuracy or complete sampling.

## Official Partial Day definition

The current Wateroffice FAQ defines grade 30 Partial Day for a daily mean with
a missing stretch exceeding 120 consecutive minutes. Its daily-symbol discussion
describes means estimated despite gaps or missing information not warranting an
Estimated symbol. The API still reports a daily value; it is not a missing day,
zero flow, proof of a turbine shutdown, or a volume measured for a known number
of hours. Do not multiply it by an invented full-day correction factor.

We preserve the **current API label**. The original 1970 coding provenance,
within-day gaps and hourly values are not supplied by these responses; the
modern definition alone cannot reconstruct their precise history or uncertainty.

## Hypotheses and unresolved questions

Different averaging clocks, calculation methods, corrections/provenance, local
routing/inflow or additional outlets are **possibilities, not findings**. Neither
the sign nor size of the difference identifies a physical transfer. No averaging,
shifting, gap filling, replacement or inflow/storage reconstruction is performed.

The next specific evidence needed is the station record history/calculation sheet
for each gauge: what DISCHARGE represents in 1970, daily averaging/date conventions,
measured versus calculated flow, any shared underlying observations, all active
outlet coverage (including Castor), and the vintage symbol definitions. No owners
were contacted. Without that evidence, 02LG006 is a defensible descriptive proxy,
not a calibrated total-reservoir outlet record; 02LG007's outlet interpretation
remains unresolved.

## Files and reproduction

`daily_comparison.csv` retains both values and English/French symbols; `statistics.csv`
separates all, unflagged, flagged and April–June comparisons. `station_comparison.csv`
retains the location and definition comparison. PNGs provide the annual overlay,
daily differences and April–June focus. `provenance.json` verifies the original pilot
and raw hashes before/after the run. All original files remain unchanged.

```bash
uv run python -m reservoir_control.systems.ottawa.discharge_comparison \\
  --pilot results/ottawa/baskatong_1970_observed_v1 \\
  --output results/ottawa/baskatong_discharge_comparison_review
```

Source: Environment and Climate Change Canada, HYDAT / MSC GeoMet, and the
contributing agencies listed above. Review date: {evidence['review_date']}.

'''
    text += '\n'.join(f"- [{key}]({value['url']})" for key, value in evidence['sources'].items())+'\n'
    return text


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results/ottawa/baskatong_1970_observed_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh output directory')
    before = snapshot(args.pilot)
    manifest = json.loads((args.pilot/'raw/manifest.json').read_text())
    for name, details in manifest['files'].items():
        if digest(args.pilot/'raw'/name) != details['sha256']: raise ValueError('Source raw checksum mismatch')
    source_provenance = json.loads((args.pilot/'provenance.json').read_text())
    if any(digest(args.pilot/name) != h for name, h in source_provenance['output_sha256'].items()):
        raise ValueError('Original pilot output checksum mismatch')
    payloads = {s: json.loads((args.pilot/'raw'/f'{s}_daily.json').read_text()) for s in STATIONS}
    stations = {s: json.loads((args.pilot/'raw'/f'{s}_station.json').read_text())['features'][0] for s in STATIONS}
    rows = align(payloads)
    flagged = rows.any_source_flag
    summary = dict(subsets=[statistics(rows, 'all'), statistics(rows[~flagged], 'both_unflagged'),
        statistics(rows[flagged], 'any_flagged'), statistics(rows[(rows.DATE >= '1970-04-01') & (rows.DATE < '1970-07-01')], 'april_june')],
        straight_line_distance_km=location_distance(stations),
        flagged_first_date=rows.loc[flagged, 'DATE'].min(), flagged_last_date=rows.loc[flagged, 'DATE'].max(),
        difference_definition='02LG007 minus 02LG006, m3/s', no_series_average_or_transfer_inference=True)
    table = station_table(stations); evidence = json.loads(EVIDENCE.read_text())
    args.output.mkdir(parents=True)
    rows.to_csv(args.output/'daily_comparison.csv', index=False)
    table.to_csv(args.output/'station_comparison.csv', index=False)
    pd.DataFrame(summary['subsets']).to_csv(args.output/'statistics.csv', index=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (args.output/'source_review.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    figures(rows, args.output)
    (args.output/'report.md').write_text(report(summary, table, evidence))
    after = snapshot(args.pilot)
    if before != after: raise RuntimeError('Original pilot changed during comparison')
    provenance = dict(original_pilot_sha256=before, original_pilot_unchanged=True,
        implementation_sha256=digest(__file__), parser_sha256=digest(Path(__file__).with_name('historical_pilot.py')),
        evidence_sha256=digest(EVIDENCE), output_sha256=snapshot(args.output))
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Report: {args.output / "report.md"}')


if __name__ == '__main__': main()
