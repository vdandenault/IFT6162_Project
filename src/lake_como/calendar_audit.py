"""Audit pinned ordinal profiles without silently assigning Gregorian dates.

File integrity/structure only, including held-out source rows; no controller
execution, forecast fitting, or test performance assessment is performed.
"""
import argparse
import calendar
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lake_como.audit import audit
from lake_como.mpc_hindsight import digest
from lake_como.paths import DAM_DATA, PROJECT_ROOT
from lake_como.splits import split_years


def describe_year(year, values):
    """Validate the raw 366-column layout and identify unused model samples."""
    values = np.asarray(values, dtype=float)
    expected = 366 if calendar.isleap(year) else 365
    if values.shape != (366,):
        raise ValueError('Expected raw ordinal columns 0 through 365')
    if not np.array_equal(np.isfinite(values), np.arange(366) < expected):
        raise ValueError(f'{year}: gaps or nonfinite samples inconsistent with the year length')
    return dict(year=int(year), leap_year=calendar.isleap(year), source_samples=expected,
        consumed_model_samples=365, unused_source_columns=[365] if expected == 366 else [],
        unused_last_value=float(values[365]) if expected == 366 else None,
        date_assumption='If ordinal column 0 is January 1 in a Gregorian year',
        assumed_last_consumed_date=str(date(year, 1, 1) + timedelta(days=364)),
        assumed_omitted_date=str(date(year, 1, 1) + timedelta(days=365)) if expected == 366 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROJECT_ROOT/'results/simulator/calendar_audit_v1')
    root = p.parse_args().output
    if root.exists() and any(root.iterdir()):
        p.error('Choose an empty output directory; prior audits are preserved')
    root.mkdir(parents=True, exist_ok=True)
    original = audit()
    if original['errors']:
        raise ValueError(original['errors'])
    metadata_path = DAM_DATA/'metadata_dam.json'
    metadata = json.loads(metadata_path.read_text())
    distributions = {}
    for item in metadata['distribution']:
        relative = item['contentUrl'].split('/dam/', 1)[1]
        path = DAM_DATA/relative
        distributions[relative] = dict(actual_sha256=digest(path), metadata_sha256=item['sha256'])
    if not all(v['actual_sha256'] == v['metadata_sha256'] for v in distributions.values()):
        raise ValueError('Pinned file differs from metadata checksum')
    records = []
    unique = {}
    headers = ['year'] + [str(i) for i in range(366)]
    for signal in ['inflow', 'demand', 'level']:
        frames = []
        for part in ['train', 'test']:
            frame = pd.read_csv(DAM_DATA/signal/f'water_{signal}_{part}.csv')
            if frame.columns.tolist() != headers:
                raise ValueError('Unexpected raw ordinal header')
            for row in frame.itertuples(index=False, name=None):
                record = describe_year(int(row[0]), row[1:])
                record.update(signal=signal, upstream_file_split=part,
                    chronological_split=next(s for s in ['train','validation','test'] if int(row[0]) in split_years(s)))
                records.append(record)
            frames.append(frame.set_index('year'))
        combined = pd.concat(frames).sort_index()
        if not combined.index.is_unique:
            raise ValueError('Duplicate source year')
        unique[signal] = len(combined.iloc[:, :365].drop_duplicates())
    frame = pd.DataFrame(records)
    frame.to_csv(root/'source_years.csv', index=False)
    year_summary = frame.query('signal=="inflow"').groupby('chronological_split').agg(
        years=('year','count'), leap_years=('leap_year','sum'), source_samples=('source_samples','sum'),
        simulated_samples=('consumed_model_samples','sum')).reset_index()
    year_summary['unused_samples'] = year_summary.source_samples - year_summary.simulated_samples
    year_summary.to_csv(root/'split_calendar.csv', index=False)
    report = dict(complete=True, source_integrity_errors=original['errors'],
        checksum_matches=len(distributions), distributions=distributions,
        metadata_sha256=digest(metadata_path), metadata_description=metadata.get('description'),
        confirmed=dict(years=65, leap_years=int(frame.query('signal=="inflow"').leap_year.sum()),
            raw_columns='year plus ordinal columns 0–365; 365/366 finite samples match Gregorian year lengths',
            unique_first_365_profiles=unique, minimum_environmental_flow_samples=365,
            implementation='DamEnv and the continuity adapter consume ordinal columns 0–364',
            date_metadata='Fields say day 0, day 1, etc.; no explicit dates, gauge datum, agency, or demand-construction method'),
        inference='The 365/366 pattern supports a Gregorian interpretation but does not prove the January-1 anchor. If that anchor holds, leap day is retained, December 31 omitted, and the 365-day seasonal/MEF schedules shift after February.',
        decision='Keep the current 365-sample benchmark and continuity protocol unchanged and labeled as model years. Do not describe it as dropping February 29. A future real-calendar version must explicitly map source dates and define 366-step seasonal/MEF treatment.',
        unresolved=['Original timestamp/date anchor and preprocessing', 'Level measurement datum/units confirmation',
                    'Whether demand is measured consumption, requested supply or a reconstructed target',
                    'Operational provenance of the 365-row environmental-flow schedule'],
        scope='Raw-file integrity/calendar audit across source rows only; no held-out controller evaluation or parameter fitting',
        test_evaluated=False, environment_modified=False)
    (root/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(year_summary.to_string(index=False))
    print(json.dumps({k: report[k] for k in ['checksum_matches','confirmed','decision']}, indent=2))


if __name__ == '__main__':
    main()
