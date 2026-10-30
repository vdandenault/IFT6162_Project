import calendar
import csv
import hashlib
import json
import math
import subprocess

from lake_como.paths import PROJECT_ROOT, UPSTREAM_ROOT, DAM_DATA


def audit():
    errors, files, profiles = [], {}, {}
    for split in ('train', 'test'):
        profiles[split] = {}
        for signal in ('demand', 'inflow', 'level'):
            path = DAM_DATA / signal / f'water_{signal}_{split}.csv'
            with path.open() as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                rows = list(reader)
            if header != ['year'] + [str(i) for i in range(366)]:
                errors.append(f'{path.name}: unexpected columns')
            values, masks, years, counts = [], {}, [], {}
            for row in rows:
                year = int(row['year'])
                years.append(year)
                daily = [float(row[str(i)]) if row[str(i)] else float('nan') for i in range(366)]
                mask = [math.isfinite(v) for v in daily]
                expected = 366 if calendar.isleap(year) else 365
                if mask != [True] * expected + [False] * (366 - expected):
                    errors.append(f'{path.name}: {year} unexpected missing/nonfinite pattern')
                if any(math.isinf(v) for v in daily):
                    errors.append(f'{path.name}: {year} infinity')
                if signal != 'level' and any(v < 0 for v in daily if math.isfinite(v)):
                    errors.append(f'{path.name}: {year} negative flow')
                masks[year] = mask
                counts[str(year)] = sum(mask)
                values.extend(v for v in daily if math.isfinite(v))
            if len(set(years)) != len(years):
                errors.append(f'{path.name}: duplicate years')
            profiles[split][signal] = (years, masks)
            files[str(path.relative_to(UPSTREAM_ROOT))] = {
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'years_in_file_order': years, 'valid_days_by_year': counts,
                'min': min(values), 'max': max(values),
                'units': 'm (model datum)' if signal == 'level' else 'm3/s (simulator interpretation)',
            }
        reference = profiles[split]['inflow']
        for signal in ('demand', 'level'):
            if profiles[split][signal] != reference:
                errors.append(f'{split}: {signal} year order or missing-value masks differ from inflow')
    train = set(profiles['train']['inflow'][0])
    test = set(profiles['test']['inflow'][0])
    if train & test or train | test != set(range(1946, 2011)):
        errors.append('Incorrect split overlap or historical coverage')
    if len(train) != 52 or len(test) != 13:
        errors.append('Incorrect split sizes')
    mef = DAM_DATA / 'MEF.csv'
    with mef.open() as f:
        flows = [float(r['MEF']) for r in csv.DictReader(f)]
    if len(flows) != 365 or not all(math.isfinite(x) and x >= 0 for x in flows):
        errors.append('Invalid minimum environmental flow schedule')
    files[str(mef.relative_to(UPSTREAM_ROOT))] = {
        'sha256': hashlib.sha256(mef.read_bytes()).hexdigest(),
        'days': len(flows), 'min': min(flows), 'max': max(flows), 'units': 'm3/s',
    }
    return {
        'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source': 'https://github.com/vdandenault/gym4ReaL',
        'original_source': 'https://github.com/Daveonwave/gym4ReaL',
        'calendar_policy': 'Reproduce upstream: strip trailing NaN, simulate first 365 samples; leap-year final sample unused. Keep ordinal simulation days, do not claim Gregorian alignment.',
        'caveats': [
            'Train/test years are mixed across 1946-2010, not chronological.',
            'Original measurement agency, demand construction, and preprocessing remain unverified.',
            'Paper data license CC BY-NC 4.0 conflicts with metadata CC BY 4.0.',
            '365-day seasonal phase and MEF do not explicitly account for leap day.',
        ],
        'config_sha256': {str(p.relative_to(UPSTREAM_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((UPSTREAM_ROOT / 'gym4real/envs/dam').glob('*.yaml'))},
        'errors': errors, 'files': files,
    }


def main():
    report = audit()
    out = PROJECT_ROOT / 'reports' / 'data_audit.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(report["files"])} files audited; {len(report["errors"])} errors. Report: {out.relative_to(PROJECT_ROOT)}')
    for error in report['errors']:
        print(error)
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
