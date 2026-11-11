"""Acquire a bounded, separate modern ARPA measurement audit (no parameter fit).

Run with uv run python scripts/lake_como/operational_measurements.py.
Requires curl; preserves raw responses and provenance in ignored results/.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('results/simulator/modern_measurements_2021_v2'))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    sources = []

    def fetch(url, name, as_json=True):
        path = root / name
        subprocess.run(['curl', '-L', '--fail', '--retry', '2', '--max-time',
                        '60', '-sS', url, '-o', str(path)], check=True)
        sources.append(dict(url=url, file=name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return json.loads(path.read_text()) if as_json else path

    base = 'https://www.dati.lombardia.it/'
    quality_path = fetch(base + 'api/views/ujnt-2rtf/files/'
                         '6d78e4c2-6be0-4e0c-8a28-3ef22dd974b8?download=true',
                         'quality_codes.csv', as_json=False)
    quality = dict(line.split(';', 1) for line in
                   quality_path.read_text(encoding='utf-8-sig').splitlines() if line)
    registry = fetch(base + 'resource/nf78-nj6b.json?$limit=10000', 'registry.json')
    stations = [r for r in registry if r['idsensore'] in ('8386', '8383', '10805')]
    assert len(stations) == 3, 'Expected three uniquely identified sensors'
    (root / 'selected_stations.json').write_text(json.dumps(stations, indent=2)+'\n')
    reports = []
    frames = []
    for dataset, sensor in [('3e8b-w7ay', '8386'), ('3e8b-w7ay', '8383'),
                            ('ujnt-2rtf', '10805')]:
        fetch(base + f'api/views/{dataset}.json', f'{sensor}_metadata.json')
        rows = []
        for offset in (0, 50000):
            query = urlencode({'$where': f"idsensore='{sensor}' AND data >= '2021-01-01T00:00:00' AND data < '2022-01-01T00:00:00'",
                               '$order': 'data', '$limit': 50000, '$offset': offset})
            page = fetch(base + f'resource/{dataset}.json?' + query,
                         f'{sensor}_{offset}.json')
            rows.extend(page)
            if len(page) < 50000:
                break
        # At most 52,560 ten-minute observations in this fixed non-leap year.
        assert 0 < len(rows) <= 52560
        frame = pd.DataFrame(rows)
        # ARPA documents fixed solar time UTC+1 and end-of-measurement timestamps.
        frame['time'] = pd.to_datetime(frame['data']).dt.tz_localize('Etc/GMT-1')
        assert not frame['time'].duplicated().any()
        assert frame['time'].is_monotonic_increasing
        frame['value'] = pd.to_numeric(frame['valore'], errors='raise')
        assert np.isfinite(frame['value']).all()
        assert set(frame.stato).issubset(quality), 'Unknown ARPA quality code'
        # Preserve all raw values; processed/reconstructed and estimated data
        # are not interchangeable with good recorded observations (code 1).
        frame[['time', 'value', 'stato']].to_csv(root / f'{sensor}.csv', index=False)
        reports.append(dict(sensor=sensor, rows=len(frame), expected_grid_rows=52560,
                            missing_grid_rows=52560-len(frame),
                            first=str(frame.time.min()), last=str(frame.time.max()),
                            interval_counts_seconds=dict(Counter(
                                str(v) for v in frame.time.diff().dt.total_seconds().dropna())),
                            quality_codes={k: dict(count=v, description=quality[k])
                                           for k, v in Counter(frame.stato).items()},
                            min=float(frame.value.min()), max=float(frame.value.max())))
        indexed = frame.set_index('time')
        frames.append(pd.concat([indexed['value'].rename(sensor),
                                 indexed['stato'].rename(sensor+'_quality')], axis=1))
    aligned = pd.concat(frames, axis=1).dropna()
    assert len(aligned) > 0
    aligned = aligned.rename(columns={'8386': 'malgrate_level_cm',
                                      '8383': 'lavello_level_cm',
                                      '10805': 'lavello_discharge_m3_s'})
    aligned.to_csv(root / 'synchronized_observations.csv')
    recorded = aligned.loc[(aligned[['8386_quality', '8383_quality',
                                     '10805_quality']] == '1').all(axis=1)]
    recorded.to_csv(root / 'synchronized_code1_observations.csv')
    groups = recorded.groupby('lavello_level_cm')['lavello_discharge_m3_s'].nunique()
    report = dict(scope='Separate 2021 observational audit; no controller evaluation or calibration fit.',
                  timezone='Fixed UTC+1, end of measurement interval per ARPA documentation',
                  documentation_url=base+'stories/s/auv9-c2sj/',
                  sensors=reports, synchronized_rows=len(aligned),
                  synchronized_code1_rows=len(recorded),
                  code1_lavello_level_groups=len(groups),
                  code1_single_valued_discharge_groups=int((groups == 1).sum()),
                  operationally_calibrated=False,
                  unresolved=['Gauge-zero elevations and datum compatibility',
                              'Registry start dates conflict with earlier archive observations',
                              'Discharge measured versus rating-derived and rating validity',
                              'No gate commands or synchronized independent outlet flow',
                              'No verified diversions/returns or reach-specific thresholds',
                              'No independently measured irrigation demand'],
                  sources=sources,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(root), 'synchronized_rows': len(aligned),
                      'operationally_calibrated': False}))


if __name__ == '__main__':
    main()
