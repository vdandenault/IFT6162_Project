"""Versioned year membership for controller training and held-out evaluation."""
import calendar
import json
from reservoir_control.systems.lake_como.paths import PROJECT_ROOT

SPLIT_MANIFEST = PROJECT_ROOT / 'configs' / 'splits.json'


def split_years(split, *, include_leap_years=True):
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    groups = manifest['splits']
    if split not in groups:
        raise ValueError(f'Unknown split {split!r}; choose from {list(groups)}')
    all_years = [y for years in groups.values() for y in years]
    if len(all_years) != len(set(all_years)) or set(all_years) != set(range(1946, 2011)):
        raise ValueError('Split manifest must cover 1946–2010 exactly once')
    return [y for y in groups[split] if include_leap_years or not calendar.isleap(y)]
