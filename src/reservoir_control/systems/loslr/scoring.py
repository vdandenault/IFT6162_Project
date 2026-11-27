import hashlib
import json
import logging
import sys
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.study_config import ONTARIO_CONFIG, load_settings

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ONTARIO_CONFIG


@dataclass(frozen=True)
class ScoreSettings:
    ontario_cap_m: float = 75.5
    pointe_claire_cap_m: float = 22.1
    ontario_weight: float = 1.
    pointe_claire_weight: float = 1.
    ontario_scale_m: float = 1.
    pointe_claire_scale_m: float = 1.

    def __post_init__(self):
        if not all(isfinite(value) for value in asdict(self).values()):
            raise ValueError('Finite scoring settings are required')
        if min(self.ontario_weight, self.pointe_claire_weight) < 0:
            raise ValueError('Scoring weights must be nonnegative')
        if self.ontario_weight + self.pointe_claire_weight == 0:
            raise ValueError('At least one positive scoring weight is required')
        if min(self.ontario_scale_m, self.pointe_claire_scale_m) <= 0:
            raise ValueError('Positive level scales are required')


def period_cost(ontario_level_m, pointe_claire_level_m, settings=ScoreSettings()):
    """Scalar or aligned-array cost; period reward is its negative."""
    ontario = np.asarray(ontario_level_m, dtype=float)
    downstream = np.asarray(pointe_claire_level_m, dtype=float)
    if ontario.shape != downstream.shape or not np.isfinite(ontario).all() or not np.isfinite(downstream).all():
        raise ValueError('Aligned finite Ontario and Pointe-Claire levels are required')
    upstream_excess = np.maximum(ontario - settings.ontario_cap_m, 0)
    downstream_excess = np.maximum(downstream - settings.pointe_claire_cap_m, 0)
    return (settings.ontario_weight * upstream_excess / settings.ontario_scale_m
            + settings.pointe_claire_weight * downstream_excess / settings.pointe_claire_scale_m)


def score_trajectories(frame, settings=ScoreSettings()):
    """Score matched complete 48-period years, retaining each physical component."""
    required = ['controller', 'year', 'quarter_month', 'ontario_end_m', 'pointe_claire_m', 'release_m3_s']
    if any(name not in frame for name in required):
        raise ValueError(f'Trajectories require columns {required}')
    if frame[required].isna().any().any():
        raise ValueError('Scoring requires complete finite trajectories')
    if not np.isfinite(frame[required[1:]].to_numpy(float)).all():
        raise ValueError('Scoring requires finite trajectory values')
    records, coverage = [], {}
    for (controller, year), rows in frame.groupby(['controller', 'year'], sort=True):
        rows = rows.sort_values('quarter_month')
        if rows.quarter_month.tolist() != list(range(1, 49)) or year != int(year):
            raise ValueError(f'{controller}, {year} requires exactly 48 distinct ordered quarter-months')
        coverage.setdefault(controller, set()).add(int(year))
        upstream = (rows.ontario_end_m - settings.ontario_cap_m).clip(lower=0)
        downstream = (rows.pointe_claire_m - settings.pointe_claire_cap_m).clip(lower=0)
        cost = float(period_cost(rows.ontario_end_m, rows.pointe_claire_m, settings).sum())
        records.append(dict(
            controller=controller, year=int(year), cost=cost, score=-cost,
            ontario_excess_m_periods=float(upstream.sum()),
            pointe_claire_excess_m_periods=float(downstream.sum()),
            ontario_periods_above=int((upstream > 1e-9).sum()),
            pointe_claire_periods_above=int((downstream > 1e-9).sum()),
            ontario_peak_m=float(rows.ontario_end_m.max()),
            pointe_claire_peak_m=float(rows.pointe_claire_m.max()),
            ontario_end_m=float(rows.ontario_end_m.iloc[-1]),
            mean_release_m3_s=float(rows.release_m3_s.mean()),
        ))
    if not records:
        raise ValueError('At least one complete trajectory is required')
    if any(years != next(iter(coverage.values())) for years in coverage.values()):
        raise ValueError('Controllers must cover the same years for a matched comparison')
    return pd.DataFrame(records)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Apply one declared physical score to Ontario controller trajectories."
    )
    parser.add_argument('--input', type=Path, action='append', help='Repeat for additional controller CSVs')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh scoring output directory')
    raw = load_settings(args.config, "scoring")
    if raw.get('schema_version') != 1:
        raise ValueError('Unsupported scoring configuration version')
    settings = ScoreSettings(**{name: raw[name] for name in ScoreSettings.__dataclass_fields__})
    paths = args.input or [ROOT / 'results/loslr/mpc_v2/comparison.csv']
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        identity = next((name for name in ('controller', 'arm', 'policy') if name in frame), None)
        if identity is None:
            raise ValueError(f'{path} needs a controller, arm or policy column')
        frame['controller'] = frame[identity]
        frames.append(frame)
    annual = score_trajectories(pd.concat(frames, ignore_index=True), settings)
    means = annual.drop(columns='year').groupby('controller', as_index=False).mean(numeric_only=True)
    means['years'] = annual.groupby('controller').size().reindex(means.controller).to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    annual.to_csv(args.output / 'annual.csv', index=False)
    means.to_csv(args.output / 'comparison.csv', index=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(
        settings=asdict(settings), interpretation='Higher score is better; score equals negative total period cost',
        scope='Normalized stage exceedance on experimental caps; other operating objectives remain separate metrics',
        mean_annual=means.to_dict('records'),
        inputs_sha256={str(path.resolve()): digest(path) for path in paths},
        config_sha256=digest(args.config), scorer_sha256=digest(Path(__file__)),
        outputs_sha256={path.name: digest(path) for path in args.output.glob('*.csv')},
    )
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    annual.to_csv(sys.stdout, index=False)
    LOG.info("Saved shared scores to %s", args.output)


if __name__ == '__main__':
    main()
