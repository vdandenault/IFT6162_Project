from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from reservoir_control.evaluation.provenance import digest
from reservoir_control.study_config import OTTAWA_EVIDENCE, load_settings
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_METADATA = OTTAWA_EVIDENCE
LEVEL_COLUMNS = ['timestamp', 'level', 'unit', 'aggregation', 'statistic', 'quality_flag', 'source_id']
FLOW_COLUMNS = ['interval_start', 'interval_end', 'mean_discharge', 'unit', 'aggregation',
                'statistic', 'quality_flag', 'source_id']


def convert(value, unit, quantity):
    factors = {'level': {'m': 1., 'cm': .01, 'ft': .3048},
               'flow': {'m3/s': 1., 'L/s': .001, 'ft3/s': .028316846592}}
    if quantity not in factors or unit not in factors[quantity]:
        raise ValueError(f'Unsupported {quantity} unit: {unit}')
    if pd.isna(value) or str(value).strip() == '': return np.nan
    number = float(value)
    return number*factors[quantity][unit] if np.isfinite(number) else np.nan


def instant(text):
    """Reject date-only/naive timestamps, including unsupported 24:00 labels."""
    value = pd.Timestamp(text)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError('Offset-aware ISO timestamp required; resolve date labels using owner metadata')
    return value.tz_convert('UTC')


def daily_boundaries(start, end, timezone):
    ZoneInfo(timezone)  # Validate the explicitly supplied clock, never infer from coordinates.
    bounds = pd.date_range(start, end, freq='D', tz=timezone)
    if len(bounds) < 2 or bounds[0].isoformat()[:10] != start or bounds[-1].isoformat()[:10] != end:
        raise ValueError('Ordered, date-only target endpoints required')
    return bounds.tz_convert('UTC')


def validate_contract(contract):
    if contract.get('schema_version') != 1: raise ValueError('Unsupported input contract')
    if not isinstance(contract.get('timezone'), str):
        raise ValueError('Declare the source daily timezone; do not infer it from reservoir location')
    ZoneInfo(contract['timezone'])
    for name in ('day_definition_evidence', 'level_datum', 'datum_alignment_evidence', 'outlet_definition'):
        if not isinstance(contract.get(name), str) or not contract[name].strip():
            raise ValueError(f'Explicit {name} required, including an unresolved description if necessary')
    if any(not isinstance(contract.get(k), bool) for k in ('datum_verified', 'total_outlet_verified', 'timing_verified')):
        raise ValueError('Explicit datum, outlet and timing verification booleans required')
    if not contract.get('usable_quality_flags'): raise ValueError('Declare usable owner quality flags')
    sources = contract.get('sources', {})
    if not sources: raise ValueError('Input source provenance required')
    for source in sources.values():
        if (source.get('local_processing_and_copy_permitted') is not True
                or not all(source.get(k) for k in ('owner', 'url', 'terms_url', 'permission_basis'))):
            raise ValueError('Verified permission and source/terms provenance required before copying records')
    return contract


@dataclass
class StorageMapping:
    minimum_m: float
    maximum_m: float
    usable_volume_m3: float | None = None
    curve: pd.DataFrame | None = None

    def __post_init__(self):
        if not np.isfinite([self.minimum_m, self.maximum_m]).all() or self.minimum_m >= self.maximum_m:
            raise ValueError('Ordered finite storage-reference elevations required')
        if self.curve is not None and self.usable_volume_m3 is not None:
            raise ValueError('Curve and assumed linear usable volume are separate modes')
        if self.usable_volume_m3 is not None and (not np.isfinite(self.usable_volume_m3) or self.usable_volume_m3 <= 0):
            raise ValueError('An explicit positive scenario K is required')
        if self.curve is not None:
            if not {'elevation_m', 'volume_m3'} <= set(self.curve):
                raise ValueError('Curve requires elevation_m and volume_m3')
            h, v = self.curve[['elevation_m', 'volume_m3']].to_numpy(float).T
            if (len(h) < 2 or not np.isfinite([h, v]).all() or (v < 0).any()
                    or (np.diff(h) <= 0).any() or (np.diff(v) <= 0).any()
                    or h[0] > self.minimum_m or h[-1] < self.maximum_m):
                raise ValueError('Strictly increasing finite curve must cover both operating endpoints')

    @property
    def mode(self):
        return 'curve_based' if self.curve is not None else ('approximate' if self.usable_volume_m3 is not None else 'normalized')

    def values(self, level):
        if not np.isfinite(level): return np.nan, np.nan
        fraction = (level-self.minimum_m)/(self.maximum_m-self.minimum_m)
        if self.curve is not None:
            h, v = self.curve[['elevation_m', 'volume_m3']].to_numpy(float).T
            if not h[0] <= level <= h[-1]: return fraction, np.nan  # no extrapolation
            return fraction, float(np.interp(level, h, v)-np.interp(self.minimum_m, h, v))
        return fraction, fraction*self.usable_volume_m3 if self.usable_volume_m3 is not None else np.nan


def _schema(frame, required, contract):
    if not set(required) <= set(frame): raise ValueError(f'Required columns: {required}')
    if frame[['source_id', 'quality_flag', 'aggregation', 'statistic', 'unit']].isna().any().any():
        raise ValueError('Every record requires source, quality, aggregation, statistic and units')
    unknown = set(frame.source_id)-set(contract['sources'])
    if unknown: raise ValueError(f'Unregistered sources: {sorted(unknown)}')


def _boundary_level(observed, side, duplicate, storage, datum_verified, good_quality):
    value = observed['_value'] if observed is not None else np.nan
    finite = np.isfinite(value)
    aggregation = observed.aggregation if observed is not None else ''
    boundary = aggregation == 'instantaneous_boundary'
    accepted_quality = observed is not None and observed.quality_flag in good_quality

    flags = []
    if not finite:
        flags.append(f'missing_level_{side}')
    if duplicate:
        flags.append(f'duplicate_level_{side}')
    if observed is not None:
        if not boundary:
            flags.append(f'not_boundary_level_{side}')
        if not accepted_quality:
            flags.append(f'level_quality_{side}')
        if finite and not storage.minimum_m <= value <= storage.maximum_m:
            flags.append(f'level_out_of_range_{side}')

    valid = finite and boundary and accepted_quality and not duplicate
    position, volume = np.nan, np.nan
    if finite and accepted_quality and datum_verified:
        position, volume = storage.values(value)
    return {
        f'reported_level_{side}_m': value,
        f'level_{side}_m': value if boundary else np.nan,
        f'level_{side}_source_id': observed.source_id if observed is not None else '',
        f'level_{side}_source_quality': observed.quality_flag if observed is not None else '',
        f'water_level_position_{side}': position,
        f'storage_{side}_m3': volume if valid else np.nan,
        f'level_{side}_aggregation': aggregation,
    }, valid, flags


def _balance_segments(frame):
    valid = frame.reconstruction_available
    starts = frame.index[valid & ~valid.shift(fill_value=False)]
    ends = frame.index[valid & ~valid.shift(-1, fill_value=False)]
    checks = []
    for start, end in zip(starts, ends, strict=True):
        block = frame.loc[start:end]
        delta = float(block.storage_end_m3.iloc[-1] - block.storage_start_m3.iloc[0])
        inflow = float((block.net_inflow_m3_s * block.duration_seconds).sum())
        outlet = float((block.mean_outlet_discharge_m3_s * block.duration_seconds).sum())
        checks.append(dict(first_interval=int(start), intervals=len(block), storage_change_m3=delta,
                           net_inflow_volume_m3=inflow, outlet_volume_m3=outlet,
                           closure_error_m3=delta - (inflow - outlet)))
    return checks


def prepare(levels, flows, metadata, contract, storage, *, jump_m_per_day=.75,
            maximum_inflow_m3_s=None):
    """Align exact daily boundaries. Missing, duplicated and misaligned data stay missing.

    Returned estimates are accounting-derived; quality warnings are retained and
    never repaired by interpolation, clipping or replacing missing values with zero.
    """
    validate_contract(contract)
    if not np.isfinite(jump_m_per_day) or jump_m_per_day <= 0:
        raise ValueError('Positive diagnostic jump threshold required')
    if maximum_inflow_m3_s is not None and (not np.isfinite(maximum_inflow_m3_s) or maximum_inflow_m3_s <= 0):
        raise ValueError('Positive explicit inflow plausibility threshold required')
    _schema(levels, LEVEL_COLUMNS, contract)
    _schema(flows, FLOW_COLUMNS, contract)
    non_observed_levels = levels[levels.statistic != 'observed'].copy()
    non_observed_flows = flows[flows.statistic != 'observed'].copy()
    levels = levels[levels.statistic == 'observed'].copy()
    flows = flows[flows.statistic == 'observed'].copy()
    levels['_time'] = [instant(x) for x in levels.timestamp]
    levels['_value'] = [convert(v, u, 'level') for v, u in zip(levels.level, levels.unit)]
    flows['_start'] = [instant(x) for x in flows.interval_start]
    flows['_end'] = [instant(x) for x in flows.interval_end]
    flows['_value'] = [convert(v, u, 'flow') for v, u in zip(flows.mean_discharge, flows.unit)]
    if any(a >= b for a, b in zip(flows['_start'], flows['_end'])):
        raise ValueError('Discharge interval end must follow start')
    target = metadata['target']
    boundaries = daily_boundaries(target['start_date'], target['end_date_exclusive'], contract['timezone'])
    pairs = list(zip(boundaries[:-1], boundaries[1:]))
    grid = set(pairs)
    duplicate_l = set(levels.loc[levels.duplicated('_time', keep=False), '_time'])
    duplicate_flows = flows.loc[flows.duplicated(['_start', '_end'], keep=False)]
    duplicate_q = set(zip(duplicate_flows['_start'], duplicate_flows['_end']))
    outside_l = levels[~levels['_time'].isin(boundaries)]
    misaligned = flows.loc[np.asarray([p not in grid for p in zip(flows['_start'], flows['_end'])], dtype=bool)]
    lmap = {r['_time']: r for _, r in levels.iterrows() if r['_time'] not in duplicate_l}
    qmap = {(r['_start'], r['_end']): r for _, r in flows.iterrows()
            if (r['_start'], r['_end']) in grid and (r['_start'], r['_end']) not in duplicate_q}
    good_quality = set(contract['usable_quality_flags'])
    definition_flags = [
        flag
        for key, flag in (
            ('datum_verified', 'datum_unverified'),
            ('timing_verified', 'timing_unverified'),
            ('total_outlet_verified', 'outlet_boundary_unverified'),
        )
        if not contract[key]
    ]
    records = []
    for start, end in pairs:
        flags = definition_flags.copy()
        row = dict(interval_start=start.isoformat(), interval_end=end.isoformat(),
                   local_date=start.tz_convert(contract['timezone']).date().isoformat(),
                   duration_seconds=(end-start).total_seconds(), storage_mode=storage.mode)
        valid_levels = True
        for side, time in (('start', start), ('end', end)):
            fields, valid, level_flags = _boundary_level(
                lmap.get(time), side, time in duplicate_l, storage, contract['datum_verified'], good_quality
            )
            row.update(fields)
            flags.extend(level_flags)
            valid_levels &= valid
        observed = qmap.get((start, end))
        q = observed['_value'] if observed is not None else np.nan
        row['mean_outlet_discharge_m3_s'] = q
        row['discharge_source_id'] = '' if observed is None else observed.source_id
        row['discharge_source_quality'] = '' if observed is None else observed.quality_flag
        valid_q = observed is not None and np.isfinite(q)
        if not valid_q:
            flags.append('missing_discharge')
        if (start, end) in duplicate_q:
            flags.append('duplicate_discharge')
            valid_q = False
        if observed is not None:
            if observed.aggregation != 'interval_mean':
                flags.append('not_mean_discharge')
                valid_q = False
            if observed.quality_flag not in good_quality:
                flags.append('discharge_quality')
                valid_q = False
        if q < 0:
            flags.append('negative_outlet_discharge')
            valid_q = False
        if any(a < end and b > start for a, b in zip(misaligned['_start'], misaligned['_end'])):
            flags.append('misaligned_or_overlapping_discharge')
            valid_q = False
        if valid_levels:
            jump = abs(row['level_end_m']-row['level_start_m'])*86400/row['duration_seconds']
            if jump > jump_m_per_day: flags.append('suspect_level_jump')
        volumes = np.asarray([row['storage_start_m3'], row['storage_end_m3']])
        if storage.mode == 'normalized': flags.append('usable_volume_unestablished')
        elif valid_levels and contract['datum_verified'] and not np.isfinite(volumes).all():
            flags.append('storage_curve_domain')
        can_compute = valid_levels and valid_q and not definition_flags and np.isfinite(volumes).all()
        net = float((volumes[1]-volumes[0])/row['duration_seconds']+q) if can_compute else np.nan
        if np.isfinite(net) and net < 0: flags.append('negative_net_inflow')
        if np.isfinite(net) and maximum_inflow_m3_s is not None and net > maximum_inflow_m3_s:
            flags.append('net_inflow_above_scenario_threshold')
        row['net_inflow_m3_s'] = net
        row['reconstruction_available'] = can_compute
        row['quality_flags'] = ';'.join(sorted(set(flags)))
        records.append(row)
    result = pd.DataFrame(records)
    counts = Counter(flag for value in result.quality_flags for flag in value.split(';') if flag)
    summary = dict(expected_intervals=len(pairs), expected_boundary_levels=len(boundaries),
        observed_level_rows=len(levels), observed_discharge_rows=len(flows),
        non_observed_level_rows=len(non_observed_levels), non_observed_discharge_rows=len(non_observed_flows),
        duplicate_level_rows=int(levels.duplicated('_time', keep=False).sum()),
        duplicate_discharge_rows=int(flows.duplicated(['_start', '_end'], keep=False).sum()),
        off_grid_level_rows=len(outside_l), misaligned_discharge_rows=len(misaligned),
        reconstruction_intervals=int(result.reconstruction_available.sum()), flag_counts=dict(counts),
        storage_mode=storage.mode, assumed_usable_volume_m3=storage.usable_volume_m3,
        accounting_segments=_balance_segments(result), independent_validation=False,
        timing='Explicit local daily clock converted to UTC; actual elapsed seconds, including DST where declared')
    return result, summary, non_observed_levels, non_observed_flows


def main():
    parser = ArgumentParser(description="Reconstruct a daily reservoir balance from local observations.")
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', type=Path)
    parser.add_argument('--outflows', type=Path)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--storage-curve', type=Path)
    parser.add_argument('--usable-volume-m3', type=float)
    parser.add_argument('--scenario-id')
    parser.add_argument('--volume-rationale')
    parser.add_argument('--jump-m-per-day', type=float, default=.75)
    parser.add_argument('--maximum-inflow-m3-s', type=float, help='Explicit diagnostic threshold, not an inferred physical bound')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    supplied = [args.levels is not None, args.outflows is not None, args.contract is not None]
    if any(supplied) and not all(supplied):
        parser.error('Supply levels, outflows and contract together')
    if not any(supplied) and (args.storage_curve or args.usable_volume_m3 is not None):
        parser.error('Storage scenarios require actual input files and a contract')
    metadata = load_settings(args.metadata, "baskatong_metadata")
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported reservoir metadata')
    inputs = {'metadata': args.metadata}
    frame = None
    if all(supplied):
        contract = validate_contract(json.loads(args.contract.read_text()))
        inputs.update(levels=args.levels, outflows=args.outflows, contract=args.contract)
        curve = None
        if args.storage_curve:
            proof = contract.get('storage_curve', {})
            if (proof.get('source_id') not in contract['sources']
                    or proof.get('datum') != contract['level_datum']
                    or not proof.get('provenance') or not contract['datum_verified']):
                raise ValueError('Sourced curve requires registered permission, provenance and the aligned vertical datum')
            curve = pd.read_csv(args.storage_curve)
            inputs['storage_curve'] = args.storage_curve
        if args.usable_volume_m3 is not None and not (args.scenario_id and args.volume_rationale):
            parser.error('A linear K requires --scenario-id and --volume-rationale; inventory capacity is not a default')
        bounds = metadata.get('level_position_reference', metadata['operating_bounds'])
        storage = StorageMapping(bounds['minimum_m'], bounds['maximum_m'], args.usable_volume_m3, curve)
        levels = pd.read_csv(args.levels, keep_default_na=False)
        outflows = pd.read_csv(args.outflows, keep_default_na=False)
        frame, summary, non_observed_levels, non_observed_flows = prepare(
            levels, outflows, metadata, contract, storage,
            jump_m_per_day=args.jump_m_per_day, maximum_inflow_m3_s=args.maximum_inflow_m3_s,
        )
        summary.update(status='processed_inputs', scenario_id=args.scenario_id,
                       volume_rationale=args.volume_rationale, timezone=contract['timezone'],
                       jump_m_per_day=args.jump_m_per_day, maximum_inflow_m3_s=args.maximum_inflow_m3_s)
    else:
        summary = dict(status='assessment_only', historical_reconstruction_completed=False,
                       local_observations_ingested=0, independent_validation=False,
                       expected_intervals=metadata['target']['expected_daily_intervals'],
                       expected_boundary_levels=metadata['target']['expected_boundary_levels'])
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metadata, args.output/'reservoir_metadata.json')
    pd.DataFrame(metadata['sources']).to_csv(args.output/'source_coverage.csv', index=False)
    if frame is not None:
        raw = args.output / 'raw'
        raw.mkdir()
        for name, path in inputs.items():
            if name != 'metadata':
                shutil.copy2(path, raw / f'{name}{path.suffix}')
        folder = args.output / summary['storage_mode']
        folder.mkdir()
        frame.to_csv(folder/'daily.csv', index=False)
        non_observed_levels.to_csv(folder/'non_observed_levels.csv', index=False)
        non_observed_flows.to_csv(folder/'non_observed_outflows.csv', index=False)
    else:
        pd.DataFrame(columns=LEVEL_COLUMNS).to_csv(args.output/'levels_template.csv', index=False)
        pd.DataFrame(columns=FLOW_COLUMNS).to_csv(args.output/'outflows_template.csv', index=False)
    summary['provenance'] = dict(inputs_sha256={str(p): digest(p) for p in inputs.values()},
                                 implementation_sha256=digest(__file__),
                                 no_network_downloads=True, observed_and_climatology_separated=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
