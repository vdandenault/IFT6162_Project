"""Daily, observation-based reservoir balance; no controller or training dependency.

No downloader is provided for the restricted ORRPB observation service. Inputs
must have a documented processing licence and explicit measurement conventions.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_METADATA = ROOT/'configs/studies/ottawa/baskatong.json'
LEVEL_COLUMNS = ['timestamp', 'level', 'unit', 'aggregation', 'statistic', 'quality_flag', 'source_id']
FLOW_COLUMNS = ['interval_start', 'interval_end', 'mean_discharge', 'unit', 'aggregation',
                'statistic', 'quality_flag', 'source_id']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    _schema(levels, LEVEL_COLUMNS, contract); _schema(flows, FLOW_COLUMNS, contract)
    levels, flows = levels.copy(), flows.copy()
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
    duplicate_q = set(zip(*[flows.loc[flows.duplicated(['_start', '_end'], keep=False), c]
                            for c in ('_start', '_end')]))
    outside_l = levels[~levels['_time'].isin(boundaries)]
    misaligned = flows.loc[np.asarray([p not in grid for p in zip(flows['_start'], flows['_end'])], dtype=bool)]
    lmap = {r['_time']: r for _, r in levels.iterrows() if r['_time'] not in duplicate_l}
    qmap = {(r['_start'], r['_end']): r for _, r in flows.iterrows()
            if (r['_start'], r['_end']) in grid and (r['_start'], r['_end']) not in duplicate_q}
    good_quality = set(contract['usable_quality_flags'])
    records = []
    for start, end in pairs:
        flags = []
        row = dict(interval_start=start.isoformat(), interval_end=end.isoformat(),
                   local_date=start.tz_convert(contract['timezone']).date().isoformat(),
                   duration_seconds=(end-start).total_seconds(), storage_mode=storage.mode)
        valid_levels = True
        for side, time in (('start', start), ('end', end)):
            observed = lmap.get(time)
            value = observed['_value'] if observed is not None else np.nan
            row[f'reported_level_{side}_m'] = value
            row[f'level_{side}_m'] = (value if observed is not None and observed.aggregation == 'instantaneous_boundary' else np.nan)
            row[f'level_{side}_source_id'] = '' if observed is None else observed.source_id
            row[f'level_{side}_source_quality'] = '' if observed is None else observed.quality_flag
            valid = observed is not None and np.isfinite(value)
            if not valid: flags.append(f'missing_level_{side}')
            if time in duplicate_l: flags.append(f'duplicate_level_{side}'); valid = False
            if observed is not None:
                if observed.aggregation != 'instantaneous_boundary':
                    flags.append(f'not_boundary_level_{side}'); valid = False
                if observed.quality_flag not in good_quality:
                    flags.append(f'level_quality_{side}'); valid = False
                if np.isfinite(value) and not storage.minimum_m <= value <= storage.maximum_m:
                    flags.append(f'level_out_of_range_{side}')
            fraction, volume = storage.values(value) if valid and contract['datum_verified'] else (np.nan, np.nan)
            # A level position is not a storage fraction. Daily means can support
            # a descriptive position, while remaining ineligible for a balance.
            position = (storage.values(value)[0] if observed is not None and np.isfinite(value)
                        and observed.quality_flag in good_quality and contract['datum_verified'] else np.nan)
            row[f'water_level_position_{side}'], row[f'storage_{side}_m3'] = position, volume
            row[f'level_{side}_aggregation'] = '' if observed is None else observed.aggregation
            valid_levels &= valid
        observed = qmap.get((start, end))
        q = observed['_value'] if observed is not None else np.nan
        row['mean_outlet_discharge_m3_s'] = q
        row['discharge_source_id'] = '' if observed is None else observed.source_id
        row['discharge_source_quality'] = '' if observed is None else observed.quality_flag
        valid_q = observed is not None and np.isfinite(q)
        if not valid_q: flags.append('missing_discharge')
        if (start, end) in duplicate_q: flags.append('duplicate_discharge'); valid_q = False
        if observed is not None:
            if observed.aggregation != 'interval_mean': flags.append('not_mean_discharge'); valid_q = False
            if observed.quality_flag not in good_quality: flags.append('discharge_quality'); valid_q = False
        if np.isfinite(q) and q < 0: flags.append('negative_outlet_discharge'); valid_q = False
        if any(a < end and b > start for a, b in zip(misaligned['_start'], misaligned['_end'])):
            flags.append('misaligned_or_overlapping_discharge'); valid_q = False
        if not contract['datum_verified']: flags.append('datum_unverified')
        if not contract['timing_verified']: flags.append('timing_unverified')
        if not contract['total_outlet_verified']: flags.append('outlet_boundary_unverified')
        if valid_levels:
            jump = abs(row['level_end_m']-row['level_start_m'])*86400/row['duration_seconds']
            if jump > jump_m_per_day: flags.append('suspect_level_jump')
        volumes = np.asarray([row['storage_start_m3'], row['storage_end_m3']])
        if storage.mode == 'normalized': flags.append('usable_volume_unestablished')
        elif valid_levels and contract['datum_verified'] and not np.isfinite(volumes).all():
            flags.append('storage_curve_domain')
        can_compute = valid_levels and valid_q and contract['timing_verified'] and contract['datum_verified'] and contract['total_outlet_verified'] and np.isfinite(volumes).all()
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
    # Telescoping check only within contiguous valid segments; never bridge a gap.
    segments, current = [], []
    for index, valid in enumerate(result.reconstruction_available):
        if valid: current.append(index)
        elif current: segments.append(current); current = []
    if current: segments.append(current)
    checks = []
    for indices in segments:
        block = result.iloc[indices]
        delta = float(block.storage_end_m3.iloc[-1]-block.storage_start_m3.iloc[0])
        inflow_volume = float((block.net_inflow_m3_s*block.duration_seconds).sum())
        outlet_volume = float((block.mean_outlet_discharge_m3_s*block.duration_seconds).sum())
        checks.append(dict(first_interval=int(indices[0]), intervals=len(indices), storage_change_m3=delta,
                           net_inflow_volume_m3=inflow_volume, outlet_volume_m3=outlet_volume,
                           closure_error_m3=delta-(inflow_volume-outlet_volume)))
    summary = dict(expected_intervals=len(pairs), expected_boundary_levels=len(boundaries),
        observed_level_rows=len(levels), observed_discharge_rows=len(flows),
        non_observed_level_rows=len(non_observed_levels), non_observed_discharge_rows=len(non_observed_flows),
        duplicate_level_rows=int(levels.duplicated('_time', keep=False).sum()),
        duplicate_discharge_rows=int(flows.duplicated(['_start', '_end'], keep=False).sum()),
        off_grid_level_rows=len(outside_l), misaligned_discharge_rows=len(misaligned),
        reconstruction_intervals=int(result.reconstruction_available.sum()), flag_counts=dict(counts),
        storage_mode=storage.mode, assumed_usable_volume_m3=storage.usable_volume_m3,
        accounting_segments=checks, independent_validation=False,
        timing='Explicit local daily clock converted to UTC; actual elapsed seconds, including DST where declared')
    return result, summary, non_observed_levels, non_observed_flows


def plots(frame, metadata, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bounds = metadata['operating_bounds']
    windows = [('daily_diagnostics', None, None),
               ('winter_spring_2017', '2016-12-01', '2017-07-01'),
               ('winter_spring_2019', '2018-12-01', '2019-07-01')]
    for name, start, end in windows:
        rows = frame if start is None else frame[(frame.local_date >= start) & (frame.local_date < end)]
        if rows.empty: continue
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, layout='constrained')
        dates = pd.to_datetime(rows.local_date)
        series = [('reported_level_start_m', 'Reported level (m)'),
                  ('mean_outlet_discharge_m3_s', 'Reported outlet (m³/s)'),
                  ('storage_start_m3', 'Useful storage (m³)'),
                  ('net_inflow_m3_s', 'Derived net inflow (m³/s)')]
        if rows.storage_mode.iloc[0] == 'normalized':
            series[2] = ('water_level_position_start', 'Normalized water-level position')
        for ax, (column, label) in zip(axes, series):
            ax.plot(dates, rows[column], color='#247ca1', linewidth=1)
            ax.set_ylabel(label); ax.grid(alpha=.2)
            if not rows[column].notna().any(): ax.text(.5, .5, 'Not available — no substituted values', transform=ax.transAxes, ha='center')
        if bounds.get('datum_matches_series', True):
            axes[0].axhline(bounds['minimum_m'], color='#a55b3e', linestyle='--')
            axes[0].axhline(bounds['maximum_m'], color='#a55b3e', linestyle='--')
        axes[3].axhline(0, color='gray', linewidth=.7)
        present = rows[['reported_level_start_m', 'mean_outlet_discharge_m3_s']].notna().all(axis=1)
        axes[4].step(dates, present.astype(int), where='post', label='Observed level + discharge present', color='#268a65')
        axes[4].step(dates, rows.reconstruction_available.astype(int), where='post', label='Estimate computable')
        warning = rows.quality_flags.str.len().gt(0).astype(int)
        axes[4].step(dates, warning, where='post', label='Quality/definition flag', alpha=.65)
        axes[4].set_ylim(-.1, 1.2); axes[4].set_ylabel('Coverage / flags'); axes[4].legend(loc='upper right')
        fig.suptitle(f"{metadata['reservoir']}: {rows.storage_mode.iloc[0]} — accounting diagnostic, not independent validation")
        fig.savefig(folder/f'{name}.png', dpi=150); plt.close(fig)


def report(metadata, summary):
    assessment = summary['status'] == 'assessment_only'
    title = f"# {metadata['reservoir']} daily water-balance pilot\n\n"
    lead = ('**Source assessment complete; historical reconstruction not completed.**\n'
            'No suitable aligned, licensed observation pair was obtained. No historical values were generated.\n'
            if assessment else
            '**Input processing completed.** Estimates use the supplied measurement contract; inspect coverage and flags before use.\n')
    text = title+lead+f"\nEvidence reviewed: {metadata['assessment_date']}. Target: {metadata['target']['start_date']} through {metadata['target']['end_date_exclusive']} (exclusive), closing at midnight.\n\n"
    text += ('## Measurement boundary\n\n'
        'ORRPB lists 3,049 million m³ and a 207.60–223.14 m operating band. '
        'The inventory does not establish usable volume over those elevations. '
        'A different administrative retention figure in the Québec dam inventory is not a substitute storage curve. '
        'Datum realization, historical clock and instantaneous-versus-mean level need confirmation.\n\n'
        'Cabonga releases can enter via Gens de Terre. Mercier and the separately listed Castor structure '
        'must be reconciled with the total-outlet record; turbine discharge alone is insufficient. '
        'Reconstructed net inflow can therefore contain upstream regulated releases, plus net losses/gains; '
        'it is not necessarily natural catchment runoff.\n\n'
        '## Access and actual coverage\n\n'
        'No restricted ORRPB observation export was made. ECCC 02LG007 lists levels only through 1994 '
        'and flows through 1971; downstream 02LG006 ends in 1993. These archives do not cover the target. '
        'The owner open products describe a ten-day window, not a 2016–2019 archive. '
        'The C-6882 disclosure preview contains some 2018 means, but full access returned 403 and '
        'coverage, boundary timing and reuse permission remain unverified. A date selector and '
        'climatological statistics are not observation coverage.\n\n')
    text += '## Processing status\n\n'+f"- Status: `{summary['status']}`.\n"
    if not assessment:
        text += (f"- Storage mode: `{summary['storage_mode']}`; reconstructed intervals: "
                 f"{summary['reconstruction_intervals']} / {summary['expected_intervals']}.\n"
                 f"- Duplicate level/discharge rows: {summary['duplicate_level_rows']} / {summary['duplicate_discharge_rows']}.\n"
                 f"- Off-grid levels / misaligned discharges: {summary['off_grid_level_rows']} / {summary['misaligned_discharge_rows']}.\n"
                 f"- Excluded non-observed level/flow statistics: {summary['non_observed_level_rows']} / {summary['non_observed_discharge_rows']}.\n"
                 f"- Flags: `{json.dumps(summary['flag_counts'], sort_keys=True)}`.\n")
        if summary['storage_mode'] == 'approximate':
            text += f"- K = {summary['assumed_usable_volume_m3']:g} m³ is an explicit scenario, not measured storage capacity.\n"
    else:
        text += ('- Target needs 1,461 daily discharge intervals and 1,462 boundary levels. '
                 'Locally ingested observations: zero. This is not a claim that the owner has no records.\n'
                 '- No cleaned historical dataset or reconstruction plots are claimed. '
                 'Header-only templates and an input contract are provided.\n')
    text += ('\n## Interpretation and next requirement\n\n'
        'Only aligned mean outlet flow and instantaneous boundary levels are eligible for balance estimates. '
        'The pipeline requires an explicit daily timezone, uses elapsed seconds (including declared DST), '
        'excludes conflicting duplicates and misaligned intervals, and does not interpolate or clip. '
        'Negative inflows and suspect jumps remain visible. Normalized water-level positions do not require K; '
        'volumetric inflow requires a sourced curve or an explicitly assumed usable K.\n\n'
        'Cumulative closure is checked separately for contiguous computable segments. Because inflow is '
        'derived from the same storage and release observations, small closure residuals are an accounting '
        'identity, not independent validation of inflow, hydraulics or forecast skill.\n\n'
        '**Extension verdict:** the ingestion and accounting method can be reused for other reservoirs, '
        'but this pilot does not yet establish a calibrated historical reservoir model. Each site needs '
        'its own datum, timing, complete outlet boundary and storage mapping; serial-reservoir transfers '
        'must not be counted twice.\n\n'+metadata['next_specific_requirement']+'\n\n'
        '## Sources\n\n')
    text += '\n'.join(f"- [{s['id']}]({s['url']}): {s['target_coverage']}" for s in metadata['sources'])+'\n'
    return text


def main():
    parser = ArgumentParser(description=__doc__)
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
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh output directory')
    supplied = [args.levels is not None, args.outflows is not None, args.contract is not None]
    if any(supplied) and not all(supplied): parser.error('Supply levels, outflows and contract together')
    if not any(supplied) and (args.storage_curve or args.usable_volume_m3 is not None):
        parser.error('Storage scenarios require actual input files and a contract')
    metadata = json.loads(args.metadata.read_text())
    if metadata.get('schema_version') != 1: raise ValueError('Unsupported reservoir metadata')
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
            curve = pd.read_csv(args.storage_curve); inputs['storage_curve'] = args.storage_curve
        if args.usable_volume_m3 is not None and not (args.scenario_id and args.volume_rationale):
            parser.error('A linear K requires --scenario-id and --volume-rationale; inventory capacity is not a default')
        bounds = metadata.get('level_position_reference', metadata['operating_bounds'])
        storage = StorageMapping(bounds['minimum_m'], bounds['maximum_m'], args.usable_volume_m3, curve)
        frame, summary, non_l, non_q = prepare(pd.read_csv(args.levels, keep_default_na=False), pd.read_csv(args.outflows, keep_default_na=False),
            metadata, contract, storage, jump_m_per_day=args.jump_m_per_day,
            maximum_inflow_m3_s=args.maximum_inflow_m3_s)
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
        raw = args.output/'raw'; raw.mkdir()
        for name, path in inputs.items():
            if name != 'metadata': shutil.copy2(path, raw/f'{name}{path.suffix}')
        folder = args.output/summary['storage_mode']; folder.mkdir()
        frame.to_csv(folder/'daily.csv', index=False)
        non_l.to_csv(folder/'non_observed_levels.csv', index=False)
        non_q.to_csv(folder/'non_observed_outflows.csv', index=False)
        plots(frame, metadata, folder)
    else:
        pd.DataFrame(columns=LEVEL_COLUMNS).to_csv(args.output/'levels_template.csv', index=False)
        pd.DataFrame(columns=FLOW_COLUMNS).to_csv(args.output/'outflows_template.csv', index=False)
    summary['provenance'] = dict(inputs_sha256={str(p): digest(p) for p in inputs.values()},
                                 implementation_sha256=digest(__file__),
                                 no_network_downloads=True, observed_and_climatology_separated=True)
    (args.output/'report.md').write_text(report(metadata, summary))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: summary[k] for k in ('status', 'expected_intervals', 'independent_validation')}, indent=2))
    print(f'Report: {args.output / "report.md"}')


if __name__ == '__main__': main()
