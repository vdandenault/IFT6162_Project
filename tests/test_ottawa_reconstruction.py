"""Synthetic unit fixtures only: no invented Baskatong historical observations."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from reservoir_control.systems.ottawa.reconstruction import (
    DEFAULT_METADATA, StorageMapping, convert, daily_boundaries, instant, prepare,
)


class DailyBalanceTests(unittest.TestCase):
    def fixture(self, start='2021-03-13', end='2021-03-16', zone='America/Toronto'):
        metadata = json.loads(DEFAULT_METADATA.read_text())
        metadata['reservoir'] = 'Synthetic unit-test reservoir, not historical observations'
        metadata['target'] = {'start_date': start, 'end_date_exclusive': end}
        boundaries = daily_boundaries(start, end, zone)
        levels = pd.DataFrame([dict(timestamp=t.isoformat(), level=220+i*.1, unit='m',
            aggregation='instantaneous_boundary', statistic='observed', quality_flag='test_good',
            source_id='synthetic_test') for i, t in enumerate(boundaries)])
        flows = pd.DataFrame([dict(interval_start=a.isoformat(), interval_end=b.isoformat(),
            mean_discharge=10., unit='m3/s', aggregation='interval_mean', statistic='observed',
            quality_flag='test_good', source_id='synthetic_test') for a, b in zip(boundaries[:-1], boundaries[1:])])
        contract = dict(schema_version=1, timezone=zone, day_definition_evidence='Synthetic fixture clock',
            timing_verified=True, level_datum='Synthetic aligned datum', datum_alignment_evidence='Fixture only',
            datum_verified=True, total_outlet_verified=True, outlet_definition='Synthetic single outlet',
            usable_quality_flags=['test_good'], sources={'synthetic_test': dict(owner='Unit test',
            url='urn:synthetic:unit-test', terms_url='urn:synthetic:unit-test:permission',
            permission_basis='Generated exclusively for numerical tests', local_processing_and_copy_permitted=True)})
        storage = StorageMapping(207.60, 223.14, usable_volume_m3=15_540_000.)
        return levels, flows, metadata, contract, storage

    def run_fixture(self, fixture): return prepare(*fixture)

    def test_units_dst_and_cumulative_balance(self):
        fixture = self.fixture()
        levels, flows = fixture[:2]
        levels['level'] *= 100; levels['unit'] = 'cm'
        flows['mean_discharge'] *= 1000; flows['unit'] = 'L/s'
        rows, summary, _, _ = self.run_fixture(fixture)
        self.assertEqual(rows.duration_seconds.tolist(), [86400., 82800., 86400.])
        expected = .1*1e6/rows.duration_seconds+10
        np.testing.assert_allclose(rows.net_inflow_m3_s, expected, atol=1e-10)
        self.assertLess(abs(summary['accounting_segments'][0]['closure_error_m3']), 1e-7)
        self.assertFalse(summary['independent_validation'])
        self.assertAlmostEqual(convert(1., 'ft', 'level'), .3048)
        self.assertAlmostEqual(convert(1., 'ft3/s', 'flow'), .028316846592)
        self.assertEqual((daily_boundaries('2021-11-06', '2021-11-09', 'America/Toronto')[2]
                         -daily_boundaries('2021-11-06', '2021-11-09', 'America/Toronto')[1]).total_seconds(), 90000.)
        with self.assertRaises(ValueError): instant('2021-03-13')
        with self.assertRaises(ValueError): convert(1., 'unknown', 'flow')

    def test_missing_final_boundary_and_leap_days(self):
        levels, flows, meta, contract, storage = self.fixture()
        rows, summary, _, _ = prepare(levels.iloc[:-1], flows, meta, contract, storage)
        self.assertTrue(pd.isna(rows.net_inflow_m3_s.iloc[-1]))
        self.assertIn('missing_level_end', rows.quality_flags.iloc[-1])
        self.assertEqual(summary['reconstruction_intervals'], 2)
        self.assertEqual(len(daily_boundaries('2016-01-01', '2020-01-01', 'UTC')), 1462)
        self.assertEqual(len(daily_boundaries('2020-02-28', '2020-03-02', 'UTC')), 4)

    def test_gaps_duplicates_and_climatology_are_not_repaired(self):
        levels, flows, meta, contract, storage = self.fixture()
        levels = pd.concat([levels, levels.iloc[[1]].assign(level=225.)], ignore_index=True)
        flows = pd.concat([flows, flows.iloc[[2]].assign(statistic='median', mean_discharge=999.)], ignore_index=True)
        flows.loc[1, 'mean_discharge'] = np.nan
        rows, summary, _, stats = prepare(levels, flows, meta, contract, storage)
        self.assertEqual(summary['duplicate_level_rows'], 2)
        self.assertEqual(len(stats), 1)
        self.assertFalse(rows.reconstruction_available.iloc[:2].any())
        self.assertTrue(pd.isna(rows.mean_outlet_discharge_m3_s.iloc[1]))
        self.assertEqual(rows.mean_outlet_discharge_m3_s.iloc[2], 10.)
        self.assertIn('duplicate_level_end', rows.quality_flags.iloc[0])

    def test_duplicate_flow_and_misaligned_overlap(self):
        levels, flows, meta, contract, storage = self.fixture()
        duplicate = pd.concat([flows, flows.iloc[[0]]], ignore_index=True)
        rows, summary, _, _ = prepare(levels, duplicate, meta, contract, storage)
        self.assertEqual(summary['duplicate_discharge_rows'], 2)
        self.assertTrue(pd.isna(rows.net_inflow_m3_s.iloc[0]))
        flows.loc[1, 'interval_start'] = (instant(flows.interval_start.iloc[1])+pd.Timedelta(hours=1)).isoformat()
        rows, summary, _, _ = prepare(levels, flows, meta, contract, storage)
        self.assertEqual(summary['misaligned_discharge_rows'], 1)
        self.assertIn('misaligned_or_overlapping_discharge', rows.quality_flags.iloc[1])
        self.assertTrue(pd.isna(rows.net_inflow_m3_s.iloc[1]))

    def test_no_volume_without_K_and_no_unverified_boundary_inflow(self):
        levels, flows, meta, contract, _ = self.fixture()
        normalized = StorageMapping(207.6, 223.14)
        rows, _, _, _ = prepare(levels, flows, meta, contract, normalized)
        self.assertTrue(rows.water_level_position_start.notna().all())
        self.assertTrue(rows.storage_start_m3.isna().all())
        self.assertTrue(rows.net_inflow_m3_s.isna().all())
        for unverified in ('datum_verified', 'total_outlet_verified', 'timing_verified'):
            bad = deepcopy(contract); bad[unverified] = False
            rows, _, _, _ = prepare(levels, flows, meta, bad, StorageMapping(207.6, 223.14, 1e9))
            self.assertTrue(rows.net_inflow_m3_s.isna().all())

    def test_daily_mean_levels_and_bad_quality_do_not_become_boundaries(self):
        levels, flows, meta, contract, storage = self.fixture()
        levels.loc[1, 'aggregation'] = 'daily_mean'
        levels.loc[2, 'quality_flag'] = 'suspect'
        rows, summary, _, _ = prepare(levels, flows, meta, contract, storage)
        self.assertEqual(summary['reconstruction_intervals'], 0)
        self.assertTrue(pd.isna(rows.level_end_m.iloc[0]))
        self.assertEqual(rows.reported_level_end_m.iloc[0], 220.1)
        self.assertTrue(np.isfinite(rows.water_level_position_end.iloc[0]))
        self.assertTrue(pd.isna(rows.storage_end_m3.iloc[0]))
        self.assertIn('not_boundary_level_end', rows.quality_flags.iloc[0])

    def test_negative_inflow_and_large_jump_are_retained(self):
        levels, flows, meta, contract, storage = self.fixture()
        levels['level'] = [223., 221., 220., 219.]
        rows, summary, _, _ = prepare(levels, flows, meta, contract, storage)
        self.assertLess(rows.net_inflow_m3_s.iloc[0], 0)
        self.assertIn('negative_net_inflow', rows.quality_flags.iloc[0])
        self.assertIn('suspect_level_jump', rows.quality_flags.iloc[0])
        levels['level'] = [224., 224.1, 224.2, 224.3]
        rows, _, _, _ = prepare(levels, flows, meta, contract, storage)
        self.assertGreater(rows.water_level_position_start.iloc[0], 1.)
        self.assertIn('level_out_of_range_start', rows.quality_flags.iloc[0])
        self.assertTrue(np.isfinite(rows.net_inflow_m3_s.iloc[0]))

    def test_curve_reference_and_no_extrapolation(self):
        curve = pd.DataFrame({'elevation_m': [207., 210., 220., 224.],
                              'volume_m3': [1e7, 2e7, 3e7, 5e7]})
        model = StorageMapping(207.6, 223.14, curve=curve)
        self.assertAlmostEqual(model.values(207.6)[1], 0.)
        self.assertAlmostEqual(model.values(220.)[1], 3e7-(1e7+.6/3*1e7))
        self.assertTrue(np.isnan(model.values(225.)[1]))
        with self.assertRaises(ValueError): StorageMapping(207.6, 223.14, curve=curve.iloc[::-1])
        with self.assertRaises(ValueError): StorageMapping(207.6, 223.14, 1e9, curve)

    def test_permission_and_unknown_source_gate(self):
        levels, flows, meta, contract, storage = self.fixture()
        bad = deepcopy(contract); bad['sources']['synthetic_test']['local_processing_and_copy_permitted'] = False
        with self.assertRaises(ValueError): prepare(levels, flows, meta, bad, storage)
        levels.loc[0, 'source_id'] = 'unregistered'
        with self.assertRaises(ValueError): prepare(levels, flows, meta, contract, storage)

    def test_all_missing_inputs_and_gap_segments(self):
        levels, flows, meta, contract, storage = self.fixture()
        rows, summary, _, _ = prepare(levels.iloc[:0], flows.iloc[:0], meta, contract, storage)
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary['reconstruction_intervals'], 0)
        self.assertTrue(rows.mean_outlet_discharge_m3_s.isna().all())
        rows, summary, _, _ = prepare(levels, flows.drop(index=1), meta, contract, storage)
        self.assertEqual([s['intervals'] for s in summary['accounting_segments']], [1, 1])

    def test_cli_preserves_literal_owner_quality_flags(self):
        levels, flows, metadata, contract, _ = self.fixture()
        levels['quality_flag'] = 'NA'; flows['quality_flag'] = 'NA'
        contract['usable_quality_flags'] = ['NA']
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, value in [('metadata', metadata), ('contract', contract)]:
                (folder/f'{name}.json').write_text(json.dumps(value))
            levels.to_csv(folder/'levels.csv', index=False)
            flows.to_csv(folder/'outflows.csv', index=False)
            subprocess.run([sys.executable, '-m', 'reservoir_control.systems.ottawa.reconstruction',
                '--metadata', str(folder/'metadata.json'), '--contract', str(folder/'contract.json'),
                '--levels', str(folder/'levels.csv'), '--outflows', str(folder/'outflows.csv'),
                '--output', str(folder/'processed')], check=True, capture_output=True)
            clean = pd.read_csv(folder/'processed/normalized/daily.csv', keep_default_na=False)
            self.assertEqual(clean.level_start_source_quality.tolist(), ['NA']*3)
            self.assertEqual(clean.discharge_source_quality.tolist(), ['NA']*3)
            self.assertTrue((folder/'processed/normalized/daily_diagnostics.png').exists())

    def test_assessment_cli_does_not_create_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'assessment'
            subprocess.run([sys.executable, '-m', 'reservoir_control.systems.ottawa.reconstruction',
                            '--output', str(folder)], check=True, capture_output=True)
            summary = json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['status'], 'assessment_only')
            self.assertFalse(summary['historical_reconstruction_completed'])
            self.assertEqual(len(pd.read_csv(folder/'levels_template.csv')), 0)
            self.assertFalse(list(folder.rglob('daily.csv')))
            rerun = subprocess.run([sys.executable, '-m', 'reservoir_control.systems.ottawa.reconstruction',
                                    '--output', str(folder)], capture_output=True)
            self.assertNotEqual(rerun.returncode, 0)


if __name__ == '__main__': unittest.main()
