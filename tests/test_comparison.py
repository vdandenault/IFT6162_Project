"""Guard paired-year alignment, uncertainty units, and unchanged rollout behavior."""
import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np
from reservoir_control.systems.lake_como.compare import paired_metrics, summaries, verify_matched_inputs, annual_metrics, validate_checkpoint
from reservoir_control.systems.lake_como.reproduce import evaluate
from reservoir_control.systems.lake_como.splits import split_years


class ComparisonChecks(unittest.TestCase):
    def test_pairing_uses_year_keys_and_rejects_missing_or_duplicate_years(self):
        frame=pd.DataFrame(dict(controller=['rule','rule','ppo','ppo'],
                                year=[1990,1989,1989,1990],full_return=[1,10,13,5]))
        pairs=paired_metrics(frame,metrics=['full_return']).set_index('year')
        self.assertEqual(pairs.loc[1989,'delta_full_return'],3)
        self.assertEqual(pairs.loc[1990,'delta_full_return'],4)
        with self.assertRaisesRegex(ValueError,'identical year'):
            paired_metrics(frame.iloc[:-1],metrics=['full_return'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            paired_metrics(pd.concat([frame,frame.iloc[:1]]),metrics=['full_return'])

    def test_seed_sd_is_across_seed_means_not_seed_year_samples(self):
        frame=pd.DataFrame(dict(controller=['ppo_1','ppo_1','ppo_2','ppo_2','ead','ead'],
                                family=['ppo']*4+['ead']*2,year=[1988,1989]*3,
                                full_return=[0,100,10,110,0,0]))
        for metric in ['objective_return','flood_days','low_water_days','deficit_million_m3','terminal_storage_m3']:
            frame[metric]=frame.full_return
        _,yearly,summary=summaries(frame)
        summary=summary.set_index('family')
        self.assertAlmostEqual(summary.loc['ppo','seed_sd_full_return'],np.std([50,60],ddof=1))
        self.assertAlmostEqual(summary.loc['ppo','year_sd_full_return'],np.std([5,105],ddof=1))
        self.assertTrue(np.isnan(summary.loc['ead','seed_sd_full_return']))
        self.assertEqual(len(yearly),4)

    def test_timing_does_not_change_control_and_forcing_mismatch_is_rejected(self):
        original,_=evaluate('ead',split='validation',protocol='chronological',correct_terminal_level=True)
        timed,_=evaluate('ead',split='validation',protocol='chronological',correct_terminal_level=True,
                          sequential_years=True,measure_decisions=True)
        pd.testing.assert_frame_equal(original,timed[original.columns])
        self.assertEqual(timed.year.drop_duplicates().tolist(),split_years('validation'))
        annual=annual_metrics(timed)
        np.testing.assert_allclose(annual.filter(regex='^weighted_').sum(axis=1),annual.full_return)
        a=timed.assign(controller='a');b=timed.assign(controller='b')
        verify_matched_inputs(pd.concat([a,b]),split_years('validation'))
        b=b.copy();b.loc[b.index[0],'inflow_m3_s']+=1
        with self.assertRaises(AssertionError):
            verify_matched_inputs(pd.concat([a,b]),split_years('validation'))

    def test_training_split_evaluation_visits_each_year_once(self):
        frame,_=evaluate('mean',split='train',protocol='chronological',correct_terminal_level=True,
                         sequential_years=True)
        self.assertEqual(frame.year.drop_duplicates().tolist(),split_years('train'))
        self.assertEqual(len(frame),42*365)

    def test_custom_physical_controller_uses_shared_evaluation(self):
        reference,_=evaluate('constant',split='validation',protocol='chronological',sequential_years=True)
        custom,_=evaluate('custom',split='validation',protocol='chronological',sequential_years=True,
                         controller_factory=lambda env: lambda obs, info: np.array([100.],dtype=np.float32))
        np.testing.assert_array_equal(reference.full_reward,custom.full_reward)
        np.testing.assert_array_equal(reference.level_m,custom.level_m)

    def test_checkpoint_requires_matching_training_membership_and_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)/'step_1000000.pt'
            checkpoint.write_bytes(b'checkpoint identity')
            metadata=dict(complete=True,job=dict(seed=123,common=dict(protocol='chronological',correct_terminal_level=True)),
                          training_years=split_years('train'),versions={},
                          checkpoint_sha256={'1000000':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            path=checkpoint.parent/'run.json';path.write_text(json.dumps(metadata))
            self.assertEqual(validate_checkpoint(123,checkpoint)['training_seed'],123)
            metadata['training_years']=[1998];path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'training membership'):
                validate_checkpoint(123,checkpoint)
            metadata['training_years']=split_years('train');path.write_text(json.dumps(metadata))
            checkpoint.write_bytes(b'modified checkpoint')
            with self.assertRaisesRegex(ValueError,'hash differs'):
                validate_checkpoint(123,checkpoint)


if __name__=='__main__':unittest.main()
