"""Consolidated plans must retain their immutable predeclared payloads."""
import json
from pathlib import Path
import tempfile
import unittest
from lake_como.experiment_plans import record_plan, PLAN_MANIFEST
from lake_como.paths import PROJECT_ROOT


class PlanChecks(unittest.TestCase):
    def test_existing_plan_is_immutable_and_new_plan_preserves_other_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'experiments/plans.json'
            record_plan('first',{'seed':123},manifest=path)
            before=path.read_bytes()
            record_plan('first',{'seed':123},manifest=path)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'plan differs'):
                record_plan('first',{'seed':124},manifest=path)
            self.assertEqual(path.read_bytes(),before)
            record_plan('second',{'steps':1000000},manifest=path)
            self.assertEqual(json.loads(path.read_text()),{'first':{'seed':123},'second':{'steps':1000000}})

    def test_selected_checkpoint_settings_agree_with_selected_study(self):
        selected=json.loads((PROJECT_ROOT/'configs/ppo.json').read_text())
        plans=json.loads(PLAN_MANIFEST.read_text())
        study=plans['duration_1m']
        expected=dict(study['common']);expected['steps']=max(study['budgets'])
        self.assertEqual(selected['training'],expected)
        self.assertEqual(selected['scheduler'],study['scheduler'])
        self.assertEqual(selected['learner_seeds'],study['seeds'])
        self.assertIn(f"step_{expected['steps']}.pt",selected['checkpoint_template'])


if __name__=='__main__':unittest.main()
