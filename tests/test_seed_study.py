"""Ensure seed uncertainty is computed across policies, not pooled year/seed rows."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from lake_como.seed_study import summarize


class SeedStudyChecks(unittest.TestCase):
    def test_aggregate_seed_means_and_pair_by_year(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest=[]
            for i, seed in enumerate(range(123,128)):
                directory=root/str(seed);directory.mkdir()
                rows=[]
                # Reverse order so pairing cannot accidentally rely on row position.
                for year in reversed(range(2000,2013)):
                    for policy in ['random','mean','max','ead','ppo']:
                        value = (i+1)*10+(year-2000) if policy=='ppo' else 2
                        rows.append(dict(policy=policy,seed=123,year=year,objective_return=value,
                                         full_return=value-1,flood_days=0,low_water_days=0,
                                         deficit_million_m3=1))
                pd.DataFrame(rows).to_csv(directory/'annual.csv',index=False)
                manifest.append(dict(training_seed=seed,evaluation_dir=str(directory),training_seconds=1))
            with redirect_stdout(StringIO()):
                means, aggregate, paired = summarize(root,manifest)
            np.testing.assert_allclose(means.objective_return,[16,26,36,46,56])
            self.assertAlmostEqual(aggregate.loc['objective_return','mean'],36)
            self.assertAlmostEqual(aggregate.loc['objective_return','std'],np.std([16,26,36,46,56],ddof=1))
            self.assertEqual(len(paired),20)
            np.testing.assert_allclose(paired[paired.training_seed==123].mean_objective_advantage,14)
            self.assertTrue((paired.years_with_higher_objective==13).all())


if __name__=='__main__':
    unittest.main()
