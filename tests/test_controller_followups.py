import numpy as np
import pandas as pd

from bilevel_reservoirs.solvers.diagnose_ppo_menu import release_probabilities
from reservoir_control.systems.loslr.slon_upper_bound_study import make_paths, select


def test_gaussian_release_mass_accounts_for_squash_and_unreachable_bins():
    mass = release_probabilities(np.array([1000., 9000., 11000., 18000.]),
                                 neutral=10000., mean=0., sigma=1., action_low=-2000., action_high=2000.)
    np.testing.assert_allclose(mass, [0., .5, .5, 0.], rtol=0, atol=1e-12)


def test_slon_margin_changes_only_first_lead_and_gate_checks_both_delays():
    bank = pd.DataFrame({
        "issue_index": [10] * 4, "lead": [1, 2, 3, 4], "delay": [0] * 4,
        "persistence": [1100.] * 4, "seasonal_history": [1000., 1200., 1300., 1400.],
    })
    bounds = {"seasonal_upper_all": 100.}
    forecasts = make_paths(bank, 0, "seasonal_half_upper", bounds)
    np.testing.assert_array_equal(forecasts[10], [1050., 1200., 1300., 1400.])
    summary = pd.DataFrame({
        "method": ["persistence", "persistence", "candidate", "candidate"], "delay": [0, 1, 0, 1],
        "periods": [960] * 4, "upstream": [1., 1., 1.05, 1.11], "downstream": [10., 10., 9., 8.],
    })
    assert select(summary, ("candidate",)) is None
    summary.loc[3, "upstream"] = 1.09
    assert select(summary, ("candidate",)) == "candidate"
