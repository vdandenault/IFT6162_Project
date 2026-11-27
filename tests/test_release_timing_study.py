import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.release_timing_study import acceptable, passes


def test_witness_requires_improved_downstream_and_matched_upstream_and_end_level():
    reference = np.array([0., .5, 74.8])
    assert acceptable(np.array([0., .4, 74.801]), reference)
    assert not acceptable(np.array([.001, .4, 74.8]), reference)
    assert not acceptable(np.array([0., .5, 74.8]), reference)
    assert not acceptable(np.array([0., .4, 74.803]), reference)


def test_causal_gate_does_not_hide_a_worse_ontario_year_in_average():
    data = pd.DataFrame({"delay": [0, 0, 1, 1],
                         "up_delta_m_periods": [-.2, .1, 0., 0.],
                         "down_delta_m_periods": [-.1, -.1, -.1, -.1],
                         "terminal_delta_m": [0.] * 4})
    assert not passes(data)
    data.loc[1, "up_delta_m_periods"] = 0.
    assert passes(data)
