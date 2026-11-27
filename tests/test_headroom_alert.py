import numpy as np
import pandas as pd

from reservoir_control.systems.loslr.headroom_alert import FEATURES, MODELS, scored_alerts


def test_held_out_opportunities_cannot_change_fits_thresholds_or_scores():
    data = []
    for year in (1985, 1995, 2005):
        for delay in (0, 1):
            for i in range(1, 19):
                data.append({"issue_year": year, "issue_qm": i, "target_year": year,
                             "target_qm": i + 1, "delay": delay, "lead": 2, "can_act": True,
                             "persistence": i / 100., "seasonal": float(np.cos(i)),
                             "early_headroom_m": .1, "planned_release_change_m3_s": i * 15.,
                             "issue_roughness": 1., "local_opportunity": i % 3 == 0})
    original = pd.DataFrame(data)
    _, before, original_fits = scored_alerts(original, [2005], [])
    changed = original.copy()
    heldout = changed.issue_year.eq(2005)
    changed.loc[heldout, "local_opportunity"] = ~changed.loc[heldout, "local_opportunity"]
    _, after, changed_fits = scored_alerts(changed, [2005], [])
    assert original_fits == changed_fits
    fields = [*FEATURES, *[f"{model}_alert" for model in MODELS]]
    pd.testing.assert_frame_equal(before[fields], after[fields])
