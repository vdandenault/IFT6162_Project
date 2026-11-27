import numpy as np
import pandas as pd

from reservoir_control.core.contracts import DecisionTime
from reservoir_control.systems.loslr.data import SupplyForecast
from reservoir_control.systems.loslr.model import DecisionInputs
from reservoir_control.systems.loslr.planning import OntarioPlanningAdapter
from reservoir_control.systems.loslr.slon_mpc_replay import SLONReplayAdapter, forecast_bank


def test_replay_injects_only_slon_at_each_lead():
    time = DecisionTime(95, 2000, 48, 657000.)
    original = DecisionInputs(SupplyForecast((6200., 6300., 6400., 6500.), 7100., 1250.), 0, 1.1, .99)
    adapter = object.__new__(SLONReplayAdapter)
    adapter.paths = {time.index: [1000., 1100., 1200., 1300.]}
    reference_stages = tuple(OntarioPlanningAdapter.forecast_stages(adapter, original, time))
    stages = tuple(adapter.forecast_stages(original, time))
    assert [decision.forecast.slon_m3_s for decision, _ in stages] == adapter.paths[time.index]
    for (reference, ref_time), (decision, stage_time) in zip(reference_stages, stages, strict=True):
        assert ref_time == stage_time
        assert decision.forecast.nts_m3_s == reference.forecast.nts_m3_s
        assert decision.forecast.annual_nts_m3_s == reference.forecast.annual_nts_m3_s
        assert (decision.ice, decision.longsault_roughness, decision.pointe_claire_roughness) == (
            reference.ice, reference.longsault_roughness, reference.pointe_claire_roughness)


def test_last_archived_decision_has_four_causal_leads():
    k = np.arange(48 * 4)
    frame = pd.DataFrame({"Year": 1901 + k // 48, "QM": k % 48 + 1,
                          "stlouisontOut": 1000. + k + 70. * np.sin(k / 3)})
    protocol = {"training_years": [1901, 1903]}
    bank, _ = forecast_bank(frame, protocol)
    tail = bank[bank.issue_index.eq(len(frame) - 1)]
    for delay, rows in tail.groupby("delay"):
        rows = rows.sort_values("lead")
        assert rows.target_year.tolist() == [1904, 1905, 1905, 1905]
        assert rows.target_qm.tolist() == [48, 1, 2, 3]
        assert rows.persistence.eq(frame.stlouisontOut.iloc[-2 - delay]).all()
        assert np.isfinite(rows.seasonal_history).all()
