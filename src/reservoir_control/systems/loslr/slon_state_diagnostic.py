"""Separate immediate SLON forecast effects from inherited-state effects."""

from argparse import ArgumentParser
from copy import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
from .data import forcing_from_row, load_inputs
from .native import INPUT_FILE, NativeReference, sha256
from .planning import MPCSettings
from .slon_mpc_replay import FLOW_TOL, METHODS, SLONReplayAdapter, replace_slon, start_sessions

EXPOSURE_TOL = 1e-9
HYBRIDS = ("seasonal_lead_1", "seasonal_leads_2_to_4")


def inherited_states(reference, frame, trace, selected):
    starts = start_sessions(reference, frame, selected.year.unique())
    states = {}
    for (year, delay, method), rows in trace[trace.year.isin(starts)].groupby(["year", "delay", "method"]):
        session = copy(starts[year])
        issue_keys = set(selected.loc[selected.year.eq(year) & selected.delay.eq(delay), "period"])
        for row in rows.sort_values("period").itertuples(index=False):
            decision, time = session.information()
            assert (time.year, time.period) == (year, row.period)
            forcing = forcing_from_row(frame.iloc[session.index])
            state = session.state
            np.testing.assert_allclose([state.level_m, state.previous_release_m3_s],
                                       [row.ontario_start_m, row.previous_release_m3_s], rtol=0, atol=1e-8)
            if row.period in issue_keys:
                states[(year, row.period, delay, method)] = (state, decision, time, forcing)
            session.state = session.model.advance_storage(state, row.release_m3_s, forcing.nts_m3_s, forcing.ice, time)
            session.index += 1
            np.testing.assert_allclose(session.state.level_m, row.ontario_end_m, rtol=0, atol=1e-8)
    return states, next(iter(starts.values())).model


def replay_cell(model, settings, snapshot, slon):
    state, causal, time, forcing = snapshot
    adapter = SLONReplayAdapter(model, settings, {time.index: slon})
    plan = RecedingHorizonSearch(adapter, beam_width=settings.beam_width).plan(
        state, replace_slon(causal, slon[0]), time)
    result = model.transition(state, plan.actions[0], forcing, time, decision=replace_slon(causal, slon[0]))
    assert not result.info["f_limit_uses_realized_slon"] and not result.info["m_limit_uses_realized_slon"]
    release = result.metrics["executed_release_m3_s"]
    stage = result.metrics["pointe_claire_level_m"]
    np.testing.assert_allclose(release, plan.steps[0].release_m3_s, rtol=0, atol=FLOW_TOL)
    return {
        "ontario_start_m": state.level_m, "previous_release_m3_s": state.previous_release_m3_s,
        "requested_release_m3_s": plan.actions[0].m3_s, "release_m3_s": release,
        "ontario_end_m": result.state.level_m, "pointe_claire_m": stage,
        "realized_slon_m3_s": forcing.slon_m3_s, "forecast_slon_m3_s": slon[0],
        "predicted_pc_headroom_m": settings.pointe_claire_max_m - plan.steps[0].pointe_claire_m,
        "plan_feasible": plan.feasible, "regime": result.info["flow_regime"],
        "f_limit_m3_s": 10 * result.info["native_limits"]["fFlow"],
        "downstream_exposure_m": max(0., stage - settings.pointe_claire_max_m),
        "forecast_path_m3_s": json.dumps(list(slon)),
        "forecast_stage_path_m": json.dumps([step.pointe_claire_m for step in plan.steps]),
    }


def decompose(pp, ps, sp, ss):
    """Average the two orders of swapping state and forecast."""
    immediate = ((ps - pp) + (ss - sp)) / 2
    inherited = ((sp - pp) + (ss - ps)) / 2
    return immediate, inherited, (ss - sp) - (ps - pp)


def analyze(model, settings, selected, trace, bank, states, protocol):
    archive = trace.set_index(["year", "period", "delay", "method"])
    paths = bank.sort_values("lead").groupby(["issue_year", "issue_qm", "delay"])
    rows, attribution = [], []
    for target in selected.itertuples(index=False):
        key = target.year, target.period, target.delay
        frame = paths.get_group(key)
        assert frame.lead.tolist() == [1, 2, 3, 4]
        p, s = (tuple(frame[method]) for method in METHODS)
        variants = dict(zip((*METHODS, *HYBRIDS), (p, s, (s[0], *p[1:]), (p[0], *s[1:])), strict=True))
        cohort = next(name for name, years in protocol["cohorts"].items() if target.year in years)
        base = {"year": target.year, "period": target.period, "delay": target.delay, "cohort": cohort}
        cells = {}
        for state_method in METHODS:
            for forecast_method, slon in variants.items():
                row = replay_cell(model, settings, states[(*key, state_method)], slon)
                cells[(state_method, forecast_method)] = row
                rows.append({**base, "inherited_state": state_method, "forecast": forecast_method, **row})
                if state_method == forecast_method:
                    original = archive.loc[(*key, state_method)]
                    checked = ("requested_release_m3_s", "release_m3_s", "pointe_claire_m", "ontario_end_m")
                    np.testing.assert_allclose([row[field] for field in checked], original[list(checked)].astype(float),
                                               rtol=0, atol=1e-8)
                    assert row["plan_feasible"] == original.plan_feasible and row["regime"] == original.regime
        response = {**base, "realized_slon_m3_s": cells[(METHODS[0], METHODS[0])]["realized_slon_m3_s"]}
        for field in ("downstream_exposure_m", "release_m3_s"):
            pp, ps, sp, ss = (cells[(state, forecast)][field] for state in METHODS for forecast in METHODS)
            immediate, inherited, interaction = decompose(pp, ps, sp, ss)
            np.testing.assert_allclose(immediate + inherited, ss - pp, atol=1e-10, rtol=0)
            response.update({f"{field}_observed_delta": ss - pp, f"{field}_forecast_effect": immediate,
                             f"{field}_inherited_effect": inherited, f"{field}_interaction": interaction})
        response["lead_one_forecast_delta_m3_s"] = s[0] - p[0]
        response["seasonal_lead_one_error_m3_s"] = s[0] - response["realized_slon_m3_s"]
        attribution.append(response)
    return pd.DataFrame(rows), pd.DataFrame(attribution)


def report(attribution, cells, threshold):
    lines = [
        "# Fixed-state diagnostic of increased Lake St. Louis exposure", "",
        "The sample is every period where seasonal history increased realized Pointe-Claire "
        "exposure relative to persistence in the archived Ontario-only replay. Each forecast is "
        "rerun on both inherited states with identical realized inputs, limits, candidate search "
        "and objective. Archived decisions reproduce on the matching state and forecast.", "",
        "The immediate forecast effect averages the two forecast swaps, one per inherited state. "
        "The inherited-state effect averages the two state swaps, one per forecast. Their sum "
        "equals the archived change. Interaction is the difference between the two forecast effects. "
        "The selected periods do not estimate net effects across all decisions.", "",
        "| Cohort | Data delay | Decisions | Exposure increase | Immediate forecast | Inherited state |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (cohort, delay), group in attribution.groupby(["cohort", "delay"], sort=False):
        values = (group[f"downstream_exposure_m_{name}"].sum() for name in
                  ("observed_delta", "forecast_effect", "inherited_effect"))
        lines.append(f"| {cohort} | {delay} | {len(group)} | " + " | ".join(f"{v:+.3f}" for v in values) + " |")
    q = attribution.filter(regex="^downstream_exposure_m_").sum()
    lines += ["", f"Across {len(attribution)} selected decisions, exposure increased by "
              f"{q.downstream_exposure_m_observed_delta:.3f} metre-periods. "
              f"The immediate swap accounts for {q.downstream_exposure_m_forecast_effect:+.3f} "
              f"and the inherited state for {q.downstream_exposure_m_inherited_effect:+.3f}.", ""]
    keys = ["cohort", "year", "period", "delay", "inherited_state"]
    wide = cells.pivot(index=keys, columns="forecast", values=["release_m3_s", "downstream_exposure_m", "regime",
                                                               "forecast_slon_m3_s", "realized_slon_m3_s"])
    flow = wide["release_m3_s"]
    downstream = wide["downstream_exposure_m"]
    harmful = downstream.seasonal_history.sub(downstream.persistence).gt(EXPOSURE_TOL)
    less_slon = wide["forecast_slon_m3_s"].seasonal_history.lt(wide["forecast_slon_m3_s"].persistence)
    under = wide["forecast_slon_m3_s"].seasonal_history.lt(wide["realized_slon_m3_s"].seasonal_history)
    high = wide["realized_slon_m3_s"].seasonal_history.ge(threshold)
    any_f = wide["regime"][list(METHODS)].eq("F").any(axis=1)
    lines += [
        "## Forecast swaps at fixed states", "",
        "The two hybrid probes replace either lead 1 or leads 2–4 on their own. "
        "Counts check whether a hybrid reproduces the seasonal-history first release on states "
        "where the full forecast swap increased downstream exposure. They can overlap.", "",
        "| Sample | Harmful swaps | Seasonal lead 1 lower | Seasonal lead 1 below realized | High realized SLON | F regime in either path | Lead 1 alone reproduces release | Leads 2–4 alone reproduce release |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, mask in (
        ("Development", wide.index.get_level_values("cohort") == "development"),
        ("All selected states", np.ones(len(wide), dtype=bool)),
    ):
        picked = harmful & mask
        counts = [picked.sum(), *(int((picked & condition).sum()) for condition in (
            less_slon, under, high, any_f,
            flow.seasonal_lead_1.sub(flow.seasonal_history).abs().le(FLOW_TOL),
            flow.seasonal_leads_2_to_4.sub(flow.seasonal_history).abs().le(FLOW_TOL),
        ))]
        lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", f"High flow uses the frozen {threshold:,.0f} m³/s threshold. "
              "Native F regime is reported separately from the MPC scoring threshold. "
              "No forecast bound is fitted or used in this diagnostic.", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/loslr/slon_mpc_replay_verified_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {name: args.archive / name for name in ("protocol.json", "trajectories.csv", "forecast_bank.csv")}
    hashes = {name: sha256(path) for name, path in files.items()}
    protocol = json.loads(files["protocol.json"].read_text())
    trace = pd.read_csv(files["trajectories.csv"])
    bank = pd.read_csv(files["forecast_bank.csv"])
    selection = trace.pivot(index=["year", "period", "delay"], columns="method", values="downstream_exposure_m")
    selected = selection[selection.seasonal_history.sub(selection.persistence).gt(EXPOSURE_TOL)].reset_index()
    reference = NativeReference()
    assert sha256(reference.root / INPUT_FILE) == protocol["source_sha256"]
    frame = load_inputs(reference)
    states, model = inherited_states(reference, frame, trace, selected)
    cells, attribution = analyze(model, MPCSettings(**protocol["settings"]), selected, trace, bank, states, protocol)
    assert len(cells) == 8 * len(selected) and len(attribution) == len(selected)
    assert hashes == {name: sha256(path) for name, path in files.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(args.output / "replays.csv", index=False)
    attribution.to_csv(args.output / "attribution.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps({
        "archive": str(args.archive), "sha256": hashes, "selection_exposure_tolerance_m": EXPOSURE_TOL,
    }, indent=2) + "\n")
    text = report(attribution, cells, protocol["slon_high_flow_threshold_m3_s"])
    (args.output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
