"""Test paired Ontario release shifts offline, then screen a causal controller."""

from argparse import ArgumentParser
from copy import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
from .data import forcing_from_row, load_inputs
from .model import ReleaseRequest
from .native import INPUT_FILE, NativeReference, sha256
from .planning import MPCSettings
from .slon_mpc_replay import SLONReplayAdapter, replace_slon, start_sessions

OFFSETS = (200., 400., 700.)
LOOKBACK = 4
END_TOL_M = .002
TOL = 1e-8


def headroom_exposure(stage, level, settings):
    return max(0., level - settings.ontario_max_m), max(0., stage - settings.pointe_claire_max_m)


def execute(start, forcing, settings, forecasts, requests, *, trace_prefix=(), planner=None):
    """Replay from a frozen state with the original native forecast-based projection."""
    session = copy(start)
    rows = list(trace_prefix)
    states = []
    for request in requests:
        causal, time = session.information()
        decision = replace_slon(causal, forecasts[time.index][0])
        if request is None:
            request = planner.plan(session.state, decision, time).actions[0]
        states.append((copy(session), request))
        result = session.model.transition(session.state, request, forcing[session.index], time, decision=decision)
        stage = result.metrics["pointe_claire_level_m"]
        state = result.state
        up, down = headroom_exposure(stage, state.level_m, settings)
        rows.append({
            "period": time.period, "request": request.m3_s, "release": result.metrics["executed_release_m3_s"],
            "ontario_end_m": state.level_m, "pointe_claire_m": stage, "up": up, "down": down,
            "regime": result.info["flow_regime"],
        })
        session.state, session.index = state, session.index + 1
    return rows, states


def totals(rows):
    return np.array([sum(row["up"] for row in rows), sum(row["down"] for row in rows),
                     rows[-1]["ontario_end_m"]])


def acceptable(candidate, base):
    return (candidate[0] <= base[0] + TOL and candidate[1] < base[1] - TOL
            and abs(candidate[2] - base[2]) <= END_TOL_M + TOL)


def hindsight(start, forcing, settings, forecast, archived):
    adapter = SLONReplayAdapter(start.model, settings, forecast)
    planner = RecedingHorizonSearch(adapter, beam_width=settings.beam_width)
    control, states = execute(start, forcing, settings, forecast, [None] * 48, planner=planner)
    requests = [action for _, action in states]
    for name, source in (("request", "requested_release_m3_s"), ("release", "release_m3_s"),
                         ("ontario_end_m", "ontario_end_m"),
                         ("pointe_claire_m", "pointe_claire_m")):
        np.testing.assert_allclose([r[name] for r in control], archived[source], rtol=0, atol=TOL)
    base = totals(control)
    best, best_values, witness = control, base, None
    trials = 0
    for later, row in enumerate(control):
        if row["down"] <= TOL:
            continue
        for earlier in range(max(0, later - LOOKBACK), later):
            if control[earlier]["pointe_claire_m"] >= settings.pointe_claire_max_m:
                continue
            for offset in OFFSETS:
                shifted = requests.copy()
                # Shift the executed reference volume. This avoids asking a
                # clipped original request to move below its previous ceiling.
                shifted[earlier] = ReleaseRequest(control[earlier]["release"] + offset)
                shifted[later] = ReleaseRequest(max(0., control[later]["release"] - offset))
                path, _ = execute(states[earlier][0], forcing, settings, forecast, shifted[earlier:],
                                  trace_prefix=control[:earlier])
                values = totals(path)
                trials += 1
                if acceptable(values, base) and values[1] < best_values[1] - TOL:
                    best, best_values, witness = path, values, (earlier + 1, later + 1, offset)
    return base, best_values, best, witness, trials


def visibility(start, witness, archived, persistence, seasonal, settings):
    if witness is None:
        return {}
    earlier, later, _ = witness
    lead = later - earlier
    if lead >= 4:
        return {"within_four_period_plan": False, "forecast_warned_of_exposure": False}
    # Score the archived later release with the earlier issue's tail and
    # roughness. The release is diagnostic context, not a causal input.
    session = copy(start)
    session.index += earlier - 1
    decision, time = session.information()
    slon = max(persistence[time.index][lead], seasonal[time.index][lead])
    release = archived.release_m3_s.iloc[later - 1]
    stage = session.model.reference.utils.round_d(
        16.57 + (decision.pointe_claire_roughness * (release + slon) / 604.) ** .58, 2,
    )
    return {"within_four_period_plan": True, "forecast_stage_at_cut_m": stage,
            "forecast_warned_of_exposure": stage > settings.pointe_claire_max_m + TOL}


def forecast_plan(adapter, initial, stages, actions):
    state = initial
    upstream = downstream = 0.
    for (decision, time), action in zip(stages, actions, strict=True):
        step = adapter.forecast_step(state, action, decision, time)
        state = step.state
        up, down = headroom_exposure(step.pointe_claire_m, state.level_m, adapter.settings)
        upstream += up
        downstream += down
    return np.array([upstream, downstream, state.level_m])


def causal_replay(start, forcing, settings, persistence, seasonal):
    session = copy(start)
    nominal = SLONReplayAdapter(start.model, settings, persistence)
    search = RecedingHorizonSearch(nominal, beam_width=settings.beam_width)
    stress_paths = {key: (path[0], *np.maximum(path[1:], seasonal[key][1:]))
                    for key, path in persistence.items()}
    stress = SLONReplayAdapter(start.model, settings, stress_paths)
    rows = []
    for period in range(1, 49):
        causal, time = session.information()
        decision = replace_slon(causal, persistence[time.index][0])
        plan = search.plan(session.state, decision, time)
        action = plan.actions[0]
        stages = tuple(stress.forecast_stages(decision, time))
        base = forecast_plan(stress, session.state, stages, plan.actions)
        best = base
        # Compare the regular MPC plan with an earlier release paired with
        # an offsetting reduction within the same forecast window.
        if base[1] > TOL:
            for later in range(1, min(len(stages), 49 - period)):
                for offset in OFFSETS:
                    alternative = list(plan.actions)
                    alternative[0] = ReleaseRequest(plan.steps[0].release_m3_s + offset)
                    alternative[later] = ReleaseRequest(max(0., plan.steps[later].release_m3_s - offset))
                    values = forecast_plan(stress, session.state, stages, alternative)
                    if acceptable(values, base) and values[1] < best[1] - TOL:
                        action, best = alternative[0], values
        result = session.model.transition(session.state, action, forcing[session.index], time, decision=decision)
        stage, state = result.metrics["pointe_claire_level_m"], result.state
        up, down = headroom_exposure(stage, state.level_m, settings)
        rows.append({"period": period, "request": action.m3_s, "release": result.metrics["executed_release_m3_s"],
                     "ontario_end_m": state.level_m, "pointe_claire_m": stage, "up": up, "down": down,
                     "regime": result.info["flow_regime"], "used_shift": action is not plan.actions[0]})
        session.state, session.index = state, session.index + 1
    return rows


def paths_for(bank, delay, method):
    wide = bank[bank.delay.eq(delay)].pivot(index="issue_index", columns="lead", values=method)
    assert wide.shape[1] == 4 and wide.notna().all().all()
    return dict(zip(wide.index, wide.to_numpy(), strict=True))


def compare_causal(starts, years, forcing, settings, archived, bank):
    rows, traces = [], []
    for delay in (0, 1):
        p, s = (paths_for(bank, delay, method) for method in ("persistence", "seasonal_history"))
        for year in years:
            ref = archived[archived.year.eq(year) & archived.delay.eq(delay)].sort_values("period")
            assert ref.period.tolist() == list(range(1, 49))
            path = causal_replay(starts[year], forcing, settings, p, s)
            base = np.array([ref.ontario_exposure_m.sum(), ref.downstream_exposure_m.sum(), ref.ontario_end_m.iloc[-1]])
            delta = totals(path) - base
            rows.append({"year": year, "delay": delay, "shift_decisions": sum(r["used_shift"] for r in path),
                         "up_delta_m_periods": delta[0], "down_delta_m_periods": delta[1],
                         "terminal_delta_m": delta[2]})
            traces.extend({"year": year, "delay": delay, **row} for row in path)
    return pd.DataFrame(rows), pd.DataFrame(traces)


def passes(summary):
    for _, group in summary.groupby("delay"):
        if (group.up_delta_m_periods.max() > TOL or group.down_delta_m_periods.sum() >= -TOL
                or group.terminal_delta_m.abs().max() > END_TOL_M + TOL):
            return False
    return True


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/loslr/slon_mpc_replay_verified_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = {name: args.archive / name for name in ("protocol.json", "forecast_bank.csv", "trajectories.csv")}
    hashes = {name: sha256(path) for name, path in saved.items()}
    protocol = json.loads(saved["protocol.json"].read_text())
    years, reserved = (protocol["cohorts"][name] for name in ("development", "reserved"))
    reference = NativeReference()
    assert sha256(reference.root / INPUT_FILE) == protocol["source_sha256"]
    frame = load_inputs(reference)
    forcing = {i: forcing_from_row(row) for i, row in frame.loc[frame.Year.isin(years + reserved)].iterrows()}
    settings = MPCSettings(**protocol["settings"])
    starts = start_sessions(reference, frame, years + reserved)
    bank = pd.read_csv(saved["forecast_bank.csv"])
    archived = pd.read_csv(saved["trajectories.csv"]).query("method == 'persistence'")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "archive_sha256": hashes, "offsets_m3_s": OFFSETS, "lookback_periods": LOOKBACK,
        "terminal_match_tolerance_m": END_TOL_M, "development": years, "reserved": reserved,
        "causal_tail": "max(persistence, seasonal history) at leads 2–4; no change to lead 1 or realized forcing",
    }, indent=2) + "\n")
    rows, witness_rows = [], []
    for delay in (0, 1):
        forecast, seasonal = (paths_for(bank, delay, method) for method in ("persistence", "seasonal_history"))
        for year in years:
            ref = archived[archived.year.eq(year) & archived.delay.eq(delay)].sort_values("period")
            if ref.downstream_exposure_m.sum() <= TOL:
                continue
            base, values, path, witness, trials = hindsight(starts[year], forcing, settings, forecast, ref)
            rows.append({"year": year, "delay": delay, "trials": trials, "found": witness is not None,
                         "reference_up": base[0], "reference_down": base[1], "up_delta": values[0] - base[0],
                         "down_delta": values[1] - base[1], "terminal_delta_m": values[2] - base[2],
                         "earlier_period": witness[0] if witness else None, "later_period": witness[1] if witness else None,
                         "request_shift_m3_s": witness[2] if witness else None,
                         **visibility(starts[year], witness, ref, forecast, seasonal, settings)})
            if witness:
                witness_rows.extend({"year": year, "delay": delay, **row} for row in path)
    checks = pd.DataFrame(rows)
    checks.to_csv(args.output / "hindsight.csv", index=False)
    if witness_rows:
        pd.DataFrame(witness_rows).to_csv(args.output / "hindsight_witnesses.csv", index=False)
    lines = [
        "# Earlier release timing under the agreed shoreline rule", "",
        "This standalone Ontario replay keeps historical SLON and supply as realized forcing. "
        "Each hindsight trial shifts one release earlier in the year, from a downstream-exceedance period "
        "to a period with downstream headroom. The search examines complete realized trajectories and "
        "uses the same causal Plan 2014 projection inputs as the reference. This finite search can show "
        "a feasible improvement. It provides no certificate of global optimality or impossibility.", "",
        f"Every witness adds zero annual Ontario exposure, reduces downstream exposure and matches the "
        f"reference year-end level within {END_TOL_M:.3f} m. All requests are passed through the same native limits.", "",
        "| Delay | Development cases with downstream exceedance | Cases with a witness | Downstream reduction across witnesses (m·periods) |",
        "|---:|---:|---:|---:|",
    ]
    for delay, group in checks.groupby("delay"):
        lines.append(f"| {delay} | {len(group)} | {group.found.sum()} | {-group.down_delta.sum():.3f} |")
    if witness_rows:
        within = int(checks.within_four_period_plan.eq(True).sum())
        warning = int(checks.forecast_warned_of_exposure.eq(True).sum())
        lines += ["", f"At {within} of {int(checks.found.sum())} witness decisions, the later cut was within "
                  f"the four-period causal plan. The forecast warned of downstream exceedance at "
                  f"{warning} of those times when checked at the archived later release. "
                  "This visibility check uses the archived release only to diagnose the earlier forecast.", ""]
        for label, sample in (("development", years), ("reserved", reserved)):
            comparison, traces = compare_causal(starts, sample, forcing, settings, archived, bank)
            comparison.to_csv(args.output / f"causal_{label}.csv", index=False)
            traces.to_csv(args.output / f"causal_{label}_trajectories.csv", index=False)
            passed = passes(comparison)
            lines += ["", f"## Causal {label} replay", "",
                      "The planner tests a volume swap over the original four-period plan. For leads two to four, "
                      "it tests headroom under the larger of persistence and seasonal history SLON. "
                      "The first-period forecast and execution remain on persistence.", "",
                      "| Delay | Years | Shifts requested | Total Ontario change | Worst-year Ontario change | Total downstream change | Maximum year-end level change (m) |",
                      "|---:|---:|---:|---:|---:|---:|---:|"]
            for delay, group in comparison.groupby("delay"):
                lines.append(f"| {delay} | {len(group)} | {group.shift_decisions.sum()} | "
                             f"{group.up_delta_m_periods.sum():+.3f} | {group.up_delta_m_periods.max():+.3f} | "
                             f"{group.down_delta_m_periods.sum():+.3f} | {group.terminal_delta_m.abs().max():.3f} |")
            lines += ["", f"Passed both availability settings: {passed}. Exposure changes are in metre-periods.", ""]
            if not passed:
                if label == "development":
                    lines += ["The reserved replay did not trigger.", ""]
                break
    else:
        lines += ["", "The finite hindsight search found no witness. The causal comparison did not trigger.", ""]
    assert hashes == {name: sha256(path) for name, path in saved.items()}
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
