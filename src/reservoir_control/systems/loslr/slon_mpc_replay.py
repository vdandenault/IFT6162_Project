"""Isolate SLON forecast effects in matched historical Ontario MPC replays."""

from argparse import ArgumentParser
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
from .causal import CausalOntarioSession
from .data import forcing_from_row, load_inputs
from .model import OntarioModel, Plan2014
from .native import INPUT_FILE, NativeReference, sha256
from .planning import MPCSettings, OntarioPlanningAdapter
from .slon_forecast import DELAYS, LEADS, aligned_pairs, fit_forecasts, load_protocol

METHODS = ("persistence", "seasonal_history")
FLOW_TOL = 1e-5


def forecast_bank(frame, protocol):
    """Recreate the frozen fits and issue all four leads at each replay decision."""
    # Future calendar labels are known. Three extra labels let the planner issue
    # its ordinary four-lead path even on the last archived decision.
    year = frame.Year.iloc[-1]
    padding = pd.DataFrame({"Year": year + 1, "QM": range(1, max(LEADS)), "stlouisontOut": np.nan})
    extended = pd.concat([frame, padding], ignore_index=True)
    assert frame.QM.iloc[-1] == 48
    training = protocol["training_years"]
    seasonal = frame.loc[frame.Year.between(*training)].groupby("QM").stlouisontOut.mean()
    parts, fits = [], {}
    for lead in LEADS:
        delayed = aligned_pairs(extended, lead, DELAYS[-1])
        eligible = delayed.drop(columns="target_m3_s").notna().all(axis=1)
        for delay in DELAYS:
            pairs = aligned_pairs(extended, lead, delay)
            forecasts, coefficients = fit_forecasts(pairs.loc[eligible], training, seasonal)
            fits[f"lead_{lead}_delay_{delay}"] = coefficients
            parts.append(forecasts.loc[forecasts.issue_index.lt(len(frame)), [
                "issue_index", "issue_year", "issue_qm", "last_available_year", "last_available_qm",
                "target_year", "target_qm", "persistence", "seasonal_history",
            ]].assign(lead=lead, delay=delay))
    return pd.concat(parts, ignore_index=True), fits


def verify_bank(bank, fits, benchmark, source_sha256):
    frozen = json.loads((benchmark / "protocol.json").read_text())
    if frozen["source_sha256"] != source_sha256:
        raise ValueError("The forecast benchmark used a different LOSLR source")
    for key, coefficients in fits.items():
        saved = frozen["coefficients"][key]
        if saved.keys() != coefficients.keys():
            raise ValueError("The forecast benchmark used different features")
        np.testing.assert_allclose(list(coefficients.values()), [saved[name] for name in coefficients],
                                   rtol=0., atol=1e-9)
    archived = pd.read_csv(benchmark / "predictions.csv")
    keys = ["issue_index", "lead", "delay"]
    shared = archived.merge(bank, on=keys, validate="one_to_one", suffixes=("_saved", "_replay"))
    if len(shared) != len(archived):
        raise ValueError("The new bank must cover every archived issue")
    for method in METHODS:
        np.testing.assert_allclose(shared[f"{method}_saved"], shared[f"{method}_replay"], rtol=0., atol=1e-8)


def replace_slon(decision, value):
    return replace(decision, forecast=replace(decision.forecast, slon_m3_s=float(value)))


class SLONReplayAdapter(OntarioPlanningAdapter):
    """Inject exactly one preissued SLON path into the existing planning stages."""

    def __init__(self, model, settings, paths):
        super().__init__(model, settings)
        self.paths = paths

    def forecast_stages(self, decision, start):
        stages = super().forecast_stages(decision, start)
        for (estimate, time), slon in zip(stages, self.paths[start.index], strict=True):
            yield replace_slon(estimate, slon), time


def start_sessions(reference, frame, years):
    baseline = CausalOntarioSession(OntarioModel(reference), frame)
    rule = Plan2014(reference)
    starts = {}
    for year in sorted(years):
        while baseline.information()[1].year < year:
            decision, time = baseline.information()
            baseline.step(rule.action(baseline.state, decision.forecast, time))
        assert baseline.information()[1].period == 1
        starts[year] = copy(baseline)
    return starts


def replay_year(start, frame, settings, paths, year, method, delay):
    session = copy(start)
    adapter = SLONReplayAdapter(session.model, settings, paths)
    controller = RecedingHorizonSearch(adapter, beam_width=settings.beam_width)
    cap = settings.pointe_claire_max_m
    stage_flow = 604. * (cap - 16.57) ** (1 / .58)
    records = []
    for period in range(1, 49):
        causal, time = session.information()
        assert (time.year, time.period) == (year, period)
        slon = paths[time.index]
        if delay == 0 and method == "persistence":
            np.testing.assert_allclose(slon, causal.forecast.slon_m3_s, rtol=0., atol=FLOW_TOL)
        decision = replace_slon(causal, slon[0])
        state = session.state
        plan = controller.plan(state, decision, time)
        # Bv7 uses decision estimates. Historical SLON enters only after the
        # request has been chosen, through the realized routing transition.
        forcing = forcing_from_row(frame.iloc[session.index])
        result = session.model.transition(state, plan.actions[0], forcing, time, decision=decision)
        release = result.metrics["executed_release_m3_s"]
        stage = result.metrics["pointe_claire_level_m"]
        combined = result.metrics["lake_st_louis_flow_m3_s"]
        np.testing.assert_allclose(release, plan.steps[0].release_m3_s, rtol=0., atol=FLOW_TOL)
        np.testing.assert_allclose(combined, release + forcing.slon_m3_s, rtol=0., atol=FLOW_TOL)
        assert not result.info["f_limit_uses_realized_slon"] and not result.info["m_limit_uses_realized_slon"]
        session.state = result.state
        session.index += 1
        records.append({
            "year": year, "period": period, "method": method, "delay": delay,
            "ontario_start_m": state.level_m, "ontario_end_m": result.state.level_m,
            "pointe_claire_m": stage, "lake_st_louis_flow_m3_s": combined,
            "previous_release_m3_s": state.previous_release_m3_s,
            "requested_release_m3_s": plan.actions[0].m3_s, "release_m3_s": release,
            "regime": result.info["flow_regime"], "plan_feasible": plan.feasible,
            "plan_max_excess_m": plan.max_excess_m,
            "predicted_pc_headroom_m": cap - plan.steps[0].pointe_claire_m,
            "min_planned_pc_headroom_m": min(cap - step.pointe_claire_m for step in plan.steps),
            "realized_pc_headroom_m": cap - stage,
            "realized_pc_flow_headroom_m3_s": stage_flow / forcing.native_metadata["ptclaireR"] - combined,
            "ontario_exposure_m": max(0., result.state.level_m - settings.ontario_max_m),
            "downstream_exposure_m": max(0., stage - cap),
            "forecast_nts_m3_s": causal.forecast.nts_m3_s[0],
            "realized_nts_m3_s": forcing.nts_m3_s, "realized_slon_m3_s": forcing.slon_m3_s,
            **{f"forecast_slon_lead_{j}_m3_s": value for j, value in enumerate(slon, 1)},
        })
    return records


def summarize(trace, cohorts, threshold):
    result, pairs = [], []
    for cohort, years in cohorts.items():
        rows = trace[trace.year.isin(years)]
        for delay in DELAYS:
            timely = rows[rows.delay.eq(delay)]
            for method, group in timely.groupby("method"):
                high = group[group.realized_slon_m3_s.ge(threshold)]
                result.append({
                    "cohort": cohort, "delay": delay, "method": method,
                    "completed_years": int(group.groupby("year").size().eq(48).sum()),
                    "years": len(years), "executed_periods": len(group),
                    "infeasible_plans": int((~group.plan_feasible).sum()),
                    "mean_release_m3_s": group.release_m3_s.mean(),
                    "min_predicted_pc_headroom_m": group.predicted_pc_headroom_m.min(),
                    "min_realized_pc_headroom_m": group.realized_pc_headroom_m.min(),
                    "min_realized_pc_flow_headroom_m3_s": group.realized_pc_flow_headroom_m3_s.min(),
                    "negative_realized_pc_headroom_periods": int(group.realized_pc_headroom_m.lt(-1e-9).sum()),
                    "upstream_exposure_m_periods": group.ontario_exposure_m.sum(),
                    "downstream_exposure_m_periods": group.downstream_exposure_m.sum(),
                    "high_flow_periods": len(high), "high_flow_downstream_exposure_m_periods": high.downstream_exposure_m.sum(),
                })
            paired = timely.pivot(index=["year", "period"], columns="method", values="release_m3_s")
            delta = paired.seasonal_history - paired.persistence
            pairs.append({"cohort": cohort, "delay": delay, "periods": len(delta),
                          "changed_releases": int(delta.abs().gt(FLOW_TOL).sum()),
                          "mean_absolute_release_change_m3_s": delta.abs().mean(),
                          "mean_signed_release_change_m3_s": delta.mean(),
                          "max_absolute_release_change_m3_s": delta.abs().max()})
    return pd.DataFrame(result), pd.DataFrame(pairs)


def report(summary, pairs, settings):
    lines = [
        "# Ontario MPC replay with historical aggregate SLON", "",
        "Both forecast methods use the existing standalone Ontario beam-search MPC with the same "
        "four-period horizon, candidate-generation rules, Plan 2014 limits, tracking and terminal weights. "
        "Historical SLON and Ontario net total supply provide the realized forcing for every arm "
        "in this Ontario-only replay. Each year starts at the same state "
        "warmed by a continuous causal Plan 2014 replay.", "",
        "Lead-one SLON estimates enter the M and F limits at execution. All four forecasts enter planning. "
        "Ontario supply forecasts remain causal persistence with the same trailing annual mean. "
        "One period of delay withholds the latest completed SLON period only.", "",
        f"Stage headroom is {settings.pointe_claire_max_m:.2f} m minus Pointe-Claire stage. "
        f"Upstream exposure is the sum of positive level differences above {settings.ontario_max_m:.2f} m. "
        "Downstream exposure uses the fixed Pointe-Claire stage cap. The CSV also includes flow headroom "
        "relative to the unrounded rating curve. Completion means 48 executed actions. "
        "The existing search executes the least-violating candidate when no feasible plan is found.", "",
        "| Case | Delay | Method | Years complete | Infeasible plans | Worst realized downstream headroom (m) | Ontario exposure | Lake St. Louis exposure | High-flow downstream exposure |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.cohort} | {row.delay} | {row.method} | {row.completed_years}/{row.years} | "
                     f"{row.infeasible_plans} | {row.min_realized_pc_headroom_m:+.2f} | "
                     f"{row.upstream_exposure_m_periods:.3f} | {row.downstream_exposure_m_periods:.3f} | "
                     f"{row.high_flow_downstream_exposure_m_periods:.3f} |")
    lines += [
        "", "Exposure totals are in metre-periods over all years in a cohort.", "",
        "| Case | Delay | Releases changed | Mean absolute change (m³/s) | Largest change (m³/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(f"| {row.cohort} | {row.delay} | {row.changed_releases}/{row.periods} | "
                     f"{row.mean_absolute_release_change_m3_s:.1f} | {row.max_absolute_release_change_m3_s:.0f} |")
    lines += ["", "Historical SLON includes multiple tributaries and reflects historic regulation. "
              "This experiment treats it as fixed downstream forcing. It cannot score a counterfactual "
              "change in Ottawa reservoir operation.", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("results/loslr/slon_forecasts_verified_v1"))
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    first, last = protocol["development_years"]
    cohorts = {"development": list(range(first, last + 1)), "reserved": protocol["reserved_test_years"],
               **{str(year): [year] for year in protocol["development_events"]}}
    years = set().union(*cohorts.values())
    reference = NativeReference(args.checkout)
    settings = MPCSettings()
    frame = load_inputs(reference)
    args.output.mkdir(parents=True, exist_ok=False)
    bank, fits = forecast_bank(frame.iloc[48:][["Year", "QM", "stlouisontOut"]].reset_index(drop=True), protocol)
    source_sha256 = sha256(reference.root / INPUT_FILE)
    verify_bank(bank, fits, args.benchmark, source_sha256)
    starts = start_sessions(reference, frame, years)
    bank = bank[bank.issue_year.isin(years)]
    trace_parts = []
    for delay in DELAYS:
        delayed = bank[bank.delay.eq(delay)]
        for method in METHODS:
            paths = delayed.pivot(index="issue_index", columns="lead", values=method).loc[:, list(LEADS)]
            assert paths.notna().all().all()
            indexed = dict(zip(paths.index, paths.to_numpy(), strict=True))
            for year, start in starts.items():
                trace_parts.extend(replay_year(start, frame, settings, indexed, year, method, delay))
            print(f"Completed {len(starts)} years, {method}, delay {delay}", flush=True)
    trace = pd.DataFrame(trace_parts)
    shared = ["forecast_nts_m3_s", "realized_nts_m3_s", "realized_slon_m3_s"]
    trace["initial_ontario_m"] = trace.groupby(["year", "method", "delay"]).ontario_start_m.transform("first")
    checks = trace.groupby(["year", "period"])[["initial_ontario_m", *shared]].nunique()
    assert checks.eq(1).all().all()
    assert trace.groupby(["year", "period"]).size().eq(len(DELAYS) * len(METHODS)).all()
    threshold = json.loads((args.benchmark / "protocol.json").read_text())["high_flow_threshold_m3_s"]
    summary, pairs = summarize(trace, cohorts, threshold)
    annual, _ = summarize(trace, {str(year): [year] for year in sorted(years)}, threshold)
    for name, data in (("trajectories", trace), ("forecast_bank", bank), ("summary", summary),
                       ("paired_releases", pairs), ("annual", annual)):
        data.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "settings": asdict(settings), "cohorts": cohorts, "methods": METHODS, "delays": DELAYS, "leads": LEADS,
        "source_sha256": source_sha256, "benchmark_sha256": {name: sha256(args.benchmark / name)
                         for name in ("protocol.json", "predictions.csv")},
        "slon_high_flow_threshold_m3_s": threshold, "completion": "48 executed actions",
    }, indent=2) + "\n")
    (args.output / "report.md").write_text(report(summary, pairs, settings))


if __name__ == "__main__":
    main()
