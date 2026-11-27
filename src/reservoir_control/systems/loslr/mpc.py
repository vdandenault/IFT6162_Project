import argparse
from copy import copy
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
from .planning import MPCSettings, OntarioPlanningAdapter
from .causal import CausalOntarioSession
from .model import OntarioModel, Plan2014
from .data import load_inputs
from .native import NativeReference, sha256


def summarize(frame, settings):
    result = {}
    for (year, policy), rows in frame.groupby(["year", "policy"]):
        metrics = {"periods": len(rows)}
        for name, cap in (("ontario_end_m", settings.ontario_max_m),
                          ("pointe_claire_m", settings.pointe_claire_max_m)):
            excess = (rows[name] - cap).clip(lower=0)
            metrics[name] = {
                "peak_m": float(rows[name].max()),
                "periods_above": int((excess > 1e-9).sum()),
                "excess_m_periods": float(excess.sum()),
            }
        metrics["mean_release_m3_s"] = float(rows.release_m3_s.mean())
        metrics["mean_abs_release_change_m3_s"] = float(rows.release_change_m3_s.mean())
        metrics["ontario_end_of_year_m"] = float(rows.ontario_end_m.iloc[-1])
        metrics["nts_forecast_mae_m3_s"] = float((rows.forecast_nts_m3_s - rows.realized_nts_m3_s).abs().mean())
        metrics["slon_forecast_mae_m3_s"] = float((rows.forecast_slon_m3_s - rows.realized_slon_m3_s).abs().mean())
        if policy == "mpc":
            metrics["mean_tracking_cost"] = float(rows.tracking_cost.mean())
            metrics["mean_terminal_cost"] = float(rows.terminal_cost.mean())
            metrics["plans_without_feasible_candidate"] = int((~rows.plan_feasible.astype(bool)).sum())
            metrics["median_plan_seconds"] = float(rows.plan_seconds.median())
            metrics["max_predicted_excess_m"] = float(rows.predicted_max_excess_m.max())
            metrics["pointe_claire_forecast_mae_m"] = float(
                (rows.predicted_pointe_claire_m - rows.pointe_claire_m).abs().mean()
            )
        result.setdefault(str(year), {})[policy] = metrics
    return result


def compare(reference, frame, years, settings):
    model = OntarioModel(reference)
    baseline = CausalOntarioSession(model, frame)
    rule = Plan2014(reference)
    controller = RecedingHorizonSearch(
        OntarioPlanningAdapter(model, settings), beam_width=settings.beam_width,
    )
    records = []

    for year in years:
        while baseline.information()[1].year < year:
            decision, time = baseline.information()
            baseline.step(rule.action(baseline.state, decision.forecast, time))

        # The input frame is read-only. Each branch advances its own state and index.
        mpc = copy(baseline)
        for policy, session in (("rule_curve", baseline), ("mpc", mpc)):
            for _ in range(48):
                decision, time = session.information()
                previous_release = session.state.previous_release_m3_s
                row = {"year": time.year, "quarter_month": time.period, "policy": policy,
                       "ontario_start_m": session.state.level_m}
                if policy == "mpc":
                    started = perf_counter()
                    plan = controller.plan(session.state, decision, time)
                    row.update(plan_seconds=perf_counter() - started, plan_feasible=plan.feasible,
                               predicted_max_excess_m=plan.max_excess_m,
                               predicted_pointe_claire_m=plan.steps[0].pointe_claire_m)
                    row.update(tracking_cost=plan.tracking_cost, terminal_cost=plan.terminal_cost)
                    excesses = []
                    for lead, step in enumerate(plan.steps, start=1):
                        upstream_margin = settings.ontario_max_m - step.state.level_m
                        downstream_margin = settings.pointe_claire_max_m - step.pointe_claire_m
                        row[f"ontario_margin_qm{lead}_m"] = upstream_margin
                        row[f"pointe_claire_margin_qm{lead}_m"] = downstream_margin
                        excesses.extend((max(0.0, -upstream_margin), max(0.0, -downstream_margin)))
                    if plan.feasible != (max(excesses) <= 1e-9):
                        raise RuntimeError("Plan feasibility disagrees with forecast stage margins")
                    action = plan.actions[0]
                else:
                    action = rule.action(session.state, decision.forecast, time)
                realized = frame.iloc[session.index]
                result = session.step(action)
                release = result.metrics["executed_release_m3_s"]
                if policy == "mpc" and release != plan.steps[0].release_m3_s:
                    raise RuntimeError("Planned and executed first releases differ")
                row.update(
                    ontario_end_m=result.state.level_m,
                    pointe_claire_m=result.metrics["pointe_claire_level_m"],
                    request_m3_s=action.m3_s,
                    release_m3_s=release,
                    release_change_m3_s=abs(release - previous_release),
                    flow_regime=result.info["flow_regime"],
                    forecast_nts_m3_s=decision.forecast.nts_m3_s[0],
                    forecast_slon_m3_s=decision.forecast.slon_m3_s,
                    realized_nts_m3_s=float(realized.ontNTS),
                    realized_slon_m3_s=float(realized.stlouisontOut),
                )
                records.append(row)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Ontario MPC and the causal rule curve on matched historical years."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--years", type=int, nargs="+", default=[1995, 2017, 2019])
    args = parser.parse_args()
    years = sorted(set(args.years))
    if any(not 1901 <= year <= 2020 for year in years):
        parser.error("Years must be in 1901–2020")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    reference = NativeReference(args.checkout)
    settings = MPCSettings()
    rows = compare(reference, load_inputs(reference), years, settings)
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / "comparison.csv"
    rows.to_csv(trace, index=False)
    summary = {
        "study_settings": vars(settings),
        "horizon_quarter_months": 4,
        "forecast": "Shared past-only persistence and trailing annual mean",
        "initialization": "Both controllers start each year from the continuous causal rule-curve state",
        "upstream": reference.provenance(),
        "trace_sha256": sha256(trace),
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("mpc.py", "planning.py", "model.py", "causal.py", "adapter.py", "data.py", "native.py")
        } | {"controllers/mpc/receding_horizon.py": sha256(Path(__file__).parents[2] / "controllers/mpc/receding_horizon.py")},
        "results": summarize(rows, settings),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["results"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
