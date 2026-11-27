"""Score early Pointe-Claire headroom alerts on issue-time Ontario release plans."""

from argparse import ArgumentParser
from copy import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_control.controllers.mpc.receding_horizon import RecedingHorizonSearch
from .data import load_inputs
from .model import ReleaseRequest
from .native import INPUT_FILE, NativeReference, sha256
from .planning import MPCSettings
from .release_timing_study import paths_for
from .slon_mpc_replay import SLONReplayAdapter, replace_slon, start_sessions
from .slon_warning import FALSE_ALARM_BUDGET, fit_score, metrics

FIT = (1981, 1990)
CALIBRATION = (1991, 2000)
LEADS = (2, 3, 4)
SHIFT_M3_S = 200.
TOL = 1e-8
MODELS = ("persistence", "seasonal", "fitted_headroom")
FEATURES = ["persistence", "seasonal", "early_headroom_m", "planned_release_change_m3_s",
            "issue_roughness", "season_sin", "season_cos"]


def pointe_claire(reference, roughness, release, slon):
    return reference.utils.round_d(16.57 + (roughness * (release + slon) / 604.) ** .58, 2)


def issue_rows(reference, frame, settings, bank, archive, years):
    starts = start_sessions(reference, frame, years)
    output = []
    for delay in (0, 1):
        persistence, seasonal = (paths_for(bank, delay, name) for name in ("persistence", "seasonal_history"))
        planner = RecedingHorizonSearch(SLONReplayAdapter(next(iter(starts.values())).model, settings, persistence),
                                        beam_width=settings.beam_width)
        for year in years:
            session = copy(starts[year])
            archived = archive[archive.year.eq(year) & archive.delay.eq(delay)].sort_values("period")
            assert archived.period.tolist() == list(range(1, 49))
            for row in archived.itertuples(index=False):
                causal, time = session.information()
                state = session.state
                decision = replace_slon(causal, persistence[time.index][0])
                plan = planner.plan(state, decision, time)
                base_release = plan.steps[0].release_m3_s
                np.testing.assert_allclose(
                    [state.level_m, state.previous_release_m3_s, plan.actions[0].m3_s, base_release],
                    [row.ontario_start_m, row.previous_release_m3_s, row.requested_release_m3_s, row.release_m3_s],
                    rtol=0, atol=TOL,
                )
                current = frame.iloc[session.index]
                roughness = decision.pointe_claire_roughness
                cap = settings.pointe_claire_max_m
                extra = base_release + SHIFT_M3_S
                projected = session.model.release_limits(state, ReleaseRequest(extra), time, decision=decision)["ontFlow"] * 10
                early_pred = pointe_claire(reference, roughness, extra, decision.forecast.slon_m3_s)
                early_actual = pointe_claire(reference, current.ptclaireR, extra, current.stlouisontOut)
                can_act = early_pred <= cap + TOL and projected >= extra - TOL
                for lead in LEADS:
                    target_index = session.index + lead - 1
                    if target_index >= len(frame):
                        continue
                    target = frame.iloc[target_index]
                    # Hold the issued release fixed in all scores and labels.
                    release = plan.steps[lead - 1].release_m3_s
                    actual = pointe_claire(reference, target.ptclaireR, release, target.stlouisontOut)
                    output.append({
                        "issue_year": year, "issue_qm": time.period, "target_year": int(target.Year),
                        "target_qm": int(target.QM), "lead": lead, "delay": delay, "planned_release_m3_s": release,
                        "planned_release_change_m3_s": release - base_release,
                        "persistence": pointe_claire(reference, roughness, release,
                                                        persistence[time.index][lead - 1]) - cap,
                        "seasonal": pointe_claire(reference, roughness, release,
                                                     seasonal[time.index][lead - 1]) - cap,
                        "early_headroom_m": cap - early_pred, "issue_roughness": roughness,
                        "can_act": can_act, "early_realized_headroom_m": cap - early_actual,
                        "target_counterfactual_headroom_m": cap - actual,
                        "exceedance": actual > cap + TOL,
                        "local_opportunity": actual > cap + TOL and early_actual <= cap + TOL,
                    })
                session.state = session.model.advance_storage(
                    state, base_release, current.ontNTS, current.iceInd, time,
                )
                session.index += 1
                np.testing.assert_allclose(session.state.level_m, row.ontario_end_m, rtol=0, atol=TOL)
        print(f"Reconstructed issue-time release plans for {len(years)} years at delay {delay}", flush=True)
    return pd.DataFrame(output)


def scored_alerts(issues, reserved, events):
    phase = 2 * np.pi * (issues.target_qm - 1) / 48
    matching = issues.groupby(["issue_year", "issue_qm", "lead"]).delay.transform("size").eq(2)
    observed = issues.assign(season_sin=np.sin(phase), season_cos=np.cos(phase))[matching]
    results, alerts, records = [], [], {}
    for (delay, lead), rows in observed.groupby(["delay", "lead"]):
        rows = rows[rows.can_act].copy()
        # Restrict the target to within-year shifts and keep training targets
        # within their split. No cross-year move can satisfy an annual gate.
        rows = rows[rows.issue_year.eq(rows.target_year)]
        train = rows[rows.issue_year.between(*FIT)]
        rows["fitted_headroom"], fit = fit_score(train, rows, FEATURES, label="local_opportunity")
        negative = rows[rows.issue_year.between(*CALIBRATION) & ~rows.local_opportunity]
        record = {"fit": fit, "training_issues": len(train), "cutoffs": {}}
        for model in MODELS:
            cutoff = float(np.nextafter(negative[model].quantile(1. - FALSE_ALARM_BUDGET,
                                                                  interpolation="higher"), np.inf))
            rows[f"{model}_alert"] = rows[model].ge(cutoff)
            record["cutoffs"][model] = cutoff
            for cohort, years in (("reserved", reserved), ("flood_years", events)):
                sample = rows[rows.issue_year.isin(years)]
                results.append({"cohort": cohort, "delay": delay, "lead": lead,
                                **metrics(sample, model, cutoff, label="local_opportunity")})
        alerts.append(rows)
        records[f"delay_{delay}_lead_{lead}"] = record
    return pd.DataFrame(results), pd.concat(alerts, ignore_index=True), records


def heldout_totals(scored, reserved):
    results = []
    for delay, data in scored[scored.issue_year.isin(reserved)].groupby("delay"):
        positives = data[data.local_opportunity]
        keys = ["target_year", "target_qm"]
        for model in MODELS:
            col = f"{model}_alert"
            warned = positives.groupby(keys)[col].any()
            negative = data[~data.local_opportunity]
            false = negative[col].sum()
            results.append({"delay": delay, "model": model, "positive_issue_lead_pairs": len(positives),
                            "unique_target_periods": len(warned), "warned_periods": int(warned.sum()),
                            "false_alarms": int(false), "negative_issue_lead_pairs": len(negative),
                            "false_alarm_rate": false / len(negative)})
    return pd.DataFrame(results)


def report(totals, scored):
    lines = [
        "# Pointe-Claire headroom alerts under an issued Ontario plan", "",
        "Each lead uses the Ontario release planned at the issue time by the unchanged persistence MPC. "
        "That release stays fixed in the forecast and retrospective stage calculation. Only the realized "
        "SLON and roughness enter the retrospective label. The stage is a model counterfactual at the "
        "issued release, rather than an observed stage at that release.", "",
        "An eligible issue can add 200 m³/s immediately under the causal native limits and forecast "
        "Pointe-Claire headroom. The positive label requires that this addition also fits with realized "
        "SLON and roughness, and that a later issued-plan release exceeds the 22.10 m study threshold. "
        "This establishes a local opportunity. The later cut, year-end balance and whole-year exposure "
        "remain to be checked in a controller replay.", "",
        "The fitted alert score uses forecast headroom, issued release changes, current spare headroom, "
        "roughness and season. It fits on 1981–1990. For each score, calibration on 1991–2000 selects "
        "a threshold targeting at most 10% false alarms. The 18 reserved years and 2017 and 2019 "
        "remain outside both steps. Delays withhold the same SLON periods as the archived study.", "",
        "| Delay | Score | Later periods with a local opportunity | Warned at any lead | False alarms |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in totals.itertuples(index=False):
        lines.append(f"| {row.delay} | {row.model} | {row.unique_target_periods} | "
                     f"{row.warned_periods} | {row.false_alarms} ({row.false_alarm_rate:.1%}) |")
    passed = []
    for _, group in totals.groupby("delay"):
        fit = group[group.model.eq("fitted_headroom")].iloc[0]
        baseline = group[~group.model.eq("fitted_headroom")]
        passed.append(fit.warned_periods > baseline.warned_periods.max()
                      and fit.false_alarm_rate <= baseline.false_alarm_rate.min() + .02)
    lines += ["", "Affected periods are unique within each delay. Each delay has its own issued Ontario "
              "plans and thus its own local opportunities. False alarms are counted per issue and lead. "
              "The gate requires more unique periods warned than either comparator at both delays, "
              "with false alarms within two percentage points of the lower comparator rate.", "",
              f"Fitted alert passed the gate: {len(passed) == 2 and all(passed)}. "
              "This benchmark does not change controller inputs or release decisions.", "",
              f"The scored dataset contains {len(scored):,} eligible issue/lead pairs across all splits.", ""]
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/loslr/slon_mpc_replay_verified_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {name: args.archive / name for name in ("protocol.json", "forecast_bank.csv", "trajectories.csv")}
    hashes = {name: sha256(path) for name, path in files.items()}
    protocol = json.loads(files["protocol.json"].read_text())
    cohorts = protocol["cohorts"]
    years = sorted({year for group in cohorts.values() for year in group})
    reference = NativeReference()
    assert sha256(reference.root / INPUT_FILE) == protocol["source_sha256"]
    frame = load_inputs(reference)
    bank = pd.read_csv(files["forecast_bank.csv"])
    archive = pd.read_csv(files["trajectories.csv"]).query("method == 'persistence'")
    settings = MPCSettings(**protocol["settings"])
    args.output.mkdir(parents=True, exist_ok=False)
    issues = issue_rows(reference, frame, settings, bank, archive, years)
    scores, alerts, fits = scored_alerts(issues, cohorts["reserved"], [2017, 2019])
    totals = heldout_totals(alerts, cohorts["reserved"])
    assert hashes == {name: sha256(path) for name, path in files.items()}
    for name, data in (("issues", issues), ("scores", scores), ("alerts", alerts), ("heldout_totals", totals)):
        data.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "archive_sha256": hashes, "settings": protocol["settings"], "fit": FIT, "calibration": CALIBRATION,
        "leads": LEADS, "shift_m3_s": SHIFT_M3_S, "false_alarm_budget": FALSE_ALARM_BUDGET, "fits": fits,
    }, indent=2) + "\n")
    text = report(totals, alerts)
    (args.output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
