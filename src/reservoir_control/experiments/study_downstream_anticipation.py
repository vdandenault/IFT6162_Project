import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from reservoir_control.evaluation.provenance import digest
from reservoir_control.experiments.final_coupled_env import HardCoupledEnv
from reservoir_control.experiments.final_coupled_study import (
    ARMS,
    CONFIG,
    NativeReference,
    OntarioModel,
    annual_cases,
    load_inputs,
    load_protocol,
    start_states,
    stress_cases,
)
from reservoir_control.experiments.fixed_schedule_replay import json_value
from reservoir_control.study_config import load_follower_config, load_settings

LOG = logging.getLogger(__name__)

ARCHIVES = ("events", "reserved", "confirmation")
COST_FIELDS = ["full_cost", "upstream_exposure_m_periods", "downstream_exposure_m_periods"]


def screen_case(model, case, costs, trace, settings):
    """Reproduce external MPC, auditing a frozen set of exposure-prone decisions."""
    trace = trace.sort_values("period").reset_index(drop=True)
    slack = trace.capacity_m3_s - trace.st_louis_flow_m3_s
    stage_slack = trace.downstream_target_m - trace.pointe_claire_m
    near = (slack <= settings["flow_headroom_m3_s"]) | (stage_slack <= settings["stage_headroom_m"])
    env = HardCoupledEnv(model, [case], costs, recovery_mode="peak_protected")
    env.reset(options={"case": 0})
    comparisons, candidates = [], []
    for k, archived in enumerate(trace.itertuples(index=False)):
        # Archived outcomes select diagnostic states offline. The planner sees
        # exactly its ordinary decision-time inputs and issued forecast.
        potential = near.iloc[k : k + 4].any() or archived.status != "executed"
        audit = []
        if potential:
            request = env.neutral_request()
            plan, _, stage, _ = env.plan("stackelberg_mpc", request, request, candidate_audit=audit)
        _, _, done, _, row = env.apply("external_mpc")
        assert row["status"] == archived.status and row["period"] == archived.period
        assert done == (k == len(trace) - 1)
        np.testing.assert_allclose(
            row["ottawa_start_storage_m3"],
            archived.ottawa_start_storage_m3,
            rtol=0,
            atol=env.tolerance.storage_m3,
        )
        np.testing.assert_allclose(
            row["ottawa_start_queue_m3"],
            json.loads(archived.ottawa_start_queue_m3),
            rtol=0,
            atol=env.tolerance.storage_m3,
        )
        np.testing.assert_allclose(
            row["ontario_start_m"], archived.ontario_start_m, rtol=0, atol=1e-6
        )
        if pd.notna(archived.ontario_applied_m3_s):
            for field in (
                "ontario_applied_m3_s",
                "ottawa_executed_release_m3_s",
                "st_louis_flow_m3_s",
            ):
                np.testing.assert_allclose(
                    row[field], getattr(archived, field), rtol=0, atol=env.tolerance.flow_m3_s
                )
        if not potential:
            continue
        key = {"case": case["name"], "year": case["year"], "period": archived.period}
        candidates.extend({**key, **candidate} for candidate in audit)
        entry = {
            **key,
            "stage": stage,
            "eligible_candidates": len(audit),
            "schedule_changed": False,
            "first_action_changed": False,
            "scores_changed": False,
            "max_downstream_stage_change_m": 0.0,
            "downstream_excess_changed": False,
            "tight_with_routed_variation": False,
        }
        if audit:
            a = min(audit, key=lambda x: (x["external_score"], x["schedule_m3_s"]))
            b = min(audit, key=lambda x: (x["response_score"], x["schedule_m3_s"]))
            np.testing.assert_array_equal(plan.ontario_executable_release_m3_s, b["schedule_m3_s"])
            np.testing.assert_allclose(
                a["schedule_m3_s"][0],
                row["ontario_applied_m3_s"],
                rtol=0,
                atol=env.tolerance.flow_m3_s,
            )
            assert stage == row["recovery_stage"]
            room = np.array([x["downstream_headroom_m3_s"] for x in audit])
            arrivals = np.array([x["ottawa_arrival_m3_s"][: room.shape[1]] for x in audit])
            ranges = np.ptp(arrivals, axis=0)
            stage_change = np.array([x["downstream_stage_m"] for x in audit]) - np.array(
                [x["nominal_downstream_stage_m"] for x in audit]
            )
            entry.update(
                min_forecast_headroom_m3_s=room.min(),
                max_routed_spread_m3_s=ranges.max(),
                max_downstream_stage_change_m=np.abs(stage_change).max(),
                tight_with_routed_variation=bool(
                    np.any(
                        (room.min(axis=0) <= settings["flow_headroom_m3_s"])
                        & (ranges >= settings["meaningful_routed_difference_m3_s"])
                    )
                ),
                scores_changed=any(
                    abs(x["external_score"] - x["response_score"]) > 1e-9 for x in audit
                ),
                downstream_excess_changed=any(
                    not np.allclose(
                        x["downstream_excess_m"],
                        x["nominal_downstream_excess_m"],
                        atol=1e-8,
                        rtol=0,
                    )
                    for x in audit
                ),
                schedule_changed=a["schedule_m3_s"] != b["schedule_m3_s"],
                first_action_changed=abs(a["schedule_m3_s"][0] - b["schedule_m3_s"][0])
                > env.tolerance.flow_m3_s,
                external_first_m3_s=a["schedule_m3_s"][0],
                stackelberg_first_m3_s=b["schedule_m3_s"][0],
            )
        else:
            assert plan is None and row["status"] != "executed"
        comparisons.append(entry)
    return comparisons, candidates


def paired_rollouts(model, cases, costs):
    results, trace = [], []
    env = HardCoupledEnv(model, cases, costs, recovery_mode="peak_protected")
    for index, case in enumerate(cases):
        for arm in ("external_mpc", "stackelberg_mpc"):
            env.reset(options={"case": index})
            done = False
            started = perf_counter()
            while not done:
                _, _, done, _, _ = env.apply(arm)
            runtime = perf_counter() - started
            executed = [row for row in env.records if "ontario_applied_m3_s" in row]
            complete = (
                len(executed) == len(case["rows"]) and env.records[-1]["status"] == "executed"
            )
            key = {"case": case["name"], "arm": arm}
            trace.extend({**key, **row} for row in env.records)
            results.append(
                {
                    **key,
                    "completed": complete,
                    "status": env.records[-1]["status"],
                    "executed_periods": len(executed),
                    "episode_runtime_seconds": runtime,
                    "full_cost": sum(row["cost"] for row in executed) if complete else None,
                    "upstream_exposure_m_periods": sum(
                        max(0.0, row["ontario_end_m"] - costs["ontario_flood_cap_m"])
                        for row in executed
                    )
                    if complete
                    else None,
                    "downstream_exposure_m_periods": sum(
                        max(0.0, row["pointe_claire_m"] - row["downstream_target_m"])
                        for row in executed
                    )
                    if complete
                    else None,
                    "storage_violation_m3": max(
                        (row["ottawa_storage_violation_m3"] for row in executed), default=0.0
                    ),
                    "downstream_violation_m3_s": max(
                        (row["st_louis_flow_m3_s"] - row["capacity_m3_s"] for row in executed),
                        default=0.0,
                    ),
                    "terminal_transit_m3": executed[-1]["ottawa_end_transit_m3"]
                    if executed
                    else None,
                }
            )
    return pd.DataFrame(results), pd.DataFrame(trace)


def final_comparison(tables, archived_pairs):
    """Use a common set of completed years, giving each year equal weight."""
    labels = {
        "rule_curve": "Filtered Plan 2014",
        "external_mpc": "External-ranked MPC",
        "stackelberg_mpc": "Stackelberg MPC",
        "ppo": "PPO (five-seed mean)",
    }
    lines = [
        "# Ontario–Ottawa controller comparison",
        "",
        "Filtered Plan 2014, external-ranked MPC, Stackelberg MPC and PPO share peak-protected recovery, "
        "Ontario release limits and the hard Ottawa follower. Ontario inputs are historical and Ottawa "
        "inflows, storage and routing are synthetic. Full costs and exposure are compared over complete "
        "matched years. PPO uses the five frozen checkpoints and each year contributes equally.",
        "",
        "| Sample | Controller | Completed runs | Operating cost | Ontario shoreline (m·periods) | Lake St. Louis (m·periods) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, table in tables.items():
        finished = table[
            table.status.eq("completed") & table.executed_periods.eq(table.planned_periods)
        ]
        assert not table.duplicated(["case", "arm", "seed"]).any()
        assert (table.groupby("case").size() == 8).all()
        common = finished.groupby("case").size().eq(8)
        matched = finished[finished.case.isin(common[common].index)]
        assert not matched.empty and matched.full_cost.notna().all()
        for arm in ARMS:
            group = table[table.arm == arm]
            completed = finished[finished.arm == arm]
            means = matched[matched.arm == arm].groupby("case")[COST_FIELDS].mean().mean()
            lines.append(
                f"| {label} | {labels[arm]} | {len(completed)}/{len(group)} | "
                + " | ".join(f"{value:.4f}" for value in means)
                + " |"
            )
    lines += [
        "",
        "The flood-year improvements in operating cost and Ontario shoreline exposure occur in "
        "the two highlighted years. Across the 18 annual replays, the mean trade-off for external MPC "
        "is lower cost and slightly higher shoreline exposure than filtered Plan 2014.",
        "",
        "## Existing paired MPC runs",
        "",
        "The new diagnostic checked archived decisions offline. The original peak-protected runs "
        "already provide a paired closed-loop comparison under the reference assumptions. "
        "Timings are median episode wall times on jointly completed cases and include simulation "
        "and bookkeeping.",
        "",
        "| Sample | External complete | Stackelberg complete | Joint completions | Identical executed release paths | External time (s) | Stackelberg time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, archive in archived_pairs.items():
        runs = pd.read_csv(archive / "runs.csv")
        completed = runs[runs.status.eq("completed")]
        a, b = (
            completed[completed.arm.eq(arm)].set_index("case")
            for arm in ("external_mpc", "stackelberg_mpc")
        )
        common = a.index.intersection(b.index)
        trace = pd.read_csv(archive / "trajectories.csv")
        fields = ["ontario_applied_m3_s", "ottawa_executed_release_m3_s"]
        paths = [
            trace[trace.case.isin(common) & trace.arm.eq(arm)]
            .set_index(["case", "period"])[fields]
            .sort_index()
            for arm in ("external_mpc", "stackelberg_mpc")
        ]
        assert paths[0].index.equals(paths[1].index)
        same = np.isclose(paths[0], paths[1], rtol=0.0, atol=1e-5).all(axis=1)
        count = pd.Series(same, index=paths[0].index).groupby("case").all().sum()
        lines.append(
            f"| {label} | {len(a)} | {len(b)} | {len(common)} | {count} | "
            f"{a.loc[common].runtime_seconds.median():.3f} | {b.loc[common].runtime_seconds.median():.3f} |"
        )
    lines += [
        "",
        "[2017 and 2019 figure](../ottawa_ranking_sensitivity_verified_v1/flood_years.png) · "
        "[Interactive replay](../../../background/ontario_peak_protected.html)",
        "",
    ]
    return lines


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Audit response-aware ranking near downstream limits and assemble final comparisons."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("results/loslr")
    settings = load_settings(CONFIG, "downstream_anticipation")
    design = load_settings(CONFIG, "annual")
    base = load_follower_config()
    costs = {**load_protocol(), "incomplete_penalty": design["incomplete_penalty"]}
    folders = {name: root / f"final_recovery_peak_protected_{name}_v1" for name in ARCHIVES}
    sources = [CONFIG]
    for folder in folders.values():
        sources.extend(folder / name for name in ("plan.json", "runs.csv", "trajectories.csv"))
    provenance = {str(path): digest(path) for path in sources}
    plans = {
        name: json.loads((folder / "plan.json").read_text()) for name, folder in folders.items()
    }
    assert all(
        plan["recovery"] == "peak_protected" and plan["design"] == design for plan in plans.values()
    )
    assert plans["confirmation"]["years"] == [1961, 2017, 2019]
    reference = NativeReference()
    model = OntarioModel(reference)
    frame = load_inputs(reference)
    years = set(settings["separate_annual_years"]).union(
        *(plan["years"] for plan in plans.values())
    )
    states = start_states(model, frame, sorted(years))
    cases = {
        name: annual_cases(frame, states, plans[name]["years"], base, design)
        for name in ("events", "reserved")
    }
    cases["confirmation"] = stress_cases(root, frame, states, base, CONFIG)

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    comparisons, candidates = [], []
    all_cases = {}
    for label, batch in cases.items():
        archived = pd.read_csv(folders[label] / "trajectories.csv")
        external = archived[archived.arm == "external_mpc"]
        assert set(external.case) == {case["name"] for case in batch}
        for case in batch:
            name = case["name"]
            all_cases[name] = case
            found, options = screen_case(
                model, case, costs, external[external.case == name], settings
            )
            comparisons.extend({"sample": label, **row} for row in found)
            candidates.extend(options)
        decisions = sum(row["sample"] == label for row in comparisons)
        LOG.info(
            "%s: reproduced %d rollouts and audited %d decisions", label, len(batch), decisions
        )
    summary = pd.DataFrame(comparisons)
    summary.to_csv(output / "decisions.csv", index=False)
    with (output / "candidates.jsonl").open("w") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate, default=json_value) + "\n")

    changed = summary[summary.schedule_changed].sort_values(
        ["first_action_changed", "case", "period"], ascending=[False, True, True]
    )
    selected = changed.case.drop_duplicates().head(settings["maximum_triggered_cases"]).tolist()
    lines = [
        "# Does anticipating Ottawa change Ontario's choice?",
        "",
        "We reproduced the archived external MPC trajectories for the two flood years, "
        "18 annual replays and 36 synthetic confirmation stress windows. We audited a decision "
        f"when an executed period in its four-lead window came within {settings['flow_headroom_m3_s']} m³/s "
        f"of the study flow cap or {settings['stage_headroom_m']} m of the Pointe-Claire target, "
        "and at every stopped decision. Recorded "
        "outcomes select states offline. Both planners still use their original issued inputs.",
        "",
        f"A tight lead with meaningful Ottawa variation has an eligible candidate within "
        f"{settings['flow_headroom_m3_s']} m³/s of its stricter native or study flow ceiling and at least "
        f"{settings['meaningful_routed_difference_m3_s']} m³/s of routed Ottawa variation across eligible "
        "schedules at that same lead.",
        "",
        "| Sample | Audited decisions | With eligible plans | Tight lead with routed variation | Downstream exposure changes | Candidate score changes | Winning schedule changes | First release changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in summary.groupby("sample", sort=False):
        counts = [
            len(group),
            group.eligible_candidates.gt(0).sum(),
            group.tight_with_routed_variation.sum(),
            group.downstream_excess_changed.sum(),
            group.scores_changed.sum(),
            group.schedule_changed.sum(),
            group.first_action_changed.sum(),
        ]
        lines.append(f"| {name} | " + " | ".join(map(str, counts)) + " |")
    lines.append("")
    stage_changes = summary.max_downstream_stage_change_m.gt(1e-8)
    lines += [
        f"Using the follower response changed at least one forecast Pointe-Claire level at "
        f"{stage_changes.sum()} audited decisions. The largest change was "
        f"{summary.max_downstream_stage_change_m.max():.3f} m.",
        "",
    ]
    if candidates and all(
        max(c["downstream_excess_m"] + c["nominal_downstream_excess_m"]) == 0.0 for c in candidates
    ):
        lines += [
            f"All {len(candidates):,} common eligible candidates had zero forecast exposure "
            "above the Pointe-Claire target under both flow assumptions. The operating cost "
            "only charges downstream levels above that target.",
            "",
        ]
    if selected:
        transfer = annual_cases(frame, states, settings["separate_annual_years"], base, design)
        runs, trace = paired_rollouts(
            model, [*(all_cases[name] for name in selected), *transfer], costs
        )
        runs.to_csv(output / "paired_runs.csv", index=False)
        trace.to_csv(output / "paired_trajectories.csv", index=False)
        for label, subset in (
            ("Triggered cases", runs[runs.case.isin(selected)]),
            ("Separate annual cases", runs[~runs.case.isin(selected)]),
        ):
            pairs = subset.pivot(index="case", columns="arm", values=COST_FIELDS)
            delta = (
                pairs.xs("stackelberg_mpc", axis=1, level=1)
                - pairs.xs("external_mpc", axis=1, level=1)
            ).dropna()
            complete = subset.groupby("arm").completed.sum()
            timing = subset.groupby("arm").episode_runtime_seconds.median()
            changes = (
                ", ".join(f"{name} {value:+.5f}" for name, value in delta.mean().items())
                if len(delta)
                else "unavailable"
            )
            lines += [
                f"**{label}.** External MPC completed {complete.external_mpc} and Stackelberg "
                f"completed {complete.stackelberg_mpc} of {len(pairs)}. The {len(delta)} jointly "
                f"completed pairs had mean Stackelberg-minus-external changes of {changes}. "
                f"Median episode wall times were {timing.external_mpc:.2f} s and "
                f"{timing.stackelberg_mpc:.2f} s respectively.",
                "",
            ]
    else:
        lines += [
            "The paired closed-loop comparison was conditional on a change in the winning "
            "schedule. The archived sensitivity screen did not trigger it.",
            "",
        ]
    lines += [
        "The stress windows test the modeled mechanism with synthetic Ottawa inflows. "
        "These findings apply to the tested candidate family, forecasts, cost and constraints.",
        "",
    ]
    (output / "diagnostic.md").write_text("\n".join(lines))
    events, reserved = (pd.read_csv(folders[name] / "runs.csv") for name in ("events", "reserved"))
    tables = {str(year): events[events.year == year] for year in (2017, 2019)}
    tables["18 annual replays"] = reserved
    archives = {
        label: folders[name]
        for label, name in (
            ("2017 and 2019", "events"),
            ("18 annual replays", "reserved"),
            ("Synthetic confirmation", "confirmation"),
        )
    }
    (output / "report.md").write_text("\n".join(final_comparison(tables, archives)))
    assert provenance == {name: digest(Path(name)) for name in provenance}
    (output / "provenance.json").write_text(
        json.dumps(
            {"settings": settings, "archive_sha256": provenance, "selected_cases": selected},
            indent=2,
        )
        + "\n"
    )
    LOG.info("Saved anticipation results to %s", output)


if __name__ == "__main__":
    main()
