"""Measure PPO request interventions and probe the filter at replayed states."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilevel_reservoirs.solvers.final_coupled_env import HardCoupledEnv
from bilevel_reservoirs.solvers.final_coupled_study import (
    CONFIG, DESIGN, NativeReference, OntarioModel, annual_cases, load_inputs,
    load_protocol, start_states,
)
from bilevel_reservoirs.solvers.fixed_schedule_replay import digest

SAMPLES = ("events", "reserved", "confirmation")
PROBE_PERIODS = (5, 12, 16, 20, 24, 28, 33, 40, 46)
RESIDUALS = tuple(range(-2000, 2001, 500))
FLOW_TOLERANCE = 1e-5


def differences(archive):
    rows = pd.read_csv(archive)
    ppo = rows.loc[rows.arm.eq("ppo")].copy()
    executed = ppo.ontario_applied_m3_s.notna()
    if not ppo.requested_release_m3_s.notna().all():
        raise ValueError(f"PPO request missing in {archive}")
    ppo["change_m3_s"] = ppo.ontario_applied_m3_s - ppo.requested_release_m3_s
    np.testing.assert_allclose(ppo.loc[executed, "change_m3_s"].abs(),
                               ppo.loc[executed, "intervention_m3_s"], rtol=0, atol=FLOW_TOLERANCE)
    assert ppo.loc[~executed, "intervention_m3_s"].isna().all()
    external = rows.loc[rows.arm.eq("external_mpc")].set_index(["case", "period"]).ontario_applied_m3_s
    ppo["external_release_m3_s"] = ppo.set_index(["case", "period"]).index.map(external)
    return ppo


def probe_flood_cases(trace, costs, config, design):
    years = sorted(trace.year.unique().tolist())
    reference = NativeReference()
    model = OntarioModel(reference)
    frame = load_inputs(reference)
    states = start_states(model, frame, years)
    cases = annual_cases(frame, states, years, config, design)
    entries = []
    env = HardCoupledEnv(model, cases, costs, recovery_mode="peak_protected")
    for index, case in enumerate(cases):
        for seed, rows in trace[trace.case.eq(case["name"])].groupby("seed"):
            env.reset(options={"case": index})
            for archived in rows.sort_values("period").itertuples(index=False):
                neutral = env.neutral_request()
                residual = archived.requested_release_m3_s - neutral
                assert env.action_space.low[0] - FLOW_TOLERANCE <= residual <= env.action_space.high[0] + FLOW_TOLERANCE
                if archived.period in PROBE_PERIODS:
                    for offset in RESIDUALS:
                        request = max(0., neutral + offset)
                        plan, _, stage, _ = env.plan("ppo", request, neutral)
                        entries.append({
                            "case": case["name"], "seed": seed, "period": archived.period,
                            "neutral_release_m3_s": neutral, "residual_m3_s": offset,
                            "request_m3_s": request,
                            "planned_first_release_m3_s": None if plan is None else plan.ontario_executable_release_m3_s[0],
                            "recovery_stage": stage, "archived_ppo_release_m3_s": archived.ontario_applied_m3_s,
                        })
                _, _, done, _, row = env.apply("ppo", residual)
                assert row["status"] == archived.status and row["period"] == archived.period
                assert done == (archived.period == len(rows))
                assert row["recovery_stage"] == archived.recovery_stage
                for field, tolerance in (
                    ("ontario_start_m", 1e-6), ("ottawa_start_storage_m3", env.tolerance.storage_m3),
                    ("ontario_applied_m3_s", env.tolerance.flow_m3_s),
                    ("ottawa_executed_release_m3_s", env.tolerance.flow_m3_s),
                ):
                    np.testing.assert_allclose(row[field], getattr(archived, field), rtol=0, atol=tolerance)
    return pd.DataFrame(entries)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive_root = Path("results/loslr")
    sources = {name: archive_root / f"final_recovery_peak_protected_{name}_v1/trajectories.csv"
               for name in SAMPLES}
    hashes = {str(path): digest(path) for path in [*sources.values(), CONFIG, DESIGN]}
    traces = {name: differences(file) for name, file in sources.items()}
    config, design = (json.loads(path.read_text()) for path in (CONFIG, DESIGN))
    costs = {**load_protocol(), "incomplete_penalty": design["incomplete_penalty"]}
    probes = probe_flood_cases(traces["events"], costs, config, design)
    counts = probes.groupby(["case", "seed", "period"]).agg(
        proposals=("request_m3_s", "nunique"), successful=("planned_first_release_m3_s", "count"),
        distinct_releases=("planned_first_release_m3_s", lambda release: release.round(4).nunique()),
    )
    assert counts.successful.eq(len(RESIDUALS)).all() and counts.proposals.eq(len(RESIDUALS)).all()
    summary = []
    for name, frame in traces.items():
        executed = frame[frame.change_m3_s.notna()]
        delta = executed.change_m3_s
        comparable = executed.external_release_m3_s.notna()
        gap = (executed.ontario_applied_m3_s - executed.external_release_m3_s).loc[comparable]
        summary.append({
            "sample": name, "executed": len(executed), "stopped_before_execution": len(frame) - len(executed),
            "changed": int(delta.abs().gt(FLOW_TOLERANCE).sum()), "changed_at_least_100": int(delta.abs().ge(100).sum()),
            "reduced_at_least_100": int(delta.le(-100).sum()), "raised_at_least_100": int(delta.ge(100).sum()),
            "median_absolute_m3_s": delta.abs().median(), "p90_absolute_m3_s": delta.abs().quantile(.9),
            "mpc_comparable_periods": len(gap), "mpc_same_release": int(gap.abs().le(FLOW_TOLERANCE).sum()),
        })
    args.output.mkdir(parents=True, exist_ok=False)
    pd.concat(traces, names=["sample"]).reset_index(level=0)[
        ["sample", "case", "seed", "period", "status", "recovery_stage", "requested_release_m3_s",
         "ontario_applied_m3_s", "change_m3_s", "external_release_m3_s"]
    ].to_csv(args.output / "requests.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    probes.to_csv(args.output / "probes.csv", index=False)
    counts.to_csv(args.output / "probe_states.csv")
    assert hashes == {name: digest(Path(name)) for name in hashes}
    (args.output / "provenance.json").write_text(json.dumps({
        "sources_sha256": hashes, "probe_periods": PROBE_PERIODS, "probe_residuals_m3_s": RESIDUALS,
        "flow_tolerance_m3_s": FLOW_TOLERANCE,
    }, indent=2) + "\n")
    lines = [
        "# PPO release requests and the shared filter", "",
        "The request is the archived nonnegative Ontario release requested by the frozen PPO policy. "
        "The executed release includes the full shared schedule selection and safety checks. "
        "Stops with no executed action are counted separately. A 100 m³/s threshold distinguishes "
        "substantial edits from small differences.", "",
        "| Sample | Executed | Stops | Changed by ≥100 m³/s | Reduced by ≥100 m³/s | Raised by ≥100 m³/s | Median absolute edit | 90th percentile edit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['sample']} | {row['executed']:,} | {row['stopped_before_execution']} | "
            f"{row['changed_at_least_100']:,} ({row['changed_at_least_100']/row['executed']:.1%}) | "
            f"{row['reduced_at_least_100']:,} | {row['raised_at_least_100']:,} | "
            f"{row['median_absolute_m3_s']:,.0f} m³/s | {row['p90_absolute_m3_s']:,.0f} m³/s |")
    singleton = counts.distinct_releases.eq(1).sum()
    lines += [
        "", f"At {len(counts)} fixed inherited states from 2017 and 2019, we applied nine residual proposals "
        "spanning the PPO action range from −2,000 to +2,000 m³/s. "
        f"All nine selected the same first release at {singleton} states. The median number of distinct "
        f"first releases was {counts.distinct_releases.median():g} out of nine. "
        "The replay reproduced all archived flood-year releases and stages before probing. "
        "Probes form plans at the archived state and do not advance the simulation.", "",
        "The per-period external MPC release provides context in the CSV. PPO and MPC may reach "
        "different inherited states, so these paired-period values do not identify a policy effect. "
        "The fixed-state grid checks compression within PPO's proposal space. It is a diagnostic "
        "on the two flood replays and does not establish how often this happens in training.", "",
        "Next, measure the same compression on development trajectories before designing a matched "
        "feasible-schedule PPO comparison.", "",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
