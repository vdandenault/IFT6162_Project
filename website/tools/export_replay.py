"""Export verified 2017 and 2019 event runs for the static flood replay."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

POLICIES = {"plan2014": "rule_curve", "ordinary": "external_mpc", "anticipating": "stackelberg_mpc"}
YEARS = (2017, 2019)
SERIES = {
    "lake_ontario_level_m": "ontario_end_m",
    "lake_st_louis_level_m": "pointe_claire_m",
    "lake_ontario_release_m3_s": "ontario_applied_m3_s",
    "ottawa_release_m3_s": "ottawa_executed_release_m3_s",
    "lake_st_louis_limit_m": "downstream_target_m",
    "ottawa_storage_m3": "ottawa_end_storage_m3",
}
FORCING = (
    "design",
    "ontario_realized_supply_m3_s",
    "ontario_forecast_supply_m3_s",
    "ottawa_net_inflow_m3_s",
    "ottawa_original_forecast_inflow_m3_s",
    "background_m3_s",
    "capacity_m3_s",
)
START = (
    "ontario_start_m",
    "ontario_previous_release_m3_s",
    "ontario_previous_ice",
    "ottawa_start_storage_m3",
    "ottawa_start_queue_m3",
)


def read_csv(path):
    """Read the private simulator CSV with its original column names."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_sources(repository, revision, hashes):
    """Match the run's code and configuration hashes to the named Git revision."""
    for original_path, expected in hashes.items():
        parts = original_path.replace("\\", "/").split("/")
        start = max(i for i, part in enumerate(parts[:-1]) if part in {"configs", "src"})
        relative = "/".join(parts[start:])
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{revision}:{relative}"]
        )
        # A Windows checkout can record CRLF while Git stores LF.
        lf = blob.replace(b"\r\n", b"\n")
        if expected not in {
            hashlib.sha256(b).hexdigest() for b in (lf, lf.replace(b"\n", b"\r\n"))
        }:
            raise ValueError(f"{relative} differs from {revision}")


def period_metrics(rows, lake_limit):
    """Calculate peaks and exposure against each period's applicable study target."""
    metrics = {}
    for site, level_key in (
        ("lake_ontario", "ontario_end_m"),
        ("lake_st_louis", "pointe_claire_m"),
    ):
        levels = [float(row[level_key]) for row in rows]
        limits = (
            [lake_limit] * len(rows)
            if site == "lake_ontario"
            else [float(row["downstream_target_m"]) for row in rows]
        )
        excess = [max(0.0, h - limit) for h, limit in zip(levels, limits, strict=True)]
        metrics[f"{site}_peak_m"] = max(levels)
        metrics[f"{site}_periods_above"] = sum(x > 1e-9 for x in excess)
        metrics[f"{site}_exposure_m_qm"] = sum(excess)
    return metrics


def may_flow_components(rows):
    """Check the one-step arrival and confluence balance, then select the late-May example."""
    prior_release = float(rows[0]["ottawa_start_transit_m3"]) / 657_000
    for row in rows:
        arrival = float(row["ottawa_arrival_m3_s"])
        total = float(row["ontario_applied_m3_s"]) + arrival + float(row["background_m3_s"])
        if (
            abs(arrival - prior_release) > 1e-5
            or abs(total - float(row["st_louis_flow_m3_s"])) > 1e-5
        ):
            raise ValueError(f"Routing mismatch in {row['case']}, quarter-month {row['period']}")
        prior_release = float(row["ottawa_executed_release_m3_s"])
    may_end = rows[19]  # Fourth quarter-month of May.
    return {
        name: round(float(may_end[name]), 6)
        for name in ("ottawa_arrival_m3_s", "background_m3_s", "st_louis_flow_m3_s")
    }


def checked_case(rows, summary, lake_limit):
    """Reconcile an entire episode before selecting the public series and April–June impacts."""
    periods = list(range(1, 49))
    if (
        summary["status"] != "completed"
        or int(summary["executed_periods"]) != 48
        or int(summary["planned_periods"]) != 48
        or any(row["status"] != "executed" for row in rows)
        or any([int(r[field]) for r in rows] != periods for field in ("period", "quarter_month"))
    ):
        raise ValueError(f"Incomplete or unordered case: {summary['case']} {summary['arm']}")

    annual = period_metrics(rows, lake_limit)
    checks = {
        "full_cost": sum(float(r["cost"]) for r in rows),
        "upstream_exposure_m_periods": annual["lake_ontario_exposure_m_qm"],
        "downstream_exposure_m_periods": annual["lake_st_louis_exposure_m_qm"],
        "energy_value": -sum(float(r["negative_energy_value"]) for r in rows),
        "ottawa_end_storage_m3": float(rows[-1]["ottawa_end_storage_m3"]),
        # The published reducer scores delivery below 900 m³/s in 657,000-second periods.
        "ottawa_service_shortfall_million_m3": sum(
            max(0.0, 900.0 - float(r["ottawa_executed_release_m3_s"])) * 657_000 / 1e6 for r in rows
        ),
    }
    for name, value in checks.items():
        if not math.isclose(value, float(summary[name]), abs_tol=1e-8, rel_tol=1e-12):
            raise ValueError(f"{summary['case']} {summary['arm']}: {name} differs from runs.csv")

    spring = rows[12:24]  # QMs 13–24 inclusive (four periods per month).
    impacts = period_metrics(spring, lake_limit)
    impacts.update(
        upstream_peak_buildings=max(float(r["upstream_coastal"]) for r in spring),
        downstream_peak_buildings=max(float(r["downstream_coastal"]) for r in spring),
        hydropower_value_usd=-sum(float(r["negative_energy_value"]) for r in spring),
        boating_cost_usd=sum(float(r["recreational_boating"]) for r in spring),
    )
    record = {}
    for name, key in SERIES.items():
        values = [float(r[key]) for r in rows]
        # The headline compares executed releases for equality before display rounding.
        record[name] = (
            values if name == "lake_ontario_release_m3_s" else [round(v, 6) for v in values]
        )
    record.update(
        may_end_flow=may_flow_components(rows),
        annual_cost=round(checks["full_cost"], 9),
        annual_ottawa_service_shortfall_million_m3=round(
            checks["ottawa_service_shortfall_million_m3"], 6
        ),
        annual_ottawa_end_storage_m3=round(checks["ottawa_end_storage_m3"], 3),
        spring={name: round(value, 8) for name, value in impacts.items()},
    )
    return record


def export(root, repository, revision):
    """Read the three event-run directories and compare their shared experimental conditions."""
    years = {str(year): {"policies": {}} for year in YEARS}
    matched_conditions = {}
    source_hashes = {}
    shared_plan = None
    for policy, arm in POLICIES.items():
        directory = root / f"{arm}_events"
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        if (
            plan["mode"] != "development"
            or plan["years"] != list(YEARS)
            or plan["controllers"] != [arm]
        ):
            raise ValueError(f"Expected the two event-only development years for {arm}")
        check_sources(repository, revision, plan["sources_sha256"])
        settings = {
            k: plan[k]
            for k in ("costs", "design", "recovery", "upstream", "candidate_offsets_m3_s")
        }
        if shared_plan is not None and settings != shared_plan:
            raise ValueError("Experimental settings differ across controllers")
        shared_plan = settings
        lake_limit = plan["costs"]["ontario_flood_cap_m"]
        summaries = read_csv(directory / "runs.csv")
        trace_file = directory / "trajectories.csv"
        all_rows = read_csv(trace_file)
        source_hashes[policy] = hashlib.sha256(trace_file.read_bytes()).hexdigest()
        for year in YEARS:
            case = f"annual_{year}"
            (summary,) = (r for r in summaries if r["case"] == case and r["arm"] == arm)
            rows = [r for r in all_rows if r["case"] == case and r["arm"] == arm]
            if int(summary["year"]) != year or any(int(r["year"]) != year for r in rows):
                raise ValueError(f"{case} has a mismatched calendar year")
            result = checked_case(rows, summary, lake_limit)
            conditions = (
                [rows[0][key] for key in START],
                [[r[key] for key in FORCING] for r in rows],
            )
            if conditions != matched_conditions.setdefault(year, conditions):
                raise ValueError(f"Initial state, forcing or issued forecasts differ for {year}")
            years[str(year)]["design"] = summary["design"]
            years[str(year)]["policies"][policy] = result
    return {
        "study": "Matched flood-year development replays",
        "revision": revision,
        "native_revision": shared_plan["upstream"]["commit"],
        "recovery": shared_plan["recovery"],
        "trace_sha256": source_hashes,
        "lake_ontario_limit_m": lake_limit,
        "years": years,
    }


def main():
    """Validate the input and atomically replace the compact JSON used by the website."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", type=Path, help="Directory containing the three <controller>_events directories"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Team Git checkout containing the simulator revision",
    )
    parser.add_argument("--revision", required=True, help="Commit used to generate the event runs")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", "--verify", f"{args.revision}^{{commit}}"],
        text=True,
    ).strip()
    replay = export(args.runs, args.repository, revision)
    payload = json.dumps(
        replay,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload + "\n")
    try:
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    for year, case in replay["years"].items():
        policies = case["policies"]
        ordinary = policies["ordinary"]["lake_ontario_release_m3_s"]
        anticipating = policies["anticipating"]["lake_ontario_release_m3_s"]
        matches = sum(a == b for a, b in zip(ordinary, anticipating, strict=True))
        print(
            f"{year}: identical executed Lake Ontario releases on {matches} of {len(ordinary)} dates"
        )


if __name__ == "__main__":
    main()
