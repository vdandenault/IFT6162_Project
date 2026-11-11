"""Run native Plan 2014, verify the adapter, and write an auditable baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .adapter import OntarioSession
from .data import baseline_parameters, input_audit, load_inputs
from .model import OntarioModel, Plan2014
from .native import NativeReference, UPSTREAM_COMMIT, sha256

# Independent native imports execute in a child interpreter whose working
# directory is the upstream checkout. Its legacy cwd/import behavior cannot
# mutate the application interpreter. No native code is copied into this repo.
_NATIVE_WORKER = r'''
import importlib, json, sys, tomllib, warnings
from pathlib import Path
import pandas as pd
from optimizationSimulation import simulation
from functions.release import ruleCurve
from functions.limits import Bv7
from functions.limits.septemberRule import septemberRule
from functions.routing import stlaw
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
settings=json.loads(sys.argv[1]); out=Path(settings["output"])
data=pd.read_csv(settings["input"],sep="\t")
data=data.loc[data.Year<=settings["last_year"]].reset_index(drop=True)
result=simulation(data,{},ruleCurve.getReleaseFunctionInputs,ruleCurve.releaseFunction,
 septemberRule,Bv7.getPlanLimitsInputs,Bv7.planLimits,
 stlaw.getStLawrenceRoutingInputs,stlaw.stLawrenceRouting,settings["parameters"])
result.to_csv(out/"native_trace.csv",index=False)
pi_status={"complete":False}
try:
 config=tomllib.loads(Path(settings["config"]).read_text())
 names=config["performanceIndicators"]["objectiveModels"]
 modules=[importlib.import_module("objectiveFunctions.legacyPIs.functions."+name) for name in names]
 objectives=importlib.import_module("objectiveFunctions.legacyPIs.objectiveSimulation")
 scored=result.iloc[48:].reset_index(drop=True)
 mapping={name:scored[name].values for name in scored}
 frames=objectives.objectiveSimulation(mapping,modules,"simulation")
 for name,frame in zip(names,frames,strict=True):frame.to_csv(out/("pi_"+name+".csv"),index=False)
 values=objectives.objectiveSimulation(mapping,modules,"netAnnualAverage")
 pi_status={"complete":True,"native_signed_objectives":dict(zip(names,[float(x) for x in values],strict=True)),
 "aggregation":"Native netAnnualAverage; maximization indicators negated as upstream defines"}
except Exception as error:
 pi_status={"complete":False,"failure_type":type(error).__name__,"failure":str(error)}
(out/"performance_indicators.json").write_text(json.dumps(pi_status,indent=2,allow_nan=False)+"\n")
'''


def adapter_trace(reference: NativeReference, frame: pd.DataFrame) -> pd.DataFrame:
    session = OntarioSession(OntarioModel(reference), frame)
    policy = Plan2014(reference)
    rows = []
    while not session.done:
        forecast, time = session.information()
        state = session.state
        action = policy.action(state, forecast, time)
        transition = session.step(action)
        route = transition.info["native_routing"]
        rows.append({
            "Year": time.year, "QM": time.period,
            "ontLevelBOQ": state.level_m,
            "ontLevelEOQ": transition.metrics["ontario_level_end_m"],
            "ontLevelMOQ": transition.state.mean_level_history_m[-1],
            "ontLevel": transition.metrics["ontario_level_mean_m"],
            "ontFlow": transition.metrics["executed_release_m3_s"] / 10,
            "flowRegime": transition.info["flow_regime"],
            "rfFlow": action.rulecurve_flow_m3_s / 10,
            "rfRegime": action.rulecurve_regime,
            "pprFlow": action.preproject_flow_m3_s / 10,
            "rfOutput": action.rulecurve_adjustment_native,
            **route,
        })
    return pd.DataFrame(rows)


def compare_traces(native: pd.DataFrame, adapted: pd.DataFrame) -> dict:
    if len(native) != len(adapted):
        raise ValueError("Native and adapted trajectory lengths differ")
    errors, mismatches = {}, {}
    for name in adapted.columns:
        if name in ("flowRegime", "rfRegime"):
            count = int(np.count_nonzero(native[name].to_numpy() != adapted[name].to_numpy()))
            mismatches[name] = count
        else:
            left, right = native[name].to_numpy(float), adapted[name].to_numpy(float)
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f"Nonfinite comparison column {name}")
            errors[name] = float(np.max(np.abs(left - right)))
    return {"complete": all(v <= 1e-10 for v in errors.values()) and not any(mismatches.values()),
            "decisions": len(adapted), "absolute_tolerance": 1e-10,
            "max_absolute_errors_native_units": errors, "regime_mismatches": mismatches}


def event_summary(frame: pd.DataFrame, years: list[int]) -> dict:
    result = {}
    for year in years:
        annual = frame.loc[frame.Year == year]
        spring = annual.loc[annual.QM.between(9, 24)]
        result[str(year)] = {
            "periods": len(annual),
            "annual_peak_ontario_mean_level_m": float(annual.ontLevel.max()),
            "annual_peak_pointe_claire_level_m": float(annual.ptclaireLevel.max()),
            "spring_definition": "QM9–24 (March–June), quarter-month model",
            "spring_peak_ontario_mean_level_m": float(spring.ontLevel.max()),
            "spring_peak_pointe_claire_level_m": float(spring.ptclaireLevel.max()),
            "spring_f_limit_periods": int((spring.flowRegime == "F").sum()),
        }
    return result


def archived_glrrm_comparison(reference: NativeReference, native: pd.DataFrame) -> dict:
    relative = "output/data/baseline/Plan2014_GLRRM/simulation/historic/1900_2020/sim.csv"
    reference._verify_file(relative)
    archived = pd.read_csv(reference.root / relative)
    joined = native.merge(archived, on=["Year", "QM"], suffixes=("_python", "_glrrm"), validate="one_to_one")
    differences = {}
    for name in ("ontFlow", "ontLevel", "ptclaireLevel"):
        differences[name] = float(np.max(np.abs(joined[name + "_python"] - joined[name + "_glrrm"])))
    event_peaks = {}
    for year in (1995, 2017, 2019):
        selected = joined.loc[joined.Year == year]
        if selected.empty:
            continue
        event_peaks[str(year)] = {
            "python_ontario_mean_peak_m": float(selected.ontLevel_python.max()),
            "archived_glrrm_ontario_mean_peak_m": float(selected.ontLevel_glrrm.max()),
            "python_pointe_claire_peak_m": float(selected.ptclaireLevel_python.max()),
            "archived_glrrm_pointe_claire_peak_m": float(selected.ptclaireLevel_glrrm.max()),
        }
    return {"reference": relative, "reference_sha256": sha256(reference.root / relative),
            "matched_periods": len(joined), "max_absolute_differences_native_units": differences,
            "event_peaks": event_peaks,
            "status": "Not numerically identical. Version/rounding/limit and forcing differences require investigation before published Plan2014 equivalence claims."}


def study_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text())
    fixed = {"schema_version": 1, "system": "loslr", "controller": "plan2014_reference",
             "upstream_commit": UPSTREAM_COMMIT,
             "input": "input/historic/1900_2020/12month_sqAR.txt",
             "coefficient_source": "output/data/baseline/Plan2014_GLRRM/config.toml",
             "release": "ruleCurve", "limits": "Bv7", "routing": "stlaw",
             "september_rule": "on", "training_enabled": False,
             "initialization": {"level_m": 74.55, "previous_release_m3_s": 5950.0,
                                "previous_ice": 2, "history_periods": 48}}
    for key, expected in fixed.items():
        if recipe.get(key) != expected:
            raise ValueError(f"Unsupported reference recipe {key}: expected {expected!r}")
    if not 1900 <= recipe["last_year"] <= 2020:
        raise ValueError("Recipe last_year outside 1900..2020")
    if any(not 1900 <= year <= 2020 for year in recipe["event_years"]):
        raise ValueError("Recipe event years outside supplied data")
    return recipe


def run(output: Path, checkout: Path | None = None, last_year: int | None = None,
        config: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing baseline artifacts: {output}")
    config = config or Path(__file__).resolve().parents[4] / "configs/studies/loslr/native_baseline.json"
    recipe = study_recipe(config)
    last_year = recipe["last_year"] if last_year is None else last_year
    reference = NativeReference(checkout)
    frame = load_inputs(reference, last_year)
    provenance = reference.provenance()
    output.mkdir(parents=True, exist_ok=True)
    settings = {"output": str(output.resolve()), "input": str(reference.root / recipe["input"]),
                "config": str(reference.root / recipe["coefficient_source"]),
                "parameters": baseline_parameters(reference), "last_year": last_year}
    child = subprocess.run([sys.executable, "-c", _NATIVE_WORKER, json.dumps(settings)],
                           cwd=reference.root, text=True, capture_output=True)
    (output / "native_stdout.log").write_text(child.stdout)
    (output / "native_stderr.log").write_text(child.stderr)
    if child.returncode:
        raise RuntimeError(f"Native baseline failed; see {output / 'native_stderr.log'}")
    adapted = adapter_trace(reference, frame)
    adapted.to_csv(output / "adapter_trace.csv", index=False)
    native = pd.read_csv(output / "native_trace.csv").iloc[48:].reset_index(drop=True)
    parity = compare_traces(native, adapted)
    pi = json.loads((output / "performance_indicators.json").read_text())
    summary = {"adapter_parity": parity, "input_audit": input_audit(frame),
               "parameters": settings["parameters"],
               "events": event_summary(native, [y for y in recipe["event_years"] if y <= last_year]),
               "performance_indicators": pi,
               "status": "Software parity only; not independent hydraulic validation or causal forecast validation",
               "archived_glrrm_comparison": archived_glrrm_comparison(reference, native),
               "study_recipe": recipe, "resolved_last_year": last_year,
               "reporting_duration_seconds": 365.0 * 86400.0 / 48,
               "reporting_duration_status": "Fixed nominal reporting convention; not inferred physical duration or a replacement for native balance 2970",
               "training_run": False, "defaults_changed": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    own_files = sorted([*Path(__file__).parent.glob("*.py"),
                        Path(__file__).resolve().parents[2] / "core/contracts.py"])
    manifest = {"upstream": provenance, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "source_hashes": {str(p.relative_to(Path(__file__).resolve().parents[4])): sha256(p) for p in own_files},
                "study_config_hash": sha256(config),
                "native_worker_sha256": hashlib.sha256(_NATIVE_WORKER.encode()).hexdigest(),
                "output_hashes": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not parity["complete"]:
        raise RuntimeError(f"Adapter parity failed; inspect {output / 'summary.json'}")
    if not pi["complete"]:
        raise RuntimeError(f"Native performance indicators failed; inspect {output / 'performance_indicators.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/loslr/native_baseline_v1"))
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--last-year", type=int, help="Optional 1900..2020 override of recipe endpoint")
    parser.add_argument("--config", type=Path, help="Validated reference recipe; algorithm variants require new runner support")
    args = parser.parse_args()
    summary = run(args.output, args.checkout, args.last_year, args.config)
    print(json.dumps({"output": str(args.output), "parity": summary["adapter_parity"]["complete"],
                      "decisions": summary["adapter_parity"]["decisions"],
                      "performance_indicators_complete": summary["performance_indicators"]["complete"],
                      "events": summary["events"]}, indent=2))


if __name__ == "__main__":
    main()
