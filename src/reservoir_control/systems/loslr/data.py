from __future__ import annotations

import tomllib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from reservoir_control.study_config import REPO_ROOT
from reservoir_control.systems.loslr.native import (
    BASELINE_CONFIG,
    INPUT_FILE,
    POLICIES_FILE,
    NativeReference,
    default_checkout,
)

# Upstream uses this balance convention, not variable Gregorian quarter lengths.
BALANCE_CONVERSION = 2970.0
PERIODS_PER_YEAR = 48


def trace_path(data_root: Path, name: Path) -> Path:
    """Prefer a project's CSV input, then resolve legacy tab-separated traces."""
    local = data_root / "input" / name
    for suffix in (".csv", ".txt"):
        path = local.with_suffix(suffix)
        if path.is_file():
            return path
    if data_root.resolve() == REPO_ROOT / "data":
        return (default_checkout() / "input" / name).with_suffix(".txt")
    return local.with_suffix(".csv")


def read_trace(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="," if path.suffix == ".csv" else "\t")


def read_policies(data_root: Path) -> pd.DataFrame:
    path = data_root / "baseline" / "satisficingPolicies.csv"
    if not path.is_file() and data_root.resolve() == REPO_ROOT / "data":
        path = default_checkout() / POLICIES_FILE
    return pd.read_csv(path, index_col="ID")


def quarter_start(year, qm):
    month, quarter = divmod(qm - 1, 4)
    # Legacy IJC quarter-month day boundaries (Plan 1958-A, section II).
    return pd.Timestamp(year, month + 1, (1, 8, 15, 22)[quarter] if month == 1 else (1, 9, 16, 24)[quarter])


@dataclass(frozen=True)
class SupplyForecast:
    nts_m3_s: tuple[float, float, float, float]
    annual_nts_m3_s: float
    slon_m3_s: float
    source: str = "upstream supplied; historical fitting/issuance not independently verified"


@dataclass(frozen=True)
class Forcing:
    """Realized physical inputs are separate from the policy's forecast object."""
    nts_m3_s: float
    slon_m3_s: float
    forecast: SupplyForecast
    ice: int
    native_metadata: Mapping[str, float]
    use_supplied_forecast_for_f_limit: bool


def baseline_parameters(reference: NativeReference) -> dict[str, float]:
    config = tomllib.loads(reference.verify_file(BASELINE_CONFIG).decode("utf-8"))
    section = config["decisionVariables"]
    original = dict(zip(section["dvName"], section["baselineValue"], strict=True))
    # Native current ruleCurve.py reduced the older baseline's parameter schema.
    # Match by parameter meaning, never pass its 14-vector into the 9-vector API.
    expected = {"rcThreshold": 7011.0, "wetDenominator": 1541.0, "dryDenominator": 1294.0}
    for key, value in expected.items():
        if original[key] != value:
            raise ValueError(f"Native ruleCurve hardcodes {key}={value}; baseline config differs")
    return {
        "wetIndicatorThreshold": original["lfWetThreshold"],
        "wetConfidenceThreshold": original["lf99Conf"],
        "C1": original["C1"], "C1*": original["C1"] + original["addC1"],
        "P1": original["P1"], "C2": original["C2"], "P2": original["P2"],
        "dryLevelThreshold": original["adjLevel"], "dryFlowAdjustment": original["adjFlow"],
    }


def load_inputs(reference: NativeReference, last_year: int | None = None) -> pd.DataFrame:
    if last_year is not None and not 1900 <= last_year <= 2020:
        raise ValueError("last_year must be in the supplied simulation range1900..2020")
    frame = pd.read_csv(BytesIO(reference.verify_file(INPUT_FILE)), sep="\t")
    if last_year is not None:
        frame = frame.loc[frame.Year <= last_year].copy()
    frame = frame.reset_index(drop=True)
    if len(frame) < 49 or int(frame.QM.iloc[0]) != 1:
        raise ValueError("Reference input needs its complete leading initialization year")
    for year, group in frame.groupby("Year", sort=False):
        if tuple(group.QM.astype(int)) != tuple(range(1, 49)):
            raise ValueError(f"Year {year} lacks a complete, ordered 48-period input")
    numeric = ["ontNTS", "stlouisontOut", "iceInd", "forNTS", "slonFlow_QM1", "foreInd"]
    numeric += [f"ontNTS_QM{k}" for k in range(1, 5)]
    if not np.isfinite(frame.loc[48:, numeric].to_numpy(dtype=float)).all():
        raise ValueError("Simulated rows contain missing required physical/forecast inputs")
    return frame


def forcing_from_row(row: Mapping) -> Forcing:
    forecast = SupplyForecast(
        tuple(float(row[f"ontNTS_QM{k}"]) for k in range(1, 5)),
        float(row["forNTS"]), float(row["slonFlow_QM1"]),
    )
    metadata = {key: float(value) for key, value in row.items()
                if isinstance(value, (int, float, np.number))}
    return Forcing(float(row["ontNTS"]), float(row["stlouisontOut"]), forecast,
                   int(row["iceInd"]), metadata, bool(row["foreInd"] == 1))


def input_audit(frame: pd.DataFrame) -> dict:
    scored = frame.iloc[48:]
    return {
        "rows": len(frame), "initialization_year": int(frame.Year.iloc[0]),
        "simulation_years": [int(scored.Year.min()), int(scored.Year.max())],
        "periods_per_year": 48,
        "foreInd_counts": {str(k): int(v) for k, v in scored.foreInd.value_counts().items()},
        "limit_information_dependencies": {"M_limit": "Uses realized SLON even when foreInd=1", "F_limit": "Uses forecast SLON when foreInd=1, otherwise realized SLON"},
        "forecast_causality": "Supplied forecast parity only; full-record fitting and issuance audit pending",
        "ontario_flow_native_units": "10 m3/s",
        "supply_and_slon_native_units": "m3/s",
        "level_units": "m; gauge datum confirmation pending",
        "balance": "delta_level = round_d((NTS_m3_s/10 - outflow_m3_s/10)/2970, 6)",
        "calendar": "48 model quarter-months/year; native fixed conversion retained",
    }
