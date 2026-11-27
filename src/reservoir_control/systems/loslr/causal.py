from pathlib import Path
import argparse
import json

import pandas as pd

from .adapter import OntarioSession, decision_time
from .baseline import adapter_trace, archived_glrrm_comparison, event_summary
from .data import SupplyForecast, forcing_from_row, load_inputs
from .model import DecisionInputs, OntarioModel, Plan2014
from .native import NativeReference, sha256


class CausalOntarioSession(OntarioSession):
    """Use a past-only warmup to initialize a continuous causal replay."""

    def reset(self):
        super().reset()
        if len(self._frame) <= 96:
            raise ValueError("Causal replay needs an initialization year, warmup year and scored periods")
        policy = Plan2014(self.model.reference)
        for _ in range(48):
            decision, time = self.information()
            self.step(policy.action(self.state, decision.forecast, time))
        return self.state

    def information(self):
        if self.done:
            raise RuntimeError("Replay is complete")
        previous = self._frame.iloc[self.index - 1]
        recent = self._frame.iloc[self.index - 48:self.index]
        forecast = SupplyForecast(
            (float(previous.ontNTS),) * 4,
            float(recent.ontNTS.mean()),
            float(previous.stlouisontOut),
            "Previous completed period persistence and trailing 48-period NTS mean",
        )
        # The synthetic 1899 initialization has no roughness estimates.
        # Use a fixed unit coefficient at the first warmup decision.
        if self.index == 48:
            longsault, pointe_claire = 1.0, 1.0
        else:
            longsault, pointe_claire = float(previous.longsaultR), float(previous.ptclaireR)
        decision = DecisionInputs(forecast, self.state.previous_ice, longsault, pointe_claire)
        return decision, decision_time(self._frame.iloc[self.index], self.index - 48)

    def step(self, action):
        decision, time = self.information()
        result = self.model.transition(
            self.state, action, forcing_from_row(self._frame.iloc[self.index]), time,
            decision=decision,
        )
        self.state = result.state
        self.index += 1
        return result


def replay(reference, frame):
    session = CausalOntarioSession(OntarioModel(reference), frame)
    policy = Plan2014(reference)
    rows = []
    while not session.done:
        decision, time = session.information()
        action = policy.action(session.state, decision.forecast, time)
        result = session.step(action)
        rows.append({
            "Year": time.year, "QM": time.period,
            "ontLevel": result.metrics["ontario_level_mean_m"],
            "ptclaireLevel": result.metrics["pointe_claire_level_m"],
            "ontFlow": result.metrics["executed_release_m3_s"] / 10,
            "flowRegime": result.info["flow_regime"],
            "forecast_nts_m3_s": decision.forecast.nts_m3_s[0],
            "forecast_slon_m3_s": decision.forecast.slon_m3_s,
            "forecast_annual_nts_m3_s": decision.forecast.annual_nts_m3_s,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Plan 2014 rule-curve benchmark with past-only decision inputs."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)

    reference = NativeReference(args.checkout)
    frame = load_inputs(reference)
    native = adapter_trace(reference, frame)
    original = archived_glrrm_comparison(reference, native)
    archive = pd.read_csv(reference.root / original["reference"])
    aligned = frame.iloc[48:].merge(archive, on=["Year", "QM"], suffixes=("_python", "_archive"),
                           validate="one_to_one")
    columns = ["ontNTS", "stlouisontOut", "iceInd", "ptclaireR", "foreInd", "forNTS"]
    differences = {
        name: int(((aligned[f"{name}_python"] - aligned[f"{name}_archive"]).abs() > 1e-8).sum())
        for name in columns
    }
    outputs = native.merge(archive, on=["Year", "QM"], suffixes=("_python", "_archive"),
                           validate="one_to_one")
    largest_gap = outputs.loc[(outputs.ontFlow_python - outputs.ontFlow_archive).abs().idxmax()]
    substituted = frame.copy()
    annual = archive.set_index(["Year", "QM"]).forNTS
    substituted.loc[48:, "forNTS"] = annual.reindex(
        pd.MultiIndex.from_frame(frame.loc[48:, ["Year", "QM"]])
    ).to_numpy()
    replacement = archived_glrrm_comparison(reference, adapter_trace(reference, substituted))
    causal = replay(reference, frame)
    summary = {
        "matched_periods": len(aligned),
        "input_difference_counts": differences,
        "native_comparison": original,
        "largest_release_gap": {
            "year": int(largest_gap.Year), "quarter_month": int(largest_gap.QM),
            "python_regime": largest_gap.flowRegime_python,
            "archive_regime": largest_gap.flowRegime_archive,
            "python_m3_s": float(largest_gap.ontFlow_python * 10),
            "archive_m3_s": float(largest_gap.ontFlow_archive * 10),
        },
        "archived_m_minus_periods": int(archive.flowRegime.eq("M-").sum()),
        "archived_annual_forecast_only_comparison": replacement,
        "causal_scored_periods": len(causal),
        "causal_years": [int(causal.Year.min()), int(causal.Year.max())],
        "causal_events": event_summary(causal, [1995, 2017, 2019]),
        "native_events": event_summary(native, [1995, 2017, 2019]),
        "warmup": "1899 supplied initialization and 1900 past-only warmup. Scoring begins in 1901.",
        "flow_trace_units": "ontFlow in 10 m³/s",
        "upstream": reference.provenance(),
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("causal.py", "model.py", "adapter.py", "baseline.py", "data.py", "native.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / "causal_trace.csv"
    causal.to_csv(trace, index=False)
    summary["trace_sha256"] = sha256(trace)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
