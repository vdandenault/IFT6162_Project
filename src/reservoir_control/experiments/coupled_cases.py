import json
from dataclasses import replace

import numpy as np
import pandas as pd

import reservoir_control.systems.ottawa.forecasts as forecasts
from reservoir_control.study_config import load_settings
from reservoir_control.systems.loslr.physics import FLOW_SCALE


def prepared_case(
    name, year, trajectory, start, base, inflows, capacities, design_label, *, bank=None
):
    trajectory = trajectory.copy()
    p = len(trajectory)
    background = 2000.0
    reference = base["reference_release_m3_s"]
    trajectory["ontLevelBOQ"] = start.level_m
    # The fifth Ontario commitment is anchored by the known initial release
    # and the declared synthetic capacity. No later observed release enters.
    trajectory["ontFlow"] = (
        np.minimum(start.previous_release_m3_s, np.asarray(capacities) - background - reference)
        / FLOW_SCALE
    )
    scenario = {
        "id": name,
        "realized_net_inflow_m3_s": list(inflows),
        "downstream_capacity_m3_s": list(capacities),
        "routing_delay_periods": 1,
        "initial_transit_volumes_m3": [reference * base["clock"]["duration_seconds"]],
    }
    config = {
        **base,
        "known_other_downstream_forecast_m3_s": [background] * p,
        "scenarios": [scenario],
    }
    if bank is None:
        noise = config["forecast_noise"]
        bank = forecasts.forecast_bank(
            inflows,
            "noisy",
            year,
            noise["initial_observed_inflow_estimate_m3_s"],
            noise["standard_deviation_m3_s"],
        )
    return {
        "name": name,
        "year": year,
        "design": design_label,
        "trajectory": trajectory,
        "rows": trajectory.to_dict("records"),
        "start": start,
        "config": config,
        "scenario": scenario,
        "bank": bank,
    }


def annual_cases(frame, states, years, base, design):
    packed = []
    for year in years:
        state, index = states[year]
        window = frame.iloc[index : index + 48].copy()
        if len(window) != 48 or not window.Year.eq(year).all():
            raise ValueError(f"Incomplete annual forcing for {year}")
        for j in range(48):
            previous = frame.iloc[index + j - 1]
            i = window.index[j]
            for lead in range(1, 5):
                window.loc[i, f"ontNTS_QM{lead}"] = previous.ontNTS
            window.loc[i, "forNTS"] = frame.ontNTS.iloc[index + j - 48 : index + j].mean()
            for key in ("iceInd", "ptclaireR", "longsaultR", "saundershwR", "saunderstwR"):
                window.loc[i, key] = previous[key]
            window.loc[i, "foreInd"] = 1
        shapes = list(design["annual_design"])
        shape = shapes[year % len(shapes)]
        inflows = np.interp(np.arange(48), np.arange(12) * 4, design["annual_design"][shape])
        packed.append(
            prepared_case(
                f"annual_{year}",
                year,
                window,
                state,
                base,
                inflows,
                [design["annual_downstream_capacity_m3_s"]] * 48,
                shape,
            )
        )
    return packed


def stress_cases(root, frame, states, base, design_path=None):
    directory = root / "ottawa_two_path_wider_fresh_v1"
    frozen = (
        load_settings(design_path, "confirmation_stress")
        if design_path
        else json.loads((directory / "settings.json").read_text())["design"]
    )
    caps = next(
        case["downstream_capacity_m3_s"]
        for case in base["scenarios"]
        if case["id"] == "one_period_delay"
    )
    cases = []
    for year in (1961, 2017, 2019):
        source = root / f"ottawa_replan_guard_{year}"
        rows = pd.read_csv(source / "ontario_inputs.csv")
        history = json.loads((source / "settings.json").read_text())["ontario_window"][
            "initial_history"
        ]
        state, index = states[year]
        offset = int(rows.QM.iloc[0]) - 1
        # Retain the archived initial commitments and warm the rule-curve
        # history with completed historical mean levels up to this window.
        prior = frame.ontLevelMOQ.iloc[index + offset - 48 : index + offset].to_numpy(dtype=float)
        if np.isfinite(prior).all():
            state = replace(state, mean_level_history_m=tuple(prior))
        state = replace(
            state,
            level_m=rows.ontLevelBOQ.iloc[0],
            previous_release_m3_s=history["previous_flow"] * FLOW_SCALE,
            previous_ice=history["previous_ice"],
        )
        for j in range(len(rows)):
            i = index + offset + j
            rows.loc[j, "forNTS"] = frame.ontNTS.iloc[i - 48 : i].mean()
        for pair in frozen["pairs"]:
            for shape in ("pulse", "plateau"):
                for seed in frozen["seeds"]:
                    stem = f"{pair['name']}_{shape}_{seed}"
                    inflows = pair[shape]
                    if design_path:
                        noise = base["forecast_noise"]
                        raw = forecasts.forecast_bank(
                            inflows,
                            "noisy",
                            seed,
                            noise["initial_observed_inflow_estimate_m3_s"],
                            noise["standard_deviation_m3_s"],
                        )
                        bank = forecasts.recession_bank(inflows, raw, jump_minimum_m3_s=700.0)
                    else:
                        with np.load(directory / str(year) / f"{stem}_forecasts.npz") as archive:
                            bank = archive["receding"]
                    cases.append(
                        prepared_case(
                            f"stress_{year}_{stem}",
                            year,
                            rows,
                            state,
                            base,
                            pair[shape],
                            caps,
                            shape,
                            bank=bank,
                        )
                    )
    return cases
