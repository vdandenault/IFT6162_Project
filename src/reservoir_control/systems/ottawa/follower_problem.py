import numpy as np

from reservoir_control.study_config import COUPLED_CONFIG, load_settings
from reservoir_control.systems.ottawa.hard_follower import FollowerWeights, HardFollowerProblem

CONFIG = COUPLED_CONFIG


def load_follower_config(path=CONFIG):
    return load_settings(path, "follower")


def make_problem(config, scenario, step, storage, previous, transit, forecast):
    horizon = len(forecast)
    window = slice(step, step + horizon)
    durations = np.full(horizon, config["clock"]["duration_seconds"])
    routing_matrix, arrivals = transit.forecast(durations)
    return HardFollowerProblem(
        initial_storage_m3=storage,
        previous_release_m3_s=previous,
        net_inflow_m3_s=forecast,
        ontario_executable_release_m3_s=config["known_ontario_executable_schedule_m3_s"][window],
        other_downstream_m3_s=config["known_other_downstream_forecast_m3_s"][window],
        routing_matrix=routing_matrix,
        in_transit_contribution_m3_s=arrivals,
        downstream_capacity_m3_s=scenario["downstream_capacity_m3_s"][window],
        storage_minimum_m3=np.full(horizon + 1, config["storage_minimum_m3"]),
        storage_maximum_m3=np.full(horizon + 1, config["storage_maximum_m3"]),
        release_minimum_m3_s=np.full(horizon, config["release_minimum_m3_s"]),
        release_maximum_m3_s=np.full(horizon, config["release_maximum_m3_s"]),
        storage_target_m3=np.full(horizon + 1, config["storage_target_m3"]),
        release_target_m3_s=np.full(horizon, config["release_target_m3_s"]),
        duration_seconds=durations,
        storage_scale_m3=config["storage_scale_m3"],
        flow_scale_m3_s=config["flow_scale_m3_s"],
        weights=FollowerWeights(**config["weights"]),
    )


def validate_config(c):
    if c["schema_version"] != 1 or c["clock"]["duration_seconds"] != 365 * 86400 / 48:
        raise ValueError("This study explicitly uses 48 equal periods per 365-day year (657000 s)")
    if c["horizon"] < 1 or c["reference_release_m3_s"] < 0:
        raise ValueError("Positive horizon and nonnegative reference required")
    n = len(c["known_ontario_executable_schedule_m3_s"])
    if n == 0 or len(c["known_other_downstream_forecast_m3_s"]) != n:
        raise ValueError("Aligned supplied downstream conditions required")
    for case in c["scenarios"]:
        if len(case["realized_net_inflow_m3_s"]) != n or len(case["downstream_capacity_m3_s"]) != n:
            raise ValueError("Complete scenario arrays required")
        if case["routing_delay_periods"] != len(case["initial_transit_volumes_m3"]):
            raise ValueError("Explicit transit queue required")
    if (
        not np.isfinite(c["forecast_noise"]["standard_deviation_m3_s"])
        or c["forecast_noise"]["standard_deviation_m3_s"] < 0
    ):
        raise ValueError("Nonnegative finite noise amplitude required")
    return c
