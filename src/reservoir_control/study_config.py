import json
from math import isfinite
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
COUPLED_CONFIG = CONFIG_DIR / "coupled.json"
ONTARIO_CONFIG = CONFIG_DIR / "ontario.json"


def load_settings(path: Path, section: str) -> dict:
    """Read a named section or a standalone recipe supplied with a CLI flag."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "study_settings" in data:
        data = data["study_settings"][section]
    return data


def load_follower_config(path: Path = COUPLED_CONFIG) -> dict:
    return load_settings(path, "follower")


def validate_follower_config(config: dict) -> dict:
    if config["schema_version"] != 1 or config["clock"]["duration_seconds"] != 365 * 86400 / 48:
        raise ValueError("This study explicitly uses 48 equal periods per 365-day year (657000 s)")
    if config["horizon"] < 1 or config["reference_release_m3_s"] < 0:
        raise ValueError("Positive horizon and nonnegative reference required")
    periods = len(config["known_ontario_executable_schedule_m3_s"])
    if periods == 0 or len(config["known_other_downstream_forecast_m3_s"]) != periods:
        raise ValueError("Aligned supplied downstream conditions required")
    for case in config["scenarios"]:
        if (
            len(case["realized_net_inflow_m3_s"]) != periods
            or len(case["downstream_capacity_m3_s"]) != periods
        ):
            raise ValueError("Complete scenario arrays required")
        if case["routing_delay_periods"] != len(case["initial_transit_volumes_m3"]):
            raise ValueError("Explicit transit queue required")
    noise = config["forecast_noise"]["standard_deviation_m3_s"]
    if not isfinite(noise) or noise < 0:
        raise ValueError("Nonnegative finite noise amplitude required")
    return config
