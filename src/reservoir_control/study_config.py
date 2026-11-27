import json
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
