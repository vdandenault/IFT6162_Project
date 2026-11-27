import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"Frozen source is missing: {path}")
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
