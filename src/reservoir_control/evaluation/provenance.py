import hashlib
from pathlib import Path
from typing import Mapping


def verify_sources(hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file():
            raise ValueError(f'Frozen source is missing: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {path}')
