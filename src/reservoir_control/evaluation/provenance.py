"""Strict source verification for frozen experiments."""
import hashlib
from pathlib import Path
from typing import Mapping


def verify_sources(hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file():
            if '/src/lake_como/' in path.as_posix():
                raise ValueError(
                    'This frozen study uses the pre-migration Lake Como source tree. '
                    'Restore its original source/environment snapshot with '
                    'scripts/migration/preserve_como.py --restore-to <empty-directory>. '
                    'Keep the recorded source hashes when reproducing the historical study.'
                )
            raise ValueError(f'Frozen source is missing: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {path}')
