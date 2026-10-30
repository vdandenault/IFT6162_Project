"""Data locations, independent of the caller's working directory.

This research project uses editable installs from a recursive Git checkout.
"""
from importlib.resources import files
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GYM4REAL_ROOT = Path(str(files('gym4real')))
UPSTREAM_ROOT = GYM4REAL_ROOT.parent
DAM_DATA = GYM4REAL_ROOT / 'data' / 'dam'
DAM_CONFIG = GYM4REAL_ROOT / 'envs' / 'dam'
RESULTS = PROJECT_ROOT / 'results' / 'phase1'
