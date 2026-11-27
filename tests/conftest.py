from importlib.util import find_spec
from pathlib import Path

ONTARIO_TESTS = Path(__file__).with_name("integration")


def pytest_ignore_collect(collection_path):
    if collection_path.is_relative_to(ONTARIO_TESTS) and find_spec("core") is None:
        return True
    return None
