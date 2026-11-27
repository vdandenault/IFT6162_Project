"""Scoped access to the unmodified, pinned LOSLR reference implementation.

The upstream project is deliberately an external checkout, not vendored code.
Its legacy absolute utility imports and path side effects are contained here.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import subprocess
import sys
import threading
from functools import cache
from pathlib import Path
from types import ModuleType

UPSTREAM_URL = "https://github.com/ksemmendinger/loslr_regulation_optimization"
UPSTREAM_COMMIT = "8b8de9e89487552f8590f4d42d87484e5edc6872"
BASELINE_CONFIG = "output/data/baseline/Plan2014_GLRRM/config.toml"
INPUT_FILE = "input/historic/1900_2020/12month_sqAR.txt"
_LOCK = threading.RLock()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_checkout() -> Path:
    return Path(__file__).resolve().parents[4] / "external/loslr_regulation_optimization"


@cache
def native_reference() -> NativeReference:
    return NativeReference()


class NativeReference:
    """Load only inspected upstream modules without changing global import paths."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else default_checkout()
        self.root = self.root.resolve()
        if not (self.root / ".git").exists():
            raise FileNotFoundError(
                f"Missing external LOSLR checkout: {self.root}. Clone {UPSTREAM_URL} "
                f"there and checkout {UPSTREAM_COMMIT}."
            )
        revision = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != UPSTREAM_COMMIT:
            raise ValueError(f"LOS LR revision {revision} does not match {UPSTREAM_COMMIT}")
        # The native worker also imports performance indicators and their data.
        # Reject changed tracked implementation/data throughout these paths.
        check = subprocess.run(["git", "-C", str(self.root), "diff", "--quiet", "HEAD", "--",
                                "functions", "optimizationSimulation.py", "objectiveFunctions",
                                BASELINE_CONFIG, INPUT_FILE])
        if check.returncode:
            raise ValueError("Native checkout has modified tracked implementation or inputs")
        self._modules: dict[str, ModuleType] = {}
        self.utils = self.load("functions/utils.py")
        self.release = self.load("functions/release/ruleCurve.py")
        self.limits = self.load("functions/limits/Bv7.py")
        self.september = self.load("functions/limits/septemberRule.py")
        self.routing = self.load("functions/routing/stlaw.py")
        self.simulation = self.load("optimizationSimulation.py")

    def load(self, relative: str) -> ModuleType:
        if relative in self._modules:
            return self._modules[relative]
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Native module must lie inside the pinned checkout")
        self._verify_file(relative)
        original_import = builtins.__import__

        def native_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "functions.utils":
                return self.utils
            return original_import(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location(
            "_loslr_reference_" + relative.replace("/", "_").replace(".", "_"), path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": native_import}
        # Native files append '.' themselves. Restore even if execution fails;
        # no native module name or absolute 'functions' alias enters sys.modules.
        with _LOCK:
            original_path = sys.path[:]
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path[:] = original_path
        self._modules[relative] = module
        return module

    def _verify_file(self, relative: str) -> None:
        expected = subprocess.check_output(
            ["git", "-C", str(self.root), "show", f"{UPSTREAM_COMMIT}:{relative}"]
        )
        if (self.root / relative).read_bytes() != expected:
            raise ValueError(f"Native file differs from pinned source: {relative}")

    def verify_inputs(self) -> None:
        for relative in (BASELINE_CONFIG, INPUT_FILE):
            self._verify_file(relative)

    def provenance(self) -> dict:
        self.verify_inputs()
        for relative in self._modules:
            self._verify_file(relative)
        pi_files = subprocess.check_output(
            ["git", "-C", str(self.root), "ls-files", "objectiveFunctions/legacyPIs"], text=True
        ).splitlines()
        files = [*self._modules, BASELINE_CONFIG, INPUT_FILE, *pi_files]
        return {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "files": {p: sha256(self.root / p) for p in sorted(files)},
            "licensing": "No repository license identified; external checkout only, redistribution unresolved",
        }
