from __future__ import annotations

import builtins
import hashlib
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

UPSTREAM_URL = "https://github.com/ksemmendinger/loslr_regulation_optimization"
UPSTREAM_COMMIT = "8b8de9e89487552f8590f4d42d87484e5edc6872"
BASELINE_CONFIG = "output/data/baseline/Plan2014_GLRRM/config.toml"
INPUT_FILE = "input/historic/1900_2020/12month_sqAR.txt"
LEGACY_INPUT_FILE = "input/historic/1961_2020/12month_sqAR.txt"
POLICIES_FILE = (
    "output/data/ruleCurve_Bv7_offSepRule_netAnnualAverage_12month_sqAR_9dv_7obj_1961_2020_75000nfe/"
    "satisficingPolicies.csv"
)


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
                                BASELINE_CONFIG, INPUT_FILE, LEGACY_INPUT_FILE, POLICIES_FILE])
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
        source = self.verify_file(relative)
        module = ModuleType("_loslr_reference_" + relative.removesuffix(".py").replace("/", "_"))
        module.__file__ = str(path)
        # Upstream uses sys.path.append(".") to find functions.utils.
        # Give it a private path list and resolve that import directly.
        local_sys = SimpleNamespace(path=sys.path.copy())

        def native_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sys":
                return local_sys
            if name == "functions.utils":
                return self.utils
            return builtins.__import__(name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": native_import}
        exec(compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__)
        self._modules[relative] = module
        return module

    def verify_file(self, relative: str) -> bytes:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Native file must lie inside the pinned checkout")
        source = path.read_bytes()
        expected = subprocess.check_output(
            ["git", "-C", str(self.root), "show", f"{UPSTREAM_COMMIT}:{relative}"]
        )
        if source != expected:
            raise ValueError(f"Native file differs from pinned source: {relative}")
        return source

    def provenance(self) -> dict:
        verified = {p: hashlib.sha256(self.verify_file(p)).hexdigest()
                    for p in (*self._modules, BASELINE_CONFIG, INPUT_FILE)}
        pi_files = subprocess.check_output(
            ["git", "-C", str(self.root), "ls-files", "objectiveFunctions/legacyPIs"], text=True
        ).splitlines()
        hashes = {p: sha256(self.root / p) for p in pi_files} | verified
        return {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "files": dict(sorted(hashes.items())),
            "licensing": "No repository license identified; external checkout only, redistribution unresolved",
        }
