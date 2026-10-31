"""Preserve predeclared study payloads in one versioned manifest."""
import json
from lake_como.paths import PROJECT_ROOT

PLAN_MANIFEST = PROJECT_ROOT / 'configs/experiments/ppo_studies.json'


def record_plan(name, plan, *, manifest=PLAN_MANIFEST):
    """Add a new named plan; reject changes to an existing experiment."""
    plans = json.loads(manifest.read_text()) if manifest.exists() else {}
    if name in plans:
        if plans[name] != plan:
            raise ValueError(f'Predeclared {name} plan differs; use a new study name')
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    plans[name] = plan
    manifest.write_text(json.dumps(plans, indent=2) + '\n')
