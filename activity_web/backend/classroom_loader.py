from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from .startup import ensure_insightface_models_flat
from .config import ATTENDANCE_DIR


def _pipeline_script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "CLASSROOM PIPELINE" / "classroom_pipeline.py"


@lru_cache(maxsize=1)
def _load_module() -> ModuleType:
    script = _pipeline_script_path()
    spec = importlib.util.spec_from_file_location("classroom_pipeline", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load classroom pipeline from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        ensure_insightface_models_flat(["antelopev2", "buffalo_l"]) 
    except Exception:
        pass
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_classroom_pipeline():
    roster_path = ATTENDANCE_DIR / "attendance_store.json"
    return _load_module().ClassroomPipeline(roster_path=roster_path)
