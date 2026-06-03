from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pipeline_script_path() -> Path:
    return repo_root() / "ACTIVITY CLASSIFICATION PIPELINE" / "student_activity_pipeline.py"


@lru_cache(maxsize=1)
def load_pipeline_module() -> ModuleType:
    script_path = pipeline_script_path()
    spec = importlib.util.spec_from_file_location("student_activity_pipeline", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_pipeline():
    module = load_pipeline_module()
    return module.StudentActivityPipeline(classifier_path=module.default_classifier_weights())
