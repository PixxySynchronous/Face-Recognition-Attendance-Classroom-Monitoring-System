"""
Read-only lookup against the Attendance roster (attendance_store.json), for pipelines
that want to recognize an enrolled student from a face embedding without depending on
the Flask attendance_service package (which has side effects like incremental gallery
growth that shouldn't run from an anonymous video-analysis context).

Uses the same embedding space and matching formula as
activity_web/backend/attendance_service.py:match_student (max of prototype similarity
and best individual stored embedding similarity), so the same threshold applies
regardless of which pipeline produced the query embedding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Higher than Attendance's photo-based 0.28 (attendance_service.py, unaffected by this
# module) — video frames tend to be lower-quality/more-varied-angle than a still
# classroom photo, and this is a read-only "bonus" recognition, not the primary
# attendance record, so it's worth erring conservative here.
DEFAULT_MATCH_THRESHOLD = 0.35
# Minimum gap over the second-best candidate student to accept a match — guards
# against accepting a mediocre match just because it's the best of a bad bunch.
DEFAULT_MATCH_MARGIN = 0.05


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def load_roster(store_path: str | Path) -> list[dict]:
    """Load the enrolled students list from an attendance_store.json path.
    Returns [] if the file doesn't exist or can't be parsed."""
    store_path = Path(store_path)
    if not store_path.exists():
        return []
    try:
        data = json.loads(store_path.read_text())
    except Exception:
        return []
    return data.get("students", []) or []


def match_against_roster(
    embedding: np.ndarray,
    students: list[dict],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    margin: float = DEFAULT_MATCH_MARGIN,
) -> tuple[str | None, float]:
    """Compare an embedding against every enrolled student's prototype + stored
    embeddings and return (name, similarity) for the best match, requiring it to
    both clear `threshold` and beat the second-best candidate student by at least
    `margin`. Returns (None, best_similarity) if nothing qualifies."""
    if not students:
        return None, -1.0

    q = _normalize(embedding)
    best_name, best_similarity = None, -1.0
    second_best_similarity = -1.0

    for student in students:
        prototype = np.asarray(student.get("prototype") or [], dtype=np.float32)
        sim = float(np.dot(q, prototype)) if prototype.size > 0 else -1.0

        stored = student.get("embeddings") or []
        if stored:
            mat = np.asarray(stored, dtype=np.float32)
            sim = max(sim, float((mat @ q).max()))

        if sim > best_similarity:
            second_best_similarity = best_similarity
            best_similarity = sim
            best_name = student.get("name")
        elif sim > second_best_similarity:
            second_best_similarity = sim

    if (
        best_name is None
        or best_similarity < threshold
        or (best_similarity - second_best_similarity) < margin
    ):
        return None, best_similarity
    return best_name, best_similarity
