from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from insightface.app import FaceAnalysis


BACKEND_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BACKEND_DIR.parent / "runtime"
ATTENDANCE_DIR = RUNTIME_DIR / "attendance"
UPLOAD_DIR = ATTENDANCE_DIR / "uploads"
MARKED_DIR = ATTENDANCE_DIR / "marked"
STORE_PATH = ATTENDANCE_DIR / "attendance_store.json"
PHOTOS_PER_VIDEO = 32
FACE_SIMILARITY_THRESHOLD = 0.38
EMBEDDING_MODEL_NAME = "antelopev2_v3"
ENROLLMENT_MIN_DET_SCORE = 0.50
ENROLLMENT_OUTLIER_SIM_THRESHOLD = 0.50
MAX_STORED_EMBEDDINGS = 128
ENROLLMENT_SAMPLE_INTERVAL_S = 1.0   # sample one frame every 1 second for enrollment
MAX_ENROLLMENT_FRAMES = 30           # cap to avoid overly long videos
ANCHOR_CONSISTENCY_THRESHOLD = 0.35
# Degradation scales applied during enrollment to bridge the gap between
# close-up enrollment faces and small distant classroom faces.
# e.g. 0.3 → shrink face to 30% then bicubic back → simulates a distant face.
ENROLLMENT_DEGRADATION_SCALES = [0.5, 0.3]


def ensure_attendance_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MARKED_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return vector / norm


def _degrade_and_embed(fa: FaceAnalysis, frame: np.ndarray, bbox: tuple, scale: float) -> np.ndarray | None:
    """Shrink a face crop to `scale` fraction then bicubic back to original size,
    then re-run recognition on that blurry crop.  Simulates how a distant classroom
    face looks to the recognition model, so the enrollment gallery contains embeddings
    that will match small faces as well as close-up ones."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    small_w, small_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    degraded = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(degraded, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # Paste degraded crop back into a copy of the frame and re-run fa.get
    frame_copy = frame.copy()
    frame_copy[y1:y2, x1:x2] = degraded
    faces = fa.get(frame_copy)

    # Pick the face closest to the original bbox
    best_emb, best_iou = None, 0.0
    for f in faces:
        fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
        ix1, iy1 = max(x1, fx1), max(y1, fy1)
        ix2, iy2 = min(x2, fx2), min(y2, fy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2-x1)*(y2-y1) + (fx2-fx1)*(fy2-fy1) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            e = getattr(f, "normed_embedding", None)
            if e is None:
                e = getattr(f, "embedding", None)
            if e is not None:
                best_emb = _normalize(np.asarray(e, dtype=np.float32).flatten().copy())
                best_iou = iou
    return best_emb





def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = _normalize(vector_a)
    b = _normalize(vector_b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


@dataclass
class FaceSample:
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float


class AttendanceService:
    def __init__(self) -> None:
        ensure_attendance_dirs()
        self.face_analysis = FaceAnalysis(
            name="antelopev2",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.5)
        self._migrate_legacy_embeddings_if_needed()

    def _read_store(self) -> dict:
        if not STORE_PATH.exists():
            return {"students": [], "attendance": []}
        try:
            data = json.loads(STORE_PATH.read_text())
        except Exception:
            return {"students": [], "attendance": []}
        data.setdefault("students", [])
        data.setdefault("attendance", [])
        return data

    def _write_store(self, data: dict) -> None:
        ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
        STORE_PATH.write_text(json.dumps(data, indent=2))

    def _load_image(self, media_path: Path) -> np.ndarray | None:
        image = cv2.imread(str(media_path))
        return image

    def _sample_video_frames(self, media_path: Path) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            return []

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[np.ndarray] = []

        if frame_count > 0:
            indices = np.linspace(0, max(0, frame_count - 1), min(PHOTOS_PER_VIDEO, frame_count), dtype=int)
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(frame)
        else:
            while len(frames) < PHOTOS_PER_VIDEO:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(frame)

        capture.release()
        return frames

    def _frames_from_media(self, media_path: Path) -> list[np.ndarray]:
        suffix = media_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            image = self._load_image(media_path)
            return [image] if image is not None else []
        return self._sample_video_frames(media_path)

    def _sample_frames_for_enrollment(self, media_path: Path) -> list[np.ndarray]:
        """Sample one frame every ENROLLMENT_SAMPLE_INTERVAL_S seconds from a video."""
        suffix = media_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            image = self._load_image(media_path)
            return [image] if image is not None else []

        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(fps * ENROLLMENT_SAMPLE_INTERVAL_S))
        indices = list(range(0, total_frames, step))[:MAX_ENROLLMENT_FRAMES]

        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def _detect_samples(self, frame: np.ndarray) -> list[FaceSample]:
        faces = self.face_analysis.get(frame)
        samples: list[FaceSample] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            bbox = tuple(int(value) for value in face.bbox)
            samples.append(
                FaceSample(
                    embedding=_normalize(np.asarray(embedding, dtype=np.float32)),
                    bbox=bbox,
                    score=float(getattr(face, "det_score", 0.0)),
                )
            )
        return samples

    def _select_primary_sample(self, samples: list[FaceSample]) -> FaceSample | None:
        if not samples:
            return None
        return max(samples, key=lambda sample: (sample.score, (sample.bbox[2] - sample.bbox[0]) * (sample.bbox[3] - sample.bbox[1])))

    def _collect_embeddings_from_media(self, media_path: Path) -> list[tuple[np.ndarray, float]]:
        """
        Sample frames every 2 seconds, detect and crop the face, extract embedding.

        Uses anchor tracking: the first detected face is the identity anchor.
        Subsequent frames are only accepted when their best face has cosine
        similarity >= ANCHOR_CONSISTENCY_THRESHOLD with the anchor, ensuring
        all collected embeddings belong to the same person even if others
        appear in the background.
        """
        results: list[tuple[np.ndarray, float]] = []
        anchor_embedding: np.ndarray | None = None

        for frame in self._sample_frames_for_enrollment(media_path):
            if frame is None:
                continue
            samples = self._detect_samples(frame)
            samples = [s for s in samples if s.score >= ENROLLMENT_MIN_DET_SCORE]
            if not samples:
                continue

            if anchor_embedding is None:
                # First face found — lock it in as the enrollment subject
                primary = self._select_primary_sample(samples)
                if primary is None:
                    continue
                anchor_embedding = primary.embedding
                results.append((primary.embedding, float(primary.score)))
                # Also enroll degraded versions to match distant classroom conditions
                for scale in ENROLLMENT_DEGRADATION_SCALES:
                    deg = _degrade_and_embed(self.face_analysis, frame, primary.bbox, scale)
                    if deg is not None:
                        results.append((deg, float(primary.score) * 0.9))
            else:
                # Pick whichever detected face is most similar to the anchor
                best = max(samples, key=lambda s: float(np.dot(s.embedding, anchor_embedding)))
                sim = float(np.dot(best.embedding, anchor_embedding))
                if sim >= ANCHOR_CONSISTENCY_THRESHOLD:
                    results.append((best.embedding, float(best.score)))
                    # Also enroll degraded versions
                    for scale in ENROLLMENT_DEGRADATION_SCALES:
                        deg = _degrade_and_embed(self.face_analysis, frame, best.bbox, scale)
                        if deg is not None:
                            results.append((deg, float(best.score) * 0.9))

        return results

    def _aggregate_embeddings(self, embeddings: list[np.ndarray], scores: list[float] | None = None) -> np.ndarray:
        normalized = np.stack([_normalize(e) for e in embeddings], axis=0).astype(np.float32)

        # Reject outliers: drop embeddings far from the initial centroid
        if len(normalized) >= 4:
            centroid = _normalize(normalized.mean(axis=0))
            sims = normalized @ centroid
            keep_mask = sims >= ENROLLMENT_OUTLIER_SIM_THRESHOLD
            if keep_mask.sum() >= 2:
                normalized = normalized[keep_mask]
                if scores is not None:
                    scores = [s for s, k in zip(scores, keep_mask.tolist()) if k]

        # Score-weighted mean so high-confidence frames contribute more
        if scores is not None and len(scores) == len(normalized):
            w = np.clip(np.array(scores, dtype=np.float32), 1e-6, None)
            w /= w.sum()
            mean_vec = (normalized * w[:, None]).sum(axis=0)
        else:
            mean_vec = normalized.mean(axis=0)

        return _normalize(mean_vec)

    def _rebuild_embeddings_from_media_samples(self, student: dict) -> tuple[list[np.ndarray], np.ndarray] | None:
        media_samples = student.get("media_samples", []) or []
        if not media_samples:
            return None

        collected_embeddings: list[np.ndarray] = []
        collected_scores: list[float] = []
        for sample in media_samples:
            file_name = str(sample.get("file_name", "")).strip()
            if not file_name:
                continue
            media_path = UPLOAD_DIR / file_name
            if not media_path.exists():
                continue
            for embedding, score in self._collect_embeddings_from_media(media_path):
                collected_embeddings.append(embedding)
                collected_scores.append(score)

        if not collected_embeddings:
            return None

        prototype = self._aggregate_embeddings(collected_embeddings, collected_scores)
        return collected_embeddings, prototype

    def _migrate_legacy_embeddings_if_needed(self) -> None:
        store = self._read_store()
        students = store.get("students", [])
        changed = False

        for student in students:
            if student.get("embedding_model") == EMBEDDING_MODEL_NAME:
                continue
            rebuilt = self._rebuild_embeddings_from_media_samples(student)
            if rebuilt is None:
                continue
            collected_embeddings, prototype = rebuilt
            previous_observations = int(student.get("observations", 0))
            student.update(
                {
                    "observations": max(previous_observations, len(collected_embeddings)),
                    "updated_at": _now_iso(),
                    "prototype": prototype.tolist(),
                    "embeddings": [embedding.tolist() for embedding in collected_embeddings],
                    "embedding_model": EMBEDDING_MODEL_NAME,
                }
            )
            changed = True

        if changed:
            self._write_store(store)

    def list_students(self) -> list[dict]:
        store = self._read_store()
        students = store.get("students", [])
        students.sort(key=lambda student: student.get("name", "").lower())
        return students

    def list_attendance(self, limit: int = 20) -> list[dict]:
        store = self._read_store()
        attendance = store.get("attendance", [])
        return attendance[-limit:][::-1]

    def delete_student(self, student_id: str) -> dict:
        store = self._read_store()
        students = store.get("students", [])
        attendance = store.get("attendance", [])

        target_index = next((index for index, student in enumerate(students) if student.get("student_id") == student_id), None)
        if target_index is None:
            raise KeyError(f"Student not found: {student_id}")

        removed_student = students.pop(target_index)
        removed_name = str(removed_student.get("name", "")).strip().lower()
        store["attendance"] = [
            row
            for row in attendance
            if str(row.get("student_name", "")).strip().lower() != removed_name
        ]
        self._write_store(store)

        return {
            "student": self._student_public(removed_student),
            "students": self.list_students(),
            "attendance": self.list_attendance(limit=20),
        }

    def enroll_student(self, student_name: str, media_paths: list[Path]) -> dict:
        normalized_name = student_name.strip()
        if not normalized_name:
            raise ValueError("student_name cannot be empty")
        if not media_paths:
            raise ValueError("Provide at least one photo or video for enrollment")

        all_pairs: list[tuple[np.ndarray, float]] = []
        media_summaries: list[dict] = []
        for media_path in media_paths:
            pairs = self._collect_embeddings_from_media(media_path)
            media_summaries.append({"file_name": media_path.name, "frame_samples": len(pairs)})
            all_pairs.extend(pairs)

        if not all_pairs:
            raise RuntimeError(
                "No face could be detected in the supplied media. "
                "Please use a clear photo or video where the person faces the camera (frontal or slight angle). "
                "Pure side profiles, extreme angles, and very dark/blurry images cannot be processed."
            )

        new_embeddings = [e for e, _ in all_pairs]
        new_scores = [s for _, s in all_pairs]
        new_prototype = self._aggregate_embeddings(new_embeddings, new_scores)

        store = self._read_store()
        students = store["students"]

        existing = next((student for student in students if student["name"].strip().lower() == normalized_name.lower()), None)
        if existing is None:
            existing = {
                "student_id": uuid.uuid4().hex[:12],
                "name": normalized_name,
                "observations": 0,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "embeddings": [],
                "prototype": [],
                "embedding_model": EMBEDDING_MODEL_NAME,
            }
            students.append(existing)

        previous_observations = int(existing.get("observations", 0))
        previous_prototype_list = existing.get("prototype", [])

        # Blend old and new prototypes weighted by observation counts so
        # re-enrollment sessions are proportionally represented
        if previous_observations > 0 and previous_prototype_list:
            old_proto = np.asarray(previous_prototype_list, dtype=np.float32)
            total = previous_observations + len(new_embeddings)
            merged_prototype = _normalize(
                old_proto * (previous_observations / total)
                + new_prototype * (len(new_embeddings) / total)
            )
        else:
            merged_prototype = new_prototype

        # Keep individual embeddings for future re-migration, capped to avoid growth
        previous_embeddings = [np.asarray(e, dtype=np.float32) for e in existing.get("embeddings", [])]
        all_embeddings = (previous_embeddings + new_embeddings)[-MAX_STORED_EMBEDDINGS:]

        existing.update(
            {
                "name": normalized_name,
                "observations": previous_observations + len(new_embeddings),
                "updated_at": _now_iso(),
                "prototype": merged_prototype.tolist(),
                "embeddings": [e.tolist() for e in all_embeddings],
                "media_samples": media_summaries,
                "embedding_model": EMBEDDING_MODEL_NAME,
            }
        )

        self._write_store(store)

        return {
            "student": self._student_public(existing),
            "media_samples": media_summaries,
            "enrollment_quality": {
                "frames_collected": len(all_pairs),
                "mean_det_score": round(float(np.mean(new_scores)), 3),
            },
        }

    def _student_public(self, student: dict) -> dict:
        return {
            "student_id": student.get("student_id"),
            "name": student.get("name"),
            "observations": int(student.get("observations", 0)),
            "updated_at": student.get("updated_at"),
        }

    def match_student(self, embedding: np.ndarray) -> dict:
        students = self.list_students()
        if not students:
            return {"match": None, "similarity": -1.0}

        best_student: dict | None = None
        best_similarity = -1.0
        q = _normalize(embedding)

        for student in students:
            # Compare against mean prototype
            prototype = np.asarray(student.get("prototype") or [], dtype=np.float32)
            sim = float(np.dot(q, prototype)) if prototype.size > 0 else -1.0

            # Also compare against every stored individual embedding and take max.
            # This catches cases where one specific enrollment frame matches the
            # current pose/lighting better than the mean prototype does.
            stored = student.get("embeddings") or []
            if stored:
                mat = np.asarray(stored, dtype=np.float32)   # (N, 512)
                best_individual = float((mat @ q).max())
                sim = max(sim, best_individual)

            if sim > best_similarity:
                best_similarity = sim
                best_student = student

        if best_student is None:
            return {"match": None, "similarity": best_similarity}

        if best_similarity < FACE_SIMILARITY_THRESHOLD:
            return {"match": None, "similarity": best_similarity}

        return {"match": self._student_public(best_student), "similarity": best_similarity}

    def mark_attendance(self, media_path: Path) -> dict:
        if not media_path.exists():
            raise FileNotFoundError(f"File not found: {media_path}")

        suffix = media_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            image = self._load_image(media_path)
            if image is None:
                raise RuntimeError("Could not read classroom photo")
            frame = image
        else:
            frames = self._sample_video_frames(media_path)
            if not frames:
                raise RuntimeError("Could not read classroom video")
            frame = frames[0]

        detections = self._detect_samples(frame)
        marked_frame = frame.copy()
        recognized: list[dict] = []
        unknown_faces = 0
        store = self._read_store()
        attendance_log = store["attendance"]

        # Match every detected face
        all_matches = [(det, self.match_student(det.embedding)) for det in detections]

        # Per student: keep only the single highest-scoring face
        # so if two faces both exceed the threshold for the same student,
        # only the best one gets the green box — the other stays Unknown.
        best_per_student: dict[str, tuple] = {}
        for det, match in all_matches:
            if match["match"] is not None:
                name = match["match"]["name"]
                if name not in best_per_student or match["similarity"] > best_per_student[name][1]:
                    best_per_student[name] = (det, match["similarity"], match)

        best_det_ids = {id(det) for det, _, _ in best_per_student.values()}

        for det, match in all_matches:
            x1, y1, x2, y2 = det.bbox
            is_best = match["match"] is not None and id(det) in best_det_ids

            if is_best:
                student = match["match"]
                color = (0, 200, 0)
                label = f"{student['name']} {match['similarity']:.2f}"
                if student["name"] not in {r["student"]["name"] for r in recognized}:
                    attendance_log.append({
                        "student_name": student["name"],
                        "recognized_at": _now_iso(),
                        "source": "classroom_photo",
                        "confidence": round(float(match["similarity"]), 4),
                    })
                    recognized.append({
                        "student": student,
                        "confidence": round(float(match["similarity"]), 4),
                        "bbox": [x1, y1, x2, y2],
                    })
                    # Incremental gallery growth: high-confidence classroom embeddings
                    # are added to the student's gallery so future matches improve.
                    if match["similarity"] >= 0.60:
                        all_students = store.get("students", [])
                        for s in all_students:
                            if s.get("student_id") == student.get("student_id"):
                                stored = s.get("embeddings", []) or []
                                new_emb = _normalize(det.embedding).tolist()
                                stored = (stored + [new_emb])[-MAX_STORED_EMBEDDINGS:]
                                s["embeddings"] = stored
                                # Recompute prototype
                                mat = np.asarray(stored, dtype=np.float32)
                                s["prototype"] = _normalize(mat.mean(axis=0)).tolist()
                                break
            else:
                unknown_faces += 1
                color = (0, 0, 255)
                sim = match["similarity"]
                label = f"Unknown {sim:.2f}"

            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                marked_frame, label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        store["attendance"] = attendance_log
        self._write_store(store)

        marked_name = f"{media_path.stem}_marked.jpg"
        marked_path = MARKED_DIR / marked_name
        cv2.imwrite(str(marked_path), marked_frame)

        return {
            "recognized": recognized,
            "unknown_faces": unknown_faces,
            "marked_path": str(marked_path),
            "marked_url": f"/api/attendance/artifacts/{marked_name}",
            "roster": self.list_students(),
            "attendance_log": self.list_attendance(limit=20),
        }


@lru_cache(maxsize=1)
def get_attendance_service() -> AttendanceService:
    return AttendanceService()
