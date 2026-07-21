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
from insightface.utils import face_align
from .startup import ensure_insightface_models_flat
from .config import ATTENDANCE_DIR
from utils.adaface_backbone import AdaFaceWrapper, DEFAULT_CKPT_PATH as ADAFACE_CKPT_PATH


UPLOAD_DIR = ATTENDANCE_DIR / "uploads"
MARKED_DIR = ATTENDANCE_DIR / "marked"
PHOTOS_PER_VIDEO = 32

# Fixed whitelist of valid classrooms — every roster is scoped to one of these.
# Always validate a classroom id against this list before it touches a file path.
CLASSROOMS = [f"cse{i}" for i in range(1, 9)]
CLASSROOM_LABELS = {c: f"CSE {c[3:]}" for c in CLASSROOMS}


def _store_path(classroom_id: str) -> Path:
    if classroom_id not in CLASSROOMS:
        raise ValueError(f"Unknown classroom: {classroom_id!r}")
    return ATTENDANCE_DIR / f"attendance_store_{classroom_id}.json"


def migrate_legacy_flat_store() -> None:
    """One-time migration: this app used to have a single flat attendance_store.json
    shared by everyone. If it still exists and CSE 8's file doesn't, copy it over —
    the students originally enrolled there all belong to CSE 8."""
    legacy_path = ATTENDANCE_DIR / "attendance_store.json"
    cse8_path = _store_path("cse8")
    if legacy_path.exists() and not cse8_path.exists():
        cse8_path.parent.mkdir(parents=True, exist_ok=True)
        cse8_path.write_text(legacy_path.read_text())
# AdaFace IR-101's own p99 impostor threshold, derived from the human-labeled
# clean eval set (eval/impostor_scope_eval.py) — NOT glintr100's 0.38, the two
# backbones' cosine distributions aren't comparable. Below this, a face gets no
# name candidate at all ("Unknown").
FACE_SIMILARITY_THRESHOLD = 0.28
# At/above this, a match counts as confidently "Present". Between
# FACE_SIMILARITY_THRESHOLD and this, it's a real name candidate but not
# confident enough to auto-confirm — surfaced as "Suspicious" for a teacher
# to eyeball, rather than silently trusted or silently discarded.
PRESENT_SIMILARITY_THRESHOLD = 0.30
EMBEDDING_MODEL_NAME = "adaface_ir101_webface12m"
ENROLLMENT_MIN_DET_SCORE = 0.50
# Unchanged from glintr100: eval/build_gallery.py mirrors production enrollment
# and kept this same value (0.50) when building the AdaFace gallery used in
# every comparison run, so no evidence it needs to move for the new backbone.
ENROLLMENT_OUTLIER_SIM_THRESHOLD = 0.50
MAX_STORED_EMBEDDINGS = 128
MAX_ENROLLMENT_FRAMES = 30           # target/cap on sampled frames per enrollment video,
                                      # spread evenly across the clip regardless of its length
# Unchanged from glintr100 (see ENROLLMENT_OUTLIER_SIM_THRESHOLD note above) —
# eval/build_gallery.py's _ANCHOR_SIM_THRESH stayed at 0.35 for AdaFace too.
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


def _align_and_embed(adaface: AdaFaceWrapper, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Align a face via InsightFace's norm_crop (same alignment AdaFace was
    trained/calibrated against) and embed it with the AdaFace backbone."""
    aligned = face_align.norm_crop(frame, landmark=np.asarray(kps, dtype=np.float32), image_size=112)
    emb, _feat_norm = adaface.embed_aligned(aligned)
    return _normalize(emb)


def _degrade_and_embed(fa: FaceAnalysis, adaface: AdaFaceWrapper, frame: np.ndarray, bbox: tuple, scale: float) -> np.ndarray | None:
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

    # Paste degraded crop back into a copy of the frame and re-run detection
    frame_copy = frame.copy()
    frame_copy[y1:y2, x1:x2] = degraded
    faces = fa.get(frame_copy)

    # Pick the face closest to the original bbox
    best_kps, best_iou = None, 0.0
    for f in faces:
        fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
        ix1, iy1 = max(x1, fx1), max(y1, fy1)
        ix2, iy2 = min(x2, fx2), min(y2, fy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2-x1)*(y2-y1) + (fx2-fx1)*(fy2-fy1) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou and getattr(f, "kps", None) is not None:
            best_kps = f.kps
            best_iou = iou
    if best_kps is None:
        return None
    return _align_and_embed(adaface, frame_copy, best_kps)





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


@lru_cache(maxsize=1)
def _get_face_models() -> tuple[FaceAnalysis, AdaFaceWrapper]:
    """Load the shared, expensive ML models once — every classroom's
    AttendanceService reuses the same instances, only the roster JSON differs."""
    face_analysis = FaceAnalysis(
        name="antelopev2",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    face_analysis.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.5)
    adaface = AdaFaceWrapper.load(ADAFACE_CKPT_PATH)
    return face_analysis, adaface


class AttendanceService:
    def __init__(self, classroom_id: str) -> None:
        if classroom_id not in CLASSROOMS:
            raise ValueError(f"Unknown classroom: {classroom_id!r}")
        self.classroom_id = classroom_id
        self.store_path = _store_path(classroom_id)
        ensure_attendance_dirs()
        self.face_analysis, self.adaface = _get_face_models()
        self._migrate_legacy_embeddings_if_needed()

    def _read_store(self) -> dict:
        if not self.store_path.exists():
            return {"students": [], "attendance": [], "pending_reviews": []}
        try:
            data = json.loads(self.store_path.read_text())
        except Exception:
            return {"students": [], "attendance": [], "pending_reviews": []}
        data.setdefault("students", [])
        data.setdefault("attendance", [])
        data.setdefault("pending_reviews", [])
        return data

    def _write_store(self, data: dict) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(data, indent=2))

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
        """Sample up to MAX_ENROLLMENT_FRAMES frames, spread evenly across the
        whole clip, by decoding sequentially and subsampling afterward —
        never by seeking.

        This used to seek to computed positions (via CAP_PROP_POS_FRAMES),
        trusting CAP_PROP_FRAME_COUNT to know where those positions were. Both
        turned out to be unreliable for webm/matroska files recorded live by a
        browser's MediaRecorder, which streams encoded clusters via
        ondataavailable and never goes back to write a finalized duration/seek
        index: the reported frame count can be garbage (even negative), and —
        the harder-to-catch failure — even when seeking *looks* like it
        succeeded (returns frames, no error), sparse keyframes in a live
        VP8/VP9 stream mean it can silently keep landing on the same handful
        of early frames instead of actually spreading across the clip. A short
        enrollment clip is cheap to fully decode, so there's no real reason to
        seek at all here — only mark_attendance's _sample_video_frames (much
        less frequently hit, and normally handed a still photo anyway) keeps
        the seek-based path."""
        suffix = media_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            image = self._load_image(media_path)
            return [image] if image is not None else []

        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            return []

        # Flat cap generous enough for any realistic enrollment recording
        # (~100s at 30fps) regardless of what the container's own metadata
        # claims about duration/frame count.
        all_frames: list[np.ndarray] = []
        read_cap = 3000
        while len(all_frames) < read_cap:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            all_frames.append(frame)
        cap.release()

        if not all_frames:
            return []
        if len(all_frames) <= MAX_ENROLLMENT_FRAMES:
            return all_frames
        pick = np.linspace(0, len(all_frames) - 1, MAX_ENROLLMENT_FRAMES, dtype=int)
        return [all_frames[i] for i in pick]

    def _detect_samples(self, frame: np.ndarray) -> list[FaceSample]:
        faces = self.face_analysis.get(frame)
        samples: list[FaceSample] = []
        for face in faces:
            kps = getattr(face, "kps", None)
            if kps is None:
                continue
            bbox = tuple(int(value) for value in face.bbox)
            samples.append(
                FaceSample(
                    embedding=_align_and_embed(self.adaface, frame, kps),
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
                    deg = _degrade_and_embed(self.face_analysis, self.adaface, frame, primary.bbox, scale)
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
                        deg = _degrade_and_embed(self.face_analysis, self.adaface, frame, best.bbox, scale)
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

    def _add_embedding_to_gallery(self, store: dict, student_id: str, embedding: np.ndarray) -> None:
        """Append an embedding to a student's stored gallery and recompute
        their prototype. Shared by mark_attendance's automatic high-confidence
        growth and by a teacher manually confirming a suspicious match."""
        for s in store.get("students", []):
            if s.get("student_id") == student_id:
                stored = s.get("embeddings", []) or []
                new_emb = _normalize(embedding).tolist()
                stored = (stored + [new_emb])[-MAX_STORED_EMBEDDINGS:]
                s["embeddings"] = stored
                mat = np.asarray(stored, dtype=np.float32)
                s["prototype"] = _normalize(mat.mean(axis=0)).tolist()
                # observations is a lifetime counter (unlike embeddings, which is
                # capped and can evict old entries) — keep it moving so the
                # roster UI visibly reflects that this reinforced the gallery.
                s["observations"] = int(s.get("observations", 0)) + 1
                s["updated_at"] = _now_iso()
                break

    def resolve_suspicious_review(self, review_id: str, confirmed: bool) -> dict:
        """A teacher confirming or rejecting a 'suspicious' match from
        mark_attendance. Confirming reinforces the model — the embedding that
        triggered the suspicious match gets added to that student's gallery,
        the same way a high-confidence classroom match already does — and
        records the student as present. Rejecting just discards the pending
        review; nothing is added anywhere, since a wrong name is worse to
        learn from than a merely uncertain one."""
        store = self._read_store()
        pending = store.get("pending_reviews", [])

        review = next((r for r in pending if r.get("review_id") == review_id), None)
        if review is None:
            raise KeyError(f"No pending review: {review_id}")
        store["pending_reviews"] = [r for r in pending if r.get("review_id") != review_id]

        if confirmed:
            embedding = np.asarray(review["embedding"], dtype=np.float32)
            self._add_embedding_to_gallery(store, review["student_id"], embedding)
            store["attendance"].append({
                "student_name": review["student_name"],
                "recognized_at": _now_iso(),
                "source": "classroom_photo_confirmed",
                "confidence": review["similarity"],
            })

        self._write_store(store)
        return {
            "confirmed": confirmed,
            "student_name": review["student_name"],
            "roster": self.list_students(),
        }

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
        present: list[dict] = []
        suspicious: list[dict] = []
        unknown_faces = 0
        unknown_faces_detail: list[dict] = []
        store = self._read_store()
        attendance_log = store["attendance"]

        # Match every detected face
        all_matches = [(det, self.match_student(det.embedding)) for det in detections]

        # Per student: keep only the single highest-scoring face
        # so if two faces both exceed the threshold for the same student,
        # only the best one gets boxed — the other stays Unknown.
        best_per_student: dict[str, tuple] = {}
        for det, match in all_matches:
            if match["match"] is not None:
                name = match["match"]["name"]
                if name not in best_per_student or match["similarity"] > best_per_student[name][1]:
                    best_per_student[name] = (det, match["similarity"], match)

        best_det_ids = {id(det) for det, _, _ in best_per_student.values()}
        seen_student_ids: set[str] = set()

        for det, match in all_matches:
            x1, y1, x2, y2 = det.bbox
            is_best = match["match"] is not None and id(det) in best_det_ids

            if is_best:
                student = match["match"]
                similarity = match["similarity"]
                is_present = similarity >= PRESENT_SIMILARITY_THRESHOLD
                color = (0, 200, 0) if is_present else (0, 165, 255)  # green vs. amber (BGR)
                tag = "" if is_present else " (suspicious)"
                label = f"{student['name']} {similarity:.2f}{tag}"

                if student["student_id"] not in seen_student_ids:
                    seen_student_ids.add(student["student_id"])
                    entry = {"student": student, "confidence": round(float(similarity), 4), "bbox": [x1, y1, x2, y2]}
                    if is_present:
                        present.append(entry)
                        # Only confident matches get written to the attendance record —
                        # a "suspicious" match is a candidate for a teacher to review,
                        # not something to silently record as confirmed attendance.
                        attendance_log.append({
                            "student_name": student["name"],
                            "recognized_at": _now_iso(),
                            "source": "classroom_photo",
                            "confidence": round(float(similarity), 4),
                        })
                        # Incremental gallery growth: high-confidence classroom embeddings
                        # are added to the student's gallery so future matches improve.
                        if similarity >= 0.60:
                            self._add_embedding_to_gallery(store, student["student_id"], det.embedding)
                    else:
                        # Hold the embedding behind a review_id rather than acting on
                        # it now — a teacher confirming/rejecting it is what decides
                        # whether it reinforces this student's gallery (see
                        # resolve_suspicious_review). Not returned in the API
                        # response itself; only the review_id is.
                        review_id = uuid.uuid4().hex
                        store.setdefault("pending_reviews", []).append({
                            "review_id": review_id,
                            "student_id": student["student_id"],
                            "student_name": student["name"],
                            "similarity": round(float(similarity), 4),
                            "embedding": _normalize(det.embedding).tolist(),
                            "created_at": _now_iso(),
                        })
                        entry["review_id"] = review_id
                        suspicious.append(entry)
            else:
                unknown_faces += 1
                color = (0, 0, 255)
                sim = match["similarity"]
                label = f"Unknown {sim:.2f}"
                unknown_faces_detail.append({
                    "bbox": [x1, y1, x2, y2],
                    "similarity": round(float(sim), 4),
                })

            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                marked_frame, label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        # Everyone enrolled in this classroom who wasn't matched at all (present
        # or suspicious) in this photo.
        absent = [
            self._student_public(s) for s in store.get("students", [])
            if s.get("student_id") not in seen_student_ids
        ]

        store["attendance"] = attendance_log
        self._write_store(store)

        marked_name = f"{media_path.stem}_marked.jpg"
        marked_path = MARKED_DIR / marked_name
        cv2.imwrite(str(marked_path), marked_frame)

        return {
            "present": present,
            "suspicious": suspicious,
            "absent": absent,
            "unknown_faces": unknown_faces,
            "unknown_faces_detail": unknown_faces_detail,
            "marked_path": str(marked_path),
            "marked_url": f"/api/attendance/artifacts/{marked_name}",
            "roster": self.list_students(),
        }


@lru_cache(maxsize=None)
def get_attendance_service(classroom_id: str) -> AttendanceService:
    # Fix nested insightface model folders if present before initializing
    try:
        ensure_insightface_models_flat(["antelopev2"])
    except Exception:
        pass
    return AttendanceService(classroom_id)
