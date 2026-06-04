"""Cognitive engagement pipeline — implements the paper:

'AI-Driven Framework for Enhancing Student Engagement Through Real-Time
Monitoring of Attendance and Cognitive Focus in Physical Classrooms'
(Rubasinghe et al., SCSE 2025)

Signals per face per sampled frame
───────────────────────────────────
• Eye gaze direction   dlib 68-pt landmarks → solvePnP head pose
                       → left / right / up / down / center
• Blink detection      Eye Aspect Ratio (EAR) from dlib eye landmarks
• Emotion              DeepFace → happy / sad / angry / surprise / fear / neutral
• Focus frame          center-ish gaze + open eyes + non-negative emotion

Per-student aggregation (paper § III-D)
────────────────────────────────────────
  concentration_pct  = focused_frames / total_frames × 100   (eq. 2)
  attention_level    = High (≥70%) / Medium (40-69%) / Low (<40%)
  dominant_emotion   = most frequent emotion across tracked frames
  detected_time_s    = frames_seen / fps                      (eq. 1)
  attendance         = Present if detected_time_s ≥ threshold  (default 600 s)

Face detection:  InsightFace buffalo_l + SAHI at det_size=1280
                 (same fine-tuned detector used by the attendance module)
Student tracking: cosine similarity on 512-d buffalo_l embeddings
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
import bz2
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.spatial import distance as dist

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import dlib
    DLIB_AVAILABLE = True
except Exception:
    dlib = None
    DLIB_AVAILABLE = False
    print("[cognitive] warning: dlib not available; landmark-based features disabled.")

from insightface.app import FaceAnalysis
from utils.retinaface_detector import sahi_face_detect

# ── constants ─────────────────────────────────────────────────────────────────

SAMPLE_EVERY_S       = 1.0    # analyse 1 frame per second
DET_SIZE             = 1280   # buffalo_l input resolution for better small-face recall
DET_THRESH           = 0.4
IDENTITY_THRESHOLD   = 0.40   # cosine similarity to re-identify the same student

EAR_BLINK_THRESH     = 0.40   # EAR below this → eye closed / blink
GAZE_YAW_THRESH      = 22.0   # degrees — beyond this = looking left/right
GAZE_PITCH_THRESH    = 18.0   # degrees — beyond this = looking up/down

# Emotions that indicate engagement (paper: "focused" state)
_FOCUS_EMOTIONS      = {"happy", "neutral", "surprise"}
# Emotions that indicate disengagement
_DISTRACT_EMOTIONS   = {"sad", "angry", "fear", "disgust"}

HIGH_ATTENTION_PCT   = 70.0   # ≥70 % → High
MEDIUM_ATTENTION_PCT = 40.0   # 40-69 % → Medium   <40 % → Low
ATTENDANCE_SECONDS   = 600.0  # 10-min threshold (paper eq. 1)
MIN_STUDENT_FRAMES   = 3      # fewer detections → discard track as noise

CLIP_FPS             = 5.0    # export fps for face-crop clips
CLIP_MAX_FRAMES      = 60     # cap stored frames per student

# dlib eye point indices (in the 68-point model)
_LEFT_EYE_IDX  = list(range(36, 42))
_RIGHT_EYE_IDX = list(range(42, 48))

# 3-D reference face model (mm) for solvePnP head-pose estimation
# Points: nose-tip(30), chin(8), L-eye-corner(36), R-eye-corner(45),
#         L-mouth(48), R-mouth(54)
_FACE_3D = np.array([
    ( 0.0,    0.0,    0.0),
    ( 0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

_LANDMARK_2D_IDX = [30, 8, 36, 45, 48, 54]   # indices in the 68-pt array


# ── dlib model download ───────────────────────────────────────────────────────

def _ensure_dlib_model() -> Path:
    model_dir  = Path.home() / ".dlib"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "shape_predictor_68_face_landmarks.dat"
    if model_path.exists():
        return model_path
    print("  [cognitive] downloading dlib 68-pt landmark model (~95 MB)…")
    bz2_path = model_path.with_suffix(".dat.bz2")
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, bz2_path)
    with bz2.open(bz2_path, "rb") as fin, open(model_path, "wb") as fout:
        fout.write(fin.read())
    bz2_path.unlink(missing_ok=True)
    print("  [cognitive] dlib model ready.")
    return model_path


# ── signal helpers ────────────────────────────────────────────────────────────

def _compute_ear(pts: np.ndarray) -> float:
    """Eye Aspect Ratio (Soukupová & Čech, 2016).  pts: (6,2) eye landmarks."""
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


def _head_pose(landmarks_2d: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Return (yaw_deg, pitch_deg) via solvePnP.
    Positive yaw  → head turned right.
    Positive pitch → head tilted up.
    """
    focal  = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)
    cam_matrix = np.array([
        [focal, 0,      center[0]],
        [0,     focal,  center[1]],
        [0,     0,      1        ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D, landmarks_2d.astype(np.float64),
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0

    rot, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix to Euler angles (yaw=Y, pitch=X, roll=Z)
    sy   = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw   = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw   = 0.0
    return float(yaw), float(pitch)


def _gaze_label(yaw: float, pitch: float) -> str:
    if abs(yaw) <= GAZE_YAW_THRESH and abs(pitch) <= GAZE_PITCH_THRESH:
        return "center"
    if abs(yaw) > abs(pitch):
        return "right" if yaw > 0 else "left"
    return "up" if pitch > 0 else "down"


def _analyze_emotion(face_crop: np.ndarray) -> tuple[str, dict[str, float]]:
    """Run DeepFace emotion analysis on a pre-cropped face image.
    Returns (dominant_emotion, {emotion: probability}).
    """
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            face_crop,
            actions=["emotion"],
            detector_backend="skip",   # we already have the crop
            enforce_detection=False,
            silent=True,
        )
        entry = result[0] if isinstance(result, list) else result
        raw   = entry.get("emotion", {})
        total = sum(raw.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in raw.items()}
        dom   = entry.get("dominant_emotion", max(probs, key=probs.get))
        return dom.lower(), probs
    except Exception:
        return "neutral", {"neutral": 1.0}


def _is_focused(gaze: str, ear: float, emotion: str) -> bool:
    """True if this frame counts as a 'good focus' frame (paper eq. 2)."""
    eyes_open  = ear >= EAR_BLINK_THRESH
    face_fwd   = gaze in {"center"}
    good_emot  = emotion in _FOCUS_EMOTIONS
    bad_emot   = emotion in _DISTRACT_EMOTIONS
    return eyes_open and face_fwd and not bad_emot


def _attention_level(pct: float) -> str:
    if pct >= HIGH_ATTENTION_PCT:
        return "High"
    if pct >= MEDIUM_ATTENTION_PCT:
        return "Medium"
    return "Low"


# ── per-frame sample record ───────────────────────────────────────────────────

class _Sample(NamedTuple):
    gaze:    str
    ear:     float
    emotion: str
    focused: bool
    emotion_probs: dict


# ── student identity bank ─────────────────────────────────────────────────────

class _StudentIdentity:
    def __init__(self, student_id: int, embedding: np.ndarray):
        self.student_id   = student_id
        self.prototype    = embedding.copy()
        self.observations = 1

    def update(self, embedding: np.ndarray) -> None:
        merged = (self.prototype * self.observations + embedding) / (self.observations + 1)
        norm   = float(np.linalg.norm(merged))
        self.prototype    = merged / norm if norm > 0 else merged
        self.observations += 1


# ── main pipeline ─────────────────────────────────────────────────────────────

class CognitivePipeline:
    """
    Usage::

        pipeline = CognitivePipeline()
        result   = pipeline.process(video_path, output_dir)
    """

    def __init__(
        self,
        sample_every_s:     float = SAMPLE_EVERY_S,
        det_size:           int   = DET_SIZE,
        attendance_seconds: float = ATTENDANCE_SECONDS,
    ):
        self.sample_every_s     = sample_every_s
        self.attendance_seconds = attendance_seconds

        print("  [cognitive] loading InsightFace buffalo_l …")
        self._fa = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._fa.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=DET_THRESH)

        if DLIB_AVAILABLE:
            print("  [cognitive] loading dlib shape predictor …")
            try:
                model_path   = _ensure_dlib_model()
                self._predictor = dlib.shape_predictor(str(model_path))
            except Exception:
                self._predictor = None
                print("  [cognitive] warning: failed to load dlib shape predictor; landmark-based features disabled.")
        else:
            self._predictor = None

        print("  [cognitive] warming up DeepFace …")
        _analyze_emotion(np.zeros((48, 48, 3), dtype=np.uint8))
        print("  [cognitive] ready.")

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, video_path: str | Path, output_dir: str | Path) -> dict:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_step  = max(1, int(round(fps * self.sample_every_s)))
        duration_s   = total_frames / fps if total_frames > 0 else 0.0

        print(f"  [cognitive] {video_path.name}  {total_frames} frames  {duration_s:.0f}s")

        # Per-track state
        samples:     dict[int, list[_Sample]]       = defaultdict(list)
        clip_frames: dict[int, list[np.ndarray]]    = defaultdict(list)
        frame_times: dict[int, list[float]]         = defaultdict(list)
        student_bank: list[_StudentIdentity]        = []
        next_sid     = 1

        frame_idx    = 0
        sampled      = 0
        t_start      = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            elapsed_s = frame_idx / fps
            ih, iw    = frame.shape[:2]
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── face detection: SAHI + buffalo_l 1280 ────────────────────────
            faces = sahi_face_detect(self._fa.get, frame, max_patches=None)

            for face in faces:
                x1, y1, x2, y2 = (int(v) for v in face.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # ── student identity via embedding ────────────────────────
                raw_emb = getattr(face, "normed_embedding", None)
                if raw_emb is None:
                    raw_emb = getattr(face, "embedding", None)
                embedding: np.ndarray | None = None
                if raw_emb is not None:
                    embedding = np.asarray(raw_emb, dtype=np.float32)
                    norm = float(np.linalg.norm(embedding))
                    if norm > 0:
                        embedding /= norm

                sid = self._resolve_identity(embedding, student_bank, next_sid)
                if sid == next_sid:
                    next_sid += 1

                if DLIB_AVAILABLE and self._predictor is not None:
                    # ── dlib 68-point landmarks ───────────────────────────
                    try:
                        drect = dlib.rectangle(x1, y1, x2, y2)
                        shape = self._predictor(gray, drect)
                        pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)

                        # EAR
                        l_ear = _compute_ear(pts[_LEFT_EYE_IDX])
                        r_ear = _compute_ear(pts[_RIGHT_EYE_IDX])
                        ear   = (l_ear + r_ear) / 2.0

                        # Head pose → gaze
                        ref2d  = pts[_LANDMARK_2D_IDX]
                        yaw, pitch = _head_pose(ref2d, iw, ih)
                        gaze   = _gaze_label(yaw, pitch)

                        # ── DeepFace emotion on tight face crop ───────────
                        pad    = max(0, int((y2 - y1) * 0.10))
                        cy1    = max(0, y1 - pad);  cy2 = min(ih, y2 + pad)
                        cx1    = max(0, x1 - pad);  cx2 = min(iw, x2 + pad)
                        crop   = frame[cy1:cy2, cx1:cx2]
                        if crop.size == 0:
                            emotion, emot_probs = "neutral", {"neutral": 1.0}
                        else:
                            face_img = cv2.resize(crop, (96, 96))
                            emotion, emot_probs = _analyze_emotion(face_img)

                        focused = _is_focused(gaze, ear, emotion)
                        samples[sid].append(_Sample(gaze, ear, emotion, focused, emot_probs))
                        frame_times[sid].append(elapsed_s)
                    except Exception:
                        pass

                # ── save annotated crop for clip ──────────────────────────
                if len(clip_frames[sid]) < CLIP_MAX_FRAMES:
                    annotated = self._annotate_crop(
                        frame, x1, y1, x2, y2, ih, iw,
                        gaze, ear, emotion, focused, elapsed_s,
                    )
                    clip_frames[sid].append(annotated)

            sampled += 1
            frame_idx += 1

            if sampled % 20 == 0 or frame_idx >= total_frames:
                pct = (frame_idx / max(1, total_frames)) * 100
                bar = "#" * int(pct / 3) + "-" * (33 - int(pct / 3))
                print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

        cap.release()
        print()

        # ── per-student scoring ───────────────────────────────────────────────
        students = []
        for sid, s_list in samples.items():
            if len(s_list) < MIN_STUDENT_FRAMES:
                continue
            n = len(s_list)

            focused_n    = sum(s.focused for s in s_list)
            conc_pct     = round(focused_n / n * 100, 1)
            attn_level   = _attention_level(conc_pct)
            action       = "Focused" if conc_pct >= MEDIUM_ATTENTION_PCT else "Not Focused"

            # Dominant emotion
            emot_counts: dict[str, int] = defaultdict(int)
            for s in s_list:
                emot_counts[s.emotion] += 1
            dom_emotion = max(emot_counts, key=emot_counts.get)

            # Emotion distribution (averaged probabilities)
            emot_dist: dict[str, float] = defaultdict(float)
            for s in s_list:
                for k, v in s.emotion_probs.items():
                    emot_dist[k] += v
            emot_dist = {k: round(v / n, 4) for k, v in emot_dist.items()}

            # Gaze distribution
            gaze_counts: dict[str, int] = defaultdict(int)
            for s in s_list:
                gaze_counts[s.gaze] += 1
            gaze_dist = {k: round(v / n, 4) for k, v in gaze_counts.items()}

            # Detected time (paper eq. 1)
            times = frame_times[sid]
            detected_time_s = round((times[-1] - times[0]) if len(times) > 1 else 0.0, 1)
            attendance = "Present" if detected_time_s >= self.attendance_seconds else "Absent"

            # Avg EAR + blink estimate
            avg_ear     = round(float(np.mean([s.ear for s in s_list])), 4)
            blink_count = sum(1 for s in s_list if s.ear < EAR_BLINK_THRESH)

            students.append({
                "student_id":         sid,
                "student_label":      f"Student {sid:03d}",
                "detected_action":    action,
                "attention_status":   attn_level,
                "emotion":            dom_emotion,
                "emotion_dist":       emot_dist,
                "gaze_dist":          gaze_dist,
                "concentration_pct":  conc_pct,
                "frames_tracked":     n,
                "focused_frames":     focused_n,
                "detected_time_s":    detected_time_s,
                "avg_ear":            avg_ear,
                "blink_count":        blink_count,
                "attendance":         attendance,
            })

        students.sort(key=lambda s: s["concentration_pct"], reverse=True)

        # ── write clips ───────────────────────────────────────────────────────
        print("  [cognitive] writing clips …")
        for student in students:
            sid    = student["student_id"]
            frames = clip_frames.get(sid, [])
            if not frames:
                student["clip"] = None
                continue
            fname     = f"student_{sid:03d}_{student['attention_status'].lower()}.mp4"
            clip_path = clips_dir / fname
            _write_clip(frames, clip_path, fps=CLIP_FPS)
            student["clip"] = fname

        # ── class summary ─────────────────────────────────────────────────────
        scores = [s["concentration_pct"] for s in students]
        class_conc = round(float(np.mean(scores)), 1) if scores else 0.0

        # Class-level emotion distribution
        all_emot: dict[str, float] = defaultdict(float)
        for s in students:
            for k, v in s["emotion_dist"].items():
                all_emot[k] += v
        ns = max(1, len(students))
        all_emot = {k: round(v / ns, 4) for k, v in all_emot.items()}

        summary = {
            "video_name":             video_path.name,
            "duration_seconds":       round(duration_s, 1),
            "frames_analyzed":        sampled,
            "student_count":          len(students),
            "class_concentration_pct": class_conc,
            "present_count":          sum(1 for s in students if s["attendance"] == "Present"),
            "absent_count":           sum(1 for s in students if s["attendance"] == "Absent"),
            "high_attention_count":   sum(1 for s in students if s["attention_status"] == "High"),
            "medium_attention_count": sum(1 for s in students if s["attention_status"] == "Medium"),
            "low_attention_count":    sum(1 for s in students if s["attention_status"] == "Low"),
            "class_emotion_dist":     dict(all_emot),
            "processing_time_s":      round(time.time() - t_start, 1),
        }

        # ── save outputs ──────────────────────────────────────────────────────
        summary_path = output_dir / "cognitive_summary.json"
        csv_path     = output_dir / "cognitive_students.csv"

        summary_path.write_text(json.dumps({"summary": summary, "students": students}, indent=2))

        _write_csv(csv_path, students)

        total_clips = sum(1 for s in students if s.get("clip"))
        print(f"  [cognitive] done — {len(students)} students  class conc {class_conc:.1f}%  {total_clips} clips")

        return {
            "summary":      summary,
            "students":     students,
            "summary_path": str(summary_path),
            "csv_path":     str(csv_path),
            "clips_dir":    str(clips_dir),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_identity(
        embedding: np.ndarray | None,
        bank:      list[_StudentIdentity],
        next_id:   int,
    ) -> int:
        if embedding is None:
            return next_id   # new anonymous track

        best_sim, best_ident = -1.0, None
        for ident in bank:
            sim = float(np.dot(ident.prototype, embedding))
            if sim > best_sim:
                best_sim, best_ident = sim, ident

        if best_ident is not None and best_sim >= IDENTITY_THRESHOLD:
            best_ident.update(embedding)
            return best_ident.student_id

        bank.append(_StudentIdentity(next_id, embedding))
        return next_id

    @staticmethod
    def _annotate_crop(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        ih: int, iw: int,
        gaze: str, ear: float, emotion: str, focused: bool,
        elapsed_s: float,
        pad: float = 0.20,
        size: int  = 224,
    ) -> np.ndarray:
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
        cx2 = min(iw, x2 + px); cy2 = min(ih, y2 + py)
        crop = frame[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        crop  = cv2.resize(crop, (size, size))

        color = (50, 200, 50) if focused else (50, 50, 220)
        cv2.rectangle(crop, (0, 0), (size - 1, size - 1), color, 3)

        label = f"{gaze.upper()} | {emotion}"
        if not focused:
            label += " !"
        cv2.putText(crop, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        cv2.putText(crop, f"EAR:{ear:.2f}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(crop, f"{elapsed_s:.0f}s", (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        return crop


# ── clip writer ───────────────────────────────────────────────────────────────

def _write_clip(frames: list[np.ndarray], path: Path, fps: float = CLIP_FPS) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    for fourcc_str in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h))
        if writer.isOpened():
            break
    for f in frames:
        writer.write(f)
    writer.release()


# ── CSV writer ────────────────────────────────────────────────────────────────

def _write_csv(path: Path, students: list[dict]) -> None:
    fields = [
        "student_id", "student_label", "detected_action", "attention_status",
        "emotion", "concentration_pct", "focused_frames", "frames_tracked",
        "detected_time_s", "attendance", "avg_ear", "blink_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(students)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="cognitive_output")
    args = ap.parse_args()

    pipeline = CognitivePipeline()
    result   = pipeline.process(args.video, args.out)
    print(json.dumps(result["summary"], indent=2))
