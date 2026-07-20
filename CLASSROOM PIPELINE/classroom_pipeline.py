"""Classroom Analysis Pipeline

Burst sampling + face re-identification + signal-based action classification.

Every SAMPLE_EVERY_SECONDS the pipeline collects BURST_FRAMES consecutive
frames, detects every student via face + body, re-identifies them with face
embeddings across the whole video, classifies their action for that window,
saves a clip, and builds a per-student timeline.

Output
------
  summary JSON  — per-student timeline + aggregate stats
  CSV           — one row per student per window
  clips/        — student_001/window_0030s_writing.mp4 …
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.spatial import distance as dist

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

from utils.adaface_backbone import AdaFaceWrapper, DEFAULT_CKPT_PATH as ADAFACE_CKPT_PATH
from utils.roster_match import load_roster, match_against_roster

# ── constants ─────────────────────────────────────────────────────────────────

BURST_FRAMES         = 24      # consecutive frames per analysis window
SAMPLE_EVERY_SECONDS = 30.0    # one window every N seconds
# Cosine similarity for within-video re-identification (AdaFace embedding space).
# Deliberately a bit more conservative than the Attendance roster-match threshold
# (0.28, derived from eval/impostor_scope_eval.py) — a false merge here silently
# conflates two different students' timelines, which is worse than a false split
# (which just double-counts one student under two IDs).
IDENTITY_THRESHOLD   = 0.35
MIN_BURST_DETECTIONS = 6       # min frames face must appear in burst to count
FACE_IOU_THRESHOLD   = 0.30    # IoU to link face across burst frames
CLIP_FPS             = 5.0     # clip playback speed
CLIP_EXPORT_SIZE     = 256     # square pixels per clip frame

KPS_CONF      = 0.30
POSE_CONF     = 0.40
PHONE_CONF    = 0.40
FACE_THRESH   = 0.40
MOTION_THRESH = 1.2
FACE_HEAD_FRAC = 0.65          # top fraction of body bbox used to find face

EAR_BLINK_THRESH  = 0.40
MAR_TALK_THRESH       = 0.20
MIN_MOUTH_TRANSITIONS = 5    # open↔closed cycles needed to count as talking
GAZE_YAW_THRESH   = 22.0
GAZE_PITCH_THRESH = 18.0

HIGH_ATTENTION_PCT   = 70.0
MEDIUM_ATTENTION_PCT = 40.0
# Minimum mouth-closed concentration_pct to call a window "Attentive". Derived
# from a threshold sweep against a labeled dataset (activity_dataset/datasets/
# annotations_split2*.csv, n=156 scored clips): accuracy plateaus at ~67-70%
# for conc_pct thresholds from 60-95, peaking around 80-85, vs. ~60% at a
# threshold of 50 and effectively random (EAR showed no separation at all)
# under the old EAR-based formula this replaced.
ATTENTIVE_CONC_THRESH = 80.0

_FOCUS_EMOTIONS    = {"happy", "neutral", "surprise"}
_DISTRACT_EMOTIONS = {"sad", "angry", "fear", "disgust"}

# YOLOv8-pose COCO keypoint indices
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SH,  _R_SH  = 5,  6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP,   _R_HIP   = 11, 12

# Index mapping into InsightFace's `landmark_2d_106` output (replaces the old
# dlib 68-point scheme, which required a separate ~95MB model that was never
# actually installed in this environment — see git history for the dlib version).
# These indices were derived empirically (not from memorized docs): a real face's
# 106 points were plotted and numbered, confirming eye clusters at 33-42/87-96,
# eyebrows at 43-51/97-105, mouth at 52-71 (outer 52-61, inner 62-71), nose at
# 72-86, and jaw contour at 0-32. Each 6/8-point subset below is ordered to match
# what `_compute_ear`/`_compute_mar` (unchanged) expect: corner, upper.., corner,
# lower.. — same shape as the dlib 68-point convention they were written for.
_EYE_LEFT_IDX_106  = [35, 41, 42, 39, 37, 36]
_EYE_RIGHT_IDX_106 = [89, 95, 96, 93, 91, 90]
_MOUTH_INNER_IDX_106 = [65, 63, 71, 67, 69, 70, 62, 66]
# nose tip, chin, left-eye outer corner, right-eye outer corner, left mouth corner, right mouth corner
_LANDMARK_2D_IDX = [80, 0, 35, 93, 52, 61]

_FACE_3D = np.array([
    (  0.0,    0.0,    0.0),
    (  0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

ACTION_COLORS = {
    "On Phone":   (0,   0,   220),
    "Sleeping":   (0,   0,   220),
    "Writing":    (0,  140,  255),
    "Talking":    (0,  200,  200),
    "Attentive":  (50, 200,   50),
    "Distracted": (120, 120, 120),
}


# ── signal helpers ────────────────────────────────────────────────────────────

def _compute_ear(pts: np.ndarray) -> float:
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


def _compute_mar(lip: np.ndarray) -> float:
    A = dist.euclidean(lip[1], lip[7])
    B = dist.euclidean(lip[2], lip[6])
    C = dist.euclidean(lip[3], lip[5])
    D = dist.euclidean(lip[0], lip[4])
    return float((A + B + C) / (3.0 * D)) if D > 0 else 0.0


def _mouth_transitions(mouth_open_flags: list[bool]) -> int:
    """Count open↔closed state changes — talking produces repeated cycling."""
    return sum(1 for a, b in zip(mouth_open_flags, mouth_open_flags[1:]) if a != b)


def _head_pose(landmarks_2d: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float]:
    focal = float(frame_w)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    cam = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D, landmarks_2d.astype(np.float64),
        cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy    = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    pitch = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
    yaw   = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))) if sy > 1e-6 else 0.0
    return yaw, pitch


def _gaze_label(yaw: float, pitch: float) -> str:
    if abs(yaw) <= GAZE_YAW_THRESH and abs(pitch) <= GAZE_PITCH_THRESH:
        return "center"
    if abs(yaw) > abs(pitch):
        return "right" if yaw > 0 else "left"
    return "up" if pitch > 0 else "down"


def _head_state_from_kps(kps: np.ndarray) -> str:
    face_vis = sum(1 for i in range(5) if float(kps[i, 2]) > KPS_CONF)
    if face_vis < 3:
        sh = float(kps[_L_SH, 2]) > KPS_CONF or float(kps[_R_SH, 2]) > KPS_CONF
        return "down" if sh else "unknown"
    l_ear_vis = float(kps[_L_EAR, 2]) > KPS_CONF
    r_ear_vis = float(kps[_R_EAR, 2]) > KPS_CONF
    if l_ear_vis != r_ear_vis:
        return "sideways"
    if float(kps[_L_EYE, 2]) > KPS_CONF and float(kps[_R_EYE, 2]) > KPS_CONF and float(kps[_NOSE, 2]) > KPS_CONF:
        eye_y   = (float(kps[_L_EYE, 1]) + float(kps[_R_EYE, 1])) / 2.0
        nose_y  = float(kps[_NOSE, 1])
        eye_sep = max(1.0, abs(float(kps[_L_EYE, 0]) - float(kps[_R_EYE, 0])))
        if nose_y > eye_y + eye_sep * 2.0:
            return "down"
    return "forward"


def _motion_in_region(prev_gray: np.ndarray, curr_gray: np.ndarray,
                      bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    p = prev_gray[y1:y2, x1:x2]
    c = curr_gray[y1:y2, x1:x2]
    if p.size == 0 or c.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _normalize(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def _attention_level(pct: float) -> str:
    if pct >= HIGH_ATTENTION_PCT:   return "High"
    if pct >= MEDIUM_ATTENTION_PCT: return "Medium"
    return "Low"


def _detect_action(phone_pct, sleeping_pct, writing_pct,
                   sideways_pct, conc_pct, mouth_transitions: int = 0) -> str:
    if phone_pct    >= 0.25: return "On Phone"
    if sleeping_pct >= 0.25: return "Sleeping"
    if writing_pct  >= 0.35: return "Writing"
    # Require actual open↔closed cycling — a yawn or smile won't fire.
    mouth_talking = mouth_transitions >= MIN_MOUTH_TRANSITIONS and sleeping_pct < 0.25
    if sideways_pct >= 0.30 or mouth_talking: return "Talking"
    if conc_pct     >= ATTENTIVE_CONC_THRESH: return "Attentive"
    return "Distracted"


def _analyze_emotion(face_crop: np.ndarray) -> tuple[str, dict]:
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            face_crop, actions=["emotion"],
            detector_backend="skip", enforce_detection=False, silent=True,
        )
        entry = result[0] if isinstance(result, list) else result
        raw   = entry.get("emotion", {})
        total = sum(raw.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in raw.items()}
        dom   = entry.get("dominant_emotion", max(probs, key=probs.get))
        return dom.lower(), probs
    except Exception:
        return "neutral", {"neutral": 1.0}


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class StudentIdentity:
    """Persistent cross-window student identity backed by face embeddings."""
    student_id:   int
    prototype:    np.ndarray
    observations: int = 1

    def update(self, embedding: np.ndarray) -> None:
        combined = (self.prototype * self.observations + embedding) / (self.observations + 1)
        n = float(np.linalg.norm(combined))
        self.prototype = combined / n if n > 0 else combined
        self.observations += 1


@dataclass
class _BurstTrack:
    """One student tracked within a single burst window."""
    track_id:    int
    n_frames:    int
    face_boxes:  list = field(default_factory=list)   # (x1,y1,x2,y2) or None per frame
    pose_boxes:  list = field(default_factory=list)   # body bbox or None per frame
    kps_list:    list = field(default_factory=list)   # keypoints or None per frame
    embeddings:  list = field(default_factory=list)   # face embedding or None per frame
    lmk106_list: list = field(default_factory=list)   # 106-pt face landmarks or None per frame
    export_crops: list = field(default_factory=list)  # annotated body crop per frame
    motions:     list = field(default_factory=list)   # optical-flow magnitudes
    phone_hits:  list = field(default_factory=list)   # bool per frame
    observations: int = 0
    last_face_box: tuple | None = None

    def __post_init__(self):
        self.face_boxes   = [None] * self.n_frames
        self.pose_boxes   = [None] * self.n_frames
        self.kps_list     = [None] * self.n_frames
        self.embeddings   = [None] * self.n_frames
        self.lmk106_list  = [None] * self.n_frames
        self.export_crops = [None] * self.n_frames
        self.motions      = [0.0]  * self.n_frames
        self.phone_hits   = [False] * self.n_frames


# ── model path helpers ────────────────────────────────────────────────────────

def _pose_model_path() -> Path:
    repo = Path(__file__).resolve().parents[1]
    for name in ("yolov8m-pose.pt", "yolov8s-pose.pt", "yolov8n-pose.pt"):
        if (repo / name).exists():
            return repo / name
    return Path("yolov8s-pose.pt")


def _phone_model_path() -> Path:
    repo = Path(__file__).resolve().parents[1]
    for name in ("yolo11m.pt", "yolov8m.pt", "yolov8n.pt"):
        if (repo / name).exists():
            return repo / name
    for alt in (
        repo / "Activity monitoring" / "Training Pipelines" / "assets" / "yolo11m.pt",
    ):
        if alt.exists():
            return alt
    return Path("yolov8n.pt")


# ── clip saving ───────────────────────────────────────────────────────────────

def _save_clip(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        encoded = path.with_suffix(".enc.mp4")
        ret = subprocess.run(
            [ffmpeg, "-y", "-i", str(tmp), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
            capture_output=True,
        )
        if ret.returncode == 0 and encoded.stat().st_size > 0:
            encoded.replace(path)
            tmp.unlink(missing_ok=True)
            return
    tmp.replace(path)


def _annotated_crop(
    frame: np.ndarray,
    face_box: tuple | None,
    pose_box: tuple | None,
    ih: int, iw: int,
    action: str,
    timestamp_s: float,
    student_label: str,
) -> np.ndarray:
    color = ACTION_COLORS.get(action, (200, 200, 200))
    size  = CLIP_EXPORT_SIZE

    # Use body bbox if available, else 2× padded face bbox
    if pose_box is not None:
        bx1, by1, bx2, by2 = pose_box
    elif face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        bw = fx2 - fx1
        bh = fy2 - fy1
        bx1 = max(0, fx1 - bw)
        bx2 = min(iw, fx2 + bw)
        by1 = max(0, fy1 - bh)
        by2 = min(ih, fy2 + int(bh * 3))
    else:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Slight padding
    pw = int((bx2 - bx1) * 0.06)
    ph = int((by2 - by1) * 0.04)
    bx1 = max(0, bx1 - pw); bx2 = min(iw, bx2 + pw)
    by1 = max(0, by1 - ph); by2 = min(ih, by2 + ph)

    crop = frame[by1:by2, bx1:bx2].copy()
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Letterbox to square
    ch, cw = crop.shape[:2]
    scale  = size / max(ch, cw, 1)
    rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop   = cv2.resize(crop, (rw, rh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    ox, oy = (size - rw) // 2, (size - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = crop

    cv2.rectangle(canvas, (0, 0), (size-1, size-1), color, 4)
    cv2.putText(canvas, action,        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color,          2, cv2.LINE_AA)
    cv2.putText(canvas, student_label, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    mm, ss = int(timestamp_s) // 60, int(timestamp_s) % 60
    cv2.putText(canvas, f"{mm:02d}:{ss:02d}", (6, size-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


# ── main pipeline ─────────────────────────────────────────────────────────────

class ClassroomPipeline:
    """
    Usage::

        pipeline = ClassroomPipeline()
        result   = pipeline.process(video_path, output_dir)
    """

    def __init__(
        self,
        pose_model_path:     str | Path | None = None,
        phone_model_path:    str | Path | None = None,
        burst_frames:        int   = BURST_FRAMES,
        sample_every_seconds: float = SAMPLE_EVERY_SECONDS,
        identity_threshold:  float = IDENTITY_THRESHOLD,
        roster_path:         str | Path | None = None,
    ):
        self.burst_frames         = burst_frames
        self.sample_every_seconds = sample_every_seconds
        self.identity_threshold   = identity_threshold

        self._roster = load_roster(roster_path) if roster_path else []
        if roster_path:
            print(f"  [classroom] loaded {len(self._roster)} enrolled student(s) from roster.")

        print("  [classroom] loading YOLOv8-pose …")
        self._pose = YOLO(str(pose_model_path or _pose_model_path()))

        print("  [classroom] loading phone detector …")
        phone_path = Path(phone_model_path or _phone_model_path())
        self._phone = YOLO(str(phone_path)) if phone_path.exists() else None
        self._phone_cls_ids: set[int] = set()
        if self._phone is not None:
            names = self._phone.names or {}
            self._phone_cls_ids = {
                cid for cid, n in names.items()
                if any(w in n.lower() for w in ("phone", "cell", "mobile"))
            }
            if not self._phone_cls_ids:
                self._phone_cls_ids = set(names.keys())

        print("  [classroom] loading InsightFace buffalo_l (detection + 106pt landmarks) …")
        self._fa = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CPUExecutionProvider"],
        )
        self._fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=FACE_THRESH)

        print("  [classroom] loading AdaFace IR-101 …")
        self._adaface = AdaFaceWrapper.load(ADAFACE_CKPT_PATH)

        print("  [classroom] warming DeepFace …")
        _analyze_emotion(np.zeros((48, 48, 3), dtype=np.uint8))
        print("  [classroom] ready.")

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, video_path: str | Path, output_dir: str | Path) -> dict:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s   = total_frames / fps

        # Frame index of each window start
        window_step   = max(1, int(round(fps * self.sample_every_seconds)))
        window_starts = set(range(0, total_frames or 10**9, window_step))

        print(f"  [classroom] {video_path.name}  {total_frames} frames  "
              f"{duration_s:.0f}s  window every {window_step} frames "
              f"({self.sample_every_seconds:.0f}s)  burst={self.burst_frames}")

        student_bank: list[StudentIdentity] = []
        next_student_id = 1
        all_window_records: list[dict] = []

        active_window: dict | None = None
        frame_idx   = 0
        t_start     = time.time()
        progress_step = max(1, (total_frames // 100) if total_frames > 0 else 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in window_starts:
                active_window = {
                    "start_frame": frame_idx,
                    "start_seconds": round(frame_idx / fps, 1),
                    "frames": [],
                }

            if active_window is not None:
                active_window["frames"].append(frame.copy())
                if len(active_window["frames"]) == self.burst_frames:
                    records, next_student_id = self._process_burst(
                        active_window["frames"],
                        active_window["start_frame"],
                        active_window["start_seconds"],
                        fps,
                        clips_dir,
                        video_path.stem,
                        student_bank,
                        next_student_id,
                    )
                    all_window_records.extend(records)
                    active_window = None

            if total_frames > 0 and (frame_idx % progress_step == 0 or frame_idx == total_frames - 1):
                pct = frame_idx / total_frames * 100
                bar = "#" * int(pct / 3) + "-" * (33 - int(pct / 3))
                print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

            frame_idx += 1

        cap.release()
        print()

        # ── aggregate per-student ─────────────────────────────────────────────
        students = self._aggregate(all_window_records, student_bank)

        summary = {
            "video_name":       video_path.name,
            "duration_seconds": round(duration_s, 1),
            "total_windows":    len({r["window_start_seconds"] for r in all_window_records}),
            "student_count":    len(students),
            "processing_time_s": round(time.time() - t_start, 1),
            "class_attentive_pct": round(
                sum(s["attentive_pct"] for s in students) / max(1, len(students)), 1
            ),
            "students": students,
            "windows":  all_window_records,
        }

        summary_path = output_dir / "classroom_summary.json"
        csv_path     = output_dir / "classroom_timeline.csv"
        summary_path.write_text(json.dumps(summary, indent=2))
        self._write_csv(csv_path, all_window_records)

        print(f"  [classroom] done — {len(students)} students  "
              f"{len(all_window_records)} window records  "
              f"{summary['class_attentive_pct']:.1f}% class attentive")

        return {
            "summary":      summary,
            "summary_path": str(summary_path),
            "csv_path":     str(csv_path),
            "clips_dir":    str(clips_dir),
        }

    # ── burst processing ──────────────────────────────────────────────────────

    def _process_burst(
        self,
        frames: list[np.ndarray],
        start_frame: int,
        start_seconds: float,
        fps: float,
        clips_dir: Path,
        video_stem: str,
        student_bank: list[StudentIdentity],
        next_student_id: int,
    ) -> tuple[list[dict], int]:

        ih, iw = frames[0].shape[:2]
        n_frames = len(frames)
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Per-frame detections
        face_dets_per_frame: list[list[dict]] = []   # [{bbox, embedding, score}]
        pose_dets_per_frame: list[list[dict]] = []   # [{bbox, kps}]
        phone_boxes_per_frame: list[list[tuple]] = []

        for fi, frame in enumerate(frames):
            # Face detection
            face_dets = []
            raw_faces = self._fa.get(frame)
            for face in (raw_faces or []):
                kps = getattr(face, "kps", None)
                emb = None
                if kps is not None:
                    aligned = face_align.norm_crop(frame, landmark=np.asarray(kps, dtype=np.float32), image_size=112)
                    emb, _feat_norm = self._adaface.embed_aligned(aligned)
                    emb = _normalize(np.asarray(emb, dtype=np.float32))
                lmk106 = getattr(face, "landmark_2d_106", None)
                if lmk106 is not None:
                    lmk106 = np.asarray(lmk106, dtype=np.float32)
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                face_dets.append({
                    "bbox": (max(0,x1), max(0,y1), min(iw,x2), min(ih,y2)),
                    "embedding": emb,
                    "lmk106": lmk106,
                    "score": float(face.det_score),
                })
            face_dets_per_frame.append(face_dets)

            # Pose detection (no tracking — within-burst IoU matching handles it)
            pose_dets = []
            pose_res = self._pose(frame, conf=POSE_CONF, verbose=False)
            if pose_res and pose_res[0].boxes is not None:
                bboxes  = pose_res[0].boxes.xyxy.cpu().numpy()
                kps_all = pose_res[0].keypoints.data.cpu().numpy() if pose_res[0].keypoints else None
                for pi, bbox in enumerate(bboxes):
                    bx1, by1, bx2, by2 = (int(v) for v in bbox)
                    kps = kps_all[pi] if kps_all is not None and pi < len(kps_all) else None
                    pose_dets.append({
                        "bbox": (max(0,bx1), max(0,by1), min(iw,bx2), min(ih,by2)),
                        "kps":  kps,
                    })
            pose_dets_per_frame.append(pose_dets)

            # Phone detection
            phone_boxes = []
            if self._phone is not None:
                ph = self._phone(frame, conf=PHONE_CONF, verbose=False)
                if ph and ph[0].boxes is not None:
                    for box, cls in zip(ph[0].boxes.xyxy.cpu().numpy(),
                                        ph[0].boxes.cls.cpu().numpy()):
                        if int(cls) in self._phone_cls_ids:
                            phone_boxes.append(tuple(int(v) for v in box[:4]))
            phone_boxes_per_frame.append(phone_boxes)

        # ── track faces within burst using IoU ────────────────────────────────
        tracks: list[_BurstTrack] = []
        next_track_id = 1

        for fi in range(n_frames):
            face_dets = face_dets_per_frame[fi]
            unmatched = set(range(len(face_dets)))

            for track in tracks:
                if track.last_face_box is None:
                    continue
                best_idx, best_iou = None, 0.0
                for di in list(unmatched):
                    iou = _iou(track.last_face_box, face_dets[di]["bbox"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, di
                if best_idx is None or best_iou < FACE_IOU_THRESHOLD:
                    continue

                det = face_dets[best_idx]
                unmatched.discard(best_idx)
                track.last_face_box  = det["bbox"]
                track.face_boxes[fi] = det["bbox"]
                track.embeddings[fi] = det["embedding"]
                track.lmk106_list[fi] = det["lmk106"]
                track.observations  += 1

                # Match to best overlapping pose detection
                best_pose, best_pose_iou = None, 0.0
                for pd in pose_dets_per_frame[fi]:
                    piou = _iou(det["bbox"], pd["bbox"])
                    if piou > best_pose_iou:
                        best_pose_iou, best_pose = piou, pd
                if best_pose is not None and best_pose_iou > 0.05:
                    track.pose_boxes[fi] = best_pose["bbox"]
                    track.kps_list[fi]   = best_pose["kps"]

                # Phone in face/body region
                check_box = track.pose_boxes[fi] or det["bbox"]
                cx, cy = (check_box[0]+check_box[2])//2, (check_box[1]+check_box[3])//2
                for pb in phone_boxes_per_frame[fi]:
                    pcx, pcy = (pb[0]+pb[2])//2, (pb[1]+pb[3])//2
                    if abs(pcx-cx) < (check_box[2]-check_box[0]) and abs(pcy-cy) < (check_box[3]-check_box[1]):
                        track.phone_hits[fi] = True

                # Motion (lower half of pose box or face region)
                if fi > 0:
                    mb = track.pose_boxes[fi] or det["bbox"]
                    track.motions[fi] = _motion_in_region(grays[fi-1], grays[fi], mb)

            # Start new tracks for unmatched detections
            for di in unmatched:
                det = face_dets[di]
                t   = _BurstTrack(track_id=next_track_id, n_frames=n_frames)
                next_track_id += 1
                t.last_face_box  = det["bbox"]
                t.face_boxes[fi] = det["bbox"]
                t.embeddings[fi] = det["embedding"]
                t.lmk106_list[fi] = det["lmk106"]
                t.observations   = 1
                for pd in pose_dets_per_frame[fi]:
                    if _iou(det["bbox"], pd["bbox"]) > 0.05:
                        t.pose_boxes[fi] = pd["bbox"]
                        t.kps_list[fi]   = pd["kps"]
                        break
                tracks.append(t)

        # ── batch identity resolution (one-to-one per burst) ──────────────────
        # Resolving each track independently and sequentially would let two different
        # real people seen in the *same* burst both match (and blend into) the same
        # existing identity. Instead resolve the whole burst as a one-to-one assignment:
        # highest-similarity (track, identity) pairs win first, and once a track or an
        # identity is claimed, neither can be reused within this burst.
        track_embeddings: dict[int, np.ndarray] = {}
        for track in tracks:
            if track.observations < MIN_BURST_DETECTIONS:
                continue
            valid_embs = [e for e in track.embeddings if e is not None]
            if not valid_embs:
                continue
            track_embeddings[track.track_id] = _normalize(np.mean(np.stack(valid_embs), axis=0))

        candidates = []  # (similarity, track_id, identity_index)
        for track_id, emb in track_embeddings.items():
            for idx, identity in enumerate(student_bank):
                sim = float(np.dot(identity.prototype, emb))
                if sim >= self.identity_threshold:
                    candidates.append((sim, track_id, idx))
        candidates.sort(key=lambda c: c[0], reverse=True)

        resolved: dict[int, tuple[int, float]] = {}   # track_id -> (student_id, similarity)
        claimed_tracks: set[int] = set()
        claimed_identities: set[int] = set()
        for sim, track_id, idx in candidates:
            if track_id in claimed_tracks or idx in claimed_identities:
                continue
            claimed_tracks.add(track_id)
            claimed_identities.add(idx)
            identity = student_bank[idx]
            identity.update(track_embeddings[track_id])
            resolved[track_id] = (identity.student_id, sim)

        for track_id, emb in track_embeddings.items():
            if track_id in resolved:
                continue
            new_id = next_student_id
            next_student_id += 1
            student_bank.append(StudentIdentity(student_id=new_id, prototype=emb.copy()))
            resolved[track_id] = (new_id, 0.0)

        # ── roster recognition — read-only lookup against enrolled Attendance
        # students, independent of the anonymous student_bank above ────────────
        recognized_by_track: dict[int, tuple[str | None, float]] = {}
        if self._roster:
            for track_id, emb in track_embeddings.items():
                recognized_by_track[track_id] = match_against_roster(emb, self._roster)

        # ── per-track classification + clip ──────────────────────────────────
        records: list[dict] = []

        for track in tracks:
            if track.track_id not in track_embeddings:
                continue

            student_id, sim = resolved[track.track_id]
            student_label = f"student_{student_id:03d}"
            recognized_name, recognized_sim = recognized_by_track.get(track.track_id, (None, -1.0))

            # ── body signals ──────────────────────────────────────────────────
            head_states  = []
            motions      = []
            phone_frames = 0
            for fi in range(n_frames):
                kps = track.kps_list[fi]
                if kps is not None:
                    head_states.append(_head_state_from_kps(kps))
                if track.motions[fi] > 0:
                    motions.append(track.motions[fi])
                if track.phone_hits[fi]:
                    phone_frames += 1

            n_hs = max(1, len(head_states))
            forward_n  = sum(1 for s in head_states if s == "forward")
            down_n     = sum(1 for s in head_states if s == "down")
            sideways_n = sum(1 for s in head_states if s == "sideways")
            avg_motion = float(np.mean(motions)) if motions else 0.0

            writing_n  = sum(
                1 for fi in range(n_frames)
                if (track.kps_list[fi] is not None
                    and _head_state_from_kps(track.kps_list[fi]) == "down"
                    and track.motions[fi] > MOTION_THRESH)
            )
            sleeping_n = sum(
                1 for fi in range(n_frames)
                if (track.kps_list[fi] is not None
                    and _head_state_from_kps(track.kps_list[fi]) == "down"
                    and track.motions[fi] <= MOTION_THRESH
                    and not track.phone_hits[fi])
            )

            phone_pct    = phone_frames / n_frames
            writing_pct  = writing_n   / n_frames
            sleeping_pct = sleeping_n  / n_frames
            sideways_pct = sideways_n  / n_hs
            forward_pct  = forward_n   / n_hs

            # ── face signals ──────────────────────────────────────────────────
            ears, mars, pitches, gazes, emotions = [], [], [], [], []
            for fi in range(n_frames):
                pts = track.lmk106_list[fi]
                if pts is None:
                    continue
                try:
                    ears.append((_compute_ear(pts[_EYE_LEFT_IDX_106]) + _compute_ear(pts[_EYE_RIGHT_IDX_106])) / 2)
                    mars.append(_compute_mar(pts[_MOUTH_INNER_IDX_106]))
                    yaw, pitch = _head_pose(pts[_LANDMARK_2D_IDX], iw, ih)
                    pitches.append(pitch)
                    gazes.append(_gaze_label(yaw, pitch))
                except Exception:
                    pass

            # Emotion — run once on middle frame face crop (expensive)
            dom_emotion, emot_probs = "neutral", {}
            mid_fi = n_frames // 2
            for search_fi in [mid_fi] + list(range(n_frames)):
                fb = track.face_boxes[search_fi]
                if fb is None:
                    continue
                fx1, fy1, fx2, fy2 = fb
                pad = max(0, int((fy2 - fy1) * 0.10))
                ec  = frames[search_fi][
                    max(0,fy1-pad):min(ih,fy2+pad),
                    max(0,fx1-pad):min(iw,fx2+pad),
                ]
                if ec.size > 0:
                    dom_emotion, emot_probs = _analyze_emotion(cv2.resize(ec, (96, 96)))
                break

            # Aggregate face signals
            avg_ear = float(np.mean(ears)) if ears else None
            mouth_flags    = [m >= MAR_TALK_THRESH for m in mars]
            mouth_open_pct = sum(mouth_flags) / max(1, len(mouth_flags))
            n_mouth_trans  = _mouth_transitions(mouth_flags)
            center_gaze_n  = sum(1 for g in gazes if g == "center")
            # Concentration is driven by mouth-closed fraction, not EAR/gaze.
            # A 300-clip calibration run against a real labeled dataset
            # (activity_dataset/datasets/annotations_split2*.csv) showed EAR has
            # ~zero separation between attentive and non-attentive clips (means
            # 0.199 vs 0.213), while mouth_open_pct separates them cleanly
            # (medians 0.0 vs ~0.96) — attentive clips keep the mouth closed,
            # everything else (talking/phone/distracted/head down/head side)
            # tends to have it open.
            if mouth_flags:
                conc_pct = round((1.0 - mouth_open_pct) * 100, 1)
            else:
                # fallback: body signals, no face detected this window
                conc_pct = round(forward_pct * 100, 1)

            action     = _detect_action(phone_pct, sleeping_pct, writing_pct,
                                        sideways_pct, conc_pct, n_mouth_trans)
            attn_level = _attention_level(conc_pct)

            # ── build annotated clip frames ───────────────────────────────────
            export_frames = []
            for fi in range(n_frames):
                crop = _annotated_crop(
                    frames[fi],
                    track.face_boxes[fi],
                    track.pose_boxes[fi],
                    ih, iw,
                    action, start_seconds, student_label,
                )
                export_frames.append(crop)

            # Save clip
            mm, ss = int(start_seconds) // 60, int(start_seconds) % 60
            clip_name = f"window_{mm:02d}m{ss:02d}s_{action.lower().replace(' ', '_')}.mp4"
            student_clip_dir = clips_dir / student_label
            clip_path = student_clip_dir / clip_name
            _save_clip(export_frames, clip_path)

            records.append({
                "window_start_frame":   start_frame,
                "window_start_seconds": start_seconds,
                "student_id":           student_id,
                "student_label":        student_label,
                "identity_similarity":  round(float(sim), 4),
                "recognized_name":      recognized_name,
                "recognized_similarity": round(float(recognized_sim), 4) if recognized_name else None,
                "action":               action,
                "attention_status":     attn_level,
                "concentration_pct":    conc_pct,
                "emotion":              dom_emotion,
                "phone_pct":            round(phone_pct, 3),
                "writing_pct":          round(writing_pct, 3),
                "sleeping_pct":         round(sleeping_pct, 3),
                "sideways_pct":         round(sideways_pct, 3),
                "mouth_open_pct":       round(mouth_open_pct, 3),
                "avg_ear":              round(avg_ear, 4) if avg_ear is not None else None,
                "avg_motion":           round(avg_motion, 3),
                "face_frames":          len(ears),
                "body_frames":          len(head_states),
                "observations":         track.observations,
                "clip":                 f"{student_label}/{clip_name}",
            })

        return records, next_student_id

    # ── aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, records: list[dict], student_bank: list[StudentIdentity]) -> list[dict]:
        by_student: dict[int, list[dict]] = {}
        for r in records:
            by_student.setdefault(r["student_id"], []).append(r)

        # Merge anonymous IDs that both confidently matched the same enrolled student.
        # Within-video re-identification (student_bank, one evolving prototype per ID)
        # and roster recognition (a full multi-embedding enrollment gallery) use
        # different thresholds and can disagree about whether two appearances are
        # "the same track" even when the roster match agrees they're the same real
        # person — e.g. lighting/pose drift can push two bursts of the same person
        # below the re-ID threshold while both still clear the roster threshold
        # independently. Trust the roster match and merge.
        name_to_sids: dict[str, list[int]] = {}
        for sid, wins in by_student.items():
            names = [w["recognized_name"] for w in wins if w.get("recognized_name")]
            if not names:
                continue
            majority_name = Counter(names).most_common(1)[0][0]
            name_to_sids.setdefault(majority_name, []).append(sid)

        for name, sids in name_to_sids.items():
            if len(sids) < 2:
                continue
            primary, *duplicates = sorted(sids)
            for dup in duplicates:
                by_student[primary].extend(by_student.pop(dup))

        students = []
        for sid, wins in sorted(by_student.items()):
            n = len(wins)
            action_counts: dict[str, int] = {}
            for w in wins:
                action_counts[w["action"]] = action_counts.get(w["action"], 0) + 1

            dominant_action = max(action_counts, key=action_counts.get)
            attentive_n = sum(1 for w in wins if w["action"] in ("Attentive", "Writing"))
            attn_pct    = round(attentive_n / n * 100, 1)

            concs = [w["concentration_pct"] for w in wins]
            avg_conc = round(float(np.mean(concs)), 1) if concs else 0.0

            timeline = sorted(wins, key=lambda w: w["window_start_seconds"])

            recognized_names = [w["recognized_name"] for w in wins if w.get("recognized_name")]
            recognized_name  = Counter(recognized_names).most_common(1)[0][0] if recognized_names else None

            students.append({
                "student_id":       sid,
                "student_label":    f"student_{sid:03d}",
                "recognized_name":  recognized_name,
                "windows_seen":     n,
                "dominant_action":  dominant_action,
                "attentive_pct":    attn_pct,
                "avg_concentration_pct": avg_conc,
                "action_breakdown": {
                    k: round(v / n * 100, 1) for k, v in action_counts.items()
                },
                "timeline": timeline,
            })

        students.sort(key=lambda s: s["attentive_pct"], reverse=True)
        return students

    # ── CSV ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_csv(path: Path, records: list[dict]) -> None:
        fields = [
            "window_start_seconds", "student_id", "student_label", "recognized_name",
            "action", "attention_status", "concentration_pct",
            "emotion", "phone_pct", "writing_pct", "sleeping_pct",
            "sideways_pct", "mouth_open_pct", "avg_ear", "avg_motion",
            "face_frames", "body_frames", "observations",
            "identity_similarity", "recognized_similarity", "clip",
        ]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(records, key=lambda r: (r["student_id"], r["window_start_seconds"])))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="classroom_output")
    ap.add_argument("--burst", type=int, default=BURST_FRAMES)
    ap.add_argument("--interval", type=float, default=SAMPLE_EVERY_SECONDS)
    args = ap.parse_args()
    p = ClassroomPipeline(burst_frames=args.burst, sample_every_seconds=args.interval)
    r = p.process(args.video, args.out)
    print(json.dumps(r["summary"], indent=2, default=str))
