"""Combined engagement + cognitive pipeline.

Merges:
  Body  (YOLOv8-pose + ByteTrack)  → head_state, motion, phone use
  Face  (InsightFace buffalo_l)     → identity, gaze (solvePnP), blink (EAR)
        (DeepFace)                  → emotion

Per-student signals
───────────────────
  detected_action  : On Phone | Writing | Sleeping | Talking | Attentive | Distracted
  attention_status : High (≥70%) | Medium (40-69%) | Low (<40%)
  engagement_score : body-signal weighted score  (same formula as engagement pipeline)
  concentration_pct: focused_frames / total_frames × 100   (paper eq. 2)
  detected_time_s  : frames seen / fps                      (paper eq. 1)
  attendance       : Present if detected_time_s ≥ threshold (default 600 s)
"""

from __future__ import annotations

import bz2
import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from insightface.app import FaceAnalysis
from ultralytics import YOLO

# ── constants ─────────────────────────────────────────────────────────────────

TARGET_FPS        = 1.0    # body-analysis sample rate (frames/second)
FACE_EVERY        = 1      # run face analysis every N sampled body frames
MOTION_THRESH     = 1.2    # optical-flow magnitude → "moving" (writing)
NOTEBOOK_CONF     = 0.30   # confidence threshold for book/notebook YOLO detections
KPS_CONF          = 0.3
POSE_CONF         = 0.4
PHONE_CONF        = 0.40
FACE_DET_THRESH   = 0.4
FACE_HEAD_FRAC    = 0.65   # top fraction of body bbox to use as head search region
MIN_TRACK_FRAMES  = 5
CLIP_SAVE_EVERY   = 1
CLIP_MAX_FRAMES   = 60
CLIP_FPS          = 5.0

EAR_BLINK_THRESH     = 0.25
GAZE_YAW_THRESH      = 22.0
GAZE_PITCH_THRESH    = 18.0
WRITING_PITCH_THRESH = -10.0  # degrees; negative = looking down at desk
_FOCUS_EMOTIONS      = {"happy", "neutral", "surprise"}
_DISTRACT_EMOTIONS   = {"sad", "angry", "fear", "disgust"}
HIGH_ATTENTION_PCT   = 70.0
MEDIUM_ATTENTION_PCT = 40.0
ATTENDANCE_SECONDS   = 600.0

# YOLOv8-pose COCO keypoint indices
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SH, _R_SH = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP,   _R_HIP   = 11, 12

# Optical flow around wrist → writing range (too still = not writing; too fast = gesturing)
WRIST_MOTION_MIN  = 0.35
WRIST_MOTION_MAX  = 5.0
WRIST_PATCH_PX    = 28   # half-side of patch around each wrist

# dlib 68-pt eye landmark indices
_LEFT_EYE_IDX  = list(range(36, 42))
_RIGHT_EYE_IDX = list(range(42, 48))

# dlib inner-lip landmark indices for Mouth Aspect Ratio (MAR)
# 60=left-corner  61=upper-left  62=upper-mid  63=upper-right
# 64=right-corner 65=lower-right 66=lower-mid  67=lower-left
_INNER_LIP_IDX = list(range(60, 68))
MAR_TALK_THRESH      = 0.20  # MAR above this → mouth open
MIN_MOUTH_TRANSITIONS = 5    # need at least this many open↔closed cycles to count as talking

# 3-D reference face model (mm) for solvePnP
_FACE_3D = np.array([
    (  0.0,    0.0,    0.0),   # nose tip       (landmark 30)
    (  0.0, -330.0,  -65.0),   # chin           (landmark  8)
    (-225.0,  170.0, -135.0),  # left eye corner (landmark 36)
    ( 225.0,  170.0, -135.0),  # right eye corner(landmark 45)
    (-150.0, -150.0, -125.0),  # left mouth      (landmark 48)
    ( 150.0, -150.0, -125.0),  # right mouth     (landmark 54)
], dtype=np.float64)
_LANDMARK_2D_IDX = [30, 8, 36, 45, 48, 54]


# ── dlib model ────────────────────────────────────────────────────────────────

def _ensure_dlib_model() -> Path:
    model_dir  = Path.home() / ".dlib"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "shape_predictor_68_face_landmarks.dat"
    if model_path.exists():
        return model_path
    print("  [combined] downloading dlib 68-pt landmark model (~95 MB) …")
    bz2_path = model_path.with_suffix(".dat.bz2")
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, bz2_path)
    with bz2.open(bz2_path, "rb") as fin, open(model_path, "wb") as fout:
        fout.write(fin.read())
    bz2_path.unlink(missing_ok=True)
    return model_path


# ── signal helpers ────────────────────────────────────────────────────────────

def _compute_ear(pts: np.ndarray) -> float:
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


def _compute_mar(pts: np.ndarray) -> float:
    """Mouth Aspect Ratio from inner-lip landmarks (indices 60-67).

    pts must be the full 68-pt array (or at least up to index 67).
    MAR = (|61-67| + |62-66| + |63-65|) / (3 * |60-64|)
    High MAR (>= MAR_TALK_THRESH) → mouth open → talking.
    """
    lip = pts[60:68]          # 8 inner-lip points, local indices 0-7
    A = dist.euclidean(lip[1], lip[7])   # 61↔67
    B = dist.euclidean(lip[2], lip[6])   # 62↔66
    C = dist.euclidean(lip[3], lip[5])   # 63↔65
    D = dist.euclidean(lip[0], lip[4])   # 60↔64 (width)
    return float((A + B + C) / (3.0 * D)) if D > 0 else 0.0


def _mouth_transitions(mouth_open_flags: list[bool]) -> int:
    """Count open↔closed state changes — talking produces repeated cycling."""
    return sum(1 for a, b in zip(mouth_open_flags, mouth_open_flags[1:]) if a != b)


def _head_pose(landmarks_2d: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float]:
    focal      = float(frame_w)
    cx, cy     = frame_w / 2.0, frame_h / 2.0
    cam_matrix = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D, landmarks_2d.astype(np.float64),
        cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    pitch = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
    yaw   = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))) if sy > 1e-6 else 0.0
    return yaw, pitch


def _gaze_label(yaw: float, pitch: float) -> str:
    if abs(yaw) <= GAZE_YAW_THRESH and abs(pitch) <= GAZE_PITCH_THRESH:
        return "center"
    if abs(yaw) > abs(pitch):
        return "right" if yaw > 0 else "left"
    return "up" if pitch > 0 else "down"


def _analyze_emotion(face_crop: np.ndarray) -> tuple[str, dict[str, float]]:
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


def _head_state_from_kps(kps: np.ndarray) -> str:
    """kps: (17, 3) – x, y, confidence.  Returns forward | down | sideways | unknown."""
    face_visible = sum(1 for i in range(5) if float(kps[i, 2]) > KPS_CONF)
    if face_visible < 3:
        sh_vis = float(kps[_L_SH, 2]) > KPS_CONF or float(kps[_R_SH, 2]) > KPS_CONF
        return "down" if sh_vis else "unknown"

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


def _motion_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray,
                      bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    ry1 = y1 + (y2 - y1) // 2
    p, c = prev_gray[ry1:y2, x1:x2], curr_gray[ry1:y2, x1:x2]
    if p.size == 0 or c.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))


def _wrist_writing_signal(
    kps: np.ndarray,
    prev_gray: np.ndarray | None,
    curr_gray: np.ndarray,
    ih: int, iw: int,
) -> bool:
    """Return True if wrist keypoints suggest the student is writing.

    Strategy:
      1. For each visible wrist, check it is at or below hip level
         (i.e., hands are resting on the desk that's off-frame).
      2. Compute dense optical flow in a small patch around the wrist.
      3. Writing motion falls in [WRIST_MOTION_MIN, WRIST_MOTION_MAX] —
         more than 'perfectly still' but less than broad arm gestures.
    At least one wrist must satisfy both the position and motion check.
    """
    if prev_gray is None:
        return False

    # Best visible hip Y — desk surface is roughly at this height
    hip_y: float | None = None
    for hi in (_L_HIP, _R_HIP):
        if float(kps[hi, 2]) > KPS_CONF:
            hip_y = float(kps[hi, 1])
            break

    for wrist_idx in (_L_WRIST, _R_WRIST):
        if float(kps[wrist_idx, 2]) < KPS_CONF:
            continue
        wx = int(float(kps[wrist_idx, 0]))
        wy = int(float(kps[wrist_idx, 1]))
        if wx < 0 or wy < 0 or wx >= iw or wy >= ih:
            continue

        # Wrist must be at or below hip (hands on desk, desk cut off by frame)
        if hip_y is not None and wy < hip_y - 15:
            continue

        # Dense optical flow in a patch around the wrist
        x1p = max(0, wx - WRIST_PATCH_PX)
        x2p = min(iw, wx + WRIST_PATCH_PX)
        y1p = max(0, wy - WRIST_PATCH_PX)
        y2p = min(ih, wy + WRIST_PATCH_PX)
        p = prev_gray[y1p:y2p, x1p:x2p]
        c = curr_gray[y1p:y2p, x1p:x2p]
        if p.size < 64:
            continue
        flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 2, 10, 2, 5, 1.1, 0)
        mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        if WRIST_MOTION_MIN <= mag <= WRIST_MOTION_MAX:
            return True

    return False


def _bbox_contains_phone(student_bbox: tuple, phone_boxes: list) -> bool:
    sx1, sy1, sx2, sy2 = student_bbox
    for px1, py1, px2, py2 in phone_boxes:
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        if sx1 <= pcx <= sx2 and sy1 <= pcy <= sy2:
            return True
        if sx1 <= px1 <= sx2 and sy1 <= py1 <= sy2:
            return True
    return False


def _bbox_has_notebook_on_desk(
    student_bbox: tuple,
    notebook_boxes: list,
    ih: int,
    iw: int,
    head_state: str = "unknown",
) -> bool:
    """Return True if a book/notebook is detected in the student's desk region.

    When head_state == 'down' the student is looking toward the desk.
    The camera typically cuts off above the desk surface, so we project
    significantly *below* the body bbox — up to 2.5× the body height —
    to reach the area the student is gazing at.

    For all other head states we fall back to a conservative region
    (lower half of body + 25 % below).
    """
    sx1, sy1, sx2, sy2 = student_bbox
    bh = sy2 - sy1
    bw = sx2 - sx1

    if head_state == "down":
        # Gaze projected desk region: start from just below mid-body,
        # extend 2.5× the body height downward (desk is well below frame crop).
        desk_y1 = sy1 + int(bh * 0.40)
        desk_y2 = min(ih, sy2 + int(bh * 2.5))
        # Wider horizontal window — student may lean slightly sideways
        desk_x1 = max(0,  sx1 - int(bw * 0.35))
        desk_x2 = min(iw, sx2 + int(bw * 0.35))
    else:
        desk_y1 = sy1 + int(bh * 0.50)
        desk_y2 = min(ih, sy2 + int(bh * 0.25))
        desk_x1 = max(0,  sx1 - int(bw * 0.15))
        desk_x2 = min(iw, sx2 + int(bw * 0.15))

    for nx1, ny1, nx2, ny2 in notebook_boxes:
        ncx = (nx1 + nx2) / 2.0
        ncy = (ny1 + ny2) / 2.0
        if desk_x1 <= ncx <= desk_x2 and desk_y1 <= ncy <= desk_y2:
            return True
        # Also match on any corner falling in the region (partially visible notebook)
        for px, py in ((nx1, ny1), (nx2, ny1), (nx1, ny2), (nx2, ny2)):
            if desk_x1 <= px <= desk_x2 and desk_y1 <= py <= desk_y2:
                return True
    return False


def _attention_level(pct: float) -> str:
    if pct >= HIGH_ATTENTION_PCT:   return "High"
    if pct >= MEDIUM_ATTENTION_PCT: return "Medium"
    return "Low"


def _detect_action(phone_pct: float, sleeping_pct: float, writing_pct: float,
                   sideways_pct: float, conc_pct: float, mouth_transitions: int = 0) -> str:
    if phone_pct    >= 0.25: return "On Phone"
    if sleeping_pct >= 0.25: return "Sleeping"
    if writing_pct  >= 0.35: return "Writing"
    # Talking: head turned sideways OR mouth cycles open↔closed (real speech, not a yawn/smile).
    # Require MIN_MOUTH_TRANSITIONS crossings so a single open or a static grin doesn't fire.
    mouth_talking = mouth_transitions >= MIN_MOUTH_TRANSITIONS and sleeping_pct < 0.25
    if sideways_pct >= 0.30 or mouth_talking: return "Talking"
    if conc_pct     >= 50.0: return "Attentive"
    return "Distracted"


# ── per-frame sample records ──────────────────────────────────────────────────

class _BodySample(NamedTuple):
    head_state:       str
    motion:           float
    phone_detected:   bool
    notebook_on_desk: bool
    wrist_writing:    bool


class _FaceSample(NamedTuple):
    gaze:          str
    ear:           float
    emotion:       str
    focused:       bool
    emotion_probs: dict
    pitch:         float   # head pitch in degrees (negative = looking down)
    mouth_open:    bool    # MAR >= MAR_TALK_THRESH → mouth open / talking


# ── model path helpers ────────────────────────────────────────────────────────

def _default_pose_model() -> Path:
    """Return the best available pose model, downloading yolov8s-pose if needed.

    Preference: m > s > n  (accuracy).  s is the sweet-spot for classroom video.
    Ultralytics downloads the weights automatically on first use when the file
    doesn't exist — we just need to pass the bare filename as the model name.
    """
    repo = Path(__file__).resolve().parents[1]
    # Return any already-downloaded model (best first)
    for name in ("yolov8m-pose.pt", "yolov8s-pose.pt", "yolov8n-pose.pt"):
        if (repo / name).exists():
            return repo / name
    # None downloaded yet — trigger auto-download of the small model
    print("  [combined] yolov8s-pose.pt not found — downloading (~23 MB) …")
    from ultralytics import YOLO
    YOLO("yolov8s-pose.pt")          # ultralytics saves to its cache
    import shutil
    cache = Path.home() / ".config" / "Ultralytics" / "yolov8s-pose.pt"
    alt   = Path.home() / "ultralytics" / "assets" / "yolov8s-pose.pt"
    for src in (cache, alt):
        if src.exists():
            dest = repo / "yolov8s-pose.pt"
            shutil.copy2(src, dest)
            print(f"  [combined] saved to {dest}")
            return dest
    # Ultralytics keeps it in its own cache dir — just return the bare name
    # so YOLO() picks it up from cache on the next call
    return Path("yolov8s-pose.pt")


def _default_phone_model() -> Path:
    repo = Path(__file__).resolve().parents[1]
    for name in ("yolo11m.pt", "yolov8m.pt", "yolov8n.pt"):
        if (repo / name).exists():
            return repo / name
    return repo / "yolo11m.pt"


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


def _write_csv(path: Path, students: list[dict]) -> None:
    fields = [
        "track_id", "student_label", "detected_action", "attention_status",
        "engagement_score", "emotion", "concentration_pct",
        "head_forward_pct", "writing_pct", "sleeping_pct", "phone_pct", "notebook_pct", "mouth_open_pct",
        "frames_tracked", "face_frames", "detected_time_s",
        "attendance", "avg_ear", "blink_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(students)


# ── annotated body-crop for clip ──────────────────────────────────────────────

_ACTION_COLOR = {
    "On Phone":  (0,   0,   220),
    "Sleeping":  (0,   0,   220),
    "Writing":   (0,  140,  255),
    "Talking":   (0,  200,  200),
    "Attentive": (50, 200,   50),
    "Distracted":(120, 120, 120),
}


def _annotated_body_crop(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ih: int, iw: int,
    head_state: str, motion: float, phone: bool,
    gaze: str | None, emotion: str | None, elapsed_s: float,
    size: int = 224,
) -> np.ndarray:
    # Determine action and color
    if phone:
        action, color = "On Phone",  _ACTION_COLOR["On Phone"]
    elif head_state == "down" and motion <= MOTION_THRESH:
        action, color = "Sleeping",  _ACTION_COLOR["Sleeping"]
    elif head_state == "down":
        action, color = "Writing",   _ACTION_COLOR["Writing"]
    elif head_state == "sideways":
        action, color = "Talking",   _ACTION_COLOR["Talking"]
    elif head_state == "forward":
        action, color = "Attentive", _ACTION_COLOR["Attentive"]
    else:
        action, color = "Distracted", _ACTION_COLOR["Distracted"]

    # Body crop with small padding
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * 0.08), int(bh * 0.05)
    cx1 = max(0, x1 - px); cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px); cy2 = min(ih, y2 + py)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Resize to fixed square
    ch, cw = crop.shape[:2]
    scale  = size / max(ch, cw)
    rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop   = cv2.resize(crop, (rw, rh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    ox, oy = (size - rw) // 2, (size - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = crop
    crop = canvas

    cv2.rectangle(crop, (0, 0), (size - 1, size - 1), color, 4)
    line1 = f"{action}" + (f" | {gaze}" if gaze else "")
    line2 = emotion or ""
    cv2.putText(crop, line1,          (6, 20),         cv2.FONT_HERSHEY_SIMPLEX, 0.50, color,          2, cv2.LINE_AA)
    cv2.putText(crop, line2,          (6, 40),         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(crop, f"{elapsed_s:.0f}s", (6, size-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return crop


# ── main pipeline ─────────────────────────────────────────────────────────────

class CombinedPipeline:
    """
    Usage::

        pipeline = CombinedPipeline()
        result   = pipeline.process(video_path, output_dir)
    """

    def __init__(
        self,
        pose_model_path:    str | Path | None = None,
        phone_model_path:   str | Path | None = None,
        target_fps:         float = TARGET_FPS,
        face_every:         int   = FACE_EVERY,
        attendance_seconds: float = ATTENDANCE_SECONDS,
    ):
        self.target_fps         = target_fps
        self.face_every         = face_every
        self.attendance_seconds = attendance_seconds

        print("  [combined] loading YOLOv8-pose …")
        self._pose_model = YOLO(str(pose_model_path or _default_pose_model()))

        print("  [combined] loading phone detector …")
        phone_path = Path(phone_model_path or _default_phone_model())
        self._phone_model = YOLO(str(phone_path)) if phone_path.exists() else None
        self._phone_cls_ids:    set[int] = set()
        self._notebook_cls_ids: set[int] = set()
        if self._phone_model is not None:
            _names = self._phone_model.names or {}
            self._phone_cls_ids = {
                cid for cid, n in _names.items()
                if any(w in n.lower() for w in ("phone", "cell", "mobile"))
            }
            if not self._phone_cls_ids:
                self._phone_cls_ids = set(_names.keys())
            self._notebook_cls_ids = {
                cid for cid, n in _names.items()
                if any(w in n.lower() for w in ("book", "notebook", "laptop", "pen", "pencil"))
            }
            print(f"  [combined] notebook class IDs: {self._notebook_cls_ids} "
                  f"({[_names[c] for c in self._notebook_cls_ids]})")

        print("  [combined] loading InsightFace buffalo_l …")
        self._fa = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=FACE_DET_THRESH)

        print("  [combined] loading dlib shape predictor …")
        self._predictor = dlib.shape_predictor(str(_ensure_dlib_model()))

        print("  [combined] warming up DeepFace …")
        _analyze_emotion(np.zeros((48, 48, 3), dtype=np.uint8))
        print("  [combined] ready.")

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
        sample_step  = max(1, int(round(fps / self.target_fps)))
        duration_s   = total_frames / fps if total_frames > 0 else 0.0

        print(f"  [combined] {video_path.name}  {total_frames} frames  {duration_s:.0f}s  "
              f"sample every {sample_step} frames")

        body_samples:  dict[int, list[_BodySample]]  = defaultdict(list)
        face_samples:  dict[int, list[_FaceSample]]  = defaultdict(list)
        clip_frames:   dict[int, list[np.ndarray]]   = defaultdict(list)
        frame_times:   dict[int, list[float]]        = defaultdict(list)

        prev_gray: np.ndarray | None = None
        frame_idx = 0
        sampled   = 0
        t_start   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            elapsed_s = frame_idx / fps
            ih, iw    = frame.shape[:2]
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── phone + notebook detection ────────────────────────────────────
            phone_boxes:    list[tuple] = []
            notebook_boxes: list[tuple] = []
            if self._phone_model is not None:
                det_conf = min(PHONE_CONF, NOTEBOOK_CONF)
                ph_res = self._phone_model(frame, conf=det_conf, verbose=False)
                if ph_res and ph_res[0].boxes is not None:
                    for box, cls in zip(
                        ph_res[0].boxes.xyxy.cpu().numpy(),
                        ph_res[0].boxes.cls.cpu().numpy(),
                    ):
                        cid = int(cls)
                        coords = tuple(int(v) for v in box[:4])
                        if cid in self._phone_cls_ids:
                            phone_boxes.append(coords)
                        elif cid in self._notebook_cls_ids:
                            notebook_boxes.append(coords)

            # ── pose + ByteTrack ──────────────────────────────────────────────
            pose_res = self._pose_model.track(
                frame, persist=True, conf=POSE_CONF, iou=0.5,
                tracker="bytetrack.yaml", verbose=False,
            )

            do_face = (sampled % self.face_every == 0)
            result  = pose_res[0]

            if result.boxes is not None and result.boxes.id is not None:
                ids     = result.boxes.id.cpu().numpy().astype(int)
                bboxes  = result.boxes.xyxy.cpu().numpy()
                kps_all = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None

                for i, (tid, bbox) in enumerate(zip(ids, bboxes)):
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(iw, x2); y2 = min(ih, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    kps           = kps_all[i] if kps_all is not None else None
                    state         = _head_state_from_kps(kps) if kps is not None else "unknown"
                    motion        = _motion_magnitude(prev_gray, curr_gray, (x1, y1, x2, y2)) if prev_gray is not None else 0.0
                    phone         = _bbox_contains_phone((x1, y1, x2, y2), phone_boxes)
                    notebook      = _bbox_has_notebook_on_desk((x1, y1, x2, y2), notebook_boxes, ih, iw, state)
                    wrist_writing = _wrist_writing_signal(kps, prev_gray, curr_gray, ih, iw) if kps is not None else False

                    body_samples[tid].append(_BodySample(state, motion, phone, notebook, wrist_writing))
                    frame_times[tid].append(elapsed_s)

                    # ── face analysis inside body bbox ────────────────────────
                    face_gaze:  str | None   = None
                    face_ear:   float | None = None
                    face_emot:  str | None   = None
                    face_probs: dict | None  = None
                    face_pitch: float        = 0.0
                    face_mar:   float        = 0.0

                    if do_face:
                        bw, bh = x2 - x1, y2 - y1
                        # Search top 65% of body (head region) with slight padding
                        hx1 = max(0, x1 - int(bw * 0.15))
                        hx2 = min(iw, x2 + int(bw * 0.15))
                        hy1 = max(0, y1 - int(bh * 0.10))
                        hy2 = min(ih, y1 + int(bh * FACE_HEAD_FRAC))

                        if hx2 > hx1 and hy2 > hy1:
                            head_crop = frame[hy1:hy2, hx1:hx2]
                            if head_crop.size > 0:
                                faces = self._fa.get(head_crop)
                                if faces:
                                    best = max(faces, key=lambda f: float(f.det_score))
                                    # Translate crop-local bbox → full-frame coords
                                    fx1 = max(0, min(iw, int(best.bbox[0]) + hx1))
                                    fy1 = max(0, min(ih, int(best.bbox[1]) + hy1))
                                    fx2 = max(0, min(iw, int(best.bbox[2]) + hx1))
                                    fy2 = max(0, min(ih, int(best.bbox[3]) + hy1))

                                    if fx2 > fx1 and fy2 > fy1:
                                        # dlib on full-frame gray with translated bbox
                                        drect = dlib.rectangle(fx1, fy1, fx2, fy2)
                                        shape = self._predictor(curr_gray, drect)
                                        pts   = np.array(
                                            [[shape.part(j).x, shape.part(j).y] for j in range(68)],
                                            dtype=np.float32,
                                        )
                                        l_ear   = _compute_ear(pts[_LEFT_EYE_IDX])
                                        r_ear   = _compute_ear(pts[_RIGHT_EYE_IDX])
                                        face_ear = (l_ear + r_ear) / 2.0
                                        yaw, pitch = _head_pose(pts[_LANDMARK_2D_IDX], iw, ih)
                                        face_gaze  = _gaze_label(yaw, pitch)
                                        face_pitch = pitch
                                        face_mar   = _compute_mar(pts)

                                        # Emotion on face crop from full frame
                                        pad   = max(0, int((fy2 - fy1) * 0.10))
                                        ec    = frame[
                                            max(0, fy1-pad):min(ih, fy2+pad),
                                            max(0, fx1-pad):min(iw, fx2+pad),
                                        ]
                                        if ec.size > 0:
                                            face_emot, face_probs = _analyze_emotion(cv2.resize(ec, (96, 96)))
                                        else:
                                            face_emot, face_probs = "neutral", {"neutral": 1.0}

                    if face_gaze is not None and face_ear is not None and face_emot is not None:
                        eyes_open  = face_ear >= EAR_BLINK_THRESH
                        face_fwd   = face_gaze == "center"
                        bad_emot   = face_emot in _DISTRACT_EMOTIONS
                        focused    = bool(eyes_open) and bool(face_fwd) and not bool(bad_emot)
                        mouth_open = face_mar >= MAR_TALK_THRESH
                        face_samples[tid].append(
                            _FaceSample(face_gaze, face_ear, face_emot, focused, face_probs or {}, face_pitch, mouth_open)
                        )

                    # ── save annotated body crop for clip ─────────────────────
                    if sampled % CLIP_SAVE_EVERY == 0 and len(clip_frames[tid]) < CLIP_MAX_FRAMES:
                        clip_frames[tid].append(
                            _annotated_body_crop(
                                frame, x1, y1, x2, y2, ih, iw,
                                state, motion, phone, face_gaze, face_emot, elapsed_s,
                            )
                        )

            prev_gray = curr_gray
            sampled  += 1
            frame_idx += 1

            if sampled % 30 == 0 or frame_idx >= total_frames:
                pct = (frame_idx / max(1, total_frames)) * 100
                bar = "#" * int(pct / 3) + "-" * (33 - int(pct / 3))
                print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

        cap.release()
        print()

        # ── per-student aggregation ───────────────────────────────────────────
        students = []
        for tid in body_samples:
            b_list = body_samples[tid]
            if len(b_list) < MIN_TRACK_FRAMES:
                continue
            n = len(b_list)

            f_list_early = face_samples.get(tid, [])

            forward_n  = sum(1 for s in b_list if s.head_state == "forward")
            # Writing: head down + hand/body motion (same as original engagement pipeline)
            # Notebook-on-desk and wrist_writing are supplementary but motion is required
            writing_n  = sum(
                1 for s in b_list
                if (s.head_state == "down" and s.motion > MOTION_THRESH)
                or (s.notebook_on_desk and s.motion > MOTION_THRESH * 0.5)
                or (s.wrist_writing and s.head_state in ("down", "forward"))
            )
            sleeping_n = sum(
                1 for s in b_list
                if s.head_state == "down"
                and s.motion <= MOTION_THRESH
                and not s.notebook_on_desk
                and not s.wrist_writing
            )

            sideways_n = sum(1 for s in b_list if s.head_state == "sideways")
            phone_n    = sum(1 for s in b_list if s.phone_detected)

            notebook_n       = sum(1 for s in b_list if s.notebook_on_desk)
            head_forward_pct = round(forward_n  / n, 3)
            writing_pct      = round(writing_n  / n, 3)
            sleeping_pct     = round(sleeping_n / n, 3)
            sideways_pct     = round(sideways_n / n, 3)
            phone_pct        = round(phone_n    / n, 3)
            notebook_pct     = round(notebook_n / n, 3)

            engagement_score = round(
                head_forward_pct * 1.0
                + writing_pct    * 0.8
                - sleeping_pct   * 1.0
                - phone_pct      * 0.8,
                3,
            )

            # Face signals
            f_list = f_list_early
            if f_list:
                focused_n   = sum(1 for s in f_list if s.focused)
                conc_pct    = round(focused_n / len(f_list) * 100, 1)
                avg_ear     = round(float(np.mean([s.ear for s in f_list])), 4)
                blink_count = sum(1 for s in f_list if s.ear < EAR_BLINK_THRESH)

                emot_counts: dict[str, int] = defaultdict(int)
                for s in f_list:
                    emot_counts[s.emotion] += 1
                dom_emotion = max(emot_counts, key=emot_counts.get)

                emot_dist: dict[str, float] = defaultdict(float)
                for s in f_list:
                    for k, v in s.emotion_probs.items():
                        emot_dist[k] += v
                emot_dist = {k: round(v / len(f_list), 4) for k, v in emot_dist.items()}

                gaze_counts: dict[str, int] = defaultdict(int)
                for s in f_list:
                    gaze_counts[s.gaze] += 1
                gaze_dist = {k: round(v / len(f_list), 4) for k, v in gaze_counts.items()}
                mouth_flags    = [s.mouth_open for s in f_list]
                mouth_open_pct = round(sum(mouth_flags) / len(mouth_flags), 3)
                n_mouth_trans  = _mouth_transitions(mouth_flags)
            else:
                # No face data: estimate focus from body signals alone
                body_focus_n = sum(
                    1 for s in b_list
                    if s.head_state == "forward" and not s.phone_detected and s.motion <= MOTION_THRESH
                )
                conc_pct       = round(body_focus_n / n * 100, 1)
                avg_ear        = None
                blink_count    = None
                dom_emotion    = "unknown"
                emot_dist      = {}
                gaze_dist      = {}
                mouth_open_pct = 0.0
                n_mouth_trans  = 0

            attn_level = _attention_level(conc_pct)
            action     = _detect_action(phone_pct, sleeping_pct, writing_pct, sideways_pct, conc_pct, n_mouth_trans)

            times = frame_times[tid]
            detected_time_s = round((times[-1] - times[0]) if len(times) > 1 else 0.0, 1)
            attendance = "Present" if detected_time_s >= self.attendance_seconds else "Absent"

            students.append({
                "track_id":          int(tid),
                "student_label":     f"Student {int(tid):03d}",
                "detected_action":   action,
                "attention_status":  attn_level,
                "engagement_score":  engagement_score,
                "emotion":           dom_emotion,
                "emotion_dist":      emot_dist,
                "gaze_dist":         gaze_dist,
                "concentration_pct": conc_pct,
                "head_forward_pct":  head_forward_pct,
                "writing_pct":       writing_pct,
                "sleeping_pct":      sleeping_pct,
                "phone_pct":         phone_pct,
                "notebook_pct":      notebook_pct,
                "mouth_open_pct":    mouth_open_pct,
                "frames_tracked":    n,
                "face_frames":       len(f_list),
                "detected_time_s":   detected_time_s,
                "avg_ear":           avg_ear,
                "blink_count":       blink_count,
                "attendance":        attendance,
            })

        students.sort(key=lambda s: s["concentration_pct"], reverse=True)

        # ── write clips ───────────────────────────────────────────────────────
        print("  [combined] writing clips …")
        for student in students:
            tid    = student["track_id"]
            frames = clip_frames.get(tid, [])
            if not frames:
                student["clip"] = None
                continue
            fname     = f"student_{int(tid):03d}_{student['attention_status'].lower()}.mp4"
            clip_path = clips_dir / fname
            _write_clip(frames, clip_path, fps=CLIP_FPS)
            student["clip"] = fname

        # ── class summary ─────────────────────────────────────────────────────
        ns        = max(1, len(students))
        scores    = [s["concentration_pct"] for s in students]
        eng_sc    = [s["engagement_score"] for s in students]
        class_conc = round(float(np.mean(scores)), 1) if scores else 0.0
        class_eng  = round(float(np.mean(eng_sc)),  3) if eng_sc  else 0.0

        all_emot: dict[str, float] = defaultdict(float)
        for s in students:
            for k, v in s["emotion_dist"].items():
                all_emot[k] += v
        class_emot = {k: round(v / ns, 4) for k, v in all_emot.items()}

        summary = {
            "video_name":              video_path.name,
            "duration_seconds":        round(duration_s, 1),
            "frames_analyzed":         sampled,
            "student_count":           len(students),
            "class_concentration_pct": class_conc,
            "class_engagement_score":  class_eng,
            "present_count":           sum(1 for s in students if s["attendance"] == "Present"),
            "absent_count":            sum(1 for s in students if s["attendance"] == "Absent"),
            "high_attention_count":    sum(1 for s in students if s["attention_status"] == "High"),
            "medium_attention_count":  sum(1 for s in students if s["attention_status"] == "Medium"),
            "low_attention_count":     sum(1 for s in students if s["attention_status"] == "Low"),
            "attentive_pct":           round(sum(1 for s in students if s["detected_action"] == "Attentive") / ns, 3),
            "writing_pct":             round(sum(s["writing_pct"]    for s in students) / ns, 3),
            "sleeping_pct":            round(sum(s["sleeping_pct"]   for s in students) / ns, 3),
            "phone_pct":               round(sum(s["phone_pct"]      for s in students) / ns, 3),
            "notebook_pct":            round(sum(s["notebook_pct"]   for s in students) / ns, 3),
            "talking_pct":             round(sum(1 for s in students if s["detected_action"] == "Talking") / ns, 3),
            "class_emotion_dist":      dict(class_emot),
            "processing_time_s":       round(time.time() - t_start, 1),
        }

        summary_path = output_dir / "combined_summary.json"
        csv_path     = output_dir / "combined_students.csv"
        summary_path.write_text(json.dumps({"summary": summary, "students": students}, indent=2))
        _write_csv(csv_path, students)

        total_clips = sum(1 for s in students if s.get("clip"))
        print(f"  [combined] done — {len(students)} students  "
              f"conc {class_conc:.1f}%  eng {class_eng:.2f}  {total_clips} clips")

        return {
            "summary":      summary,
            "students":     students,
            "summary_path": str(summary_path),
            "csv_path":     str(csv_path),
            "clips_dir":    str(clips_dir),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="combined_output")
    args = ap.parse_args()
    pipeline = CombinedPipeline()
    result   = pipeline.process(args.video, args.out)
    print(json.dumps(result["summary"], indent=2))
