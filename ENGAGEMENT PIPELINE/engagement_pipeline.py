"""Engagement monitoring pipeline — parallel to the existing activity pipeline.

Architecture:
  - YOLOv8-pose  : detects all students + 17 body keypoints in one pass per sampled frame
  - ByteTrack    : maintains per-student track IDs across frames (via ultralytics)
  - Head state   : inferred from pose keypoints (nose/eye/ear visibility + geometry)
  - Motion       : optical flow magnitude in each student's lower-body ROI
  - Phone YOLO   : reuses the existing yolo11m phone detector
  - Temporal     : 60-sample rolling window (1 sample/s) per track → engagement score
  - Output       : per-student score + class summary, updated every SCORE_INTERVAL seconds

No extra model downloads required — yolov8n-pose.pt (~6 MB) is pulled by ultralytics
on first use; everything else reuses existing project assets.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ultralytics import YOLO

# ── constants ────────────────────────────────────────────────────────────────

POSE_MODEL      = "yolov8n-pose.pt"      # auto-downloaded by ultralytics on first use
SAMPLE_EVERY_S  = 1.0                    # analyse 1 frame per second
WINDOW_SIZE     = 60                     # rolling window length (samples = seconds)
SCORE_INTERVAL  = 30                     # emit a class-level score snapshot every N seconds
MOTION_THRESH   = 1.2                    # optical-flow magnitude threshold for "moving"
KPS_CONF        = 0.3                    # keypoint confidence threshold
PHONE_CONF      = 0.40                   # phone detection confidence threshold
MIN_TRACK_FRAMES = 5                    # discard tracks shorter than this (noise / false positives)

# YOLOv8-pose keypoint indices
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SH, _R_SH = 5, 6


# ── head-state from pose keypoints ───────────────────────────────────────────

def _head_state(kps: np.ndarray) -> str:
    """
    kps : (17, 3)  x, y, confidence per keypoint.
    Returns one of: 'forward' | 'down' | 'sideways' | 'unknown'

    Logic:
      • If <3 face keypoints visible → 'down' (face turned away / looking at desk)
      • If one ear visible but not the other → 'sideways'
      • If nose is significantly below eye midpoint (relative to eye separation) → 'down'
      • Otherwise → 'forward'
    """
    nose  = kps[_NOSE]
    l_eye = kps[_L_EYE];  r_eye = kps[_R_EYE]
    l_ear = kps[_L_EAR];  r_ear = kps[_R_EAR]

    face_visible = sum(kps[i, 2] > KPS_CONF for i in range(5))
    if face_visible < 3:
        # shoulders still detectable → looking down; otherwise truly unknown
        sh_vis = kps[_L_SH, 2] > KPS_CONF or kps[_R_SH, 2] > KPS_CONF
        return "down" if sh_vis else "unknown"

    # sideways: one ear visible, other not
    l_ear_vis = l_ear[2] > KPS_CONF
    r_ear_vis = r_ear[2] > KPS_CONF
    if l_ear_vis != r_ear_vis:
        return "sideways"

    # head down: nose well below eye midpoint
    if l_eye[2] > KPS_CONF and r_eye[2] > KPS_CONF and nose[2] > KPS_CONF:
        eye_y  = (l_eye[1] + r_eye[1]) / 2
        eye_sep = max(1.0, abs(l_eye[0] - r_eye[0]))
        if nose[1] > eye_y + eye_sep * 2.0:
            return "down"

    return "forward"


# ── optical flow motion in a bounding-box ROI ────────────────────────────────

def _motion_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray,
                      bbox: tuple[int, int, int, int]) -> float:
    """Mean optical-flow magnitude in the lower half of the student's bounding box."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    # Focus on lower half (hands / desk area)
    ry1 = y1 + h // 2
    p = prev_gray[ry1:y2, x1:x2]
    c = curr_gray[ry1:y2, x1:x2]
    if p.size == 0 or c.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag  = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


# ── per-sample record ─────────────────────────────────────────────────────────

class _Sample(NamedTuple):
    head_state:     str    # forward | down | sideways | unknown
    motion:         float  # optical-flow magnitude
    phone_detected: bool


# ── engagement scoring ────────────────────────────────────────────────────────

def _score_window(window: list[_Sample]) -> tuple[float, str]:
    """
    Returns (score 0–1, status label).

    Rules:
      phone             → score ≈ 0.05,  status = 'phone'
      head-down + motion → writing  → score ≈ 0.75, status = 'writing'
      head-down + still  → sleeping → score ≈ 0.10, status = 'sleeping'
      sideways sustained → distracted
      forward            → attentive
    """
    if not window:
        return 0.5, "uncertain"

    n = len(window)
    phone_pct    = sum(s.phone_detected for s in window) / n
    forward_pct  = sum(s.head_state == "forward"  for s in window) / n
    down_pct     = sum(s.head_state == "down"      for s in window) / n
    side_pct     = sum(s.head_state == "sideways"  for s in window) / n
    writing_pct  = sum(s.head_state == "down" and s.motion > MOTION_THRESH for s in window) / n
    sleeping_pct = sum(s.head_state == "down" and s.motion <= MOTION_THRESH for s in window) / n

    if phone_pct > 0.25:
        score = max(0.0, 0.10 - phone_pct * 0.1)
        return round(score, 3), "phone"

    if sleeping_pct > 0.45:
        return round(max(0.05, 0.20 - sleeping_pct * 0.2), 3), "sleeping"

    score = (
        forward_pct  * 0.85
        + writing_pct  * 0.70
        - sleeping_pct * 0.60
        - phone_pct    * 0.90
        - side_pct     * 0.20
        + 0.10          # small baseline
    )
    score = float(np.clip(score, 0.0, 1.0))

    if writing_pct > 0.35:
        return round(score, 3), "writing"
    if forward_pct > 0.55:
        return round(score, 3), "attentive"
    if side_pct > 0.40:
        return round(score, 3), "distracted"
    return round(score, 3), "uncertain"


# ── main pipeline class ───────────────────────────────────────────────────────

class EngagementPipeline:
    """
    Process a classroom video and return per-student engagement scores.

    Usage::

        pipeline = EngagementPipeline()
        result   = pipeline.process(video_path, output_dir)
    """

    def __init__(
        self,
        pose_model:     str        = POSE_MODEL,
        phone_model:    str | None = None,
        sample_every_s: float      = SAMPLE_EVERY_S,
        window_size:    int        = WINDOW_SIZE,
        score_interval: int        = SCORE_INTERVAL,
    ):
        self.pose_model_name = pose_model
        self.sample_every_s  = sample_every_s
        self.window_size     = window_size
        self.score_interval  = score_interval

        self._pose_model  = YOLO(pose_model)

        phone_path = phone_model or self._default_phone_model()
        self._phone_model = YOLO(str(phone_path)) if phone_path and Path(str(phone_path)).exists() else None
        if self._phone_model is None:
            print("  [engagement] phone detector not found — phone signal disabled")

        # Resolve which class IDs in the phone model represent phones/mobiles.
        # For a COCO-pretrained model class 67 = "cell phone"; for a single-class
        # custom detector every class counts. Fall back to all if no name matches.
        if self._phone_model is not None:
            _names = self._phone_model.names or {}
            self._phone_cls_ids: set[int] = {
                cid for cid, n in _names.items()
                if any(w in n.lower() for w in ("phone", "cell", "mobile"))
            }
            if not self._phone_cls_ids:
                self._phone_cls_ids = set(_names.keys())
        else:
            self._phone_cls_ids: set[int] = set()

    @staticmethod
    def _default_phone_model() -> Path | None:
        p = Path(_REPO_ROOT) / "Activity monitoring" / "Training Pipelines" / "assets" / "yolo11m.pt"
        return p if p.exists() else None

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, video_path: str | Path, output_dir: str | Path) -> dict:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_step  = max(1, int(round(fps * self.sample_every_s)))
        duration_s   = total_frames / fps if total_frames > 0 else 0

        print(f"  [engagement] {video_path.name}  {total_frames} frames  {duration_s:.0f}s  sample every {sample_step} frames")

        # per-track state
        windows:      dict[int, deque[_Sample]]         = defaultdict(lambda: deque(maxlen=self.window_size))
        last_bbox:    dict[int, tuple]                  = {}
        frame_counts: dict[int, int]                    = defaultdict(int)
        # clip buffer: store annotated crops for each track (max 60 frames → 1-min clip at 1fps)
        clip_frames:  dict[int, list[np.ndarray]]       = defaultdict(list)
        timeline:     list[dict]                        = []

        clips_dir = output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        CLIP_MAX_FRAMES = 60          # cap stored frames per student
        CLIP_SAVE_EVERY = 5           # store every 5th sampled frame to spread coverage

        prev_gray:     np.ndarray | None = None
        frame_idx      = 0
        samples_so_far = 0
        next_score_at  = self.score_interval
        t_start        = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            elapsed_s = frame_idx / fps
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── pose + tracking ──────────────────────────────────────────────
            pose_results = self._pose_model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.4,
            )

            # ── phone detection on full frame ────────────────────────────────
            phone_boxes: list[tuple[int, int, int, int]] = []
            if self._phone_model is not None:
                ph_res = self._phone_model.predict(frame, verbose=False, conf=PHONE_CONF)
                if ph_res and ph_res[0].boxes is not None:
                    for box, cls in zip(ph_res[0].boxes.xyxy.cpu().numpy(),
                                        ph_res[0].boxes.cls.cpu().numpy()):
                        if int(cls) in self._phone_cls_ids:
                            phone_boxes.append(tuple(int(v) for v in box[:4]))

            # ── per-student signals ──────────────────────────────────────────
            result = pose_results[0]
            if result.boxes is not None and result.boxes.id is not None:
                ids     = result.boxes.id.cpu().numpy().astype(int)
                bboxes  = result.boxes.xyxy.cpu().numpy()
                kps_all = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
                ih, iw  = frame.shape[:2]

                for i, (tid, bbox) in enumerate(zip(ids, bboxes)):
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    kps   = kps_all[i] if kps_all is not None else None
                    state = _head_state(kps) if kps is not None else "unknown"

                    motion = 0.0
                    if prev_gray is not None:
                        motion = _motion_magnitude(prev_gray, curr_gray, (x1, y1, x2, y2))

                    phone = _bbox_contains_phone((x1, y1, x2, y2), phone_boxes)

                    windows[tid].append(_Sample(state, motion, phone))
                    last_bbox[tid]    = (x1, y1, x2, y2)
                    frame_counts[tid] += 1

                    # ── save annotated crop for clip ─────────────────────────
                    if (samples_so_far % CLIP_SAVE_EVERY == 0
                            and len(clip_frames[tid]) < CLIP_MAX_FRAMES):
                        crop = _annotated_crop(frame, x1, y1, x2, y2, ih, iw,
                                               state, motion, phone, elapsed_s)
                        clip_frames[tid].append(crop)

            prev_gray = curr_gray
            samples_so_far += 1
            frame_idx += 1

            # ── progress ─────────────────────────────────────────────────────
            if samples_so_far % 30 == 0 or frame_idx >= total_frames:
                pct = (frame_idx / max(1, total_frames)) * 100
                bar = "#" * int(pct / 3) + "-" * (33 - int(pct / 3))
                print(f"\r  Progress: [{bar}] {frame_idx}/{total_frames} ({pct:.0f}%)", end="", flush=True)

            # ── periodic class snapshot ───────────────────────────────────────
            if elapsed_s >= next_score_at:
                snap = _class_snapshot(elapsed_s, windows)
                timeline.append(snap)
                next_score_at += self.score_interval

        cap.release()
        print()

        # ── final per-student results ─────────────────────────────────────────
        students = []
        for tid, window in windows.items():
            score, status = _score_window(list(window))
            w = list(window)
            n = max(1, len(w))
            students.append({
                "track_id":         int(tid),
                "engagement_score": score,
                "status":           status,
                "head_forward_pct": round(sum(s.head_state == "forward"  for s in w) / n, 3),
                "writing_pct":      round(sum(s.head_state == "down" and s.motion > MOTION_THRESH for s in w) / n, 3),
                "sleeping_pct":     round(sum(s.head_state == "down" and s.motion <= MOTION_THRESH for s in w) / n, 3),
                "phone_pct":        round(sum(s.phone_detected for s in w) / n, 3),
                "frames_tracked":   frame_counts[tid],
            })

        # Drop tracks that appear in only a handful of frames — these are noise,
        # false-positive person detections, or ByteTrack ID splits on the same student.
        students = [s for s in students if s["frames_tracked"] >= MIN_TRACK_FRAMES]
        students.sort(key=lambda s: s["engagement_score"])

        # ── compile per-student clips (split into N time-window segments) ────────
        print("  [engagement] writing clips...")
        SEGMENTS = 4   # split each student's frames into this many clips

        for student in students:
            tid    = student["track_id"]
            frames = clip_frames.get(tid, [])
            if not frames:
                student["clips"] = []
                continue

            # divide evenly; last segment gets any remainder
            seg_size   = max(1, len(frames) // SEGMENTS)
            clips_list = []
            for seg_i in range(SEGMENTS):
                start = seg_i * seg_size
                chunk = frames[start:start + seg_size] if seg_i < SEGMENTS - 1 else frames[start:]
                if not chunk:
                    continue
                fname     = f"track_{tid:03d}_part{seg_i + 1}_{student['status']}.mp4"
                clip_file = clips_dir / fname
                _write_clip(chunk, clip_file, fps=2.0)
                clips_list.append(fname)

            student["clips"] = clips_list

        # ── class-level summary ───────────────────────────────────────────────
        scores      = [s["engagement_score"] for s in students]
        class_score = round(float(np.mean(scores)), 3) if scores else 0.0

        summary = {
            "video_name":             video_path.name,
            "duration_seconds":       round(duration_s, 1),
            "frames_analyzed":        samples_so_far,
            "student_count":          len(students),
            "class_engagement_score": class_score,
            "attentive_pct":  round(sum(s["status"] == "attentive"  for s in students) / max(1, len(students)), 3),
            "writing_pct":    round(sum(s["status"] == "writing"    for s in students) / max(1, len(students)), 3),
            "sleeping_pct":   round(sum(s["status"] == "sleeping"   for s in students) / max(1, len(students)), 3),
            "phone_pct":      round(sum(s["status"] == "phone"      for s in students) / max(1, len(students)), 3),
            "processing_time_s": round(time.time() - t_start, 1),
        }

        # ── save outputs ──────────────────────────────────────────────────────
        summary_path = output_dir / "engagement_summary.json"
        csv_path     = output_dir / "engagement_students.csv"

        summary_path.write_text(json.dumps({"summary": summary, "students": students, "timeline": timeline}, indent=2))

        with csv_path.open("w", newline="") as f:
            if students:
                writer = csv.DictWriter(f, fieldnames=[k for k in students[0] if k != "clip_path"])
                writer.writeheader()
                writer.writerows({k: v for k, v in s.items() if k != "clip_path"} for s in students)

        total_clips = sum(len(s.get("clips", [])) for s in students)
        print(f"  [engagement] done — {len(students)} students, class score {class_score:.2f}, {total_clips} clips")
        return {
            "summary":      summary,
            "students":     students,
            "timeline":     timeline,
            "summary_path": str(summary_path),
            "csv_path":     str(csv_path),
            "clips_dir":    str(clips_dir),
        }


# ── helpers ───────────────────────────────────────────────────────────────────

_STATE_COLOR = {
    "forward":  (50,  200, 50),
    "down":     (50,  150, 255),
    "sideways": (200, 100, 50),
    "unknown":  (120, 120, 120),
}

def _annotated_crop(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ih: int, iw: int,
    state: str, motion: float, phone: bool,
    elapsed_s: float,
    pad: float = 0.15,
    out_size: int = 224,
) -> np.ndarray:
    """Return a square annotated crop of this student."""
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px); cy2 = min(ih, y2 + py)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    crop = cv2.resize(crop, (out_size, out_size))

    color = _STATE_COLOR.get(state, (120, 120, 120))
    label = state.upper()
    if phone:
        label += " PHONE"
        color = (0, 100, 255)

    cv2.rectangle(crop, (0, 0), (out_size - 1, out_size - 1), color, 3)
    cv2.putText(crop, label,         (6, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(crop, f"mot:{motion:.1f}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(crop, f"{elapsed_s:.0f}s",  (6, out_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return crop


def _write_clip(frames: list[np.ndarray], path: Path, fps: float = 2.0) -> None:
    """Write frames to an mp4 clip using H.264 (browser-compatible)."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    # avc1 = H.264, supported by all browsers; fall back to mp4v if unavailable
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            break
    for f in frames:
        writer.write(f)
    writer.release()


def _bbox_contains_phone(student_bbox: tuple, phone_boxes: list) -> bool:
    """True if any phone box centre falls inside the student bounding box."""
    sx1, sy1, sx2, sy2 = student_bbox
    for px1, py1, px2, py2 in phone_boxes:
        cx = (px1 + px2) / 2
        cy = (py1 + py2) / 2
        if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
            return True
    return False


def _class_snapshot(elapsed_s: float, windows: dict) -> dict:
    scores = []
    for tid, w in windows.items():
        if w:
            score, _ = _score_window(list(w))
            scores.append(score)
    return {
        "second":           round(elapsed_s, 1),
        "class_engagement": round(float(np.mean(scores)), 3) if scores else 0.0,
        "students_present": len(scores),
    }


# ── default weights path (mirrors existing pipeline convention) ───────────────

def default_engagement_pipeline() -> "EngagementPipeline":
    return EngagementPipeline()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="engagement_output")
    args = ap.parse_args()

    pipeline = EngagementPipeline()
    result   = pipeline.process(args.video, args.out)
    print(json.dumps(result["summary"], indent=2))
