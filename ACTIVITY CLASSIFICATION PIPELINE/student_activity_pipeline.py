#!/usr/bin/env python3
"""Sample a classroom video every 30 seconds, detect faces with the fine-tuned RetinaFace
ONNX model, crop each face tightly across a 15-frame window, and classify the resulting clip
with the fine-tuned 3D CNN R-18 model.

Pipeline:
1. Sample 15 consecutive frames every N seconds from the input video.
2. Run the fine-tuned RetinaFace detector on each sampled frame.
3. Group one face across the 15-frame window with IoU matching.
4. Save an extremely zoomed face clip for each grouped face.
5. Run the fine-tuned r3d-18 classifier on each clip.

The bundled classifier is binary: low_engagement vs high_engagement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models.video import r3d_18
import os

from insightface.app import FaceAnalysis
from utils.retinaface_detector import build_face_analysis_with_retinaface_detector


KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

LABELS = {0: "low_engagement", 1: "high_engagement"}
TRACK_COLORS = {
    0: (56, 176, 222),
    1: (56, 222, 126),
}
PHONE_CLASS_ID = 67
PHONE_CONF_THRESHOLD = 0.10
PHONE_CONFIRM_STREAK = 1


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Make project-root utils importable regardless of working directory
import sys as _sys
_repo_root_str = str(Path(__file__).resolve().parents[1])
if _repo_root_str not in _sys.path:
    _sys.path.insert(0, _repo_root_str)


def default_detector_weights() -> Path:
    return resolve_repo_root() / "Activity monitoring" / "Training Pipelines" / "assets" / "yolo11m.pt"


def default_classifier_weights() -> Path:
    return resolve_repo_root() / "Activity monitoring" / "models" / "best_model" / "3dcnn_r3d18_weighted.pt"


def default_face_model_name() -> str:
    return os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_l")


class Video3DClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.backbone = r3d_18(weights=None if not pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@dataclass
class WindowFaceTrack:
    track_id: int
    frame_count: int
    boxes: list[tuple[int, int, int, int] | None] = field(init=False)
    raw_crops: list[np.ndarray | None] = field(init=False)
    tensors: list[torch.Tensor | None] = field(init=False)
    scores: list[float | None] = field(init=False)
    embeddings: list[np.ndarray | None] = field(init=False)
    observations: int = 0
    last_bbox: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        self.boxes = [None] * self.frame_count
        self.raw_crops = [None] * self.frame_count
        self.tensors = [None] * self.frame_count
        self.scores = [None] * self.frame_count
        self.embeddings = [None] * self.frame_count


@dataclass
class StudentIdentity:
    student_id: int
    prototype: np.ndarray
    observations: int = 1

    def update(self, embedding: np.ndarray) -> None:
        combined = (self.prototype * self.observations + embedding) / (self.observations + 1)
        norm = float(np.linalg.norm(combined))
        self.prototype = combined / norm if norm > 0 else combined
        self.observations += 1


class StudentActivityPipeline:
    def __init__(
        self,
        classifier_path: Path,
        num_frames: int = 16,
        crop_size: int = 112,
        sample_frames: int = 24,
        sample_every_seconds: float = 30.0,
        face_det_size: int = 1280,
        face_det_thresh: float = 0.5,
        face_iou_threshold: float = 0.3,
        crop_padding: float = 0.12,
        export_crop_size: int = 224,
        export_crop_padding: float = 0.60,
        identity_threshold: float = 0.4,
        device: str | None = None,
    ):
        self.classifier_path = Path(classifier_path)
        self.num_frames = num_frames
        self.sample_frames = sample_frames
        self.sample_every_seconds = sample_every_seconds
        self.crop_size = crop_size
        self.export_crop_size = export_crop_size
        self.export_crop_padding = export_crop_padding
        self.face_det_size = face_det_size
        self.face_det_thresh = face_det_thresh
        self.face_iou_threshold = face_iou_threshold
        self.identity_threshold = identity_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.face_detector = FaceAnalysis(name=default_face_model_name(), providers=["CPUExecutionProvider"])
        self.face_detector.prepare(ctx_id=0, det_size=(self.face_det_size, self.face_det_size), det_thresh=self.face_det_thresh)
        detector_path = default_detector_weights()
        if not detector_path.exists():
            raise FileNotFoundError(f"Could not find YOLO phone detector weights at {detector_path}")
        self.phone_detector = YOLO(str(detector_path))
        self.model = self._load_classifier()
        self.model.eval()
        self.crop_padding = crop_padding

    def _load_classifier(self) -> nn.Module:
        model = Video3DClassifier(num_classes=2, pretrained=False, dropout=0.2)
        checkpoint = torch.load(self.classifier_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.to(self.device)
        return model

    @staticmethod
    def _confirm_streak(detections: list[bool]) -> int:
        best = current = 0
        for detected in detections:
            if detected:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _phone_gate(self, frames: list[np.ndarray]) -> dict:
        frame_scores: list[float] = []
        frame_hits: list[bool] = []

        for frame in frames:
            results = self.phone_detector.predict(frame, conf=0.01, verbose=False)
            best_conf = 0.0
            found = False
            if results:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if cls_id == PHONE_CLASS_ID and conf > best_conf:
                            best_conf = conf
                        if cls_id == PHONE_CLASS_ID and conf >= PHONE_CONF_THRESHOLD:
                            found = True
            frame_scores.append(best_conf)
            frame_hits.append(found)

        max_conf = float(np.max(frame_scores)) if frame_scores else 0.0
        mean_conf = float(np.mean(frame_scores)) if frame_scores else 0.0
        hit_count = int(np.sum(frame_hits))
        streak_max = self._confirm_streak(frame_hits)
        confirmed = streak_max >= PHONE_CONFIRM_STREAK or max_conf >= PHONE_CONF_THRESHOLD

        return {
            "phone_detected": confirmed,
            "phone_max_confidence": round(max_conf, 4),
            "phone_mean_confidence": round(mean_conf, 4),
            "phone_hit_count": hit_count,
            "phone_confirm_streak": streak_max,
        }

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        from utils.retinaface_detector import sahi_face_detect
        faces = sahi_face_detect(self.face_detector.get, frame)
        detections: list[dict] = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is not None:
                embedding = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(embedding))
                if norm > 0:
                    embedding = embedding / norm
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": float(face.det_score),
                    "embedding": embedding,
                }
            )
        return detections

    @staticmethod
    def _clip_box(box: Iterable[int], width: int, height: int, padding: float = 0.18) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in box]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width - 1, x2 + pad_x)
        y2 = min(height - 1, y2 + pad_y)
        return x1, y1, x2, y2

    def _crop_to_tensor(self, frame: np.ndarray, box: Iterable[int]) -> torch.Tensor | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_box(box, width, height, padding=self.crop_padding)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy((rgb - KINETICS_MEAN) / KINETICS_STD).permute(2, 0, 1).contiguous()
        return tensor

    def _crop_for_export(self, frame: np.ndarray, box: Iterable[int]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_box(box, width, height, padding=self.export_crop_padding)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (self.export_crop_size, self.export_crop_size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _iou(box_a: Iterable[int], box_b: Iterable[int]) -> float:
        ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
        bx1, by1, bx2, by2 = [int(v) for v in box_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _fill_missing_sequence(self, sequence: list[np.ndarray | torch.Tensor | None]) -> list[np.ndarray | torch.Tensor]:
        filled: list[np.ndarray | torch.Tensor | None] = list(sequence)
        first_valid = next((index for index, item in enumerate(filled) if item is not None), None)
        if first_valid is None:
            return []

        last_seen = filled[first_valid]
        for index in range(0, first_valid):
            filled[index] = last_seen.copy() if isinstance(last_seen, np.ndarray) else last_seen.clone()

        for index in range(first_valid + 1, len(filled)):
            if filled[index] is None:
                previous = filled[index - 1]
                filled[index] = previous.copy() if isinstance(previous, np.ndarray) else previous.clone()

        for index in range(len(filled) - 1, -1, -1):
            if filled[index] is None:
                next_valid = next((filled[j] for j in range(index + 1, len(filled)) if filled[j] is not None), None)
                if next_valid is not None:
                    filled[index] = next_valid.copy() if isinstance(next_valid, np.ndarray) else next_valid.clone()

        return [item for item in filled if item is not None]

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return None
        return vector / norm

    def _resolve_student_identity(
        self,
        track_embedding: np.ndarray | None,
        student_bank: list[StudentIdentity],
        next_student_id: int,
    ) -> tuple[int, int, float]:
        embedding = self._normalize_embedding(track_embedding)
        if embedding is None:
            student_id = next_student_id
            return next_student_id + 1, student_id, 0.0

        best_identity: StudentIdentity | None = None
        best_similarity = -1.0
        for identity in student_bank:
            similarity = float(np.dot(identity.prototype, embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_identity = identity

        if best_identity is not None and best_similarity >= self.identity_threshold:
            best_identity.update(embedding)
            return next_student_id, best_identity.student_id, best_similarity

        student_id = next_student_id
        student_bank.append(StudentIdentity(student_id=student_id, prototype=embedding.copy()))
        return next_student_id + 1, student_id, 0.0

    @staticmethod
    def _track_embedding(track: WindowFaceTrack) -> np.ndarray | None:
        embeddings = [embedding for embedding in track.embeddings if embedding is not None]
        if not embeddings:
            return None
        stacked = np.stack(embeddings, axis=0).astype(np.float32)
        mean_embedding = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean_embedding))
        if norm <= 0:
            return None
        return mean_embedding / norm

    def _pad_for_classifier(self, clip_frames: list[torch.Tensor]) -> list[torch.Tensor]:
        if not clip_frames:
            return []

        padded = list(clip_frames)
        if len(padded) > self.num_frames:
            padded = padded[: self.num_frames]
        while len(padded) < self.num_frames:
            padded.append(padded[-1].clone())
        return padded

    def _save_clip(self, frames: list[np.ndarray], clip_path: Path, fps: float) -> None:
        if not frames:
            return

        # Always export at a slow fixed rate so 24 frames → ~5 seconds regardless
        # of the source video fps (60fps source would otherwise give 0.4s clips).
        export_fps = 5.0

        clip_path.parent.mkdir(parents=True, exist_ok=True)
        temp_clip_path = clip_path.with_suffix(".raw.mp4")
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(temp_clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            export_fps,
            (width, height),
        )
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        if not temp_clip_path.exists() or temp_clip_path.stat().st_size == 0:
            return

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            temp_clip_path.replace(clip_path)
            return

        encoded_path = clip_path.with_suffix(".encoded.mp4")
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(temp_clip_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(encoded_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode == 0 and encoded_path.exists() and encoded_path.stat().st_size > 0:
                encoded_path.replace(clip_path)
            else:
                temp_clip_path.replace(clip_path)
        finally:
            if temp_clip_path.exists():
                temp_clip_path.unlink()
            if encoded_path.exists() and encoded_path != clip_path:
                encoded_path.unlink()

    @torch.no_grad()
    def classify_clip(self, clip_frames: list[torch.Tensor]) -> dict:
        clip_frames = self._pad_for_classifier(clip_frames)
        clip = torch.stack(clip_frames, dim=0)  # T x C x H x W
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # 1 x C x T x H x W
        logits = self.model(clip)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        predicted_index = int(np.argmax(probabilities))
        return {
            "predicted_index": predicted_index,
            "predicted_label": LABELS[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "low_engagement": float(probabilities[0]),
            "high_engagement": float(probabilities[1]),
        }

    def process_video(self, video_path: Path, output_dir: Path, annotate: bool = True) -> dict:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        clip_dir = output_dir / "classified_clips"
        frame_index = 0
        sample_step_frames = max(1, int(round((fps if fps > 0 else 30.0) * self.sample_every_seconds)))
        window_starts = set(range(0, total_frames if total_frames > 0 else 10**9, sample_step_frames))
        # progress reporting
        progress_step = max(1, (total_frames // 100) if total_frames > 0 else 1)
        active_window: dict | None = None
        clip_records: list[dict] = []
        annotated_frames: list[np.ndarray] = []
        student_bank: list[StudentIdentity] = []
        next_student_id = 1

        def progress_line(current_frame: int) -> str:
            if total_frames <= 0:
                return "Progress: [??????????????????????????????]"
            bar_width = 30
            ratio = min(1.0, max(0.0, current_frame / total_frames))
            filled = int(round(bar_width * ratio))
            bar = "#" * filled + "-" * (bar_width - filled)
            return f"Progress: [{bar}] {current_frame}/{total_frames} ({ratio * 100:.1f}%)"

        def start_window(start_frame: int) -> dict:
            return {
                "start_frame": start_frame,
                "frames": [],
                "detections": [],
            }

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index in window_starts:
                active_window = start_window(frame_index)

            if active_window is not None:
                detections = self.detect_faces(frame)
                active_window["frames"].append(frame.copy())
                active_window["detections"].append(detections)

                if len(active_window["frames"]) == self.sample_frames:
                    window_records, next_student_id = self._process_window(
                        video_path=video_path,
                        window_start_frame=active_window["start_frame"],
                        frames=active_window["frames"],
                        detections_per_frame=active_window["detections"],
                        clip_dir=clip_dir,
                        fps=fps if fps > 0 else 25.0,
                        student_bank=student_bank,
                        next_student_id=next_student_id,
                    )
                    clip_records.extend(window_records)
                    if annotate:
                        annotated_frames.extend(self._render_window_annotation(active_window["frames"], active_window["detections"], window_records))
                    active_window = None

            # print progress occasionally so the user can monitor long runs
            if total_frames > 0 and (frame_index % progress_step == 0 or frame_index == total_frames - 1):
                print(f"\r{progress_line(frame_index)}", end="", flush=True)

            frame_index += 1

        cap.release()
        # finish progress line
        if total_frames > 0:
            print(f"\r{progress_line(total_frames)}")

        summary = self._build_summary(video_path, total_frames, clip_records, sample_step_frames, self.sample_frames)
        summary_path = output_dir / f"{video_path.stem}_summary.json"
        csv_path = output_dir / f"{video_path.stem}_per_student_predictions.csv"

        summary_path.write_text(json.dumps(summary, indent=2))
        self._write_csv(csv_path, summary["clips"])

        annotated_path = None
        if annotate and annotated_frames:
            annotated_path = output_dir / f"{video_path.stem}_sampled_annotated.mp4"
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps > 0 else 25.0,
                (frame_width, frame_height),
            )
            try:
                for annotated_frame in annotated_frames:
                    writer.write(annotated_frame)
            finally:
                writer.release()

        return {
            "summary_path": str(summary_path),
            "csv_path": str(csv_path),
            "clip_dir": str(clip_dir / video_path.stem),
            "annotated_video": str(annotated_path) if annotated_path else None,
            "summary": summary,
        }

    def _process_window(
        self,
        video_path: Path,
        window_start_frame: int,
        frames: list[np.ndarray],
        detections_per_frame: list[list[dict]],
        clip_dir: Path,
        fps: float,
        student_bank: list[StudentIdentity],
        next_student_id: int,
    ) -> tuple[list[dict], int]:
        tracks: list[WindowFaceTrack] = []
        next_track_id = 1

        for frame_index, (frame, detections) in enumerate(zip(frames, detections_per_frame)):
            unmatched_detection_indices = set(range(len(detections)))

            for track in tracks:
                best_match_index = None
                best_match_iou = 0.0
                for detection_index in list(unmatched_detection_indices):
                    iou = self._iou(track.last_bbox, detections[detection_index]["bbox"]) if track.last_bbox is not None else 0.0
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_index = detection_index

                if best_match_index is None or best_match_iou < self.face_iou_threshold:
                    continue

                detection = detections[best_match_index]
                unmatched_detection_indices.remove(best_match_index)
                track.last_bbox = detection["bbox"]
                track.boxes[frame_index] = detection["bbox"]
                tensor = self._crop_to_tensor(frame, detection["bbox"])
                raw_crop = self._crop_for_export(frame, detection["bbox"])
                if tensor is not None:
                    track.tensors[frame_index] = tensor
                if raw_crop is not None:
                    track.raw_crops[frame_index] = raw_crop
                track.scores[frame_index] = detection["score"]
                track.embeddings[frame_index] = detection.get("embedding")
                track.observations += 1

            for detection_index in unmatched_detection_indices:
                detection = detections[detection_index]
                track = WindowFaceTrack(track_id=next_track_id, frame_count=len(frames))
                next_track_id += 1
                track.last_bbox = detection["bbox"]
                track.boxes[frame_index] = detection["bbox"]
                tensor = self._crop_to_tensor(frame, detection["bbox"])
                raw_crop = self._crop_for_export(frame, detection["bbox"])
                if tensor is not None:
                    track.tensors[frame_index] = tensor
                if raw_crop is not None:
                    track.raw_crops[frame_index] = raw_crop
                track.scores[frame_index] = detection["score"]
                track.embeddings[frame_index] = detection.get("embedding")
                track.observations = 1
                tracks.append(track)

        records: list[dict] = []
        window_start_seconds = window_start_frame / fps if fps > 0 else 0.0
        for track in tracks:
            if track.observations < max(3, self.sample_frames // 3):
                continue

            raw_crops = self._fill_missing_sequence(track.raw_crops)
            tensors = self._fill_missing_sequence(track.tensors)
            if not raw_crops or not tensors:
                continue

            student_embedding = self._track_embedding(track)
            next_student_id, student_id, student_similarity = self._resolve_student_identity(
                student_embedding,
                student_bank,
                next_student_id,
            )
            student_label = f"student_{student_id:03d}"

            phone_gate = self._phone_gate([crop for crop in raw_crops if isinstance(crop, np.ndarray)])
            if phone_gate["phone_detected"]:
                prediction = {
                    "predicted_index": 0,
                    "predicted_label": LABELS[0],
                    "confidence": 1.0,
                    "low_engagement": 1.0,
                    "high_engagement": 0.0,
                    "decision_source": "phone_detector",
                    **phone_gate,
                }
            else:
                prediction = self.classify_clip([tensor for tensor in tensors if isinstance(tensor, torch.Tensor)])
                prediction.update({"decision_source": "3dcnn", **phone_gate})
            clip_name = f"window_{window_start_frame:06d}_{student_label}_{prediction['predicted_label']}.mp4"
            clip_path = clip_dir / video_path.stem / student_label / clip_name
            self._save_clip([crop for crop in raw_crops if isinstance(crop, np.ndarray)], clip_path, fps)
            prediction.update(
                {
                    "clip_path": str(clip_path),
                    "window_start_frame": window_start_frame,
                    "window_start_seconds": round(window_start_seconds, 2),
                    "track_id": track.track_id,
                    "student_id": student_id,
                    "student_label": student_label,
                    "student_similarity": round(float(student_similarity), 4),
                    "observations": track.observations,
                }
            )
            records.append(prediction)

        return records, next_student_id

    def _render_window_annotation(
        self,
        frames: list[np.ndarray],
        detections_per_frame: list[list[dict]],
        records: list[dict],
    ) -> list[np.ndarray]:
        annotated_frames: list[np.ndarray] = []
        record_lookup = {record["track_id"]: record for record in records}
        tracks: list[WindowFaceTrack] = []
        next_track_id = 1

        for frame_index, (frame, detections) in enumerate(zip(frames, detections_per_frame)):
            annotated = frame.copy()
            unmatched_detection_indices = set(range(len(detections)))

            for track in tracks:
                best_match_index = None
                best_match_iou = 0.0
                for detection_index in list(unmatched_detection_indices):
                    iou = self._iou(track.last_bbox, detections[detection_index]["bbox"]) if track.last_bbox is not None else 0.0
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_index = detection_index

                if best_match_index is None or best_match_iou < self.face_iou_threshold:
                    continue

                detection = detections[best_match_index]
                unmatched_detection_indices.remove(best_match_index)
                track.last_bbox = detection["bbox"]
                track.boxes[frame_index] = detection["bbox"]
                track.observations += 1
                record = record_lookup.get(track.track_id)
                if record is not None:
                    color = TRACK_COLORS.get(record["predicted_index"], (255, 255, 255))
                    x1, y1, x2, y2 = detection["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"{record['student_label']} | {record['predicted_label']} {record['confidence']:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            for detection_index in unmatched_detection_indices:
                detection = detections[detection_index]
                track = WindowFaceTrack(track_id=next_track_id, frame_count=len(frames))
                next_track_id += 1
                track.last_bbox = detection["bbox"]
                track.boxes[frame_index] = detection["bbox"]
                tracks.append(track)
                record = record_lookup.get(track.track_id)
                if record is not None:
                    color = TRACK_COLORS.get(record["predicted_index"], (255, 255, 255))
                    x1, y1, x2, y2 = detection["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            annotated_frames.append(annotated)

        return annotated_frames

    @staticmethod
    def _write_csv(csv_path: Path, rows: list[dict]) -> None:
        import csv

        fieldnames = [
            "window_start_frame",
            "window_start_seconds",
            "track_id",
            "student_id",
            "student_label",
            "student_similarity",
            "observations",
            "decision_source",
            "phone_detected",
            "phone_max_confidence",
            "phone_mean_confidence",
            "phone_hit_count",
            "phone_confirm_streak",
            "predicted_label",
            "confidence",
            "low_engagement",
            "high_engagement",
            "clip_path",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    @staticmethod
    def _build_summary(
        video_path: Path,
        total_frames: int,
        clip_records: list[dict],
        sample_step_frames: int,
        sample_frames: int,
        annotated_video: Path | None = None,
    ) -> dict:
        clips = []
        students: dict[int, dict] = {}
        for record in clip_records:
            student_id = int(record.get("student_id") or 0)
            student_label = record.get("student_label")
            if student_id > 0 and student_id not in students:
                students[student_id] = {
                    "student_id": student_id,
                    "student_label": student_label,
                    "first_window_start_frame": record.get("window_start_frame"),
                }
            clips.append(
                {
                    "window_start_frame": record.get("window_start_frame"),
                    "window_start_seconds": record.get("window_start_seconds"),
                    "track_id": record.get("track_id"),
                    "student_id": record.get("student_id"),
                    "student_label": record.get("student_label"),
                    "student_similarity": round(float(record.get("student_similarity", 0.0)), 4),
                    "observations": record.get("observations"),
                    "decision_source": record.get("decision_source"),
                    "phone_detected": bool(record.get("phone_detected", False)),
                    "phone_max_confidence": round(float(record.get("phone_max_confidence", 0.0)), 4),
                    "phone_mean_confidence": round(float(record.get("phone_mean_confidence", 0.0)), 4),
                    "phone_hit_count": int(record.get("phone_hit_count", 0)),
                    "phone_confirm_streak": int(record.get("phone_confirm_streak", 0)),
                    "predicted_label": record.get("predicted_label"),
                    "confidence": round(float(record.get("confidence", 0.0)), 4),
                    "low_engagement": round(float(record.get("low_engagement", 0.0)), 4),
                    "high_engagement": round(float(record.get("high_engagement", 0.0)), 4),
                    "clip_path": record.get("clip_path"),
                }
            )

        return {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "total_frames": total_frames,
            "sample_step_frames": sample_step_frames,
            "sample_frames": sample_frames,
            "clip_count": len(clips),
            "student_count": len(students),
            "annotated_video": str(annotated_video) if annotated_video else None,
            "clips": clips,
            "students": list(students.values()),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to the input classroom video")
    parser.add_argument(
        "--output-dir",
        default=str(resolve_repo_root() / "ACTIVITY CLASSIFICATION PIPELINE" / "outputs"),
        help="Directory for CSV, JSON, and optional annotated video outputs",
    )
    parser.add_argument(
        "--classifier-path",
        default=str(default_classifier_weights()),
        help="Path to the fine-tuned 3D CNN R-18 checkpoint",
    )
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per classifier clip for the 3D CNN")
    parser.add_argument("--sample-frames", type=int, default=24, help="Number of consecutive frames to sample every interval")
    parser.add_argument("--sample-every-seconds", type=float, default=30.0, help="Sample one window every N seconds")
    parser.add_argument("--crop-size", type=int, default=112, help="Square crop size fed into the classifier")
    parser.add_argument("--crop-padding", type=float, default=0.12, help="Fractional padding around detected face bbox for crop and clip export (0.0=tight crop)")
    parser.add_argument("--export-crop-size", type=int, default=224, help="Square crop size for saved clip frames and phone detector input")
    parser.add_argument("--export-crop-padding", type=float, default=0.60, help="Fractional padding around detected face bbox for saved clip frames and phone detection")
    parser.add_argument("--face-det-size", type=int, default=640, help="InsightFace detection input size")
    parser.add_argument("--face-det-thresh", type=float, default=0.5, help="InsightFace detection threshold")
    parser.add_argument("--face-iou-threshold", type=float, default=0.3, help="IoU threshold used to keep one face grouped across the 15-frame window")
    parser.add_argument("--identity-threshold", type=float, default=0.4, help="Cosine similarity threshold used to keep the same student ID across windows")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")
    parser.add_argument("--no-annotated-video", action="store_true", help="Skip writing the annotated video")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    pipeline = StudentActivityPipeline(
        classifier_path=Path(args.classifier_path),
        num_frames=args.num_frames,
        sample_frames=args.sample_frames,
        sample_every_seconds=args.sample_every_seconds,
        crop_size=args.crop_size,
        crop_padding=args.crop_padding,
        export_crop_size=args.export_crop_size,
        export_crop_padding=args.export_crop_padding,
        face_det_size=args.face_det_size,
        face_det_thresh=args.face_det_thresh,
        face_iou_threshold=args.face_iou_threshold,
        identity_threshold=args.identity_threshold,
        device=args.device,
    )

    result = pipeline.process_video(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        annotate=not args.no_annotated_video,
    )

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved summary: {result['summary_path']}")
    print(f"Saved CSV: {result['csv_path']}")
    print(f"Saved clips: {result['clip_dir']}")
    if result["annotated_video"]:
        print(f"Saved annotated video: {result['annotated_video']}")


if __name__ == "__main__":
    main()