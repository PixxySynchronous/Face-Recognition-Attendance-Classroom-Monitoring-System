from __future__ import annotations

import os
from math import ceil
from pathlib import Path
from itertools import product

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from insightface.utils import face_align
try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


_CFG_MNET = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}


class _PriorBox:
    def __init__(self, cfg: dict, image_size: tuple[int, int]):
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self) -> torch.Tensor:
        anchors: list[float] = []
        for k, feature_map in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def _decode(loc: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landm(pre: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    return torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )


def _py_cpu_nms(dets: np.ndarray, thresh: float) -> list[int]:
    if dets.size == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thresh)[0] + 1]

    return keep


def sahi_face_detect(
    fa_get,
    img: np.ndarray,
    slice_size: int = 640,
    overlap: float = 0.2,
    nms_thresh: float = 0.4,
    max_patches: int | None = None,
) -> list:
    """SAHI: full-image detection combined with overlapping-patch detection, merged via NMS.

    The full-image pass preserves all detections that the detector already finds
    (no regression).  The patch passes recover small/distant faces that were being
    downscaled below the detection minimum — each 640-px patch is upscaled 2× by the
    detector (det_size=1280), so a 20-px back-row face becomes 40 px and is found.

    fa_get — callable matching fa.get(img) → objects with
             .bbox, .det_score, .kps, .normed_embedding, .embedding.
    """
    import types as _types

    h, w = img.shape[:2]

    def _collect(source_faces, x0: int = 0, y0: int = 0):
        for face in source_faces:
            bbox = np.asarray(face.bbox, dtype=np.float32).ravel()[:4].copy()
            bbox[[0, 2]] += x0
            bbox[[1, 3]] += y0
            kps = getattr(face, "kps", None)
            if kps is not None:
                kps = np.asarray(kps, dtype=np.float32).copy()
                kps[:, 0] += x0
                kps[:, 1] += y0
            all_boxes.append(np.append(bbox, float(face.det_score)))
            all_kps.append(kps)
            # Copy embeddings immediately — InsightFace reuses its internal
            # output buffer across inference calls, so without a copy every
            # face in all_faces ends up pointing to the same array (the last
            # one written), making all embeddings appear identical.
            ne = getattr(face, "normed_embedding", None)
            e  = getattr(face, "embedding", None)
            import types as _t
            all_faces.append(_t.SimpleNamespace(
                bbox=bbox.copy(),
                kps=kps,
                det_score=float(face.det_score),
                normed_embedding=np.array(ne, dtype=np.float32).copy() if ne is not None else None,
                embedding=np.array(e,  dtype=np.float32).copy() if e  is not None else None,
            ))

    all_boxes: list[np.ndarray] = []
    all_faces: list = []
    all_kps: list = []

    # Pass 1 — full image (guarantees no regression vs. plain detection)
    _collect(fa_get(img))

    # Pass 2 — SAHI patches (recovers small faces lost to downscaling)
    if w > slice_size or h > slice_size:
        stride = max(1, int(slice_size * (1 - overlap)))

        def _positions(dim: int) -> list[int]:
            if dim <= slice_size:
                return [0]
            starts = list(range(0, dim - slice_size, stride))
            starts.append(dim - slice_size)
            return sorted(set(starts))

        xs = _positions(w)
        ys = _positions(h)

        # Skip patch pass if it would exceed the memory budget
        if max_patches is None or len(xs) * len(ys) <= max_patches:
            for y0 in ys:
                for x0 in xs:
                    patch = img[y0:y0 + slice_size, x0:x0 + slice_size]
                    _collect(fa_get(patch), x0=x0, y0=y0)

    if not all_boxes:
        return []

    keep = _py_cpu_nms(np.stack(all_boxes, axis=0), nms_thresh)
    return [
        _types.SimpleNamespace(
            bbox=all_boxes[i][:4],
            kps=all_kps[i],
            det_score=float(all_boxes[i][4]),
            normed_embedding=getattr(all_faces[i], "normed_embedding", None),
            embedding=getattr(all_faces[i], "embedding", None),
        )
        for i in keep
    ]


def _default_model_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []

    env_path = os.environ.get("RETINAFACE_MODEL_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        repo_root / "models" / "retinaface_finetune" / "mobile0.25_retinaface.onnx",
        repo_root / "retinaface_finetune" / "mobile0.25_retinaface.onnx",
        Path("/Users/satyam/Downloads/retinaface_finetune/mobile0.25_retinaface.onnx"),
    ])
    return candidates


def default_retinaface_model_path() -> Path:
    for candidate in _default_model_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the fine-tuned RetinaFace ONNX model. Set RETINAFACE_MODEL_PATH or place "
        "mobile0.25_retinaface.onnx in a supported location."
    )


def _local_face_embedding(img: np.ndarray, kps: np.ndarray | None) -> np.ndarray | None:
    if img is None or kps is None:
        return None

    try:
        aligned = face_align.norm_crop(img, landmark=np.asarray(kps, dtype=np.float32), image_size=112)
    except Exception:
        return None

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (16, 32), interpolation=cv2.INTER_AREA)
    embedding = gray.astype(np.float32).reshape(-1)
    embedding -= float(embedding.mean())
    std = float(embedding.std())
    if std > 0:
        embedding /= std
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding /= norm
    return embedding


class FineTunedRetinaFaceDetector:
    def __init__(self, model_path: str | Path | None = None, providers: list[str] | None = None):
        self.model_path = Path(model_path).expanduser() if model_path is not None else default_retinaface_model_path()
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.cfg = _CFG_MNET
        self.variance = self.cfg["variance"]
        self.det_thresh = 0.5
        self.input_size = (640, 640)
        self.nms_threshold = 0.4
        self.top_k = 5000
        self.keep_top_k = 750

    def prepare(self, ctx_id: int, input_size: tuple[int, int] = (640, 640), det_thresh: float = 0.5):
        self.input_size = tuple(int(v) for v in input_size)
        self.det_thresh = float(det_thresh)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        if img is None or img.size == 0:
            raise ValueError("img cannot be empty")

        height, width = img.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / max(1, width), target_h / max(1, height))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))

        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
        canvas[:resized_h, :resized_w] = resized.astype(np.float32)
        canvas -= np.array((104.0, 117.0, 123.0), dtype=np.float32)
        tensor = canvas.transpose(2, 0, 1)[None, ...]
        return tensor, scale, (height, width)

    def detect(self, img: np.ndarray, max_num: int = 0, metric: str = "default"):
        del max_num, metric

        input_tensor, scale, original_shape = self._preprocess(img)
        target_h, target_w = self.input_size
        priors = _PriorBox(self.cfg, image_size=(target_h, target_w)).forward()

        outputs = self.session.run(None, {self.input_name: input_tensor.astype(np.float32)})
        loc = torch.from_numpy(outputs[0])[0]
        conf = torch.from_numpy(outputs[1])[0]
        landms = torch.from_numpy(outputs[2])[0]

        scores = F.softmax(conf, dim=-1)[:, 1]
        keep = scores > self.det_thresh
        if not torch.any(keep):
            return np.empty((0, 5), dtype=np.float32), None

        loc = loc[keep]
        scores = scores[keep]
        landms = landms[keep]
        priors = priors[keep]

        boxes = _decode(loc, priors, self.variance)
        landms = _decode_landm(landms, priors, self.variance)

        scale_xyxy = torch.tensor([target_w, target_h, target_w, target_h], dtype=torch.float32)
        scale_landm = torch.tensor([target_w, target_h] * 5, dtype=torch.float32)
        boxes = boxes * scale_xyxy
        landms = landms * scale_landm

        boxes_np = boxes.cpu().numpy()
        landms_np = landms.cpu().numpy()
        scores_np = scores.cpu().numpy()

        order = scores_np.argsort()[::-1][: self.top_k]
        boxes_np = boxes_np[order]
        landms_np = landms_np[order]
        scores_np = scores_np[order]

        dets = np.hstack((boxes_np, scores_np[:, np.newaxis])).astype(np.float32, copy=False)
        keep_indices = _py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep_indices, :]
        landms_np = landms_np[keep_indices]

        dets = dets[: self.keep_top_k, :]
        landms_np = landms_np[: self.keep_top_k, :]

        if dets.size == 0:
            return np.empty((0, 5), dtype=np.float32), None

        dets[:, [0, 2]] /= max(scale, 1e-6)
        dets[:, [1, 3]] /= max(scale, 1e-6)
        landms_np[:, 0::2] /= max(scale, 1e-6)
        landms_np[:, 1::2] /= max(scale, 1e-6)

        height, width = original_shape
        dets[:, 0] = np.clip(dets[:, 0], 0, width - 1)
        dets[:, 1] = np.clip(dets[:, 1], 0, height - 1)
        dets[:, 2] = np.clip(dets[:, 2], 0, width - 1)
        dets[:, 3] = np.clip(dets[:, 3], 0, height - 1)
        landms_np[:, 0::2] = np.clip(landms_np[:, 0::2], 0, width - 1)
        landms_np[:, 1::2] = np.clip(landms_np[:, 1::2], 0, height - 1)

        landms_np = landms_np.reshape(-1, 5, 2)

        return dets, landms_np


def _get_mtcnn():
    """Lazy-load facenet-pytorch MTCNN once. Returns None if not installed."""
    if not hasattr(_get_mtcnn, "_instance"):
        try:
            from facenet_pytorch import MTCNN as _MTCNN
            _get_mtcnn._instance = _MTCNN(
                keep_all=True,
                device="cpu",
                min_face_size=20,
                thresholds=[0.5, 0.6, 0.7],
                post_process=False,
            )
        except Exception:
            _get_mtcnn._instance = None
    return _get_mtcnn._instance


def _get_rec_session():
    """Lazy-load buffalo_l recognition ONNX for embedding extraction on MTCNN-only faces."""
    if not hasattr(_get_rec_session, "_session"):
        rec_path = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        try:
            _get_rec_session._session = ort.InferenceSession(
                str(rec_path), providers=["CPUExecutionProvider"]
            )
        except Exception:
            _get_rec_session._session = None
    return _get_rec_session._session


def _embed_from_crop(img_bgr: np.ndarray, kps5: np.ndarray | None) -> np.ndarray | None:
    """Align a face crop and extract a buffalo_l normed embedding."""
    session = _get_rec_session()
    if session is None:
        return None
    try:
        from insightface.utils import face_align as _fa
        if kps5 is not None:
            aligned = _fa.norm_crop(img_bgr, landmark=np.asarray(kps5, dtype=np.float32), image_size=112)
        else:
            aligned = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        blob = ((aligned.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
        emb = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb
    except Exception:
        return None


def build_face_analysis_with_retinaface_detector(
    *,
    name: str = "retinaface_local",
    providers: list[str] | None = None,
    model_path: str | Path | None = None,
):
    """Return a lightweight FaceAnalysis-like adapter using the fine-tuned
    RetinaFace detector plus a local, non-Buffalo embedding extractor.

    The returned object implements `prepare(ctx_id, det_size, det_thresh)` and
    `get(img)`, and produces objects with `bbox`, `kps`, `det_score`, and
    embedding attributes for enrollment and matching.
    """

    providers = providers or ["CPUExecutionProvider"]

    # Always use InsightFace FaceAnalysis (buffalo) for detection and embeddings.
    # If InsightFace is not installed or models are missing, raise an informative error.
    if FaceAnalysis is None:
        raise RuntimeError(
            "InsightFace (FaceAnalysis) is required for buffalo backend. Install the `insightface` package and place buffalo models under ~/.insightface/models/buffalo_l/."
        )

    fa = FaceAnalysis(name="antelopev2", allowed_modules=["detection", "recognition"], providers=providers)

    class FaceAnalysisAdapter:
        def __init__(self, fa, providers):
            self._fa = fa
            self._providers = providers

        def prepare(self, ctx_id: int, det_size=(640, 640), det_thresh: float = 0.5):
            try:
                self._fa.prepare(ctx_id=ctx_id, det_size=det_size)
            except Exception:
                self._fa.prepare(ctx_id=ctx_id)

        def get(self, img, max_num=0):
            import types as _types

            # --- Pass 1: buffalo_l + SAHI (already includes embeddings) ---
            bl_faces = sahi_face_detect(self._fa.get, img, max_patches=None)

            # --- Pass 2: MTCNN (different cascade architecture, fills gaps) ---
            mtcnn_faces: list = []
            mtcnn = _get_mtcnn()
            if mtcnn is not None:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
                    if boxes is not None:
                        for box, prob, lm in zip(boxes, probs, landmarks):
                            x1, y1, x2, y2 = box
                            mtcnn_faces.append(_types.SimpleNamespace(
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                kps=lm.astype(np.float32),   # (5, 2): l_eye r_eye nose ml mr
                                det_score=float(prob),
                                normed_embedding=None,
                                embedding=None,
                            ))
                except Exception:
                    pass

            # --- Merge via NMS ---
            all_faces = bl_faces + mtcnn_faces
            if not all_faces:
                return []

            boxes_arr = np.array(
                [[*f.bbox[:4], float(f.det_score)] for f in all_faces], dtype=np.float32
            )
            keep = _py_cpu_nms(boxes_arr, 0.4)
            n_bl = len(bl_faces)

            out = []
            for idx in keep:
                f = all_faces[idx]
                if idx < n_bl:
                    # Buffalo_l face — normalize its embedding
                    emb = getattr(f, "normed_embedding", None)
                    if emb is None:
                        emb = getattr(f, "embedding", None)
                    normed = None
                    if emb is not None:
                        v = np.asarray(emb, dtype=np.float32).reshape(-1)
                        norm = float(np.linalg.norm(v))
                        normed = v / norm if norm > 0 else v
                else:
                    # MTCNN-only face — extract embedding via buffalo_l rec model
                    x1, y1, x2, y2 = (int(v) for v in f.bbox)
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    normed = _embed_from_crop(crop, f.kps) if crop.size > 0 else None
                    emb = normed

                out.append(_types.SimpleNamespace(
                    bbox=f.bbox,
                    kps=getattr(f, "kps", None),
                    det_score=f.det_score,
                    normed_embedding=normed,
                    embedding=emb,
                ))
            return out

    return FaceAnalysisAdapter(fa=fa, providers=providers)