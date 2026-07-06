"""
AdaFace IR-101 (CVLFace, minchul/cvlface_adaface_ir101_webface12m) recognition backbone.

Replaces glintr100/antelopev2 as the production embedding model. InsightFace's
`FaceAnalysis` is still used for detection (bbox + 5-pt landmarks); this module
only handles alignment + embedding.

Preprocessing:
  - color_space: RGB → convert BGR→RGB before normalising
  - normalisation: (pixel/255 - 0.5) / 0.5, applied after the BGR→RGB flip
  - alignment: insightface.utils.face_align.norm_crop, 112x112 (same as glintr100)

Output: (512-d L2-normalised embedding, pre-BN feature norm — quality proxy).
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Flatten,
                      Linear, MaxPool2d, PReLU, Sequential)


# ---------------------------------------------------------------------------
# Building blocks (CVLFace IR-101 architecture)
# ---------------------------------------------------------------------------

class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class _BlockSpec(NamedTuple):
    in_channel: int
    depth: int
    stride: int


def _get_blocks_ir101() -> list[list[_BlockSpec]]:
    """Block specs for IR-101: unit counts [3, 13, 30, 3]."""
    def _block(in_c: int, depth: int, n: int) -> list[_BlockSpec]:
        return [_BlockSpec(in_c, depth, 2)] + [_BlockSpec(depth, depth, 1)] * (n - 1)

    return [
        _block(64,  64,  3),
        _block(64,  128, 13),
        _block(128, 256, 30),
        _block(256, 512, 3),
    ]


class Backbone(nn.Module):
    """CVLFace Backbone (InsightFace-style IR). Forward returns (emb, feat_norm)."""

    def __init__(self, blocks_spec: list[list[_BlockSpec]], output_dim: int = 512,
                 dropout: float = 0.4):
        super().__init__()
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        units = [BasicBlockIR(b.in_channel, b.depth, b.stride)
                 for block in blocks_spec for b in block]
        self.body = Sequential(*units)

        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(p=dropout),
            Flatten(),
            Linear(512 * 7 * 7, output_dim, bias=True),
            BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer[0](x)
        x = self.output_layer[1](x)
        x = self.output_layer[2](x)
        x = self.output_layer[3](x)
        feat_norm = torch.norm(x, p=2, dim=1)
        x = self.output_layer[4](x)
        emb = F.normalize(x, p=2, dim=1)
        return emb, feat_norm


def ir101(output_dim: int = 512) -> Backbone:
    return Backbone(_get_blocks_ir101(), output_dim=output_dim)


def load_cvlface_checkpoint(model: Backbone, ckpt_path: str) -> None:
    """Load CVLFace model.pt, stripping the 'net.' prefix from state-dict keys."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and not isinstance(raw, nn.Module):
        sd = raw.get("state_dict", raw)
    else:
        sd = raw
    if any(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
    model.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# Preprocessing + wrapper
# ---------------------------------------------------------------------------

def _to_tensor(aligned_bgr: np.ndarray) -> torch.Tensor:
    """112x112 BGR ndarray -> (1,3,112,112) float32 tensor, RGB, normalised to [-1,1]."""
    arr = aligned_bgr[:, :, ::-1].copy()       # BGR -> RGB
    arr = arr.astype(np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5            # -> [-1, 1]
    arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1, C, H, W)
    return torch.from_numpy(arr)


class AdaFaceWrapper:
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._model.eval()

    @classmethod
    def load(cls, ckpt_path: str | Path) -> "AdaFaceWrapper":
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"AdaFace checkpoint not found: {ckpt_path}\n"
                "Run `python download_models.py` to fetch it."
            )
        model = ir101()
        load_cvlface_checkpoint(model, str(ckpt_path))
        model.eval()
        return cls(model)

    def embed_aligned(self, aligned_bgr_112: np.ndarray) -> tuple[np.ndarray, float]:
        """
        aligned_bgr_112 : 112x112 BGR ndarray (from insightface.utils.face_align.norm_crop)
        Returns (emb (512,) float32 L2-normalised, feat_norm quality proxy).
        """
        if aligned_bgr_112 is None or aligned_bgr_112.size == 0:
            raise ValueError("aligned_bgr_112 cannot be empty")
        crop = aligned_bgr_112
        if crop.shape[:2] != (112, 112):
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        tensor = _to_tensor(crop)
        with torch.inference_mode():
            emb_t, norm_t = self._model(tensor)
        emb = emb_t[0].numpy().astype(np.float32)
        norm = float(norm_t[0].item())
        return emb, norm


# ---------------------------------------------------------------------------
# Download helper — mirrors the YOLO-weight pattern in download_models.py
# ---------------------------------------------------------------------------

_HF_REPO = "minchul/cvlface_adaface_ir101_webface12m"
_HF_FILE = "pretrained_model/model.pt"

DEFAULT_CKPT_PATH = Path(__file__).resolve().parent.parent / "models" / "adaface" / "adaface_ir101_webface12m.pt"


def download_model(dest: Path = DEFAULT_CKPT_PATH, force: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    from huggingface_hub import hf_hub_download
    tmp = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE, local_dir=str(dest.parent))
    tmp_p = Path(tmp)
    if tmp_p != dest:
        tmp_p.rename(dest)
    return dest
