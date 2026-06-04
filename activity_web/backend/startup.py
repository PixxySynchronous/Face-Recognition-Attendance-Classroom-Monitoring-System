from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable


def get_ffmpeg_path() -> str | None:
    """Return an ffmpeg executable path if available, including imageio-ffmpeg fallback."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
        if ffmpeg and Path(ffmpeg).exists():
            return ffmpeg
    except Exception:
        pass
    return None


def ensure_insightface_models_flat(model_names: Iterable[str] | None = None) -> None:
    """Ensure InsightFace model directories are not nested (model/model/*).

    If a model folder was extracted into a nested directory like
    ~/.insightface/models/antelopev2/antelopev2/* this moves the inner
    contents up one level so InsightFace can find them.
    """
    home = Path.home()
    base = home / ".insightface" / "models"
    if model_names is None:
        model_names = [p.name for p in base.iterdir() if p.is_dir()] if base.exists() else []

    for name in model_names:
        model_dir = base / name
        nested = model_dir / name
        try:
            if nested.exists() and nested.is_dir():
                print(f"[startup] fixing nested insightface model folder: {nested}")
                for item in nested.iterdir():
                    target = model_dir / item.name
                    # if target exists, overwrite by moving with replace
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(model_dir))
                # remove now-empty nested dir
                try:
                    nested.rmdir()
                except Exception:
                    pass
        except Exception as exc:
            print(f"[startup] warning: failed to flatten model folder {nested}: {exc}")


def check_ffmpeg_available() -> bool:
    """Return True if ffmpeg is available via PATH or imageio-ffmpeg."""
    ffmpeg = get_ffmpeg_path()
    if ffmpeg:
        return True
    print("[startup] warning: ffmpeg not found on PATH. Clips may be encoded with a codec\n"
          "that some browsers cannot play. To ensure in-browser playback, install ffmpeg:\n"
          "  - Windows: install via Chocolatey `choco install ffmpeg` or download from https://ffmpeg.org/download.html\n"
          "  - macOS: `brew install ffmpeg`\n"
          "  - Linux: use your distro package manager (apt/yum/pacman)\n")
    return False


