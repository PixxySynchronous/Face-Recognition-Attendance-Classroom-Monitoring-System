from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from .startup import get_ffmpeg_path


def _transcode_file(src: Path, tmp_suffix: str = ".trans.mp4") -> bool:
    """Transcode `src` to H.264 mp4 using ffmpeg in-place (via temp file).
    Returns True if replaced successfully.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return False
    tmp = src.with_suffix(src.suffix + tmp_suffix)
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, check=False)
        if ret.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False
        # replace original
        tmp.replace(src)
        return True
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def transcode_clips_in_dir(dirpath: Path) -> int:
    """Transcode matching files under `dirpath`. Returns count replaced."""
    replaced = 0
    if not get_ffmpeg_path():
        return 0
    for p in dirpath.rglob("*.mp4"):
        # skip files that are already small or likely h264? We transcode anyway.
        try:
            if p.stat().st_size == 0:
                continue
            ok = _transcode_file(p)
            if ok:
                replaced += 1
        except Exception:
            continue
    return replaced


def transcode_clips_async(base_dir: Path) -> threading.Thread:
    """Start background thread to transcode clips under `base_dir` and return the Thread."""
    def _worker():
        try:
            count = transcode_clips_in_dir(base_dir)
            if count:
                print(f"[transcode] replaced {count} clip(s) with H.264-encoded MP4s")
        except Exception as exc:
            print(f"[transcode] error: {exc}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
