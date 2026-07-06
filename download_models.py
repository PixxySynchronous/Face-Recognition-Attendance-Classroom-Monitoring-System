"""
Download YOLO model weights that are stored as Git LFS pointers in the repo.
Run once after cloning: python download_models.py
"""
import urllib.request
import sys
from pathlib import Path

MODELS = [
    (
        "yolov8n-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
        6_800_000,
    ),
    (
        "yolov8s-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt",
        23_000_000,
    ),
    (
        "Activity monitoring/Training Pipelines/assets/yolo11m.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        40_000_000,
    ),
]

ROOT = Path(__file__).parent


def _is_lfs_pointer(path: Path) -> bool:
    """Return True if the file is a Git LFS pointer rather than real weights."""
    if path.stat().st_size > 1_000:
        return False
    try:
        return path.read_bytes().startswith(b"version https://git-lfs")
    except OSError:
        return False


def _download(url: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    def _progress(count, block, total):
        if total > 0:
            pct = min(count * block * 100 // total, 100)
            print(f"\r  downloading {dest.name} … {pct}%", end="", flush=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {dest.name} … (attempt {attempt}/{retries})", end="", flush=True)
            urllib.request.urlretrieve(url, tmp, _progress)
            tmp.rename(dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"\r  {dest.name} — {size_mb:.1f} MB  ✓")
            return
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            print(f"\r  attempt {attempt} failed: {exc}")
            if attempt == retries:
                print(f"  FAILED after {retries} attempts: {dest.name}")
                sys.exit(1)


def _download_adaface() -> bool:
    from utils.adaface_backbone import DEFAULT_CKPT_PATH, download_model

    if DEFAULT_CKPT_PATH.exists() and not _is_lfs_pointer(DEFAULT_CKPT_PATH):
        print(f"  {DEFAULT_CKPT_PATH.name} already present, skipping.")
        return False
    print(f"  downloading {DEFAULT_CKPT_PATH.name} from HuggingFace (minchul/cvlface_adaface_ir101_webface12m) …")
    download_model(DEFAULT_CKPT_PATH)
    size_mb = DEFAULT_CKPT_PATH.stat().st_size / 1e6
    print(f"  {DEFAULT_CKPT_PATH.name} — {size_mb:.1f} MB  ✓")
    return True


def main() -> None:
    any_downloaded = False
    for rel, url, _ in MODELS:
        dest = ROOT / rel
        if dest.exists() and not _is_lfs_pointer(dest):
            print(f"  {dest.name} already present, skipping.")
            continue
        _download(url, dest)
        any_downloaded = True

    any_downloaded = _download_adaface() or any_downloaded

    if not any_downloaded:
        print("All model weights are already in place.")
    else:
        print("\nDone. You can now start the app.")


if __name__ == "__main__":
    main()
