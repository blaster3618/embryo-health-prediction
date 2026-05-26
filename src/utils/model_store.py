"""Model weight discovery and lazy downloads for local/cloud deployments."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "model_manifest.json"
LOCAL_MODEL_DIR = PROJECT_ROOT / "saved_models"
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", PROJECT_ROOT / ".model_cache"))
DEFAULT_CLASSES = ["NonViable", "Viable"]

LEGACY_MODEL_PATHS = {
    "resnet18": PROJECT_ROOT / "resnet18" / "best.pt",
    "resnet50": PROJECT_ROOT / "resnet50" / "best.pt",
}

ProgressCallback = Callable[[int, int | None], None]


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"models": {}}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_entry(arch: str) -> dict:
    manifest = load_manifest()
    return manifest.get("models", {}).get(arch, {})


def model_filename(arch: str) -> str:
    return _model_entry(arch).get("filename", f"{arch}_best.pt")


def class_filename(arch: str) -> str:
    return _model_entry(arch).get("classes_file", f"{arch}_classes.txt")


def format_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _is_usable_model_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 1024 and not _is_lfs_pointer(path)


def expected_size(arch: str) -> int | None:
    value = _model_entry(arch).get("size_bytes")
    return int(value) if value else None


def remote_model_url(arch: str) -> str | None:
    entry = _model_entry(arch)
    env_key = f"MODEL_URL_{arch.upper()}"
    if os.getenv(env_key):
        return os.getenv(env_key)
    if entry.get("url"):
        return entry["url"]

    manifest = load_manifest()
    base_url = (
        os.getenv("MODEL_BASE_URL")
        or os.getenv("MODEL_RELEASE_BASE_URL")
        or manifest.get("base_url")
    )
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/{model_filename(arch)}"


def local_model_path(arch: str) -> Path | None:
    candidates = [
        LOCAL_MODEL_DIR / model_filename(arch),
        CACHE_DIR / model_filename(arch),
    ]
    legacy = LEGACY_MODEL_PATHS.get(arch)
    if legacy is not None:
        candidates.append(legacy)

    for path in candidates:
        if _is_usable_model_file(path):
            return path
    return None


def model_status(arch: str) -> dict:
    path = local_model_path(arch)
    if path:
        source = "cache" if CACHE_DIR in path.parents else "local"
        return {
            "available": True,
            "source": source,
            "path": str(path),
            "url": None,
            "size_bytes": path.stat().st_size,
        }

    url = remote_model_url(arch)
    return {
        "available": bool(url),
        "source": "remote" if url else "missing",
        "path": None,
        "url": url,
        "size_bytes": expected_size(arch),
    }


def available_archs(archs: Iterable[str]) -> list[str]:
    return [arch for arch in archs if model_status(arch)["available"]]


def available_models(archs: Iterable[str]) -> dict:
    return {arch: model_status(arch) for arch in archs if model_status(arch)["available"]}


def load_class_names(arch: str) -> list[str]:
    class_file = LOCAL_MODEL_DIR / class_filename(arch)
    if class_file.exists():
        with class_file.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names

    classes = _model_entry(arch).get("classes")
    return classes if classes else DEFAULT_CLASSES


def ensure_model_file(arch: str, progress: ProgressCallback | None = None) -> Path:
    existing = local_model_path(arch)
    if existing:
        return existing

    url = remote_model_url(arch)
    if not url:
        raise FileNotFoundError(
            f"No local or remote weights are configured for '{arch}'. "
            "Set MODEL_BASE_URL or MODEL_URL_<ARCH>."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / model_filename(arch)
    _download_file(url, destination, expected_size(arch), progress)
    return destination


def _download_file(
    url: str,
    destination: Path,
    expected_bytes: int | None,
    progress: ProgressCallback | None,
) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "embryo-health-prediction/1.0"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else expected_bytes
            downloaded = 0
            with tmp_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total_bytes)
    except urllib.error.HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {exc.reason}") from exc

    if expected_bytes and tmp_path.stat().st_size != expected_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {format_size(tmp_path.stat().st_size)}, "
            f"expected {format_size(expected_bytes)} for {destination.name}."
        )
    if _is_lfs_pointer(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{destination.name} downloaded as a Git LFS pointer, not model weights.")

    shutil.move(str(tmp_path), str(destination))
