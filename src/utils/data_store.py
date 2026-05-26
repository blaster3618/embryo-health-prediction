"""Evaluation dataset discovery and lazy downloads."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "data_manifest.json"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "data" / "embryo" / "test_data"
DATA_CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", PROJECT_ROOT / ".data_cache"))

ProgressCallback = Callable[[int, int | None], None]


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _entry() -> dict:
    return load_manifest().get("evaluation_data", {})


def evaluation_data_url() -> str | None:
    if os.getenv("EVALUATION_DATA_URL"):
        return os.getenv("EVALUATION_DATA_URL")

    entry = _entry()
    if entry.get("url"):
        return entry["url"]

    base_url = os.getenv("EVALUATION_DATA_BASE_URL") or entry.get("base_url")
    filename = entry.get("filename")
    if not base_url or not filename:
        return None
    return f"{base_url.rstrip('/')}/{filename}"


def expected_test_data_path() -> Path:
    configured = _entry().get("expected_path")
    return PROJECT_ROOT / configured if configured else DEFAULT_TEST_DATA_PATH


def has_test_data(path: str | Path | None = None) -> bool:
    target = Path(path) if path else expected_test_data_path()
    return (target / "bad").is_dir() and (target / "good").is_dir()


def evaluation_data_status(path: str | Path | None = None) -> dict:
    target = Path(path) if path else expected_test_data_path()
    if has_test_data(target):
        return {"available": True, "source": "local", "path": str(target), "url": None}

    url = evaluation_data_url()
    return {"available": bool(url), "source": "remote" if url else "missing", "path": str(target), "url": url}


def ensure_evaluation_data(progress: ProgressCallback | None = None) -> Path:
    target = expected_test_data_path()
    if has_test_data(target):
        return target

    url = evaluation_data_url()
    if not url:
        raise FileNotFoundError(
            "Evaluation data is not available locally and no remote evaluation data URL is configured."
        )

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_CACHE_DIR / _entry().get("filename", "embryo-test-data.zip")
    if not archive_path.exists():
        _download_file(url, archive_path, progress)

    extract_to = PROJECT_ROOT / _entry().get("extract_to", "data/embryo")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_to)

    if not has_test_data(target):
        raise RuntimeError(f"Downloaded evaluation data did not create {target}")
    return target


def _download_file(url: str, destination: Path, progress: ProgressCallback | None) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "embryo-health-prediction/1.0"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
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

    shutil.move(str(tmp_path), str(destination))
