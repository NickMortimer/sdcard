"""Extract video frames into interim caches and stamp EXIF for later ``xif`` indexing."""

from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table
from rich.text import Text

from sdcard.config import Config
from sdcard.utils.config_path_cache import resolve_config_path
from sdcard.utils.cli_xif import _normalize_extensions, _resolve_exiftool_executable


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

FRAMES_MANIFEST_NAME = "frames.csv"

_START_TIME_KEYS = (
    "MediaCreateDate",
    "TrackCreateDate",
    "CreateDate",
    "DateTimeOriginal",
)

_IDENTITY_TAG_ALLOWLIST = frozenset(
    {
        "Make",
        "Model",
        "LensModel",
        "SerialNumber",
        "CameraSerialNumber",
        "InternalSerialNumber",
        "DroneSerialNumber",
    }
)

_EXIFTOOL_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"
_STILL_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".dng"}
_STILL_FILETYPES = frozenset({"JPEG", "JPG", "TIFF", "DNG"})
_TAG_PAIR_RE = re.compile(r"^([^=]+)=(.*)$", re.DOTALL)


def _decompress_json_zstd(payload: bytes) -> dict[str, dict[str, object]]:
    """Return a JSON object previously compressed with zstd."""
    try:
        import compression.zstd as std_zstd

        raw = std_zstd.decompress(payload)
    except ImportError:
        import zstandard

        raw = zstandard.ZstdDecompressor().decompress(payload)

    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Expected extracted metadata JSON object")
    return parsed


def _parse_tag_pair(raw: str) -> tuple[str, str]:
    """Parse a ``Key=Value`` CLI tag override."""
    match = _TAG_PAIR_RE.match(raw.strip())
    if match is None:
        raise typer.BadParameter(
            f"Invalid --tag '{raw}'. Expected KEY=VALUE.",
            param_hint="--tag",
        )
    key = match.group(1).strip()
    value = match.group(2)
    if not key:
        raise typer.BadParameter(
            f"Invalid --tag '{raw}'. Tag name must not be empty.",
            param_hint="--tag",
        )
    return key, value


def _parse_tags_json(raw: str | None) -> dict[str, object]:
    """Parse optional JSON object of tag overrides."""
    if raw is None or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Invalid --tags-json: {exc}",
            param_hint="--tags-json",
        ) from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(
            "--tags-json must be a JSON object",
            param_hint="--tags-json",
        )
    return payload


def _merge_identity_tags(
    discovered: dict[str, object],
    tag_pairs: list[str] | None,
    tags_json: str | None,
) -> dict[str, object]:
    """Merge discovered identity tags with CLI overrides (CLI wins)."""
    merged = dict(discovered)
    merged.update(_parse_tags_json(tags_json))
    for pair in tag_pairs or []:
        key, value = _parse_tag_pair(pair)
        merged[key] = value
    return merged


def _identity_tags_from_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return copyable camera/drone identity tags from a metadata mapping."""
    tags: dict[str, object] = {}
    for key, value in metadata.items():
        if key not in _IDENTITY_TAG_ALLOWLIST:
            continue
        if value is None:
            continue
        if isinstance(value, str) and (
            not value.strip() or value.startswith("(Binary data")
        ):
            continue
        tags[key] = value
    return tags


def _mirror_serial_tags(tags: dict[str, object]) -> dict[str, object]:
    """
    Ensure both ``SerialNumber`` and ``CameraSerialNumber`` are set when either is.

    DJI stills often only populate ``SerialNumber``; downstream tools (p4rtk, camgeo)
    look for either name.
    """
    out = dict(tags)
    serial = out.get("SerialNumber")
    camera = out.get("CameraSerialNumber")
    if serial and not camera:
        out["CameraSerialNumber"] = serial
    elif camera and not serial:
        out["SerialNumber"] = camera
    return out


def _is_still_sidecar_entry(name: str, entry: dict[str, object]) -> bool:
    """True when a sidecar row looks like a still image, not a video."""
    file_type = str(entry.get("FileType") or "").strip().upper()
    if file_type in _STILL_FILETYPES:
        return True
    if file_type in {"MOV", "MP4", "AVI", "MKV", "M4V"}:
        return False
    return Path(name).suffix.lower() in _STILL_SUFFIXES


def _find_dcim_root(path: Path) -> Path | None:
    """Return the ``DCIM`` ancestor of ``path``, if any."""
    for parent in (path, *path.parents):
        if parent.name.upper() == "DCIM":
            return parent
    return None


def _iter_sidecar_dirs_near_video(video_dir: Path) -> list[Path]:
    """
    Directories to search for still identity, video folder first.

    Then other folders under the same ``DCIM`` that have ``exif.json.zst``
    (e.g. ``100MEDIA`` vs ``SURVEY/100_0001``).
    """
    ordered: list[Path] = []
    seen: set[Path] = set()

    def add(directory: Path) -> None:
        try:
            resolved = directory.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        if not (directory / "exif.json.zst").is_file():
            return
        seen.add(resolved)
        ordered.append(directory)

    add(video_dir)
    dcim = _find_dcim_root(video_dir)
    if dcim is None:
        return ordered

    try:
        children = sorted(
            (child for child in dcim.iterdir() if child.is_dir()),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        return ordered

    for child in children:
        add(child)
        if child.name.upper() != "SURVEY":
            continue
        try:
            survey_dirs = sorted(
                (entry for entry in child.iterdir() if entry.is_dir()),
                key=lambda path: path.name.lower(),
            )
        except OSError:
            continue
        for survey_dir in survey_dirs:
            add(survey_dir)
    return ordered


def _identity_tags_from_directory_stills(
    directory: Path,
    *,
    prefer_serial: bool = True,
) -> dict[str, object]:
    """Sample identity tags from still rows in ``directory/exif.json.zst``."""
    sidecar = _load_sidecar_metadata(directory)
    if not sidecar:
        return {}

    candidates: list[dict[str, object]] = []
    for name, entry in sidecar.items():
        if not isinstance(entry, dict) or not entry:
            continue
        if not _is_still_sidecar_entry(str(name), entry):
            continue
        tags = _identity_tags_from_metadata(entry)
        if not tags:
            continue
        candidates.append(tags)

    if not candidates:
        return {}

    if prefer_serial:
        for tags in candidates:
            if tags.get("SerialNumber") or tags.get("CameraSerialNumber"):
                return tags
    return candidates[0]


def _identity_tags_from_sibling_still(
    directory: Path,
    *,
    prefer_serial: bool = True,
) -> dict[str, object]:
    """
    Sample identity from stills near a video directory.

    Order: video directory sidecar, then other ``exif.json.zst`` folders under the
    same ``DCIM`` (``100MEDIA``, ``SURVEY/…``). Prefers a still that carries a serial.
    """
    best_without_serial: dict[str, object] = {}
    for sidecar_dir in _iter_sidecar_dirs_near_video(directory):
        tags = _identity_tags_from_directory_stills(
            sidecar_dir,
            prefer_serial=prefer_serial,
        )
        if not tags:
            continue
        if tags.get("SerialNumber") or tags.get("CameraSerialNumber"):
            return tags
        if not best_without_serial:
            best_without_serial = tags
    return best_without_serial


def _resolve_identity_tags(
    video_path: Path,
    metadata: dict[str, object],
    identity_overrides: dict[str, object],
) -> dict[str, object]:
    """
    Build identity tags: video metadata, then fill gaps from nearby stills, then CLI.

    Nearby = same directory, else other folders under the same ``DCIM``.
    Always mirrors SerialNumber ↔ CameraSerialNumber so both land on stamped frames.
    """
    discovered = _identity_tags_from_metadata(metadata)
    needed = ("Make", "Model", "SerialNumber", "CameraSerialNumber")
    if any(key not in discovered for key in needed):
        sibling = _identity_tags_from_sibling_still(video_path.parent)
        for key, value in sibling.items():
            discovered.setdefault(key, value)
    merged = {**discovered, **identity_overrides}
    return _mirror_serial_tags(merged)


def _parse_exif_datetime(raw: object) -> datetime | None:
    """Parse common ExifTool datetime strings into naive datetime."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # ExifTool may append timezone: "2026:08:03 13:00:31+08:00" or "Z"
    text = re.sub(r"([+-]\d{2}:\d{2}|Z)$", "", text).strip()
    for fmt in (
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_exif_datetime(value: datetime) -> str:
    """Format datetime for ExifTool, including fractional seconds when present."""
    base = value.strftime(_EXIFTOOL_DATETIME_FORMAT)
    if not value.microsecond:
        return base
    frac = f"{value.microsecond:06d}".rstrip("0")
    return f"{base}.{frac}"


def _subsec_time_tag(value: datetime) -> str | None:
    """Return SubSecTimeOriginal digits (no decimal), or None if whole-second."""
    if not value.microsecond:
        return None
    return f"{value.microsecond:06d}".rstrip("0") or "0"


def _fps_dirname_label(fps: float) -> str:
    """Return a filesystem-safe fps label such as ``1fps`` or ``0.5fps``."""
    if float(fps).is_integer():
        return f"{int(fps)}fps"
    text = f"{fps:.6f}".rstrip("0").rstrip(".")
    return f"{text}fps"


def _default_frames_output_root(card_store: Path) -> Path:
    """Map ``.../raw/sdcards`` → ``.../interim/frames``; else sibling ``frames``."""
    resolved = card_store.resolve()
    parts = resolved.parts
    if len(parts) >= 2 and parts[-1] == "sdcards" and parts[-2] == "raw":
        return resolved.parents[1] / "interim" / "frames"
    return resolved.parent / "frames"


def _frames_output_dir(
    video_path: Path,
    source_root: Path,
    output_root: Path,
    fps: float,
) -> Path:
    """Return ``output_root/{rel}/{stem}.{fps}fps.frames`` for a video."""
    relative = video_path.resolve().relative_to(source_root.resolve())
    return (
        output_root
        / relative.parent
        / f"{video_path.stem}.{_fps_dirname_label(fps)}.frames"
    )


def _load_sidecar_metadata(directory: Path) -> dict[str, dict[str, object]]:
    """Load ``exif.json.zst`` from a directory when present."""
    sidecar = directory / "exif.json.zst"
    if not sidecar.is_file():
        return {}
    try:
        return _decompress_json_zstd(sidecar.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _probe_video_metadata(
    video_path: Path,
    exiftool_executable: str,
) -> dict[str, object]:
    """Return ExifTool JSON metadata for one video file."""
    command = [
        exiftool_executable,
        "-api",
        "LargeFileSupport=1",
        "-j",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        rendered = shlex.join(command)
        raise RuntimeError(
            f"exiftool failed for '{rendered}' "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or not payload:
        return {}
    item = payload[0]
    return item if isinstance(item, dict) else {}


def _resolve_video_metadata(
    video_path: Path,
    exiftool_executable: str | None,
) -> tuple[dict[str, object], str]:
    """
    Resolve metadata for a video.

    Prefers the parent directory ``exif.json.zst`` entry; falls back to exiftool.
    Returns ``(metadata, source_label)``.
    """
    sidecar = _load_sidecar_metadata(video_path.parent)
    entry = sidecar.get(video_path.name)
    if isinstance(entry, dict) and entry:
        return entry, "exif.json.zst"

    if exiftool_executable is None:
        return {}, "none"

    try:
        return _probe_video_metadata(video_path, exiftool_executable), "exiftool"
    except (RuntimeError, json.JSONDecodeError, OSError):
        return {}, "none"


def _resolve_video_start(
    video_path: Path,
    metadata: dict[str, object],
) -> tuple[datetime, str]:
    """Return ``(start_time, source_label)`` for frame timestamping."""
    for key in _START_TIME_KEYS:
        parsed = _parse_exif_datetime(metadata.get(key))
        if parsed is not None:
            return parsed, key
    mtime = datetime.fromtimestamp(video_path.stat().st_mtime)
    return mtime, "mtime"


def _iter_videos(
    source_root: Path,
    allowed_suffixes: set[str],
) -> list[Path]:
    """Return video files under source_root, skipping AppleDouble junk."""
    videos: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        videos.append(path)
    return videos


def _extraction_complete(output_dir: Path) -> bool:
    """Return whether a prior run finished (manifest CSV present)."""
    return (output_dir / FRAMES_MANIFEST_NAME).is_file()


def _clear_frame_outputs(output_dir: Path) -> None:
    """Remove prior frame JPEGs and manifest so a re-extract cannot keep stale files."""
    if not output_dir.exists():
        return
    for path in output_dir.glob("frame_*.jpg"):
        path.unlink(missing_ok=True)
    manifest = output_dir / FRAMES_MANIFEST_NAME
    manifest.unlink(missing_ok=True)


def _write_extraction_csv(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> Path:
    """
    Write the completion manifest for a finished extraction.

    Presence of this file is the skip/complete signal for later runs.
    """
    manifest = output_dir / FRAMES_MANIFEST_NAME
    fieldnames = ["frame_number", "file", "elapsed_s", "frame_time"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest


def _extract_frames_with_ffmpeg(
    video_path: Path,
    output_dir: Path,
    fps: float,
    ffmpeg_executable: str,
) -> list[Path]:
    """Extract JPEG frames with ffmpeg and return sorted frame paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(pattern),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        rendered = shlex.join(command)
        raise RuntimeError(
            f"ffmpeg failed for '{rendered}' "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )
    return sorted(output_dir.glob("frame_*.jpg"))


def _stamp_frame_exif(
    frame_path: Path,
    frame_time: datetime,
    identity_tags: dict[str, object],
    exiftool_executable: str,
) -> None:
    """Write DateTimeOriginal/CreateDate plus identity tags onto one JPEG."""
    payload_tags: dict[str, object] = {
        "SourceFile": str(frame_path),
        "DateTimeOriginal": _format_exif_datetime(frame_time),
        "CreateDate": _format_exif_datetime(frame_time),
        **identity_tags,
    }
    subsec = _subsec_time_tag(frame_time)
    if subsec is not None:
        payload_tags["SubSecTimeOriginal"] = subsec
        payload_tags["SubSecTimeDigitized"] = subsec
    payload = [payload_tags]

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path = Path(handle.name)

    try:
        command = [
            exiftool_executable,
            "-overwrite_original",
            "-m",
            f"-json={temp_path}",
            str(frame_path),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            rendered = shlex.join(command)
            raise RuntimeError(
                f"exiftool failed for '{rendered}' "
                f"(exit code {result.returncode}): {result.stderr.strip()}"
            )
    finally:
        temp_path.unlink(missing_ok=True)


def _remove_frames_directory(output_dir: Path) -> None:
    """Delete an existing ``*.frames`` cache directory entirely."""
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _process_video(
    video_path: Path,
    source_root: Path,
    output_root: Path,
    fps: float,
    *,
    skip_existing: bool,
    clean: bool,
    identity_overrides: dict[str, object],
    ffmpeg_executable: str,
    exiftool_executable: str | None,
    on_progress=None,
    on_status=None,
) -> tuple[str, int, str | None]:
    """
    Extract and stamp frames for one video.

    Returns ``(status, frame_count, warning)`` where status is ``extracted`` or
    ``skipped``. Completion is marked only by writing ``frames.csv`` after all
    stamps succeed.
    """
    output_dir = _frames_output_dir(video_path, source_root, output_root, fps)
    complete = _extraction_complete(output_dir)

    if clean and output_dir.exists():
        _remove_frames_directory(output_dir)
        complete = False
    elif complete and skip_existing:
        return "skipped", 0, None

    # Incomplete prior run (JPEGs without frames.csv) or forced re-extract:
    # drop stale outputs so frame indices cannot mix across runs.
    if output_dir.exists():
        _clear_frame_outputs(output_dir)

    metadata, _meta_source = _resolve_video_metadata(video_path, exiftool_executable)
    identity_tags = _resolve_identity_tags(video_path, metadata, identity_overrides)
    start_time, start_source = _resolve_video_start(video_path, metadata)
    warning = None
    if start_source == "mtime":
        warning = f"No video start time for {video_path.name}; using file mtime"

    if exiftool_executable is None:
        raise RuntimeError(
            "exiftool is required to stamp frame DateTimeOriginal and identity tags"
        )

    if on_status is not None:
        on_status("extracting")
    frames = _extract_frames_with_ffmpeg(
        video_path,
        output_dir,
        fps,
        ffmpeg_executable,
    )
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")

    total = len(frames)
    if on_status is not None:
        on_status("stamping")
    if on_progress is not None:
        on_progress(0, total)

    manifest_rows: list[dict[str, object]] = []
    for index, frame_path in enumerate(frames, start=1):
        elapsed_s = (index - 1) / fps
        frame_time = start_time + timedelta(seconds=elapsed_s)
        _stamp_frame_exif(
            frame_path,
            frame_time,
            identity_tags,
            exiftool_executable,
        )
        manifest_rows.append(
            {
                "frame_number": index,
                "file": frame_path.name,
                "elapsed_s": f"{elapsed_s:.6f}".rstrip("0").rstrip(".") or "0",
                "frame_time": frame_time.isoformat(sep=" ", timespec="microseconds"),
            }
        )
        if on_progress is not None:
            on_progress(index, total)

    # Write last: presence means this extraction finished successfully.
    _write_extraction_csv(output_dir, manifest_rows)
    return "extracted", total, warning


def _status_style(status: str) -> str:
    """Return Rich style for a status value."""
    lowered = status.lower()
    if lowered in {"queued", "skipped"}:
        return "grey62"
    if lowered in {"running", "extracting", "stamping"}:
        return "orange1"
    if lowered == "done":
        return "green"
    if lowered in {"failed", "error"}:
        return "red"
    return "white"


def _short_path(path_text: str, max_len: int = 48) -> str:
    """Shorten long paths for fixed-width table output."""
    if len(path_text) <= max_len:
        return path_text
    return "..." + path_text[-(max_len - 3) :]


def _progress_bar(percent: int, width: int = 14) -> str:
    """Render a simple textual progress bar."""
    bounded = max(0, min(100, percent))
    filled = round((bounded / 100) * width)
    return f"{'#' * filled}{'-' * (width - filled)} {bounded:>3}%"


def _build_frames_table(
    job_order: list[Path],
    job_state: dict[Path, dict[str, object]],
    completed_jobs: int,
    total_jobs: int,
    processed_frames: int,
    total_frames: int,
    frames_per_second: float,
) -> Table:
    """Build live status table for video frame extraction jobs."""
    title = "Frame Extraction"
    if total_jobs > 0:
        job_percent = int((completed_jobs / total_jobs) * 100)
        if total_frames > 0:
            title = (
                f"Frame Extraction videos [{completed_jobs}/{total_jobs}] "
                f"{_progress_bar(job_percent, width=20)}  "
                f"frames [{processed_frames}/{total_frames}]  "
                f"{frames_per_second:.1f} frames/s"
            )
        else:
            title = (
                f"Frame Extraction videos [{completed_jobs}/{total_jobs}] "
                f"{_progress_bar(job_percent, width=20)}"
            )

    table = Table(title=title)
    table.add_column("Video")
    table.add_column("Output", no_wrap=True)
    table.add_column("Frames", no_wrap=True)
    table.add_column("Done", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    for video_path in job_order:
        state = job_state[video_path]
        status = str(state["status"])
        style = _status_style(status)
        table.add_row(
            Text(_short_path(str(video_path)), style=style),
            str(state["output"]),
            str(state["frames"]),
            str(state["done_frames"]),
            str(state["progress"]),
            Text(status, style=style),
        )
    return table


def extract_frames_tree(
    source_root: Path,
    output_root: Path,
    fps: float,
    *,
    skip_existing: bool,
    clean: bool,
    allowed_suffixes: set[str],
    identity_overrides: dict[str, object],
    ffmpeg_executable: str,
    exiftool_executable: str | None,
    workers: int = 4,
) -> dict[str, int]:
    """Walk videos under source_root and extract stamped frames under output_root."""
    extracted = 0
    skipped = 0
    failed = 0

    videos = _iter_videos(source_root, allowed_suffixes)
    if not videos:
        typer.echo("No videos found.")
        return {"extracted": 0, "skipped": 0, "failed": 0, "videos": 0}

    output_root.mkdir(parents=True, exist_ok=True)

    job_order = list(videos)
    state_lock = threading.Lock()
    start_time = time.monotonic()
    job_state: dict[Path, dict[str, object]] = {
        video_path: {
            "status": "queued",
            "output": _frames_output_dir(
                video_path, source_root, output_root, fps
            ).name,
            "frames": 0,
            "done_frames": 0,
            "progress": _progress_bar(0),
        }
        for video_path in videos
    }

    completed_jobs = 0
    total_jobs = len(videos)
    processed_frames = 0
    known_total_frames = 0
    failed_messages: list[str] = []
    warning_messages: list[str] = []

    def _on_progress(video_path: Path, done: int, total: int) -> None:
        nonlocal processed_frames, known_total_frames
        with state_lock:
            state = job_state[video_path]
            previous_done = int(state["done_frames"])
            previous_total = int(state["frames"])
            state["frames"] = total
            state["done_frames"] = done
            state["progress"] = _progress_bar(
                int((done / total) * 100) if total else 0
            )
            processed_frames += done - previous_done
            if previous_total == 0 and total > 0:
                known_total_frames += total

    def _on_status(video_path: Path, status: str) -> None:
        with state_lock:
            job_state[video_path]["status"] = status

    def _run_one(video_path: Path) -> tuple[Path, str, int, str | None, str | None]:
        with state_lock:
            job_state[video_path]["status"] = "running"
        try:
            status, frame_count, warning = _process_video(
                video_path,
                source_root,
                output_root,
                fps,
                skip_existing=skip_existing,
                clean=clean,
                identity_overrides=identity_overrides,
                ffmpeg_executable=ffmpeg_executable,
                exiftool_executable=exiftool_executable,
                on_progress=lambda done, total: _on_progress(video_path, done, total),
                on_status=lambda phase: _on_status(video_path, phase),
            )
            return video_path, status, frame_count, warning, None
        except Exception as exc:  # noqa: BLE001 - report and continue
            return video_path, "failed", 0, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_run_one, video_path): video_path for video_path in videos
        }
        with Live(
            _build_frames_table(
                job_order,
                job_state,
                completed_jobs,
                total_jobs,
                processed_frames,
                known_total_frames,
                0.0,
            ),
            refresh_per_second=4,
        ) as live:
            pending = set(futures)
            while pending:
                done, pending = wait(
                    pending, timeout=0.2, return_when=FIRST_COMPLETED
                )
                for future in done:
                    video_path, status, frame_count, warning, error = future.result()
                    with state_lock:
                        state = job_state[video_path]
                        if status == "extracted":
                            extracted += 1
                            state["status"] = "done"
                            state["frames"] = frame_count or int(state["frames"])
                            state["done_frames"] = state["frames"]
                            state["progress"] = _progress_bar(100)
                        elif status == "skipped":
                            skipped += 1
                            state["status"] = "skipped"
                            state["progress"] = _progress_bar(100)
                        else:
                            failed += 1
                            state["status"] = "failed"
                            state["progress"] = _progress_bar(100)
                            failed_messages.append(f"⛔ {video_path}: {error}")
                        if warning:
                            warning_messages.append(f"⚠️  {warning}")
                        completed_jobs += 1

                # Refresh every poll tick (like xif) so mid-job stamp progress shows.
                elapsed = max(time.monotonic() - start_time, 1e-9)
                with state_lock:
                    live.update(
                        _build_frames_table(
                            job_order,
                            job_state,
                            completed_jobs,
                            total_jobs,
                            processed_frames,
                            known_total_frames,
                            processed_frames / elapsed,
                        )
                    )

    for message in warning_messages:
        typer.echo(message, err=True)
    for message in failed_messages:
        typer.echo(message, err=True)

    return {
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
        "videos": len(videos),
    }


def frames(
    head_directory: Path | None = typer.Argument(
        None,
        help="Root directory to scan recursively for videos",
    ),
    fps: float = typer.Option(
        1.0,
        "--fps",
        min=0.001,
        help="Frames per second to extract with ffmpeg",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        min=1,
        help="Number of worker threads for per-video extraction",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip/--no-skip",
        help=(
            "Skip videos whose frames.csv completion manifest already exists "
            "in {stem}.{fps}fps.frames"
        ),
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Delete each matching *.frames directory before extracting",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Same as --clean (delete existing frames dir, then re-extract)",
    ),
    extensions: list[str] = typer.Option(
        None,
        "--ext",
        help="Only process videos with these extensions; repeat for multiple",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help=(
            "Root for mirrored frame caches. Default: if HEAD ends with "
            "raw/sdcards → ../../interim/frames; else <parent>/frames"
        ),
    ),
    card_store: bool = typer.Option(
        False,
        "--card-store",
        help="Resolve scan root from config card_store",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        help="Path to config file used with --card-store",
    ),
    tag: list[str] = typer.Option(
        None,
        "--tag",
        help="Identity EXIF override as KEY=VALUE; repeatable",
    ),
    tags_json: str | None = typer.Option(
        None,
        "--tags-json",
        help='JSON object of identity EXIF overrides, e.g. \'{"Make":"DJI"}\'',
    ),
) -> None:
    """Extract video frames into interim caches and stamp EXIF for ``sdcard xif``."""
    config_path = resolve_config_path(config_path)

    if card_store and head_directory is not None:
        raise typer.BadParameter(
            "Cannot pass HEAD_DIRECTORY with --card-store",
            param_hint="head_directory",
        )

    if head_directory is None:
        if card_store:
            config = Config(config_path)
            head_directory = config.get_path("card_store")
            typer.echo(f"🧭 Using card_store from config: {head_directory}")
        else:
            raise typer.BadParameter(
                "Missing HEAD_DIRECTORY. Provide a path or use --card-store."
            )

    if not head_directory.exists():
        raise typer.BadParameter(f"Path does not exist: {head_directory}")
    if not head_directory.is_dir():
        raise typer.BadParameter(f"Path is not a directory: {head_directory}")

    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        raise typer.BadParameter(
            "ffmpeg is required but was not found in PATH.",
            param_hint="ffmpeg",
        )

    exiftool_executable = _resolve_exiftool_executable(config_path)
    if exiftool_executable is None:
        raise typer.BadParameter(
            "exiftool is required but was not found in PATH or in {CATALOG_DIR}/bin.",
            param_hint="exiftool",
        )

    resolved_output = (
        output_dir.expanduser()
        if output_dir is not None
        else _default_frames_output_root(head_directory)
    )
    typer.echo(f"📁 Frame output root: {resolved_output}")

    if extensions:
        allowed_suffixes = _normalize_extensions(extensions)
    else:
        allowed_suffixes = set(VIDEO_SUFFIXES)

    identity_overrides = _merge_identity_tags({}, tag, tags_json)
    do_clean = clean or force
    if do_clean and not skip_existing:
        typer.echo("🧹 --clean/--force will replace existing frames dirs")

    summary = extract_frames_tree(
        head_directory,
        resolved_output,
        fps,
        skip_existing=skip_existing and not do_clean,
        clean=do_clean,
        allowed_suffixes=allowed_suffixes,
        identity_overrides=identity_overrides,
        ffmpeg_executable=ffmpeg_executable,
        exiftool_executable=exiftool_executable,
        workers=workers,
    )
    typer.echo(
        "🎞️  Frame extraction complete: "
        f"extracted {summary['extracted']}, skipped {summary['skipped']}, "
        f"failed {summary['failed']} "
        f"(videos scanned {summary['videos']})"
    )
