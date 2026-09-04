"""Extract video frames into interim caches and stamp EXIF for later ``xif`` indexing."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import typer

from sdcard.config import Config
from sdcard.utils.config_path_cache import resolve_config_path
from sdcard.utils.cli_xif import _normalize_extensions, _resolve_exiftool_executable


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

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
    """Format datetime for ExifTool DateTimeOriginal/CreateDate."""
    return value.strftime(_EXIFTOOL_DATETIME_FORMAT)


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


def _output_has_frames(output_dir: Path) -> bool:
    """Return whether an output directory already contains extracted frames."""
    if not output_dir.is_dir():
        return False
    return any(output_dir.glob("frame_*.jpg"))


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
    payload_tags = {
        "SourceFile": str(frame_path),
        "DateTimeOriginal": _format_exif_datetime(frame_time),
        "CreateDate": _format_exif_datetime(frame_time),
        **identity_tags,
    }
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
) -> str:
    """
    Extract and stamp frames for one video.

    Returns one of: ``extracted``, ``skipped``, ``cleaned``, ``failed``.
    """
    output_dir = _frames_output_dir(video_path, source_root, output_root, fps)
    has_frames = _output_has_frames(output_dir)

    if clean and output_dir.exists():
        _remove_frames_directory(output_dir)
        has_frames = False
    elif has_frames and skip_existing:
        return "skipped"

    metadata, _meta_source = _resolve_video_metadata(video_path, exiftool_executable)
    discovered = _identity_tags_from_metadata(metadata)
    identity_tags = {**discovered, **identity_overrides}
    start_time, start_source = _resolve_video_start(video_path, metadata)
    if start_source == "mtime":
        typer.echo(
            f"⚠️  No video start time for {video_path.name}; using file mtime",
            err=True,
        )

    if exiftool_executable is None:
        raise RuntimeError(
            "exiftool is required to stamp frame DateTimeOriginal and identity tags"
        )

    frames = _extract_frames_with_ffmpeg(
        video_path,
        output_dir,
        fps,
        ffmpeg_executable,
    )
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")

    for index, frame_path in enumerate(frames, start=1):
        elapsed = timedelta(seconds=(index - 1) / fps)
        frame_time = start_time + elapsed
        _stamp_frame_exif(
            frame_path,
            frame_time,
            identity_tags,
            exiftool_executable,
        )
    return "extracted"


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
    for video_path in videos:
        try:
            status = _process_video(
                video_path,
                source_root,
                output_root,
                fps,
                skip_existing=skip_existing,
                clean=clean,
                identity_overrides=identity_overrides,
                ffmpeg_executable=ffmpeg_executable,
                exiftool_executable=exiftool_executable,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            typer.echo(f"⛔ {video_path}: {exc}", err=True)
            continue

        if status == "extracted":
            extracted += 1
            typer.echo(
                f"✅ {video_path.name} → "
                f"{_frames_output_dir(video_path, source_root, output_root, fps)}"
            )
        elif status == "skipped":
            skipped += 1
            typer.echo(
                f"⏭️  skip {video_path.name} "
                f"({_frames_output_dir(video_path, source_root, output_root, fps)} exists)"
            )

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
    skip_existing: bool = typer.Option(
        True,
        "--skip/--no-skip",
        help="Skip videos whose {stem}.{fps}fps.frames dir already has frames",
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
    )
    typer.echo(
        "🎞️  Frame extraction complete: "
        f"extracted {summary['extracted']}, skipped {summary['skipped']}, "
        f"failed {summary['failed']} "
        f"(videos scanned {summary['videos']})"
    )
