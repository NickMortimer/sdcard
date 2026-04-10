import json
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table
from rich.text import Text

from sdcard.utils.cli_xif import IMAGE_SUFFIXES, _normalize_extensions


THUMBNAIL_SOURCE_TAGS = (
    ("ThumbnailOffset", "ThumbnailLength"),
    ("PreviewImageStart", "PreviewImageLength"),
    ("JpgFromRawStart", "JpgFromRawLength"),
    ("OtherImageStart", "OtherImageLength"),
    ("MPImageStart", "MPImageLength"),
)

EXCLUDED_METADATA_TAGS = {
    "SourceFile",
    "Directory",
    "FileName",
    "FilePermissions",
    "FileModifyDate",
    "FileAccessDate",
    "FileInodeChangeDate",
    "FileType",
    "FileTypeExtension",
    "MIMEType",
    "ExifToolVersion",
    "Warning",
    "Error",
    "ThumbnailImage",
    "PreviewImage",
    "JpgFromRaw",
    "OtherImage",
}

DEFAULT_OUTPUT_DIR_NAME = "thumbnails"


def _resolve_output_target(output_target: str) -> tuple[str | None, Path | None]:
    """Return sibling-name mode or mirrored-root mode for thumbnail output."""
    text = output_target.strip()
    if not text:
        raise typer.BadParameter("--output-dir-name must not be empty")

    candidate = Path(text).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return None, candidate
    return text, None


def _is_under_directory(path: Path, candidate_parent: Path) -> bool:
    """Return whether a path is equal to or inside a candidate parent directory."""
    try:
        path.relative_to(candidate_parent)
    except ValueError:
        return False
    return True


def _decompress_json_zstd(payload: bytes) -> dict[str, dict[str, object]]:
    """Return a JSON payload previously compressed with zstd."""
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


def _status_style(status: str) -> str:
    """Return Rich style for a status value."""
    lowered = status.lower()
    if lowered == "queued":
        return "grey62"
    if lowered == "running":
        return "orange1"
    if lowered == "done":
        return "green"
    if lowered == "partial":
        return "yellow"
    if lowered in {"failed", "error"}:
        return "red"
    return "white"


def _short_path(path_text: str, max_len: int = 48) -> str:
    """Shorten long paths for fixed-width table output."""
    if len(path_text) <= max_len:
        return path_text
    return "..." + path_text[-(max_len - 3):]


def _progress_bar(percent: int, width: int = 14) -> str:
    """Render a simple textual progress bar."""
    bounded = max(0, min(100, percent))
    filled = round((bounded / 100) * width)
    return f"{'#' * filled}{'-' * (width - filled)} {bounded:>3}%"


def _build_thumbnail_table(
    job_order: list[Path],
    job_state: dict[Path, dict[str, object]],
    processed_files: int,
    total_files: int,
    files_per_second: float,
) -> Table:
    """Build live status table for thumbnail directory jobs."""
    title = "Thumbnail Extraction"
    if total_files > 0:
        total_percent = int((processed_files / total_files) * 100)
        title = (
            f"Thumbnail Extraction files [{processed_files}/{total_files}] "
            f"{_progress_bar(total_percent, width=20)}"
            f"  {files_per_second:.1f} files/s"
        )

    table = Table(title=title)
    table.add_column("Directory")
    table.add_column("Files", no_wrap=True)
    table.add_column("Written", no_wrap=True)
    table.add_column("Skipped", no_wrap=True)
    table.add_column("Missing", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    for directory in job_order:
        state = job_state[directory]
        status = str(state["status"])
        style = _status_style(status)
        table.add_row(
            Text(_short_path(str(directory)), style=style),
            str(state["files"]),
            str(state["written"]),
            str(state["skipped"]),
            str(state["missing"]),
            str(state["progress"]),
            Text(status, style=style),
        )
    return table


def _parse_positive_int(value: object) -> int | None:
    """Parse a positive integer from exiftool JSON values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        integer = int(value)
        return integer if integer >= 0 else None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        integer = int(text)
    except ValueError:
        integer_text = text.split()[0]
        try:
            integer = int(integer_text)
        except ValueError:
            return None
    return integer if integer >= 0 else None


def _load_extracted_metadata(path: Path) -> dict[str, dict[str, object]]:
    """Load extracted EXIF metadata from a per-directory zstd JSON file."""
    payload = _decompress_json_zstd(path.read_bytes())
    return {
        str(name): value
        for name, value in payload.items()
        if isinstance(name, str) and isinstance(value, dict)
    }


def _thumbnail_location(
    metadata: dict[str, object],
) -> tuple[str, int, int] | None:
    """Return the first usable embedded preview byte range."""
    for start_tag, length_tag in THUMBNAIL_SOURCE_TAGS:
        start = _parse_positive_int(metadata.get(start_tag))
        length = _parse_positive_int(metadata.get(length_tag))
        if start is None or length is None or length <= 0:
            continue
        return start_tag, start, length
    return None


def _guess_thumbnail_suffix(
    payload: bytes,
    metadata: dict[str, object],
    source_tag: str,
) -> str:
    """Infer a thumbnail file extension from the embedded preview bytes."""
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"

    mime_value = metadata.get("MIMEType") or metadata.get("ThumbnailImageType")
    if isinstance(mime_value, str):
        lowered = mime_value.lower()
        if "jpeg" in lowered or "jpg" in lowered:
            return ".jpg"
        if "png" in lowered:
            return ".png"
        if "tif" in lowered or "tiff" in lowered:
            return ".tif"
        if "webp" in lowered:
            return ".webp"

    if source_tag in {"PreviewImageStart", "JpgFromRawStart", "MPImageStart"}:
        return ".jpg"
    return ".bin"


def _thumbnail_output_path(
    head_directory: Path,
    source_path: Path,
    sibling_output_dir_name: str | None,
    mirrored_output_root: Path | None,
    suffix: str,
) -> Path:
    """Return the thumbnail output path for sibling-name or mirrored-root mode."""
    filename = f"{source_path.stem}.thumbnail{suffix}"
    if mirrored_output_root is not None:
        relative_parent = source_path.parent.relative_to(head_directory)
        return mirrored_output_root / relative_parent / filename
    if sibling_output_dir_name is None:
        raise RuntimeError("Missing thumbnail output target")
    return source_path.parent / sibling_output_dir_name / filename


def _existing_thumbnail_path(
    head_directory: Path,
    source_path: Path,
    sibling_output_dir_name: str | None,
    mirrored_output_root: Path | None,
) -> Path | None:
    """Return an existing thumbnail path if the source image was processed before."""
    if mirrored_output_root is not None:
        output_directory = mirrored_output_root / source_path.parent.relative_to(
            head_directory
        )
    else:
        if sibling_output_dir_name is None:
            return None
        output_directory = source_path.parent / sibling_output_dir_name
    if not output_directory.exists():
        return None

    matches = sorted(output_directory.glob(f"{source_path.stem}.thumbnail.*"))
    if matches:
        return matches[0]
    return None


def _extract_embedded_thumbnail(source_path: Path, start: int, length: int) -> bytes:
    """Read embedded preview bytes using offsets from previously extracted metadata."""
    with source_path.open("rb") as handle:
        handle.seek(start)
        payload = handle.read(length)

    if len(payload) != length:
        raise RuntimeError(
            f"Expected {length} thumbnail bytes at offset {start}, "
            f"read {len(payload)}"
        )
    return payload


def _metadata_tag_is_copyable(tag_name: str, value: object) -> bool:
    """Return whether an extracted tag should be forwarded to exiftool JSON import."""
    if tag_name in EXCLUDED_METADATA_TAGS:
        return False
    if tag_name.endswith(("Offset", "Length", "Start")):
        return False
    if tag_name.startswith("File"):
        return False
    if isinstance(value, str) and value.startswith("(Binary data"):
        return False
    return True


def _copy_thumbnail_metadata(
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    """Write extracted metadata into the generated thumbnail file."""
    writable_tags = {
        tag_name: value
        for tag_name, value in metadata.items()
        if _metadata_tag_is_copyable(tag_name, value)
    }
    if not writable_tags:
        return

    payload = [{"SourceFile": str(output_path), **writable_tags}]

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
            "exiftool",
            "-overwrite_original",
            "-m",
            f"-json={temp_path}",
            str(output_path),
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
                f"exiftool failed for command '{rendered}' "
                f"(exit code {result.returncode}): {result.stderr.strip()}"
            )
    finally:
        temp_path.unlink(missing_ok=True)


def _process_directory_thumbnails(
    head_directory: Path,
    directory: Path,
    file_metadata: dict[Path, dict[str, object]],
    sibling_output_dir_name: str | None,
    mirrored_output_root: Path | None,
    copy_meta: bool,
    on_file_complete=None,
) -> tuple[Path, dict[str, int], str | None]:
    """Create thumbnails for all eligible files in one directory."""
    summary = {
        "files": len(file_metadata),
        "written": 0,
        "skipped": 0,
        "missing": 0,
        "failed": 0,
    }
    last_error = None

    for source_path, metadata in sorted(file_metadata.items()):
        try:
            existing_output = _existing_thumbnail_path(
                head_directory,
                source_path,
                sibling_output_dir_name,
                mirrored_output_root,
            )
            if existing_output is not None:
                summary["skipped"] += 1
                continue

            location = _thumbnail_location(metadata)
            if location is None:
                summary["missing"] += 1
                continue

            source_tag, start, length = location
            payload = _extract_embedded_thumbnail(source_path, start, length)
            suffix = _guess_thumbnail_suffix(payload, metadata, source_tag)
            output_path = _thumbnail_output_path(
                head_directory,
                source_path,
                sibling_output_dir_name,
                mirrored_output_root,
                suffix,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(payload)

            if copy_meta:
                _copy_thumbnail_metadata(output_path, metadata)

            summary["written"] += 1
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            summary["failed"] += 1
            last_error = str(exc)
        finally:
            if on_file_complete is not None:
                on_file_complete(directory, summary)

    return directory, summary, last_error


def extract_thumbnail_tree(
    head_directory: Path,
    metadata_name: str,
    output_target: str,
    workers: int,
    copy_meta: bool,
    allowed_suffixes: set[str],
) -> dict[str, int]:
    """Walk a directory tree and create thumbnails from extracted EXIF metadata."""
    sibling_output_dir_name, mirrored_output_root = _resolve_output_target(
        output_target
    )
    written = 0
    skipped = 0
    empty = 0
    missing_metadata = 0
    missing_thumbnail = 0
    invalid_metadata = 0
    failed = 0
    processed_files = 0

    directories = [head_directory]
    directories.extend(
        sorted(path for path in head_directory.rglob("*") if path.is_dir())
    )

    jobs: list[tuple[Path, dict[Path, dict[str, object]]]] = []

    for directory in directories:
        if (
            sibling_output_dir_name is not None
            and directory.name == sibling_output_dir_name
        ):
            continue
        if mirrored_output_root is not None and _is_under_directory(
            directory,
            mirrored_output_root,
        ):
            continue

        image_files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
        if not image_files:
            empty += 1
            continue

        metadata_path = directory / metadata_name
        if not metadata_path.exists():
            missing_metadata += 1
            continue

        try:
            extracted_metadata = _load_extracted_metadata(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid_metadata += 1
            typer.echo(f"⛔ Failed to read extracted metadata for {directory}: {exc}")
            continue

        file_metadata = {
            image_path: metadata
            for image_path in image_files
            if (metadata := extracted_metadata.get(image_path.name)) is not None
        }
        if not file_metadata:
            missing_thumbnail += len(image_files)
            continue

        jobs.append((directory, file_metadata))

    if not jobs:
        return {
            "written": written,
            "skipped": skipped,
            "empty": empty,
            "missing_metadata": missing_metadata,
            "missing_thumbnail": missing_thumbnail,
            "invalid_metadata": invalid_metadata,
            "failed": failed,
        }

    job_order = [directory for directory, _ in jobs]
    state_lock = threading.Lock()
    start_time = time.monotonic()
    total_files = sum(len(file_metadata) for _, file_metadata in jobs)
    job_state: dict[Path, dict[str, object]] = {
        directory: {
            "status": "queued",
            "files": len(file_metadata),
            "written": 0,
            "skipped": 0,
            "missing": 0,
            "processed": 0,
            "progress": _progress_bar(0),
        }
        for directory, file_metadata in jobs
    }

    def _on_file_complete(directory: Path, summary: dict[str, int]) -> None:
        nonlocal processed_files
        with state_lock:
            state = job_state[directory]
            state["written"] = summary["written"]
            state["skipped"] = summary["skipped"]
            state["missing"] = summary["missing"]
            current_processed = (
                summary["written"]
                + summary["skipped"]
                + summary["missing"]
                + summary["failed"]
            )
            processed_delta = current_processed - int(state["processed"])
            state["processed"] = current_processed
            processed_files += processed_delta
            total_dir_files = int(state["files"])
            percent = int((current_processed / total_dir_files) * 100)
            state["progress"] = _progress_bar(percent)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for directory, file_metadata in jobs:
            job_state[directory]["status"] = "running"
            future = executor.submit(
                _process_directory_thumbnails,
                head_directory,
                directory,
                file_metadata,
                sibling_output_dir_name,
                mirrored_output_root,
                copy_meta,
                _on_file_complete,
            )
            futures[future] = directory

        with Live(
            _build_thumbnail_table(job_order, job_state, 0, total_files, 0.0),
            refresh_per_second=4,
        ) as live:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    directory, summary, error_message = future.result()
                    written += summary["written"]
                    skipped += summary["skipped"]
                    missing_thumbnail += summary["missing"]
                    failed += summary["failed"]

                    if summary["failed"] > 0:
                        job_state[directory]["status"] = "partial"
                        if error_message:
                            typer.echo(
                                f"⛔ Failed thumbnail extraction for {directory}: {error_message}"
                            )
                    elif summary["missing"] > 0:
                        job_state[directory]["status"] = "partial"
                    else:
                        job_state[directory]["status"] = "done"

                    job_state[directory]["progress"] = _progress_bar(100)

                elapsed = max(time.monotonic() - start_time, 1e-9)
                files_per_second = processed_files / elapsed
                live.update(
                    _build_thumbnail_table(
                        job_order,
                        job_state,
                        processed_files,
                        total_files,
                        files_per_second,
                    )
                )

    return {
        "written": written,
        "skipped": skipped,
        "empty": empty,
        "missing_metadata": missing_metadata,
        "missing_thumbnail": missing_thumbnail,
        "invalid_metadata": invalid_metadata,
        "failed": failed,
    }


def thumbnail(
    head_directory: Path = typer.Argument(
        ..., help="Root directory to scan recursively for extracted thumbnails"
    ),
    metadata_name: str = typer.Option(
        "exif.json.zst",
        "--metadata-name",
        help="Per-directory extracted EXIF metadata filename",
    ),
    output_dir_name: str = typer.Option(
        DEFAULT_OUTPUT_DIR_NAME,
        "--output-dir-name",
        help="Sibling directory name or mirrored output root for thumbnails",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        min=1,
        help="Number of worker threads for directory processing",
    ),
    copy_meta: bool = typer.Option(
        False,
        "--copy-meta",
        help="Copy extracted metadata into generated thumbnail files",
    ),
    extensions: list[str] = typer.Option(
        None,
        "--ext",
        help="Only process files with these extensions; repeat for multiple",
    ),
) -> None:
    """Create thumbnail files using previously extracted EXIF metadata only."""
    if copy_meta and shutil.which("exiftool") is None:
        raise typer.BadParameter(
            "exiftool is required for --copy-meta but was not found in PATH"
        )

    if not head_directory.exists():
        raise typer.BadParameter(f"Path does not exist: {head_directory}")
    if not head_directory.is_dir():
        raise typer.BadParameter(f"Path is not a directory: {head_directory}")

    allowed_suffixes = _normalize_extensions(extensions)
    summary = extract_thumbnail_tree(
        head_directory=head_directory,
        metadata_name=metadata_name,
        output_target=output_dir_name,
        workers=workers,
        copy_meta=copy_meta,
        allowed_suffixes=allowed_suffixes,
    )
    typer.echo(
        "🖼️ Thumbnail extraction complete: "
        f"wrote {summary['written']}, skipped {summary['skipped']}, "
        f"no-metadata-dirs {summary['missing_metadata']}, "
        f"no-thumbnail {summary['missing_thumbnail']}, "
        f"invalid-metadata {summary['invalid_metadata']}, "
        f"no-images {summary['empty']}, failed {summary['failed']}"
    )