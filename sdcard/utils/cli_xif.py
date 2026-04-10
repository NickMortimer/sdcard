import json
import shlex
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table
from rich.text import Text


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".png",
    ".heic",
    ".heif",
    ".webp",
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".erf",
    ".kdc",
    ".mos",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".srw",
    ".x3f",
}


def _normalize_extensions(extensions: list[str] | None) -> set[str]:
    """Normalize optional extension filters into lowercase dotted suffixes."""
    if not extensions:
        return set(IMAGE_SUFFIXES)

    normalized: set[str] = set()
    for extension in extensions:
        text = extension.strip().lower()
        if not text:
            raise typer.BadParameter("Extension filters must not be empty")
        if not text.startswith("."):
            text = f".{text}"
        normalized.add(text)
    return normalized


def _compress_json_zstd(payload: dict[str, dict[str, object]]) -> bytes:
    """Serialize metadata to zstd-compressed JSON bytes."""
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

    try:
        import compression.zstd as std_zstd

        return std_zstd.compress(raw)
    except ImportError:
        pass

    try:
        import zstandard

        return zstandard.ZstdCompressor(level=3).compress(raw)
    except ImportError as exc:
        raise RuntimeError(
            "zstd compression is required but unavailable. "
            "Use Python 3.14+ or install the 'zstandard' package."
        ) from exc


def _extract_directory_exif(image_files: list[Path]) -> dict[str, dict[str, object]]:
    """Extract metadata for all images in a directory using exiftool JSON output."""
    command = ["exiftool", "-j", *[str(path) for path in image_files]]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        rendered = shlex.join(command)
        raise RuntimeError(
            f"exiftool failed for command '{rendered}' (exit code {result.returncode}): {result.stderr.strip()}"
        )

    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        return {}

    metadata_by_file: dict[str, dict[str, object]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        source_file = item.get("SourceFile")
        file_name = Path(source_file).name if isinstance(source_file, str) else "unknown"
        metadata_by_file[file_name] = item
    return metadata_by_file


def _chunk_paths(items: list[Path], batch_size: int) -> list[list[Path]]:
    """Split file paths into fixed-size chunks."""
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def _directory_images(
    directory: Path,
    allowed_suffixes: set[str] | None = None,
) -> list[Path]:
    """Return supported image files in a directory."""
    suffixes = IMAGE_SUFFIXES if allowed_suffixes is None else allowed_suffixes
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _status_style(status: str) -> str:
    """Return Rich style for a status value."""
    lowered = status.lower()
    if lowered == "queued":
        return "grey62"
    if lowered == "running":
        return "orange1"
    if lowered == "done":
        return "green"
    if lowered == "batch-failed":
        return "red"
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


def _build_xif_table(
    job_order: list[Path],
    job_state: dict[Path, dict[str, object]],
    completed_jobs: int,
    total_jobs: int,
    processed_files: int,
    total_files: int,
    files_per_second: float,
) -> Table:
    """Build live status table for EXIF directory jobs."""
    title = "EXIF Extraction"
    if total_files > 0:
        total_percent = int((processed_files / total_files) * 100)
        title = (
            f"EXIF Extraction files [{processed_files}/{total_files}] "
            f"{_progress_bar(total_percent, width=20)}"
            f"  {files_per_second:.1f} files/s"
        )

    table = Table(title=title)
    table.add_column("Directory")
    table.add_column("Images", no_wrap=True)
    table.add_column("Done", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    for directory in job_order:
        state = job_state[directory]
        status = str(state["status"])
        style = _status_style(status)
        table.add_row(
            Text(_short_path(str(directory)), style=style),
            str(state["images"]),
            str(state["done_files"]),
            str(state["progress"]),
            Text(status, style=style),
        )
    return table


def _extract_with_split(
    image_files: list[Path],
    on_files_done,
) -> tuple[dict[str, dict[str, object]], int, bool]:
    """Extract EXIF and recursively split failing batches to isolate bad files."""
    try:
        metadata = _extract_directory_exif(image_files)
        on_files_done(len(image_files), False)
        return metadata, 0, False
    except (RuntimeError, json.JSONDecodeError):
        if len(image_files) == 1:
            on_files_done(1, True)
            return {}, 1, True

    midpoint = len(image_files) // 2
    left = image_files[:midpoint]
    right = image_files[midpoint:]

    left_metadata, left_failed, left_had_failure = _extract_with_split(
        left,
        on_files_done,
    )
    right_metadata, right_failed, right_had_failure = _extract_with_split(
        right,
        on_files_done,
    )

    merged = dict(left_metadata)
    merged.update(right_metadata)
    return merged, left_failed + right_failed, True


def _extract_directory_metadata(
    directory: Path,
    output_name: str,
    image_files: list[Path],
    batch_size: int,
    on_batch_complete=None,
) -> tuple[Path, bool, str | None, int, bool]:
    """Extract and write EXIF metadata for one directory."""
    output_path = directory / output_name

    try:
        failed_files = 0
        batch_failed = False
        directory_metadata: dict[str, dict[str, object]] = {}
        for batch in _chunk_paths(image_files, batch_size):
            batch_metadata, batch_failed_count, had_failure = _extract_with_split(
                batch,
                lambda processed, had_batch_error: (
                    on_batch_complete(directory, processed, had_batch_error)
                    if on_batch_complete is not None
                    else None
                ),
            )
            directory_metadata.update(batch_metadata)
            failed_files += batch_failed_count
            if had_failure:
                batch_failed = True

        if not directory_metadata:
            return (
                directory,
                False,
                "No readable EXIF data extracted from files in directory",
                failed_files,
                batch_failed,
            )

        output_path.write_bytes(_compress_json_zstd(directory_metadata))
        return directory, True, None, failed_files, batch_failed
    except (OSError, RuntimeError) as exc:
        return directory, False, str(exc), failed_files, batch_failed


def extract_exif_tree(
    head_directory: Path,
    output_name: str,
    batch_size: int,
    workers: int,
    allowed_suffixes: set[str],
) -> dict[str, int]:
    """Walk a directory tree and write one EXIF JSON file per image directory."""
    written = 0
    skipped = 0
    empty = 0
    failed = 0
    failed_files = 0
    processed_files = 0

    directories = [head_directory]
    directories.extend(
        sorted(path for path in head_directory.rglob("*") if path.is_dir())
    )

    jobs: list[tuple[Path, list[Path]]] = []

    for directory in directories:
        output_path = directory / output_name
        if output_path.exists():
            skipped += 1
            continue

        image_files = _directory_images(directory, allowed_suffixes)
        if not image_files:
            empty += 1
            continue

        jobs.append((directory, image_files))

    if not jobs:
        return {
            "written": written,
            "skipped": skipped,
            "empty": empty,
            "failed": failed,
            "failed_files": failed_files,
        }

    job_order = [directory for directory, _ in jobs]
    state_lock = threading.Lock()
    start_time = time.monotonic()
    job_state: dict[Path, dict[str, object]] = {
        directory: {
            "status": "queued",
            "images": len(image_files),
            "done_files": 0,
            "progress": _progress_bar(0),
        }
        for directory, image_files in jobs
    }
    total_files = sum(len(image_files) for _, image_files in jobs)

    completed_jobs = 0
    total_jobs = len(jobs)

    def _on_batch_complete(
        directory: Path,
        processed_count: int,
        failed_batch: bool,
    ) -> None:
        nonlocal processed_files
        with state_lock:
            state = job_state[directory]
            state["done_files"] = int(state["done_files"]) + processed_count
            processed_files += processed_count
            total_dir_files = int(state["images"])
            if total_dir_files > 0:
                percent = int((int(state["done_files"]) / total_dir_files) * 100)
            else:
                percent = 100
            state["progress"] = _progress_bar(percent)
            if failed_batch:
                state["status"] = "batch-failed"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for directory, image_files in jobs:
            job_state[directory]["status"] = "running"
            future = executor.submit(
                _extract_directory_metadata,
                directory,
                output_name,
                image_files,
                batch_size,
                _on_batch_complete,
            )
            futures[future] = directory

        with Live(
            _build_xif_table(
                job_order,
                job_state,
                completed_jobs,
                total_jobs,
                processed_files,
                total_files,
                0.0,
            ),
            refresh_per_second=4,
        ) as live:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    directory, succeeded, error_message, directory_failed_files, had_batch_failure = future.result()
                    failed_files += directory_failed_files
                    if succeeded:
                        written += 1
                        if had_batch_failure:
                            job_state[directory]["status"] = "batch-failed"
                        elif directory_failed_files > 0:
                            job_state[directory]["status"] = "partial"
                        else:
                            job_state[directory]["status"] = "done"
                        job_state[directory]["progress"] = _progress_bar(100)
                    else:
                        failed += 1
                        job_state[directory]["status"] = "failed"
                        job_state[directory]["progress"] = _progress_bar(100)
                        typer.echo(
                            f"⛔ Failed EXIF extraction for {directory}: {error_message}"
                        )

                    completed_jobs += 1

                elapsed = max(time.monotonic() - start_time, 1e-9)
                files_per_second = processed_files / elapsed
                live.update(
                    _build_xif_table(
                        job_order,
                        job_state,
                        completed_jobs,
                        total_jobs,
                        processed_files,
                        total_files,
                        files_per_second,
                    )
                )

    return {
        "written": written,
        "skipped": skipped,
        "empty": empty,
        "failed": failed,
        "failed_files": failed_files,
    }


def xif(
    head_directory: Path = typer.Argument(
        ..., help="Root directory to scan recursively for image EXIF data"
    ),
    output_name: str = typer.Option(
        "exif.json.zst", "--output-name", help="Per-directory zstd JSON filename"
    ),
    batch_size: int = typer.Option(
        200,
        "--batch-size",
        min=1,
        help="Number of images sent to exiftool per call",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        min=1,
        help="Number of worker threads for directory processing",
    ),
    extensions: list[str] = typer.Option(
        None,
        "--ext",
        help="Only process files with these extensions; repeat for multiple",
    ),
) -> None:
    """Extract EXIF metadata into a JSON file in each image directory."""
    if shutil.which("exiftool") is None:
        raise typer.BadParameter("exiftool is required but was not found in PATH")

    if not head_directory.exists():
        raise typer.BadParameter(f"Path does not exist: {head_directory}")
    if not head_directory.is_dir():
        raise typer.BadParameter(f"Path is not a directory: {head_directory}")

    allowed_suffixes = _normalize_extensions(extensions)
    summary = extract_exif_tree(
        head_directory,
        output_name,
        batch_size,
        workers,
        allowed_suffixes,
    )
    typer.echo(
        "📷 EXIF extraction complete: "
        f"wrote {summary['written']}, skipped {summary['skipped']}, "
        f"no-images {summary['empty']}, failed {summary['failed']}, "
        f"failed-files {summary['failed_files']}"
    )