from datetime import datetime
from pathlib import Path
import logging
import os
import sys
import typer


def _list_file_metadata(path: Path | str) -> dict[str, dict[str, object]]:
    """Return a mapping of relative file path to metadata using native Python stat calls."""
    root = Path(path)
    file_metadata = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            abs_path = Path(dirpath) / name
            try:
                st = abs_path.stat()
            except OSError:
                continue
            rel_path = str(abs_path.relative_to(root))
            file_metadata[rel_path] = {
                "mod_time": st.st_mtime,
                "size": st.st_size,
            }
    return file_metadata


def _is_comparison_ignored(rel_path: str) -> bool:
    """Exclude metadata and trash paths from overwrite safety comparison."""
    normalized = rel_path.replace('\\', '/').strip()
    basename = Path(normalized).name.lower()
    if normalized.startswith('.Trash-'):
        return True
    if normalized.lower().endswith('.trashinfo'):
        return True
    if basename in {"readme.md", "import.yml", "file_times.txt.zst"}:
        return True
    return False


def _modtimes_differ(src_mod: object, dst_mod: object, tolerance_seconds: float = 1.0) -> bool:
    """Compare modtimes (float epoch seconds) with tolerance to avoid filesystem precision false positives."""
    if src_mod is None and dst_mod is None:
        return False
    if src_mod is None or dst_mod is None:
        return True
    try:
        return abs(float(src_mod) - float(dst_mod)) > tolerance_seconds
    except Exception:
        return src_mod != dst_mod


def _collect_destination_conflicts(source: Path, destination: Path) -> dict[str, object]:
    """Collect overlap and conflict metadata between source and destination trees."""
    if not destination.exists():
        return {
            "source_metadata": {},
            "destination_metadata": {},
            "overlapping_files": [],
            "differing_files": [],
            "likely_partial_files": [],
        }

    source_metadata = _list_file_metadata(source)
    destination_metadata = _list_file_metadata(destination)

    overlapping_files = sorted(
        rel_path
        for rel_path in (set(source_metadata) & set(destination_metadata))
        if not _is_comparison_ignored(rel_path)
    )

    differing_files = [
        rel_path
        for rel_path in overlapping_files
        if (
            _modtimes_differ(
                source_metadata.get(rel_path, {}).get("mod_time"),
                destination_metadata.get(rel_path, {}).get("mod_time"),
            )
            or source_metadata.get(rel_path, {}).get("size") != destination_metadata.get(rel_path, {}).get("size")
        )
    ]

    likely_partial_files = [
        rel_path
        for rel_path in differing_files
        if isinstance(source_metadata.get(rel_path, {}).get("size"), int)
        and isinstance(destination_metadata.get(rel_path, {}).get("size"), int)
        and destination_metadata.get(rel_path, {}).get("size") < source_metadata.get(rel_path, {}).get("size")
    ]

    return {
        "source_metadata": source_metadata,
        "destination_metadata": destination_metadata,
        "overlapping_files": overlapping_files,
        "differing_files": differing_files,
        "likely_partial_files": likely_partial_files,
    }


def _write_conflict_report(
    report_dir: Path,
    card: Path,
    destination: Path,
    import_token: str,
    conflict_data: dict[str, object],
) -> Path:
    """Write a detailed conflict report and return its path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_card = str(card).strip("/").replace("/", "_") or "card"
    report_path = report_dir / f"{timestamp}_{import_token}_{safe_card}.txt"

    source_metadata = conflict_data.get("source_metadata", {})
    destination_metadata = conflict_data.get("destination_metadata", {})
    differing_files = conflict_data.get("differing_files", [])
    likely_partial_files = set(conflict_data.get("likely_partial_files", []))

    lines = [
        "SDCard import conflict report",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"card: {card}",
        f"destination: {destination}",
        f"import_token: {import_token}",
        f"conflict_count: {len(differing_files)}",
        "",
        "files:",
    ]

    for rel_path in differing_files:
        src = source_metadata.get(rel_path, {})
        dst = destination_metadata.get(rel_path, {})
        partial_tag = " [likely_partial]" if rel_path in likely_partial_files else ""
        lines.append(
            f"- {rel_path}{partial_tag}\n"
            f"  source:      mod={src.get('mod_time')} size={src.get('size')}\n"
            f"  destination: mod={dst.get('mod_time')} size={dst.get('size')}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _prompt_for_new_token_if_destination_changed(
    source: Path,
    destination: Path,
    importdetails: dict,
    allow_overwrite: bool = False,
    prompt_overwrite: bool = True,
) -> bool:
    """Abort import and report differences when an existing destination has changed files."""
    try:
        conflict_data = _collect_destination_conflicts(source, destination)
    except Exception as exc:
        logging.warning(f"Unable to compare source and destination: {exc}")
        return False

    source_metadata = conflict_data["source_metadata"]
    destination_metadata = conflict_data["destination_metadata"]
    overlapping_files = conflict_data["overlapping_files"]
    differing_files = conflict_data["differing_files"]
    likely_partial_files = conflict_data["likely_partial_files"]

    if not overlapping_files:
        return False
    if not differing_files:
        return False

    token = importdetails.get("import_token", "unknown")
    typer.echo("\n⛔ Import stopped to prevent overwriting existing data.")
    typer.echo(f"Destination: {destination}")
    typer.echo(f"Import token: {token}")
    typer.echo(
        f"Detected {len(differing_files)} changed files out of {len(overlapping_files)} overlapping files "
        f"(modification time and/or size)."
    )

    if likely_partial_files:
        typer.echo(
            f"Likely leftovers from a failed previous copy: {len(likely_partial_files)} files "
            f"have smaller destination size than source."
        )
        typer.echo("\nLikely partial files (first 10):")
        for rel_path in likely_partial_files[:10]:
            typer.echo(
                f"- {rel_path}\n"
                f"  source size:      {source_metadata.get(rel_path, {}).get('size')}\n"
                f"  destination size: {destination_metadata.get(rel_path, {}).get('size')}"
            )
        if len(likely_partial_files) > 10:
            typer.echo(f"... and {len(likely_partial_files) - 10} more likely partial files.")

    typer.echo("\nChanged files (first 10):")
    for rel_path in differing_files[:10]:
        typer.echo(
            f"- {rel_path}\n"
            f"  source:      mod={source_metadata.get(rel_path, {}).get('mod_time')} size={source_metadata.get(rel_path, {}).get('size')}\n"
            f"  destination: mod={destination_metadata.get(rel_path, {}).get('mod_time')} size={destination_metadata.get(rel_path, {}).get('size')}"
        )
    if len(differing_files) > 10:
        typer.echo(f"... and {len(differing_files) - 10} more changed files.")

    if allow_overwrite:
        typer.echo("\n⚠️ --allow-overwrite is set; continuing with overwrite enabled.")
        return True

    stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if prompt_overwrite and stdin_is_tty:
        overwrite_confirmed = typer.confirm(
            "\nOverwrite changed files in destination? This is unsafe and may replace existing data.",
            default=False,
        )
        if overwrite_confirmed:
            typer.echo("⚠️ User confirmed overwrite for this import.")
            return True

    raise typer.Abort(
        "Destination already contains different data. Create a new import token before retrying."
    )
