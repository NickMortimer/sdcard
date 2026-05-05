from datetime import datetime
from pathlib import Path
import logging
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import threading
import typer
import uuid
import yaml
import psutil
from sdcard.config import DEFAULT_IMPORT_TEMPLATE
from sdcard.utils.import_conflicts import _collect_destination_conflicts
from sdcard.utils.import_metadata import _resolve_destination_path_template, refresh_import_yml_after_clean


def _safe_echo(message: str) -> None:
    """Echo text to the console with an ASCII-safe fallback.

    Some Windows terminals/debug consoles default to legacy encodings (e.g. cp1252)
    and raise UnicodeEncodeError for emoji characters.
    """

    try:
        typer.echo(message)
    except UnicodeEncodeError:
        sanitized = message.replace("💾 ", "").encode("ascii", "replace").decode("ascii")
        typer.echo(sanitized)


def _compress_bytes_zstd(raw_bytes: bytes) -> bytes:
    try:
        import compression.zstd as std_zstd
        return std_zstd.compress(raw_bytes)
    except ImportError:
        pass

    import zstandard
    return zstandard.ZstdCompressor(level=3).compress(raw_bytes)


def _write_file_times_manifest(root_path: Path, output_path: Path) -> None:
    lines: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            file_path = Path(dirpath) / filename
            stat_result = file_path.stat()
            timestamp = datetime.fromtimestamp(stat_result.st_mtime).isoformat(sep=" ", timespec="microseconds")
            lines.append(f"{file_path} {timestamp}")

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    compressed = _compress_bytes_zstd(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(output_path.parent)) as temp_file:
        temp_file.write(compressed)
        temp_name = temp_file.name

    Path(temp_name).replace(output_path)


def _resolve_transfer_tool_path(config) -> Path | str:
    if platform.system() == "Windows":
        bundled = config.catalog_dir / "bin" / "rclone.exe"
        if bundled.exists():
            return bundled
        resolved = shutil.which("rclone") or shutil.which("rclone.exe")
        if resolved:
            return resolved
        checked = [
            str(bundled),
            "PATH:rclone",
            "PATH:rclone.exe",
        ]
        path_hint = os.environ.get("PATH", "")
        raise typer.Abort(
            "rclone is required on Windows but was not found for this Python process.\n"
            f"Checked: {', '.join(checked)}\n"
            f"Python PATH (prefix): {path_hint[:300]}\n"
            "Install local binaries with: sdcard getbins --config-path /path/to/config.yml"
        )

    resolved = shutil.which("rsync")
    if resolved:
        return resolved
    raise typer.Abort("rsync is required but was not found in PATH")


def _rsync_dir_arg(path: Path | str) -> str:
    normalized = str(Path(path).resolve()).replace("\\", "/")
    if not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _build_rsync_command(
    rsync_path: Path | str,
    source: Path,
    destination: Path,
    dry_run: bool = False,
    clean: bool = False,
    update: bool = False,
    ignore_errors: bool = False,
) -> list[str]:
    command = [
        str(rsync_path),
        "-a",
        "--info=progress2",
    ]
    if update:
        command.append("--update")
    else:
        command.extend(["--update", "--ignore-existing"])
    if dry_run:
        command.append("--dry-run")
    if clean:
        command.append("--remove-source-files")
    if ignore_errors:
        command.append("--ignore-errors")
    command.extend([_rsync_dir_arg(source), _rsync_dir_arg(destination)])
    return command


def _build_rclone_command(
    rclone_path: Path | str,
    source: Path,
    destination: Path,
    dry_run: bool = False,
    clean: bool = False,
    update: bool = False,
    ignore_errors: bool = False,
    rclone_transfers: int = 4,
    rclone_checkers: int = 8,
    single_stream: bool = False,
) -> list[str]:
    transfers = 1 if single_stream else max(1, int(rclone_transfers))
    checkers = 1 if single_stream else max(1, int(rclone_checkers))
    command = [
        str(rclone_path),
        "copy" if not clean else "move",
        str(Path(source).resolve()),
        str(Path(destination).resolve()),
        "--progress",
        "--transfers", str(transfers),
        "--checkers", str(checkers),
    ]

    if update:
        command.append("--update")
    else:
        command.extend(["--update", "--ignore-existing"])
    if dry_run:
        command.append("--dry-run")
    if ignore_errors:
        command.append("--ignore-errors")

    return command


def _run_transfer_command(
    command: list[str],
    dry_run: bool,
    operation_label: str,
    tool_name: str,
) -> None:
    """Run transfer command, streaming progress output to stdout."""
    logging.info("running %s", shlex.join(command))
    if dry_run:
        return

    stderr_lines: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    def _drain_stderr() -> None:
        if process.stderr is None:
            return
        for raw in process.stderr:
            stderr_lines.append(raw.decode(errors="replace").rstrip("\r\n"))
        process.stderr.close()

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # Both rsync and rclone progress can use carriage returns for in-place updates.
    # Split on both \r and \n so each update prints as a separate line.
    buf = b""
    if process.stdout is not None:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                r_pos = buf.find(b"\r")
                n_pos = buf.find(b"\n")
                if r_pos == -1 and n_pos == -1:
                    break
                if r_pos == -1:
                    sep = n_pos
                elif n_pos == -1:
                    sep = r_pos
                else:
                    sep = min(r_pos, n_pos)
                line = buf[:sep].decode(errors="replace").strip()
                buf = buf[sep + 1:]
                if line:
                    print(line, flush=True)
        process.stdout.close()

    if buf:
        line = buf.decode(errors="replace").strip()
        if line:
            print(line, flush=True)

    process.wait()
    stderr_thread.join(timeout=1.0)

    if process.returncode != 0:
        _safe_echo(f"❌ {tool_name} {operation_label} failed (exit code {process.returncode}).")
        if stderr_lines:
            typer.echo(f"--- {tool_name} stderr ---\n" + "\n".join(stderr_lines))
        raise typer.Abort(f"See above for {tool_name} error details.")


def _is_source_newer_than_destination(
    source_mod: object,
    destination_mod: object,
    tolerance_seconds: float = 1.0,
) -> bool:
    """Return True when source mtime is newer than destination by tolerance."""
    if source_mod is None or destination_mod is None:
        return False
    try:
        return (float(source_mod) - float(destination_mod)) > tolerance_seconds
    except Exception:
        return False


def _precheck_conflicts(
    source: Path,
    destination: Path,
    update: bool = False,
) -> dict[str, object]:
    if not destination.exists():
        return {
            "conflicts": [],
            "likely_partial_files": [],
            "overlapping_files": [],
            "differing_files": [],
        }

    conflict_data = _collect_destination_conflicts(source, destination)
    likely_partial_files = set(conflict_data.get("likely_partial_files", []))
    source_metadata = conflict_data.get("source_metadata", {})
    destination_metadata = conflict_data.get("destination_metadata", {})
    differing_files = list(conflict_data.get("differing_files", []))
    update_allowed_files: list[str] = []
    if update:
        conflicts = []
        for rel_path in differing_files:
            if rel_path in likely_partial_files:
                continue
            source_mod = source_metadata.get(rel_path, {}).get("mod_time")
            destination_mod = destination_metadata.get(rel_path, {}).get("mod_time")
            if _is_source_newer_than_destination(source_mod, destination_mod):
                update_allowed_files.append(rel_path)
                continue
            conflicts.append(rel_path)
        conflicts = sorted(conflicts)
        update_allowed_files = sorted(update_allowed_files)
    else:
        conflicts = sorted(
            [
                rel_path
                for rel_path in differing_files
                if rel_path not in likely_partial_files
            ]
        )
    return {
        "conflicts": conflicts,
        "update_allowed_files": update_allowed_files,
        "likely_partial_files": sorted(likely_partial_files),
        "overlapping_files": conflict_data.get("overlapping_files", []),
        "differing_files": differing_files,
    }


def _remove_empty_source_directories(source_root: Path) -> None:
    for dirpath, _dirnames, _filenames in os.walk(source_root, topdown=False):
        current = Path(dirpath)
        if current == source_root:
            continue
        try:
            current.rmdir()
        except OSError:
            continue


def _source_matches_destination(
    source_file: Path,
    destination_file: Path,
    tolerance_seconds: float = 1.0,
) -> bool:
    try:
        source_stat = source_file.stat()
        destination_stat = destination_file.stat()
    except OSError:
        return False

    if source_stat.st_size != destination_stat.st_size:
        return False

    return abs(source_stat.st_mtime - destination_stat.st_mtime) <= tolerance_seconds


def _delete_source_files_matching_destination(
    source_root: Path,
    destination_root: Path,
) -> int:
    deleted_count = 0
    for dirpath, _dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            source_file = Path(dirpath) / filename
            relative_path = source_file.relative_to(source_root)
            destination_file = destination_root / relative_path

            if not destination_file.is_file():
                continue

            if not _source_matches_destination(source_file, destination_file):
                continue

            try:
                source_file.unlink()
                deleted_count += 1
            except OSError:
                continue

    return deleted_count


def _build_transfer_candidate_manifest(source_root: Path) -> dict[str, dict[str, int | float]]:
    """Capture source files that should exist at destination after transfer.

    import.yml and trash metadata files are intentionally excluded from verification.
    """
    manifest: dict[str, dict[str, int | float]] = {}
    for dirpath, _dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            source_file = Path(dirpath) / filename
            relative_path = source_file.relative_to(source_root)
            rel_text = str(relative_path).replace("\\", "/")
            lowered = rel_text.lower()
            basename = Path(lowered).name
            if basename == "import.yml":
                continue
            if lowered.startswith(".trash-") or lowered.endswith(".trashinfo"):
                continue
            try:
                stat_result = source_file.stat()
            except OSError:
                continue
            manifest[rel_text] = {
                "size": int(stat_result.st_size),
                "mod_time": float(stat_result.st_mtime),
            }
    return manifest


def _verify_destination_contains_manifest(
    destination_root: Path,
    source_manifest: dict[str, dict[str, int | float]],
) -> dict[str, object]:
    """Return verification results for source files expected at destination."""
    missing_files: list[str] = []
    for rel_path in source_manifest:
        destination_file = destination_root / Path(rel_path)
        if not destination_file.is_file():
            missing_files.append(rel_path)
    return {
        "expected_count": len(source_manifest),
        "missing_files": sorted(missing_files),
    }


def import_cards(config, card_path, copy, clean, find, file_extension, dry_run: bool, format_card=False, allow_overwrite: bool = False, check: bool = False, update: bool = False, precheck: bool = False, ignore_errors: bool = False, refresh: bool = False, rclone_transfers: int = 4, rclone_checkers: int = 8, single_stream: bool = False, verify: bool = False):
    """Copy or move (clean) SD card contents to the configured card store."""
    import pandas as pd
    _ = file_extension
    check_failures: list[str] = []
    check_checked = 0

    if check:
        copy = False
        clean = False

    for card in card_path:
        check_checked += 1
        dry_run_log_string = "DRY_RUN - " if dry_run else ""
        importyml = Path(card) / "import.yml"
        if importyml.exists():
            try:
                importdetails = yaml.safe_load(importyml.read_text(encoding='utf-8'))
            except yaml.YAMLError as exc:
                if check:
                    check_failures.append(f"{card}: corrupt import.yml ({exc})")
                    _safe_echo(f"⛔ {card}: corrupt import.yml")
                    continue
                raise typer.Abort(f"Error possible corrupt yaml {importyml}: {exc}")
        else:
            typer.echo(f"Error {importyml} not found")
            if check:
                check_failures.append(f"{card}: missing import.yml")
            continue

        importdetails_pre_clean = dict(importdetails) if isinstance(importdetails, dict) else {}

        import_metadata_changed = False
        if not importdetails.get("import_token"):
            importdetails["import_token"] = str(uuid.uuid4())[0:8]
            import_metadata_changed = True
        card_store = Path(config.data.get('card_store'))
        configured_destination_template = (
            config.data.get('import_path_template')
            or config.data.get('destination_path')
            or DEFAULT_IMPORT_TEMPLATE
        )
        importdetails.setdefault("register_date", f"{datetime.now():%Y-%m-%d}")
        if not importdetails.get("import_date"):
            importdetails["import_date"] = f"{datetime.now():%Y-%m-%d}"
            import_metadata_changed = True

        if not dry_run and import_metadata_changed:
            importyml.write_text(yaml.safe_dump(importdetails, sort_keys=False), encoding="utf-8")

        import_context = {
            **importdetails,
            "import_date": importdetails.get("import_date"),
            "import_token": importdetails.get("import_token"),
            "card_number": importdetails.get("card_number", 0),
            "card_store": str(card_store),
            "CATALOG_DIR": str(config.catalog_dir),
            "register_date": importdetails.get("register_date"),
        }
        destination_template = (
            importdetails.get('destination_path')
            or configured_destination_template
        )
        destination_path = _resolve_destination_path_template(destination_template, import_context)
        importdetails['destination_path'] = destination_path

        destination = Path(destination_path)

        if find:
            matches = list(card_store.rglob(f"*{importdetails.get('import_token')}*"))
            if matches:
                destination = max(matches)

        source = Path(card)
        source_manifest: dict[str, dict[str, int | float]] = {}
        if verify and not dry_run and (copy or clean):
            source_manifest = _build_transfer_candidate_manifest(source)
        _safe_echo(f"💾 Reading {importdetails['import_token']} from {source} to {destination}")

        if check:
            precheck_result = _precheck_conflicts(source, destination, update=update)
            conflicts = precheck_result["conflicts"]
            update_allowed_files = precheck_result["update_allowed_files"]
            likely_partial_files = precheck_result["likely_partial_files"]

            if conflicts:
                check_failures.append(
                    f"{card}: {len(conflicts)} overwrite conflict(s) detected"
                )
                _safe_echo(
                    f"⛔ {card}: {len(conflicts)} overwrite conflict(s) detected"
                )
                for rel_path in conflicts[:10]:
                    typer.echo(f"  - {rel_path}")
                if len(conflicts) > 10:
                    typer.echo(f"  ... and {len(conflicts) - 10} more")
            else:
                _safe_echo(f"✅ {card}: no overwrite conflicts detected")
                if update and update_allowed_files:
                    _safe_echo(
                        f"ℹ️ {card}: {len(update_allowed_files)} older destination file(s) can be updated"
                    )
                if likely_partial_files:
                    _safe_echo(
                        f"ℹ️ {card}: {len(likely_partial_files)} likely partial destination file(s) ignored"
                    )
            continue

        if precheck:
            precheck_result = _precheck_conflicts(source, destination, update=update)
            conflicts = precheck_result["conflicts"]
            update_allowed_files = precheck_result["update_allowed_files"]
            likely_partial_files = precheck_result["likely_partial_files"]

            if conflicts:
                _safe_echo(
                    f"⛔ Found {len(conflicts)} overwrite conflict(s) before transfer:"
                )
                for rel_path in conflicts[:10]:
                    typer.echo(f"  - {rel_path}")
                if len(conflicts) > 10:
                    typer.echo(f"  ... and {len(conflicts) - 10} more")
                if not allow_overwrite:
                    raise typer.Abort(
                        "Precheck found destination conflicts. Use --allow-overwrite to continue anyway."
                    )
                _safe_echo("⚠️ --allow-overwrite is set; continuing despite conflicts.")

            if likely_partial_files:
                _safe_echo(
                    f"ℹ️ Ignoring {len(likely_partial_files)} likely partial destination file(s)."
                )
            if update and update_allowed_files:
                _safe_echo(
                    f"ℹ️ Allowing updates for {len(update_allowed_files)} older destination file(s)."
                )

        transfer_tool_path = _resolve_transfer_tool_path(config)
        is_windows = platform.system() == "Windows"
        transfer_tool_name = "rclone" if is_windows else "rsync"

        if copy:
            logging.info(f'{dry_run_log_string}  Copy  {source} --> {destination}')
            destination.mkdir(exist_ok=True, parents=True)
            if is_windows:
                copy_command = _build_rclone_command(
                    rclone_path=transfer_tool_path,
                    source=source,
                    destination=destination,
                    dry_run=dry_run,
                    clean=False,
                    update=update,
                    ignore_errors=ignore_errors,
                    rclone_transfers=rclone_transfers,
                    rclone_checkers=rclone_checkers,
                    single_stream=single_stream,
                )
            else:
                copy_command = _build_rsync_command(
                    rsync_path=transfer_tool_path,
                    source=source,
                    destination=destination,
                    dry_run=dry_run,
                    clean=False,
                    update=update,
                    ignore_errors=ignore_errors,
                )
            _run_transfer_command(
                copy_command,
                dry_run=dry_run,
                operation_label="copy",
                tool_name=transfer_tool_name,
            )

        if clean:
            logging.info(f'{dry_run_log_string}  Clean  {source} --> {destination}')
            destination.mkdir(exist_ok=True, parents=True)
            if is_windows:
                clean_command = _build_rclone_command(
                    rclone_path=transfer_tool_path,
                    source=source,
                    destination=destination,
                    dry_run=dry_run,
                    clean=True,
                    update=update,
                    ignore_errors=ignore_errors,
                    rclone_transfers=rclone_transfers,
                    rclone_checkers=rclone_checkers,
                    single_stream=single_stream,
                )
            else:
                clean_command = _build_rsync_command(
                    rsync_path=transfer_tool_path,
                    source=source,
                    destination=destination,
                    dry_run=dry_run,
                    clean=True,
                    update=update,
                    ignore_errors=ignore_errors,
                )
            _run_transfer_command(
                clean_command,
                dry_run=dry_run,
                operation_label="clean",
                tool_name=transfer_tool_name,
            )

            if not dry_run:
                deleted_count = _delete_source_files_matching_destination(source, destination)
                if deleted_count:
                    logging.info("Deleted %s matching source file(s)", deleted_count)
                _remove_empty_source_directories(source)

                if format_card and (psutil.disk_usage(card).used < 1 * 1024**3):
                    if platform.system() == "Windows":
                        command = f"format {card} /FS:exFAT /Q /Y"
                    else:
                        command = f"mkfs.exfat {card}"
                    command = command.replace('\\', '/')
                    logging.info(f'{dry_run_log_string}  Deleting empty drive {card}')
                    command = f"rmdir {card}"
                    command = command.replace('\\', '/')
                    logging.info(f'{dry_run_log_string}  {command}')
                    if not dry_run:
                        process = subprocess.Popen(shlex.split(command))
                        process.wait()

                if refresh:
                    refresh_import_yml_after_clean(
                        importyml,
                        importdetails_pre_clean,
                        dry_run,
                    )

        if not dry_run and (copy or clean):
            persisted = dict(importdetails)
            persisted.update({
                "card_number": importdetails.get("card_number"),
                "import_token": importdetails.get("import_token"),
                "register_date": importdetails.get("register_date"),
                "import_date": importdetails.get("import_date"),
                "destination_path": importdetails.get("destination_path", destination_path),
            })
            #importyml.write_text(yaml.safe_dump(persisted), encoding="utf-8")

            # Write to imports.csv in the same directory as import.yml
            csv_dir = config.catalog_dir 
            csv_path = csv_dir / "imports.csv"
            import_token = importdetails.get("import_token")
            # Use ISO 8601 format for date_time
            date_time = datetime.now().isoformat()
            # Directory relative to config file
            config_dir = Path(config.data.get('card_store', '.')).resolve()
            try:
                rel_dir = str(csv_dir.relative_to(config_dir))
            except Exception:
                rel_dir = str(csv_dir)
            row = {"import_token": import_token, "date_time": date_time, "directory": rel_dir}
            # Append or create CSV
            try:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])
                df.to_csv(csv_path, index=False)
            except Exception as e:
                logging.warning(f"Failed to write import log to {csv_path}: {e}")

            try:
                file_times_path = destination / "file_times.txt.zst"
                _write_file_times_manifest(destination, file_times_path)
            except Exception as e:
                logging.warning(f"Failed to write file times manifest to {destination}: {e}")

            if verify:
                verify_result = _verify_destination_contains_manifest(destination, source_manifest)
                missing_files = verify_result["missing_files"]
                expected_count = verify_result["expected_count"]
                if missing_files:
                    _safe_echo(
                        f"⛔ Verification failed for {card}: {len(missing_files)} of {expected_count} expected file(s) are missing at destination"
                    )
                    for rel_path in missing_files[:10]:
                        typer.echo(f"  - {rel_path}")
                    if len(missing_files) > 10:
                        typer.echo(f"  ... and {len(missing_files) - 10} more")
                    raise typer.Abort("Verification failed after transfer.")
                _safe_echo(
                    f"✅ Verification passed for {card}: all {expected_count} expected file(s) are present at destination"
                )

    if check:
        typer.echo(f"\nChecked {check_checked} card(s).")
        if check_failures:
            typer.echo(f"Found {len(check_failures)} issue(s):")
            for issue in check_failures:
                typer.echo(f"- {issue}")
            raise typer.Exit(code=1)
        typer.echo("No overwrite issues detected.")
