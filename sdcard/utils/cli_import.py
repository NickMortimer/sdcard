import typer
import subprocess
import sys
import os
import re
import time
import platform
from rich.live import Live
from rich.table import Table
from rich.text import Text
from pathlib import Path
from sdcard.config import Config
from sdcard.utils.cards_discovery import list_sdcards, filter_empty_cards
from sdcard.utils.config_path_cache import resolve_config_path
from sdcard.utils.cli_defaults import (
    DEFAULT_ALLOW_OVERWRITE,
    DEFAULT_CARD_SIZE,
    DEFAULT_CHECK,
    DEFAULT_CLEAN,
    DEFAULT_COPY,
    DEFAULT_DRY_RUN,
    DEFAULT_FILE_EXTENSION,
    DEFAULT_FIND,
    DEFAULT_FORMAT_CARD,
    DEFAULT_FORMAT_TYPE,
    DEFAULT_IGNORE_ERRORS,
    DEFAULT_PRECHECK,
    DEFAULT_REFRESH,
    DEFAULT_RCLONE_CHECKERS,
    DEFAULT_RCLONE_TRANSFERS,
    DEFAULT_SINGLE_STREAM,
    DEFAULT_UPDATE,
    DEFAULT_VERIFY,
)
from sdcard.utils.import_transfer import import_cards


def _status_style(status: str) -> str:
    lowered = status.lower()
    if lowered in {"queued"}:
        return "grey62"
    if lowered in {"done"}:
        return "green"
    if lowered in {"reading", "copying", "running", "working"}:
        return "orange1"
    if lowered in {"error", "failed"}:
        return "red"
    return "white"


def _short_path(path_text: str, max_len: int = 40) -> str:
    if len(path_text) <= max_len:
        return path_text
    return "..." + path_text[-(max_len - 3):]


def _progress_bar(percent: int, width: int = 18) -> str:
    bounded = max(0, min(100, percent))
    filled = round((bounded / 100) * width)
    return f"{'#' * filled}{'-' * (width - filled)} {bounded:>3}%"


def _build_single_import_table(state: dict[str, str | int]) -> Table:
    table = Table(title="Import Status")
    table.add_column("Card")
    table.add_column("Destination")
    table.add_column("Speed", no_wrap=True)
    table.add_column("Progress", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    status = str(state["status"])
    style = _status_style(status)
    table.add_row(
        Text(str(state["card"]), style=style),
        str(state["destination"]),
        str(state["speed"]),
        str(state["progress_bar"]),
        Text(status, style=style),
    )
    return table


def _run_import_with_live_table(
    card_path: list[str],
    config_path: Path | None,
    copy: bool,
    clean: bool,
    refresh: bool,
    find: bool,
    allow_overwrite: bool,
    check: bool,
    update: bool,
    precheck: bool,
    card_size: int,
    format_type: str,
    dry_run: bool,
    file_extension: str,
    format_card: bool,
    ignore_errors: bool,
    rclone_transfers: int,
    rclone_checkers: int,
    single_stream: bool,
    verify: bool,
) -> None:
    command = [sys.executable, "-m", "sdcard.main", "import", *card_path]
    if config_path is not None:
        command.extend(["--config-path", str(config_path)])
    if clean:
        command.append("--clean")
    if refresh:
        command.append("--refresh")
    if copy != DEFAULT_COPY:
        command.extend(["--copy", str(copy)])
    if find != DEFAULT_FIND:
        command.extend(["--find", str(find)])
    if allow_overwrite:
        command.append("--allow-overwrite")
    if check:
        command.append("--check")
    if update:
        command.append("--update")
    if precheck:
        command.append("--precheck")
    if card_size != DEFAULT_CARD_SIZE:
        command.extend(["--card-size", str(card_size)])
    if format_type != DEFAULT_FORMAT_TYPE:
        command.extend(["--format-type", format_type])
    if dry_run:
        command.append("--dry-run")
    if file_extension != DEFAULT_FILE_EXTENSION:
        command.extend(["--file-extension", file_extension])
    if format_card:
        command.append("--format-card")
    if ignore_errors:
        command.append("--ignore-errors")
    if single_stream:
        command.append("--single-stream")
    else:
        if rclone_transfers != DEFAULT_RCLONE_TRANSFERS:
            command.extend(["--rclone-transfers", str(rclone_transfers)])
        if rclone_checkers != DEFAULT_RCLONE_CHECKERS:
            command.extend(["--rclone-checkers", str(rclone_checkers)])
    if verify:
        command.append("--verify")

    env = os.environ.copy()
    env["SDCARD_TABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    state: dict[str, str | int] = {
        "card": _short_path(card_path[0], 24),
        "destination": "-",
        "speed": "-",
        "progress_bar": _progress_bar(0),
        "status": "queued",
    }

    reading_pattern = re.compile(r"^💾 Reading (\S+) from (.+) to (.+)$")
    progress_pattern = re.compile(r"(\d{1,3})%\s+([\d.,]+\s*\S+/s)")

    with Live(_build_single_import_table(state), refresh_per_second=4) as live:
        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip("\r\n")
                reading_match = reading_pattern.match(line)
                if reading_match:
                    _token, card, destination = reading_match.groups()
                    state["card"] = _short_path(card, 24)
                    state["destination"] = _short_path(destination, 40)
                    state["speed"] = "-"
                    state["progress_bar"] = _progress_bar(0)
                    state["status"] = "reading"

                progress_match = progress_pattern.search(line)
                if progress_match:
                    percent = int(progress_match.group(1))
                    speed = progress_match.group(2).strip()
                    state["speed"] = speed
                    state["progress_bar"] = _progress_bar(percent)
                    state["status"] = "copying"

                if "error" in line.lower() or "failed" in line.lower():
                    state["speed"] = "-"
                    state["status"] = "error"

                live.update(_build_single_import_table(state))

        process.wait()
        if process.returncode == 0:
            state["status"] = "done"
            state["speed"] = "-"
            state["progress_bar"] = _progress_bar(100)
        elif str(state["status"]) != "error":
            state["status"] = "failed"
            state["speed"] = "-"
        live.update(_build_single_import_table(state))
        time.sleep(0.2)

    if process.returncode != 0:
        raise typer.Exit(code=process.returncode)

def import_command(
    card_path: list[str] = typer.Argument(None, help="One or more SD card mount points"),
    config_path: Path = typer.Option(None, help="Path to config file"),
    copy: bool = typer.Option(DEFAULT_COPY, help="Copy source (default)", show_default=True),
    clean: bool = typer.Option(DEFAULT_CLEAN, help="Move source and delete after copy"),
    refresh: bool = typer.Option(DEFAULT_REFRESH, "--refresh", help="After --clean, write a new import.yml to the card"),
    find: bool = typer.Option(DEFAULT_FIND, help="Reuse an existing destination that matches the import token"),
    allow_overwrite: bool = typer.Option(DEFAULT_ALLOW_OVERWRITE, "--allow-overwrite", help="Allow overwriting changed files at destination (unsafe)"),
    check: bool = typer.Option(DEFAULT_CHECK, "--check", help="Check all cards for conflicts only; do not copy or move"),
    update: bool = typer.Option(DEFAULT_UPDATE, "--update", help="Allow newer source files to update older destination files"),
    precheck: bool = typer.Option(DEFAULT_PRECHECK, "--precheck", help="Run destination conflict prechecks before copy/move"),
    card_size: int = typer.Option(DEFAULT_CARD_SIZE, help="Maximum card size to auto-detect"),
    format_type: str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
    dry_run: bool = typer.Option(DEFAULT_DRY_RUN, help="Show actions without changing files"),
    file_extension: str = typer.Option(DEFAULT_FILE_EXTENSION, help="extension to catalog"),
    format_card: bool = typer.Option(DEFAULT_FORMAT_CARD, help="Format drive after import and move"),
    ignore_errors: bool = typer.Option(DEFAULT_IGNORE_ERRORS, "--ignore-errors", help="Pass --ignore-errors to rsync and continue on file errors"),
    rclone_transfers: int = typer.Option(DEFAULT_RCLONE_TRANSFERS, "--rclone-transfers", min=1, help="Number of concurrent rclone file transfers (Windows only)"),
    rclone_checkers: int = typer.Option(DEFAULT_RCLONE_CHECKERS, "--rclone-checkers", min=1, help="Number of concurrent rclone checkers (Windows only)"),
    single_stream: bool = typer.Option(DEFAULT_SINGLE_STREAM, "--single-stream", help="Use single-stream copy on rclone (equivalent to transfers=1, checkers=1)"),
    verify: bool = typer.Option(DEFAULT_VERIFY, "--verify", help="Verify all transferable source files are present at destination after import"),
):
    config_path = resolve_config_path(config_path)
    config = Config(config_path)
    if not card_path:
        card_path = list_sdcards(format_type, card_size, config)
    else:
        # On Windows, "F:" is a valid drive argument but Path("F:") is not the
        # drive root directory. Normalize drive-only inputs to "F:\\".
        if platform.system() == "Windows":
            normalized: list[str] = []
            for raw in card_path:
                text = str(raw)
                if len(text) == 2 and text[1] == ":":
                    text = text + "\\"
                normalized.append(text)
            card_path = normalized

    # Skip cards that have nothing to import (only import.yml present).
    in_worker_mode = os.environ.get("SDCARD_TABLE") == "1"
    if not in_worker_mode:
        card_path, _empty = filter_empty_cards([str(p) for p in card_path])
        if not card_path:
            typer.echo("⚠️  No cards with payload files found. Nothing to import.")
            return

    # Avoid nested live tables when import is launched by turbo workers.
    if len(card_path) == 1 and not in_worker_mode:
        _run_import_with_live_table(
            card_path=card_path,
            config_path=config_path,
            copy=copy,
            clean=clean,
            refresh=refresh,
            find=find,
            allow_overwrite=allow_overwrite,
            check=check,
            update=update,
            precheck=precheck,
            card_size=card_size,
            format_type=format_type,
            dry_run=dry_run,
            file_extension=file_extension,
            format_card=format_card,
            ignore_errors=ignore_errors,
            rclone_transfers=rclone_transfers,
            rclone_checkers=rclone_checkers,
            single_stream=single_stream,
            verify=verify,
        )
        return

    import_cards(
        config=config,
        card_path=card_path,
        copy=copy,
        clean=clean,
        find=find,
        allow_overwrite=allow_overwrite,
        check=check,
        update=update,
        precheck=precheck,
        dry_run=dry_run,
        file_extension=file_extension,
        format_card=format_card,
        ignore_errors=ignore_errors,
        refresh=refresh,
        rclone_transfers=rclone_transfers,
        rclone_checkers=rclone_checkers,
        single_stream=single_stream,
        verify=verify,
    )
