import typer
import platform
import subprocess
import sys
import os
import yaml
import queue
import threading
import time
import ctypes
from pathlib import Path
from collections import defaultdict
from sdcard.config import Config
from sdcard.utils.cards_discovery import list_sdcards, filter_empty_cards
from sdcard.utils.platform_adapters import get_windows_drive_reader_map
from sdcard.utils.cli_defaults import (
    DEFAULT_CARD_SIZE,
    DEFAULT_CHECK,
    DEFAULT_CLEAN,
    DEFAULT_DEBUG,
    DEFAULT_FORMAT_TYPE,
    DEFAULT_IGNORE_ERRORS,
    DEFAULT_MAX_PROCESSES,
    DEFAULT_MAX_SD_SPEED,
    DEFAULT_PRECHECK,
    DEFAULT_QUIET_WORKERS,
    DEFAULT_REFRESH,
    DEFAULT_RCLONE_CHECKERS,
    DEFAULT_RCLONE_TRANSFERS,
    DEFAULT_SINGLE_STREAM,
    DEFAULT_UPDATE,
    DEFAULT_VERIFY,
)
from rich.live import Live
from rich.table import Table
from rich.text import Text
import re
from sdcard.utils.config_path_cache import resolve_config_path


def _parse_transfer_progress_line(line: str) -> tuple[int, str] | None:
    """Parse transfer progress percent and speed from rsync or rclone lines."""
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line)
    cleaned = cleaned.replace("\x1b", "")

    # rsync: " 53%   12.34MB/s"
    rsync_match = re.search(r"(\d{1,3})%\s+([\d.,]+\s*\S+/s)", cleaned)
    if rsync_match:
        return int(rsync_match.group(1)), rsync_match.group(2).strip()

    # rclone: "..., 61%, 12.3 MiB/s, ETA ..."
    rclone_match = re.search(r"(\d{1,3})%,\s*([\d.,]+\s*\S+/s)", cleaned)
    if rclone_match:
        return int(rclone_match.group(1)), rclone_match.group(2).strip()

    return None


def _card_passes_import_check(
    card_path: str,
    config_path: str | None,
    update: bool,
) -> tuple[bool, str]:
    command = [sys.executable, "-m", "sdcard.main", "import", card_path, "--check"]
    if update:
        command.append("--update")
    if config_path:
        command.extend(["--config-path", config_path])

    result = subprocess.run(command, capture_output=True, text=True)
    output = "\n".join(
        line for line in [result.stdout.strip(), result.stderr.strip()] if line
    )
    return result.returncode == 0, output


def _filter_cards_with_check(
    card_paths: list[str],
    config_path: str | None,
    update: bool,
) -> tuple[list[str], list[str]]:
    if not card_paths:
        return [], []

    allowed_cards: list[str] = []
    blocked_cards: list[str] = []

    print(f"🔎 Running --check preflight for {len(card_paths)} card(s)")
    for card_path in card_paths:
        import_yml_path = Path(card_path) / "import.yml"
        if not import_yml_path.exists():
            blocked_cards.append(card_path)
            print(f"⛔ Missing import.yml, skipping card: {card_path}")
            continue

        ok, output = _card_passes_import_check(card_path, config_path, update)
        if ok:
            allowed_cards.append(card_path)
            print(f"✅ Check passed: {card_path}")
            continue

        blocked_cards.append(card_path)
        print(f"⛔ Check failed, skipping card: {card_path}")
        if output:
            tail_lines = output.splitlines()[-10:]
            print("--- check output (last 10 lines) ---")
            print("\n".join(tail_lines))

    return allowed_cards, blocked_cards


def _card_has_remaining_payload_files(card_path: str) -> bool:
    """Return True when card still has payload files after a clean move.

    import.yml is expected to remain on cards, so it is ignored here.
    """
    root = Path(card_path)
    if not root.exists() or not root.is_dir():
        return False

    for dirpath, _dirnames, filenames in os.walk(root):
        base_dir = Path(dirpath)
        for filename in filenames:
            file_path = base_dir / filename
            try:
                rel = str(file_path.relative_to(root)).replace("\\", "/")
            except Exception:
                rel = filename

            lowered = rel.lower()
            basename = Path(lowered).name
            if basename == "import.yml":
                continue
            if lowered.startswith('.trash-'):
                continue
            if lowered.endswith('.trashinfo'):
                continue
            return True

    return False


def _split_card_group(card_paths: list[str], max_group_size: int = 2) -> list[list[str]]:
    """Split a card list into fixed-size groups (default: 2)."""
    if max_group_size <= 1:
        return [[card] for card in card_paths]
    return [
        card_paths[i:i + max_group_size]
        for i in range(0, len(card_paths), max_group_size)
    ]


def _run_parallel_import_workers(
    worker_assignments: dict[str, list[str]],
    clean: bool,
    config_path: str | None,
    quiet_workers: bool,
    precheck: bool,
    ignore_errors: bool = False,
    update: bool = False,
    refresh: bool = False,
    rclone_transfers: int = DEFAULT_RCLONE_TRANSFERS,
    rclone_checkers: int = DEFAULT_RCLONE_CHECKERS,
    single_stream: bool = DEFAULT_SINGLE_STREAM,
    verify: bool = DEFAULT_VERIFY,
) -> None:
    """Run import workers as child processes and show a live per-card status table."""
    processes: dict[str, subprocess.Popen | None] = {}
    output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    reader_threads: list[threading.Thread] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SDCARD_TABLE"] = "1"
    card_state: dict[str, dict[str, str | int]] = {}
    card_order: list[str] = []
    current_card_by_worker: dict[str, str | None] = {}
    worker_status: dict[str, str] = {}
    worker_verify_passed: dict[str, bool] = {}
    worker_pending_verified: dict[str, bool] = {}
    card_display_cache: dict[str, str] = {}

    def _get_windows_volume_label(card_path: str) -> str | None:
        if platform.system() != "Windows":
            return None
        try:
            mountpoint = str(card_path)
            if len(mountpoint) == 2 and mountpoint[1] == ":":
                mountpoint = mountpoint + "\\"
            if not mountpoint.endswith("\\"):
                mountpoint = mountpoint + "\\"

            kernel32 = ctypes.windll.kernel32
            volume_name_buf = ctypes.create_unicode_buffer(1024)
            fs_name_buf = ctypes.create_unicode_buffer(1024)
            serial_number = ctypes.c_uint()
            max_comp_len = ctypes.c_uint()
            file_sys_flags = ctypes.c_uint()
            result = kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(mountpoint),
                volume_name_buf,
                ctypes.sizeof(volume_name_buf),
                ctypes.byref(serial_number),
                ctypes.byref(max_comp_len),
                ctypes.byref(file_sys_flags),
                fs_name_buf,
                ctypes.sizeof(fs_name_buf),
            )
            if result and volume_name_buf.value:
                return volume_name_buf.value.strip()
        except Exception:
            return None
        return None

    def _format_card_display(card_path: str) -> str:
        cached = card_display_cache.get(card_path)
        if cached is not None:
            return cached

        label = _get_windows_volume_label(card_path)
        if label:
            drive = str(card_path)
            if len(drive) >= 2 and drive[1] == ":":
                drive = drive[:2]
            display = f"{drive}{label}"
        else:
            display = card_path

        card_display_cache[card_path] = display
        return display

    def _status_style(status: str) -> str:
        lowered = status.lower()
        if lowered in {"queued"}:
            return "grey62"
        if lowered in {"done", "complete", "verified"}:
            return "green"
        if lowered in {"reading", "copying", "running", "working"}:
            return "orange1"
        if lowered in {"error", "failed"}:
            return "red"
        return "white"

    def _build_worker_table() -> Table:
        table = Table(title="Turbo Import Workers")
        table.add_column("Worker", no_wrap=True)
        table.add_column("Card")
        table.add_column("Destination")
        table.add_column("Speed", no_wrap=True)
        table.add_column("Progress", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        for card_key in card_order:
            state = card_state[card_key]
            status = str(state["status"])
            style = _status_style(status)
            table.add_row(
                str(state["worker"]),
                Text(str(state["card"]), style=style),
                str(state["destination"]),
                str(state["speed"]),
                str(state["progress_bar"]),
                Text(status, style=style),
            )
        return table

    def _short_path(path_text: str, max_len: int = 40) -> str:
        if len(path_text) <= max_len:
            return path_text
        return "..." + path_text[-(max_len - 3):]

    def _progress_bar(percent: int, width: int = 18) -> str:
        bounded = max(0, min(100, percent))
        filled = round((bounded / 100) * width)
        return f"{'#' * filled}{'-' * (width - filled)} {bounded:>3}%"

    def _resolve_card_key(card_path: str) -> str:
        card_path_obj = Path(card_path)
        for known_card in card_state:
            if Path(known_card) == card_path_obj:
                return known_card
        return card_path

    def _update_worker_state(worker_name: str, line: str) -> None:
        if line == "__WORKER_COMPLETE__":
            card_key = current_card_by_worker.get(worker_name)
            if card_key and card_key in card_state:
                state = card_state[card_key]
                state["status"] = "complete"
                state["progress_bar"] = _progress_bar(100)
                state["speed"] = "-"
            worker_status[worker_name] = "complete"
            if worker_verify_passed.get(worker_name, False):
                worker_pending_verified[worker_name] = True
            return

        if line == "__WORKER_FAILED__":
            card_key = current_card_by_worker.get(worker_name)
            if card_key and card_key in card_state:
                state = card_state[card_key]
                state["status"] = "failed"
                state["speed"] = "-"
            worker_status[worker_name] = "failed"
            return

        if "verification passed for" in line.lower():
            worker_verify_passed[worker_name] = True
            return

        reading_match = re.match(r"^(?:💾\s*)?Reading (\S+) from (.+) to (.+)$", line)
        if reading_match:
            _token, card_path, destination_path = reading_match.groups()
            previous_card = current_card_by_worker.get(worker_name)
            if previous_card and previous_card in card_state:
                previous_state = card_state[previous_card]
                if str(previous_state["status"]) in {"reading", "copying", "working"}:
                    previous_state["status"] = "complete"
                    previous_state["progress_bar"] = _progress_bar(100)
                    previous_state["speed"] = "-"

            card_key = _resolve_card_key(card_path)
            if card_key not in card_state:
                card_state[card_key] = {
                    "worker": worker_name,
                    "card": _short_path(_format_card_display(card_path), 24),
                    "destination": "-",
                    "speed": "-",
                    "progress_bar": _progress_bar(0),
                    "status": "queued",
                }
                card_order.append(card_key)

            state = card_state[card_key]
            state["destination"] = _short_path(destination_path, 40)
            state["speed"] = "-"
            state["progress_bar"] = _progress_bar(0)
            state["status"] = "reading"
            current_card_by_worker[worker_name] = card_key
            worker_status[worker_name] = "working"
            return

        progress = _parse_transfer_progress_line(line)
        if progress:
            card_key = current_card_by_worker.get(worker_name)
            if not card_key or card_key not in card_state:
                return
            percent, speed = progress
            state = card_state[card_key]
            state["speed"] = speed
            state["progress_bar"] = _progress_bar(percent)
            state["status"] = "copying"
            worker_status[worker_name] = "working"
            return

        if "error" in line.lower() or "failed" in line.lower():
            card_key = current_card_by_worker.get(worker_name)
            if not card_key or card_key not in card_state:
                return
            state = card_state[card_key]
            state["speed"] = "-"
            state["progress_bar"] = _short_path(line.strip(), 22)
            state["status"] = "error"
            worker_status[worker_name] = "error"

    def _promote_verified_cards() -> None:
        for worker_name, pending_verified in list(worker_pending_verified.items()):
            if not pending_verified:
                continue
            card_key = current_card_by_worker.get(worker_name)
            if card_key and card_key in card_state:
                state = card_state[card_key]
                if str(state["status"]) == "complete":
                    state["status"] = "verified"
            worker_status[worker_name] = "verified"
            worker_pending_verified[worker_name] = False

    for worker_name, cards in worker_assignments.items():
        current_card_by_worker[worker_name] = None
        worker_status[worker_name] = "queued"
        worker_verify_passed[worker_name] = False
        worker_pending_verified[worker_name] = False
        for card in cards:
            if card in card_state:
                continue
            card_state[card] = {
                "worker": worker_name,
                "card": _short_path(_format_card_display(card), 24),
                "destination": "-",
                "speed": "-",
                "progress_bar": _progress_bar(0),
                "status": "queued",
            }
            card_order.append(card)

    def _build_import_command(cards: list[str]) -> list[str]:
        command = [sys.executable, "-m", "sdcard.main", "import", *cards]
        if clean:
            command.append("--clean")
        if refresh:
            command.append("--refresh")
        if precheck:
            command.append("--precheck")
        if update:
            command.append("--update")
        if config_path:
            command.extend(["--config-path", config_path])
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
        return command

    def _capture_and_print(worker_name: str, cards: list[str]) -> None:
        output: list[str] = []
        worker_failed = False
        last_return_code = 0
        for card_path in cards:
            output_queue.put((worker_name, f"🧵 {worker_name} starting {card_path}"))
            process = subprocess.Popen(
                _build_import_command([card_path]),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes[worker_name] = process
            worker_status[worker_name] = "running"
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip("\r\n")
                    output.append(line)
                    output_queue.put((worker_name, line))
                process.stdout.close()
            process.wait()
            last_return_code = process.returncode or 0
            if process.returncode != 0:
                worker_failed = True
                output_queue.put((worker_name, f"🧵 {worker_name} failed {card_path} exit={process.returncode}"))
                break
        output_queue.put((worker_name, "__WORKER_FAILED__" if worker_failed else "__WORKER_COMPLETE__"))
        processes.pop(worker_name, None)
        if output and not quiet_workers:
            return
        if output and worker_failed:
            print(f"\n❌ Worker {worker_name} failed with exit code {last_return_code}")
            print(f"--- Worker {worker_name} output ---")
            print("\n".join(output))

    for worker_name, cards in worker_assignments.items():
        if not cards:
            continue
        processes.setdefault(worker_name, None)
        thread = threading.Thread(
            target=_capture_and_print,
            args=(worker_name, list(cards)),
            daemon=True,
        )
        thread.start()
        reader_threads.append(thread)
        print(f"   ✅ Launched {worker_name} for {len(cards)} cards")

    start_deadline = time.time() + 2.0
    while processes and all(proc is None for proc in processes.values()) and time.time() < start_deadline:
        time.sleep(0.05)

    if not processes or all(proc is None for proc in processes.values()):
        print("❌ No import processes were launched")
        return

    print()
    print(f"🚀 Running {len(processes)} worker process(es) in this terminal")
    print(f"🗂️ Tracking {len(card_order)} card(s) with worker mapping")
    if quiet_workers:
        print("🔇 Worker output is hidden (--quiet-workers); showing status only")
    else:
        print("💡 Live import status is shown below as a table")

    live_context = Live(_build_worker_table(), refresh_per_second=4) if not quiet_workers else None
    if live_context is None:
        while processes or not output_queue.empty():
            try:
                while True:
                    worker_name, line = output_queue.get_nowait()
                    if line:
                        _update_worker_state(worker_name, line)
            except queue.Empty:
                pass

            for worker_name, process in list(processes.items()):
                if process is None:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue

                if return_code == 0 and worker_verify_passed.get(worker_name, False):
                    worker_pending_verified[worker_name] = True
                elif return_code == 0:
                    card_key = current_card_by_worker.get(worker_name)
                    if card_key and card_key in card_state:
                        card_state[card_key]["status"] = "complete"
                    worker_status[worker_name] = "complete"
                else:
                    worker_status[worker_name] = "failed"

                processes.pop(worker_name, None)

            _promote_verified_cards()
            if processes or not output_queue.empty():
                time.sleep(0.2)
    else:
        with live_context as live:
            while processes or not output_queue.empty():
                try:
                    while True:
                        worker_name, line = output_queue.get_nowait()
                        if line:
                            _update_worker_state(worker_name, line)
                except queue.Empty:
                    pass

                for worker_name, process in list(processes.items()):
                    if process is None:
                        continue
                    return_code = process.poll()
                    if return_code is None:
                        continue

                    left_over_cards: set[str] = set()
                    if return_code == 0 and clean:
                        for card_path in worker_assignments.get(worker_name, []):
                            if _card_has_remaining_payload_files(card_path):
                                left_over_cards.add(card_path)

                    current_card = current_card_by_worker.get(worker_name)
                    if current_card and current_card in card_state:
                        current_state = card_state[current_card]
                        if current_card in left_over_cards:
                            current_state["status"] = "failed"
                            current_state["progress_bar"] = "files left"
                            current_state["speed"] = "-"
                        elif return_code == 0 and str(current_state["status"]) in {"reading", "copying", "working"}:
                            current_state["status"] = "complete"
                            current_state["progress_bar"] = _progress_bar(100)
                            current_state["speed"] = "-"
                            if worker_verify_passed.get(worker_name, False):
                                worker_pending_verified[worker_name] = True
                        elif return_code != 0 and str(current_state["status"]) not in {"done", "complete", "verified"}:
                            current_state["status"] = "error"
                            current_state["speed"] = "-"

                    for card_key in card_order:
                        state = card_state[card_key]
                        if state["worker"] != worker_name:
                            continue
                        if card_key in left_over_cards:
                            state["status"] = "failed"
                            state["speed"] = "-"
                            state["progress_bar"] = "files left"
                            continue
                        if str(state["status"]) == "queued":
                            if return_code == 0:
                                state["status"] = "complete"
                                state["progress_bar"] = _progress_bar(100)
                            else:
                                state["status"] = "failed"

                    if return_code == 0 and not left_over_cards:
                        worker_status[worker_name] = "complete"
                    else:
                        worker_status[worker_name] = "failed"

                    if return_code == 0:
                        current_card_by_worker[worker_name] = current_card
                    processes.pop(worker_name, None)

                _promote_verified_cards()
                live.update(_build_worker_table())
                if processes or not output_queue.empty():
                    time.sleep(0.2)

        for worker_name in sorted(worker_status):
            status = worker_status[worker_name]
            status_icon = "✅" if status in {"done", "complete", "verified"} else "❌"
            print(f"{status_icon} {worker_name} {status}")

    for thread in reader_threads:
        thread.join(timeout=0.5)


def turbo(config_path: str = typer.Option(None, help="Path to config directory."),
         max_processes:int = typer.Option(DEFAULT_MAX_PROCESSES, help="Number of concurrent transfers"),
         format_type:str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
         card_size:int = typer.Option(DEFAULT_CARD_SIZE, help="maximum card size"),
         clean:bool = typer.Option(DEFAULT_CLEAN, help="move all the files to location"),
         refresh:bool = typer.Option(DEFAULT_REFRESH, "--refresh", help="After --clean, write a new import.yml to each card"),
         check: bool = typer.Option(DEFAULT_CHECK, "--check", help="Preflight each card and skip cards that have copy conflicts"),
         update: bool = typer.Option(DEFAULT_UPDATE, "--update", help="Allow newer source files to update older destination files"),
         precheck: bool = typer.Option(DEFAULT_PRECHECK, "--precheck", help="Run destination conflict prechecks before copy/move"),
         quiet_workers: bool = typer.Option(DEFAULT_QUIET_WORKERS, "--quiet-workers", help="Hide worker output and only show process status"),
         ignore_errors: bool = typer.Option(DEFAULT_IGNORE_ERRORS, "--ignore-errors", help="Pass --ignore-errors to rsync and continue on file errors"),
         rclone_transfers: int = typer.Option(DEFAULT_RCLONE_TRANSFERS, "--rclone-transfers", min=1, help="Number of concurrent rclone file transfers per worker (Windows only)"),
         rclone_checkers: int = typer.Option(DEFAULT_RCLONE_CHECKERS, "--rclone-checkers", min=1, help="Number of concurrent rclone checkers per worker (Windows only)"),
         single_stream: bool = typer.Option(DEFAULT_SINGLE_STREAM, "--single-stream", help="Use single-stream copy on rclone (equivalent to transfers=1, checkers=1)"),
         verify: bool = typer.Option(DEFAULT_VERIFY, "--verify", help="Verify all transferable source files are present at destination after import"),
         debug:bool = typer.Option(DEFAULT_DEBUG, help="Card format type"),
         max_sd_speed:float = typer.Option(DEFAULT_MAX_SD_SPEED, help="Maximum realistic SD card read speed in MB/s"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    """
    This command orchestrates the import process for ALL detected SD cards,
    launching multiple processes to handle concurrent imports.
    It supports both Linux and Windows environments.
    """
    from sdcard.utils.usb import (
        analyze_destination_capacity,
        calculate_real_transfer_rate,
        detect_thunderbolt_upstream_capacity,
        get_complete_usb_card_info,
        get_usb_hierarchy_info,
        scan_thunderbolt_ports,
    )
    config_path = str(resolve_config_path(config_path) or "") or None
    if platform.system() == "Linux":
        config = Config(config_path)
        destination_path = config.get_path('card_store')
        # Get USB card info and available ports (filtered by size and format)
        usb_cards, available_ports = get_complete_usb_card_info(destination_path, card_size, format_type)
        
        # Analyze destination drives with optional override
        destinations = analyze_destination_capacity(usb_cards, dest_write_speed, destination_path)
        
        # Scan Thunderbolt ports (for display purposes)
        thunderbolt_ports, thunderbolt_controllers = scan_thunderbolt_ports()
        
        # Use system-based Thunderbolt detection instead of boltctl analysis
        thunderbolt_bandwidth = detect_thunderbolt_upstream_capacity()
        
        # Calculate realistic transfer rates based on SD card limitations
        for i, row in usb_cards.iterrows():
            # Cap transfer rate at SD card maximum
            usb_cards.at[i, 'actual_transfer_rate'] = min(row['actual_transfer_rate'], max_sd_speed)
            
            # Add realistic single-card speed
            usb_cards.at[i, 'realistic_single_speed'] = min(row['actual_transfer_rate'], max_sd_speed)
        
        # Calculate saturation based on realistic SD card speeds
        saturation_info = None
        if thunderbolt_bandwidth:
            thunderbolt_devices = []
            total_thunderbolt_demand = 0
            
            for _, card in usb_cards.iterrows():
                if card.get('thunderbolt_connected', False):
                    thunderbolt_devices.append(card)
                    realistic_speed = min(card.get('actual_transfer_rate', 0), max_sd_speed)
                    total_thunderbolt_demand += realistic_speed
            
            upstream_capacity = thunderbolt_bandwidth.get('practical_usb_capacity_mbps', 0)
            utilization_percentage = (total_thunderbolt_demand / upstream_capacity * 100) if upstream_capacity > 0 else 0
            
            saturation_info = {
                'thunderbolt_devices': len(thunderbolt_devices),
                'total_demand_mbps': total_thunderbolt_demand,
                'upstream_capacity_mbps': upstream_capacity,
                'utilization_percentage': utilization_percentage,
                'is_saturated': utilization_percentage > 90,
                'headroom_mbps': upstream_capacity - total_thunderbolt_demand,
                'devices': thunderbolt_devices
            }
        
        if usb_cards.empty:
            print("No SD cards found")
            return
        
        # Sort by transfer speed for processing
        usb_cards = usb_cards.sort_values('realistic_single_speed', ascending=False)
        
        # Group cards by their READER for efficient dual-slot processing
        grouped_cards = {}
        for _, card in usb_cards.iterrows():
            # Skip destination drives - they are not SD cards to import
            if card.get('device_type') == 'destination':
                continue    
                
            reader_id = card.get('reader_manufacturer', 'unknown') + '_' + card.get('reader_product', 'unknown')
            
            hierarchy = get_usb_hierarchy_info(card['name'])
            if (Path(card['mountpoint']) / "import.yml").exists():
                import_yml_path = Path(card['mountpoint']) / "import.yml"
                raw_data = yaml.safe_load(import_yml_path.read_text(encoding='utf-8'))
                card['card_number'] = raw_data['card_number'] if 'card_number' in raw_data else 0
                if 'card_reader' in hierarchy:
                    reader_path = hierarchy['card_reader'].get('usb_path', 'unknown')
                    reader_id = f"{reader_id}_{reader_path}"
                
                if reader_id not in grouped_cards:
                    grouped_cards[reader_id] = []
                grouped_cards[reader_id].append(card)
            else:
                print(f"❌ Card {card['mountpoint']} does not have import.yml, skipping")
 
        # make a number of lists that match the number of terminals 
        # and distribute cards evenly across them
        # Process each reader group
        if not grouped_cards:
            print("❌ No valid card groups found for import")
            return  
        print(f"📊 Found {len(grouped_cards)} reader groups, using {max_processes} terminals")

        print()
        print("🚀 Launching import processes:")
        print("-" * 60)
        # Ensure we don't exceed max_processes
        if max_processes <= 0:
            print("❌ Invalid max_processes value, must be greater than 0")
            return

        reader_groups = defaultdict(list)

        # Round-robin distribute group values (lists of strings) to readers
        for i, group in enumerate(grouped_cards.values()):
            reader_key = f"reader_{i % max_processes}"
            group_mounts = [card['mountpoint'] for card in group]
            if check:
                allowed_cards, blocked_cards = _filter_cards_with_check(
                    group_mounts,
                    config_path,
                    update,
                )
                if blocked_cards:
                    print(f"⚠️ Skipped {len(blocked_cards)} card(s) on check failure for {reader_key}")
                group_mounts = allowed_cards
            reader_groups[reader_key].extend(group_mounts)
        worker_assignments = {
            worker_name: list(cards)
            for worker_name, cards in reader_groups.items()
            if cards
        }
        if not worker_assignments:
            print("⛔ No cards passed --check preflight")
            return
        _run_parallel_import_workers(
            worker_assignments,
            clean=clean,
            config_path=config_path,
            quiet_workers=quiet_workers,
            precheck=precheck,
            ignore_errors=ignore_errors,
            update=update,
            refresh=refresh,
            rclone_transfers=rclone_transfers,
            rclone_checkers=rclone_checkers,
            single_stream=single_stream,
            verify=verify,
        )
    else:
        # Windows version - use lightweight mountpoint discovery to avoid
        # slow WMI/USB host probing during startup.
        config = Config(config_path)
        discovered_cards = list_sdcards(format_type, card_size, config)

        # Filter for cards with import.yml
        valid_drives: list[str] = []
        for mountpoint in discovered_cards:
            import_yml_path = Path(mountpoint) / "import.yml"
            if import_yml_path.exists():
                valid_drives.append(mountpoint)
            else:
                print(f"❌ Card {mountpoint} does not have import.yml, skipping")
        
        if not valid_drives:
            print("❌ No valid Windows drives found for import")
            return

        valid_drives, _empty = filter_empty_cards(valid_drives)
        if not valid_drives:
            print("⚠️  All cards contain only import.yml — nothing to import")
            return
        
        print(f"📊 Found {len(valid_drives)} Windows drives, using {max_processes} terminals")
        print()
        print("🚀 Launching import processes:")
        print("-" * 60)
        
        # Ensure we don't exceed max_processes
        if max_processes <= 0:
            print("❌ Invalid max_processes value, must be greater than 0")
            return

        reader_map = get_windows_drive_reader_map()
        host_groups: dict[str, list[str]] = defaultdict(list)
        for drive in valid_drives:
            drive_letter = str(drive)[0:1].upper()
            host_key = reader_map.get(drive_letter, f"drive-{drive_letter}")
            host_groups[host_key].append(drive)

        # If discovery collapses all cards into one bucket, preserve throughput
        # by splitting at drive level instead of idling all but one worker.
        if len(host_groups) == 1 and len(valid_drives) > 1 and max_processes > 1:
            host_groups = defaultdict(list)
            for drive in valid_drives:
                drive_letter = str(drive)[0:1].upper()
                host_groups[f"drive-{drive_letter}"] = [drive]

        # Group host buckets into terminal assignments using round-robin,
        # while capping each grouped batch to 2 cards.
        terminal_assignments = defaultdict(list)
        assignment_index = 0
        for group_drives in host_groups.values():
            for split_drives in _split_card_group(group_drives, max_group_size=2):
                terminal_key = f"terminal_{assignment_index % max_processes}"
                assignment_index += 1
                scheduled_drives = split_drives
                if check:
                    allowed_cards, _blocked_cards = _filter_cards_with_check(
                        split_drives,
                        config_path,
                        update,
                    )
                    scheduled_drives = allowed_cards
                terminal_assignments[terminal_key].extend(scheduled_drives)
        
        # Launch worker processes directly instead of opening terminal windows
        worker_assignments = {
            worker_name: list(drives)
            for worker_name, drives in terminal_assignments.items()
            if drives
        }
        if not worker_assignments:
            print("⛔ No cards passed --check preflight")
            return
        _run_parallel_import_workers(
            worker_assignments,
            clean=clean,
            config_path=config_path,
            quiet_workers=quiet_workers,
            precheck=precheck,
            ignore_errors=ignore_errors,
            update=update,
            refresh=refresh,
            rclone_transfers=rclone_transfers,
            rclone_checkers=rclone_checkers,
            single_stream=single_stream,
            verify=verify,
        )
        
        print("✅ Import orchestration completed!")

