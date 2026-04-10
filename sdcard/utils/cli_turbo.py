import typer
import platform
import subprocess
import sys
import os
import yaml
import queue
import threading
import time
import uuid
from pathlib import Path
from collections import defaultdict
from typer.testing import CliRunner
from sdcard.config import Config
from sdcard.utils.cards_discovery import get_available_cards
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
    DEFAULT_UPDATE,
)
from sdcard.utils.usb import (
    get_complete_usb_card_info, scan_thunderbolt_ports, detect_thunderbolt_upstream_capacity,
    analyze_destination_capacity, calculate_real_transfer_rate, get_usb_hierarchy_info
)
from rich.live import Live
from rich.table import Table
from rich.text import Text
import re


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


def _write_new_import_yml_after_clean(card_path: str) -> None:
    """Write a fresh import.yml to card after successful clean, preserving instrument name.
    
    Reads the existing import.yml to extract the instrument field,
    then writes a new import.yml with card_number reset to 1 and import-related
    fields cleared (import_date, import_token).
    """
    card_root = Path(card_path)
    import_yml_path = card_root / "import.yml"
    
    if not import_yml_path.exists():
        return
    
    try:
        # Read existing import.yml to get instrument and card_number
        with open(import_yml_path, 'r') as f:
            existing = yaml.safe_load(f) or {}
        
        instrument = existing.get('instrument', 'unknown')
        card_number = existing.get('card_number', 1)
        
        # Generate new import_token
        import_token = str(uuid.uuid4())[:8].replace('-', '')
        
        # Always get destination_path from config file
        from sdcard.config import Config
        config_path = None
        # Try to find config path from parent directory
        for parent in card_root.parents:
            candidate = parent / "import.yml"
            if candidate.exists():
                config_path = str(candidate)
                break
        config = Config(config_path) if config_path else Config()
        destination_path = config.data.get('import_path_template') or config.data.get('destination_path')

        # Build new import.yml preserving instrument and card_number, with fresh token
        new_yml = {
            'field_trip_id': existing.get('field_trip_id', ''),
            'start_date': existing.get('start_date', ''),
            'end_date': existing.get('end_date', ''),
            'custodian': existing.get('custodian', ''),
            'email': existing.get('email', ''),
            'project_name': existing.get('project_name', ''),
            'instrument': instrument,
            'card_number': card_number,
            'import_token': import_token,
            'destination_path': destination_path,
            # Clear import_date so it's set fresh on next import
        }

        # Only include non-empty fields
        new_yml = {k: v for k, v in new_yml.items() if v}

        # Write the new import.yml
        with open(import_yml_path, 'w') as f:
            yaml.dump(new_yml, f, default_flow_style=False, sort_keys=False)

        print(f"✏️  Updated import.yml on {card_path} (instrument='{instrument}', card_number={card_number}, token='{import_token}', destination_path='{destination_path}')")
    except Exception as e:
        print(f"⚠️  Could not update import.yml on {card_path}: {e}")


def _run_parallel_import_workers(
    worker_assignments: dict[str, list[str]],
    clean: bool,
    config_path: str | None,
    quiet_workers: bool,
    precheck: bool,
    ignore_errors: bool = False,
    update: bool = False,
    refresh: bool = False,
) -> None:
    """Run import workers as child processes and show a live per-card status table."""
    processes: dict[str, subprocess.Popen] = {}
    output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    reader_threads: list[threading.Thread] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SDCARD_TABLE"] = "1"
    card_state: dict[str, dict[str, str | int]] = {}
    card_order: list[str] = []
    current_card_by_worker: dict[str, str | None] = {}
    worker_status: dict[str, str] = {}

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
        reading_match = re.match(r"^💾 Reading (\S+) from (.+) to (.+)$", line)
        if reading_match:
            _token, card_path, destination_path = reading_match.groups()
            previous_card = current_card_by_worker.get(worker_name)
            if previous_card and previous_card in card_state:
                previous_state = card_state[previous_card]
                if str(previous_state["status"]) in {"reading", "copying", "working"}:
                    previous_state["status"] = "done"
                    previous_state["progress_bar"] = _progress_bar(100)
                    previous_state["speed"] = "-"

            card_key = _resolve_card_key(card_path)
            if card_key not in card_state:
                card_state[card_key] = {
                    "worker": worker_name,
                    "card": _short_path(card_path, 24),
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
        progress_match = re.search(
            r"(\d{1,3})%\s+([\d.,]+\s*\S+/s)",
            line,
        )
        if progress_match:
            card_key = current_card_by_worker.get(worker_name)
            if not card_key or card_key not in card_state:
                return
            percent = int(progress_match.group(1))
            speed = progress_match.group(2).strip()
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

    for worker_name, cards in worker_assignments.items():
        current_card_by_worker[worker_name] = None
        worker_status[worker_name] = "queued"
        for card in cards:
            if card in card_state:
                continue
            card_state[card] = {
                "worker": worker_name,
                "card": _short_path(card, 24),
                "destination": "-",
                "speed": "-",
                "progress_bar": _progress_bar(0),
                "status": "queued",
            }
            card_order.append(card)

    def _stream_worker_output(worker_name: str, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            output_queue.put((worker_name, line.rstrip("\r\n")))
        process.stdout.close()

    for worker_name, cards in worker_assignments.items():
        if not cards:
            continue

        command = [sys.executable, "-m", "sdcard.main", "import", *cards]
        if clean:
            command.append("--clean")
        if precheck:
            command.append("--precheck")
        if update:
            command.append("--update")
        if config_path:
            command.extend(["--config-path", config_path])
        if ignore_errors:
            command.append("--ignore-errors")

        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[worker_name] = process
        worker_status[worker_name] = "running"
        def _capture_and_print(worker_name, process):
            output = []
            if process.stdout is not None:
                for line in process.stdout:
                    output.append(line.rstrip("\r\n"))
                    if not quiet_workers:
                        output_queue.put((worker_name, line.rstrip("\r\n")))
                process.stdout.close()
            process.wait()
            if process.returncode != 0:
                print(f"\n❌ Worker {worker_name} failed with exit code {process.returncode}")
                print(f"--- Worker {worker_name} output ---")
                print("\n".join(output))
        thread = threading.Thread(
            target=_capture_and_print,
            args=(worker_name, process),
            daemon=True,
        )
        thread.start()
        reader_threads.append(thread)
        print(f"   ✅ Launched {worker_name} for {len(cards)} cards")

    if not processes:
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
        while processes:
            for worker_name, process in list(processes.items()):
                return_code = process.poll()
                if return_code is not None:
                    left_over_cards: list[str] = []
                    if return_code == 0 and clean:
                        for card_path in worker_assignments.get(worker_name, []):
                            if _card_has_remaining_payload_files(card_path):
                                left_over_cards.append(card_path)
                            else:
                                # Clean succeeded with no remaining files, write fresh import.yml if refresh is set
                                if refresh:
                                    _write_new_import_yml_after_clean(card_path)

                    if left_over_cards:
                        print(f"❌ {worker_name} finished with files left on {len(left_over_cards)} card(s)")
                        for card_path in left_over_cards:
                            print(f"   - {card_path}")
                    else:
                        status_icon = "✅" if return_code == 0 else "❌"
                        print(f"{status_icon} {worker_name} finished with exit code {return_code}")
                    processes.pop(worker_name, None)
            if processes:
                time.sleep(0.2)
    else:
        with live_context as live:
            while processes:
                try:
                    while True:
                        worker_name, line = output_queue.get_nowait()
                        if line:
                            _update_worker_state(worker_name, line)
                    
                except queue.Empty:
                    pass

                for worker_name, process in list(processes.items()):
                    return_code = process.poll()
                    if return_code is not None:
                        left_over_cards: set[str] = set()
                        if return_code == 0 and clean:
                            for card_path in worker_assignments.get(worker_name, []):
                                if _card_has_remaining_payload_files(card_path):
                                    left_over_cards.add(card_path)
                                else:
                                    # Clean succeeded with no remaining files, write fresh import.yml if refresh is set
                                    if refresh:
                                        _write_new_import_yml_after_clean(card_path)

                        current_card = current_card_by_worker.get(worker_name)
                        if current_card and current_card in card_state:
                            current_state = card_state[current_card]
                            if current_card in left_over_cards:
                                current_state["status"] = "failed"
                                current_state["progress_bar"] = "files left"
                                current_state["speed"] = "-"
                            elif return_code == 0 and str(current_state["status"]) in {"reading", "copying", "working"}:
                                current_state["status"] = "done"
                                current_state["progress_bar"] = _progress_bar(100)
                                current_state["speed"] = "-"
                            elif return_code != 0 and str(current_state["status"]) != "done":
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
                                    state["status"] = "done"
                                    state["progress_bar"] = _progress_bar(100)
                                else:
                                    state["status"] = "failed"

                        worker_status[worker_name] = (
                            "done" if return_code == 0 and not left_over_cards else "failed"
                        )
                        if return_code == 0:
                            current_card_by_worker[worker_name] = None
                        processes.pop(worker_name, None)

                live.update(_build_worker_table())
                if processes:
                    time.sleep(0.2)

        for worker_name in sorted(worker_status):
            status = worker_status[worker_name]
            status_icon = "✅" if status == "done" else "❌"
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
         debug:bool = typer.Option(DEFAULT_DEBUG, help="Card format type"),
         max_sd_speed:float = typer.Option(DEFAULT_MAX_SD_SPEED, help="Maximum realistic SD card read speed in MB/s"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    """
    This command orchestrates the import process for ALL detected SD cards,
    launching multiple processes to handle concurrent imports.
    It supports both Linux and Windows environments.
    """
    if platform.system() == "Linux":
        runner = CliRunner()
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
        )
    else:
        # Windows version - use existing get_available_cards function
        runner = CliRunner()
        
        # Get available cards using the existing function
        drives_info = get_available_cards(format_type, card_size)
        
        # Filter for cards with import.yml and extract mountpoints
        valid_drives = []
        
        for drive_info in drives_info:
            mountpoint = drive_info['mountpoint']
            import_yml_path = Path(mountpoint) / "import.yml"
            
            if import_yml_path.exists():
                valid_drives.append(mountpoint)
            else:
                print(f"❌ Card {mountpoint} does not have import.yml, skipping")
        
        if not valid_drives:
            print("❌ No valid Windows drives found for import")
            return
        
        print(f"📊 Found {len(valid_drives)} Windows drives, using {max_processes} terminals")
        print()
        print("🚀 Launching import processes:")
        print("-" * 60)
        
        # Ensure we don't exceed max_processes
        if max_processes <= 0:
            print("❌ Invalid max_processes value, must be greater than 0")
            return

        # Group drives into terminal assignments using round-robin
        terminal_assignments = defaultdict(list)
        
        # Round-robin distribute drives to terminals
        for i, drive in enumerate(valid_drives):
            terminal_key = f"terminal_{i % max_processes}"
            if check:
                allowed_cards, _blocked_cards = _filter_cards_with_check(
                    [drive],
                    config_path,
                    update,
                )
                if not allowed_cards:
                    continue
            terminal_assignments[terminal_key].append(drive)
        
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
        )
        
        print("✅ Import orchestration completed!")

