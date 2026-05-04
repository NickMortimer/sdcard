from unicodedata import name
import pandas as pd
import subprocess
import shlex
from typer.testing import CliRunner
import typer
import platform
import sys
import os
from pathlib import Path
from sdcard.config import Config  
from sdcard.main import sdcard
from sdcard.utils.usb import (
    get_complete_usb_card_info, 
    scan_thunderbolt_ports, 
    detect_thunderbolt_upstream_capacity,
    analyze_destination_capacity,
    calculate_real_transfer_rate,
    get_usb_hierarchy_info
)
from sdcard.utils.cards_discovery import get_available_cards
from collections import defaultdict
import queue
import re
import threading
import time
from rich.live import Live
from rich.table import Table
from rich.text import Text

sdall = typer.Typer(
    name="Import all SD cards",
)


def _parse_transfer_progress(line: str) -> tuple[int, str] | None:
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


def _run_parallel_import_workers(
    worker_assignments: dict[str, list[str]],
    clean: bool,
    config_path: str | None,
    quiet_workers: bool,
    precheck: bool,
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
        progress = _parse_transfer_progress(line)
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
            command.append("--move")
        if precheck:
            command.append("--precheck")
        if config_path:
            command.extend(["--config-path", config_path])

        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.DEVNULL if quiet_workers else subprocess.PIPE,
            stderr=subprocess.DEVNULL if quiet_workers else subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[worker_name] = process
        worker_status[worker_name] = "running"
        if not quiet_workers:
            thread = threading.Thread(
                target=_stream_worker_output,
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
                        current_card = current_card_by_worker.get(worker_name)
                        if current_card and current_card in card_state:
                            current_state = card_state[current_card]
                            if return_code == 0 and str(current_state["status"]) in {"reading", "copying", "working"}:
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
                            if str(state["status"]) == "queued":
                                if return_code == 0:
                                    state["status"] = "done"
                                    state["progress_bar"] = _progress_bar(100)
                                else:
                                    state["status"] = "failed"

                        worker_status[worker_name] = "done" if return_code == 0 else "failed"
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

@sdall.command('probe')
def probe(config_path: str = typer.Option(None, help="Path to config directory."),
         format_type:str = typer.Option('exfat', help="Card format type"),
         card_size:int = typer.Option(512, help="maximum card size"),
         max_sd_speed:float = typer.Option(140.0, help="Maximum realistic SD card read speed in MB/s"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    """Probe and analyze USB device tree, SD cards, and system capabilities."""
    
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
        
        # Show analysis with emojis
        print("🔍 Port Analysis:")
        print("=" * 60)
        
        # Show destination drive analysis
        if destinations:
            print("💽 Destination Drive Analysis:")
            for dest_mount, dest_info in destinations.items():
                saturation_icon = "🔴" if dest_info['is_saturated'] else "🟢"
                override_icon = "⚙️" if dest_info.get('speed_override') else ""
                config_icon = "📋" if dest_info.get('using_global_config') else ""
                
                print(f"  🎯 {saturation_icon} {override_icon} {config_icon} {dest_mount} ({dest_info['drive_type']})")
                print(f"     Device: {dest_info['device']}")
                print(f"     Destination: {dest_info.get('destination_path', 'Unknown')}")
                
                if dest_info.get('using_global_config'):
                    print(f"     Source: Global config (--config-path)")
                else:
                    print(f"     Source: Individual card import.yml files")
                
                if dest_info.get('speed_override'):
                    print(f"     Write Speed: {dest_info['estimated_write_speed']:.0f} MB/s (USER OVERRIDE)")
                else:
                    print(f"     Estimated Write Speed: {dest_info['estimated_write_speed']:.0f} MB/s (min 200 MB/s)")
                    
                print(f"     Total Demand: {dest_info['total_demand']:.1f} MB/s from {len(dest_info['cards'])} cards")
                print(f"     Utilization: {dest_info['utilization_percentage']:.1f}%")
                
                if dest_info['is_saturated']:
                    print(f"     ⚠️  WARNING: Destination drive may be saturated!")
                else:
                    remaining = dest_info['estimated_write_speed'] - dest_info['total_demand']
                    additional_cards = remaining / max_sd_speed if max_sd_speed > 0 else 0
                    print(f"     ✅ Drive OK: Can handle ~{additional_cards:.1f} more cards")
                print()
        
        # Show Thunderbolt analysis
        if thunderbolt_bandwidth:
            print("⚡ Thunderbolt System Analysis:")
            print(f"  🔌 Controller: {thunderbolt_bandwidth.get('controller', 'Unknown')}")
            print(f"  📊 Generation: {thunderbolt_bandwidth.get('generation', 'Unknown')}")
            print(f"  💾 Max Speed: {thunderbolt_bandwidth.get('max_speed_gbps', 0):.0f} Gb/s")
            print(f"  🔬 Practical USB Capacity: {thunderbolt_bandwidth.get('practical_usb_capacity_mbps', 0):.1f} MB/s")
            
            if saturation_info and saturation_info['thunderbolt_devices'] > 0:
                saturation_icon = "🔴" if saturation_info['is_saturated'] else "🟢"
                print(f"\n📈 Bandwidth Utilization:")
                print(f"  {saturation_icon} Connected TB Devices: {saturation_info['thunderbolt_devices']}")
                print(f"  📊 Total Demand: {saturation_info['total_demand_mbps']:.1f} MB/s")
                print(f"  🎯 Utilization: {saturation_info['utilization_percentage']:.1f}%")
                print(f"  🚀 Available Headroom: {saturation_info['headroom_mbps']:.1f} MB/s")
            print()
        
        # Show card analysis
        print("📱 SD Card Analysis:")
        print("=" * 60)
        
        cards_without_config = []
        
        # Group cards by reader to detect dual-slot scenarios
        reader_groups = {}
        for _, card in usb_cards.iterrows():
            if card.get('device_type') == 'destination':
                continue
                
            reader_key = f"{card.get('reader_manufacturer', 'Unknown')}_{card.get('reader_product', 'Unknown')}"
            hierarchy = get_usb_hierarchy_info(card['name'])
            if 'card_reader' in hierarchy:
                reader_path = hierarchy['card_reader'].get('usb_path', 'unknown')
                reader_key = f"{reader_key}_{reader_path}"
            
            if reader_key not in reader_groups:
                reader_groups[reader_key] = []
            reader_groups[reader_key].append(card)
        
        for _, card in usb_cards.iterrows():
            # Show destination drives separately
            if card.get('device_type') == 'destination':
                print(f"💽 🎯 {card['mountpoint']} ({card['size']}) - DESTINATION DRIVE")
                print(f"    💾 Device: {card['name']}")
                print(f"    📂 Filesystem: {card['fstype']}")
                print(f"    🎯 This is where SD card content will be stored")
                
                # Add USB speed information for destination drive
                if card.get('reader_manufacturer'):
                    print(f"    📖 Connection: {card['reader_manufacturer']} {card['reader_product']}")
                    print(f"       USB Speed: {card['reader_speed']} ({card['reader_speed_mbps']} Mbps)")
                    usb_transfer_rate = card.get('actual_transfer_rate', 0)
                    print(f"    ⚡ USB Transfer Rate: {usb_transfer_rate:.1f} MB/s")
                else:
                    print(f"    🔌 Connection: Internal/SATA (not USB)")
                
                # Show Thunderbolt status for destination if applicable
                if card.get('thunderbolt_connected', False):
                    print(f"    ⚡ Thunderbolt: {card.get('thunderbolt_info', 'Connected')}")
                elif card.get('thunderbolt_info'):
                    print(f"    🔌 Connection: {card.get('thunderbolt_info', 'Standard connection')}")
                
                print()
                continue
            
            # Check for import.yml
            mountpoint = card['mountpoint']
            has_import_yml = (Path(mountpoint) / "import.yml").exists()
            
            if not has_import_yml:
                cards_without_config.append(mountpoint)
            
            # Find reader group to check for dual-slot performance impact
            reader_key = f"{card.get('reader_manufacturer', 'Unknown')}_{card.get('reader_product', 'Unknown')}"
            hierarchy = get_usb_hierarchy_info(card['name'])
            if 'card_reader' in hierarchy:
                reader_path = hierarchy['card_reader'].get('usb_path', 'unknown')
                reader_key = f"{reader_key}_{reader_path}"
            
            cards_in_reader = reader_groups.get(reader_key, [card])
            is_dual_slot = len(cards_in_reader) > 1
            
            usb_limited_speed = card.get('actual_transfer_rate', 0)
            is_usb_optimal = usb_limited_speed <= max([calculate_real_transfer_rate(p['speed_mbps']) for p in available_ports]) * 0.9 if available_ports else True
            realistic_speed = min(usb_limited_speed, max_sd_speed)
            
            # Apply dual-slot penalty - reading two cards is slower than one!
            if is_dual_slot:
                dual_slot_factor = 0.6  # 40% slower when reading two cards
                realistic_speed *= dual_slot_factor
            
            icon = "✅" if is_usb_optimal else "⚠️"
            tb_icon = "⚡" if card.get('thunderbolt_connected', False) else "🔌"
            speed_limited_icon = "🐌" if usb_limited_speed > max_sd_speed else ""
            config_icon = "❌" if not has_import_yml else ""
            dual_slot_icon = "👥" if is_dual_slot else ""
            
            print(f"{icon} {tb_icon} {speed_limited_icon} {config_icon} {dual_slot_icon} {card['mountpoint']} ({card['size']})")
            
            if not has_import_yml:
                print(f"    ❌ WARNING: Missing import.yml configuration file!")
                print(f"       This card will be SKIPPED during processing")
            
            if is_dual_slot:
                print(f"    👥 DUAL-SLOT READER: {len(cards_in_reader)} cards in same reader")
            
            if card.get('thunderbolt_connected', False):
                print(f"    ⚡ Thunderbolt: {card.get('thunderbolt_info', 'Connected')}")
            else:
                print(f"    🔌 Connection: {card.get('thunderbolt_info', 'Standard USB')}")
            
            if card.get('reader_manufacturer'):
                print(f"    📖 Reader: {card['reader_manufacturer']} {card['reader_product']}")
                print(f"       USB Speed: {card['reader_speed']} ({card['reader_speed_mbps']} Mbps)")
            
            print(f"    ⚡ USB Transfer Rate: {usb_limited_speed:.1f} MB/s")
            if is_dual_slot:
                original_speed = min(usb_limited_speed, max_sd_speed)
                print(f"    🎯 Single-card Speed: {original_speed:.1f} MB/s")
                print(f"    👥 Dual-slot Speed: {realistic_speed:.1f} MB/s (reduced due to sharing)")
            else:
                print(f"    🎯 Realistic Speed: {realistic_speed:.1f} MB/s (SD card limited)")
            print()
            
    else:
        # Windows version - use existing get_available_cards function        
        # Get available cards using the existing function
        drives_info = get_available_cards(format_type, card_size)
        
        # Filter for cards with import.yml and extract mountpoints
        valid_drives = []
        cards_without_config = []
        
        for drive_info in drives_info:
            mountpoint = drive_info['mountpoint']
            import_yml_path = Path(mountpoint) / "import.yml"
            
            if import_yml_path.exists():
                valid_drives.append(mountpoint)
            else:
                cards_without_config.append(mountpoint)
        
        # Show warning for cards without config
        if cards_without_config:
            print("⚠️  CONFIGURATION WARNINGS:")
            print("=" * 60)
            print(f"❌ {len(cards_without_config)} SD cards missing import.yml configuration:")
            for card_path in cards_without_config:
                print(f"   • {card_path}")
            print()
            print("💡 These cards will be SKIPPED during import processing.")
            print("   Create import.yml files to include them in the import.")
            print()
        
        if not valid_drives:
            print("❌ No valid Windows drives found for import")
            return
        
        print(f"📊 Found {len(valid_drives)} Windows drives with import.yml")
        for drive in valid_drives:
            print(f"   ✅ {drive}")
    
    print("✅ Probe analysis completed!")

@sdall.command('import_all')
def import_all(config_path: str = typer.Option(None, help="Path to config directory."),
         max_processes:int = typer.Option(4, help="Number of concurrent transfers"),
         format_type:str = typer.Option('exfat', help="Card format type"),
         card_size:int = typer.Option(512, help="maximum card size"),
         clean:bool = typer.Option(False, help="move all the files to location"),
         precheck: bool = typer.Option(False, "--precheck", help="Run destination conflict prechecks before copy/move"),
         quiet_workers: bool = typer.Option(False, "--quiet-workers", help="Hide worker output and only show process status"),
         debug:bool = typer.Option(False, help="Card format type"),
        max_sd_speed:float = typer.Option(140.0, help="Maximum realistic SD card read speed in MB/s (auto-detect if not specified)"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    
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
            reader_groups[reader_key].extend([card['mountpoint'] for card in group])
        worker_assignments = {
            worker_name: list(cards)
            for worker_name, cards in reader_groups.items()
            if cards
        }
        _run_parallel_import_workers(
            worker_assignments,
            clean=clean,
            config_path=config_path,
            quiet_workers=quiet_workers,
            precheck=precheck,
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
            terminal_assignments[terminal_key].append(drive)
        
        # Launch worker processes directly instead of opening terminal windows
        worker_assignments = {
            worker_name: list(drives)
            for worker_name, drives in terminal_assignments.items()
            if drives
        }
        _run_parallel_import_workers(
            worker_assignments,
            clean=clean,
            config_path=config_path,
            quiet_workers=quiet_workers,
            precheck=precheck,
        )
        
        print("✅ Import orchestration completed!")

# === AI DO NOT EDIT START ===
if __name__ == "__main__":
    sdall()
# === AI DO NOT EDIT END ===