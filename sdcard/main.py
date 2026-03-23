#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import typer
import platform
import subprocess
from sdcard.utils.file_system import list_sdcards
from sdcard.utils.cards import get_usb_info
from sdcard.utils.cards import get_card_number_from_import_yml,get_available_cards
from typer import Typer
import sys
from pathlib import Path
import os
import yaml
from sdcard.config import Config
from sdcard.utils.usb import (
    get_complete_usb_card_info, 
    scan_thunderbolt_ports, 
    detect_thunderbolt_upstream_capacity,
    analyze_destination_capacity,
    calculate_real_transfer_rate,
    get_usb_hierarchy_info,
    get_probe_destinations_info,
    get_probe_cards_info
)
from sdcard.utils.cards import get_available_cards
from collections import defaultdict
import pandas as pd
from typer.testing import CliRunner
import typer
import shlex


__author__ = "SDCard Maintainers"
__license__ = "MIT"
__version__ = "0.2"
__status__ = "Development"

sdcard = typer.Typer(
    name="SDCard manager",
    help="""SD card manager \n
        A Python CLI for managing sdcards""",
    short_help="SD Card Manager",
    no_args_is_help=True,
)



@sdcard.command('probe')
def probe(config_path: str = typer.Option(None, help="Path to config directory."),
         format_type:str = typer.Option('exfat', help="Card format type"),
         card_size:int = typer.Option(512, help="maximum card size"),
         max_sd_speed:float = typer.Option(140.0, help="Maximum realistic SD card read speed in MB/s"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    """Probe and analyze USB device tree, SD cards, and system capabilities."""
    usb_cards, available_ports, thunderbolt_bandwidth, destination_path = get_probe_cards_info(
        config_path, format_type, card_size, max_sd_speed
    )
    destinations = get_probe_destinations_info(
        usb_cards, dest_write_speed, destination_path
    )
    
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
        card['card_number'] = get_card_number_from_import_yml(Path(mountpoint) / "import.yml")
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
        
        print(f"{icon} {tb_icon} {speed_limited_icon} {config_icon} {dual_slot_icon} {card['mountpoint']} ({card['size']}) #{card['card_number']}")
        
        if not has_import_yml:
            print(f"    ❌ WARNING: Missing import.yml configuration file!")
            print(f"       This card will be SKIPPED during processing")
        
        if is_dual_slot:
            print(f"    👥 DUAL-SLOT READER: {len(cards_in_reader)} cards in same reader")
        
        if card.get('thunderbolt_connected', False):
            print(f"    ⚡ Thunderbolt: {card.get('thunderbolt_info', 'Connected')}")
        else:
            print(f"    🔌 Connection: {card.get('thunderbolt_info', 'Standard USB')}")
        if card.get('hub_manufacturer'):
            print(f"    📖 Connection: {card['hub_manufacturer']} {card['reader_product']}")
            print(f"       USB Speed: {card['hub_speed']} ({card['hub_speed_mbps']} Mbps)")
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

    print("✅ Probe analysis completed!")


@sdcard.command('register')
def register_command(
    card_path: list[str] = typer.Argument(None, help="One or more SD card mount points"),
    card_number: list[str] = typer.Option(None, help="Set card number metadata on the card"),
    config_path: str = typer.Option(None, help="Path to config directory."),
        all: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        dry_run: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        overwrite:bool = typer.Option(False, help="Overwrite import.yaml"),
    prompt_card_details: bool = typer.Option(False, "--card-details", help="Prompt for optional card manufacturer and rated UHS metadata"),
        cardsize:int = typer.Option(512, help="maximum card size"),
        format_type:str = typer.Option('exfat', help="Card format type"),
):
    """
    register sd cards
    """
    from sdcard.utils.cards import list_sdcards,register_cards  
    config = Config(config_path)
    if all and (not card_path ):
        card_path = list_sdcards(format_type,cardsize)
        if card_number is None:
            card_number = ['0'] * len(card_path)
    else:
        card_path = [Path(path) for path in card_path]
    register_cards(
        config,
        card_number=card_number,
        card_path=card_path,
        dry_run=dry_run,
        overwrite=overwrite,
        prompt_card_details=prompt_card_details,
    )

@sdcard.command('list')
def list_cards(
    format_type: str = typer.Option("exfat", help="Card format type"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show reader and speed details"),
):
    """List available SD cards"""
    if not verbose:
        cards = get_available_cards(format_type)
        for card in cards:
            typer.echo(f"Reader Card: {card['card_number']} {card['host']}: {card['mountpoint']} ({card['size_gb']}GB)")
        return cards

    usb_cards, _ = get_complete_usb_card_info(destination_path=None, format_type=format_type)
    listed_cards = []

    for _, card in usb_cards.iterrows():
        if card.get('device_type') == 'destination':
            continue

        mountpoint = card['mountpoint']
        import_yml_path = Path(mountpoint) / "import.yml"
        card_number = get_card_number_from_import_yml(import_yml_path)

        listed_card = {
            'card_number': card_number,
            'mountpoint': mountpoint,
            'size': card.get('size'),
            'fstype': card.get('fstype'),
            'reader_manufacturer': card.get('reader_manufacturer') or 'Unknown',
            'reader_product': card.get('reader_product') or 'Unknown',
            'reader_speed': card.get('reader_speed') or 'Unknown',
            'reader_speed_mbps': card.get('reader_speed_mbps') or 0,
            'actual_transfer_rate': card.get('actual_transfer_rate') or 0,
            'thunderbolt_connected': bool(card.get('thunderbolt_connected', False)),
            'thunderbolt_info': card.get('thunderbolt_info') or 'Standard USB',
        }
        listed_cards.append(listed_card)

        typer.echo(
            f"Reader Card: {card_number} {mountpoint} "
            f"({listed_card['size']}, {listed_card['fstype']})"
        )
        typer.echo(
            f"  Reader: {listed_card['reader_manufacturer']} {listed_card['reader_product']}"
        )
        typer.echo(
            f"  USB Speed: {listed_card['reader_speed']} ({listed_card['reader_speed_mbps']} Mbps)"
        )
        typer.echo(
            f"  Estimated Transfer: {listed_card['actual_transfer_rate']:.1f} MB/s"
        )
        typer.echo(
            f"  Connection: {listed_card['thunderbolt_info']}"
        )

    return listed_cards


@sdcard.command('import')
def import_command(
        card_path: list[str] = typer.Argument(None, help="One or more SD card mount points"),
        config_path: Path = typer.Option(None, help="Path to config file"),
        copy: bool = typer.Option(True, help="Copy source (default)", show_default=True),
        move: bool = typer.Option(False, help="Move source and delete after copy"),
        find: bool = typer.Option(True, help="Reuse an existing destination that matches the import token"),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite", help="Allow overwriting changed files at destination (unsafe)"),
    check: bool = typer.Option(False, "--check", help="Check all cards for conflicts only; do not copy or move"),
        card_size: int = typer.Option(512, help="Maximum card size to auto-detect"),
        format_type: str = typer.Option('exfat', help="Card format type"),
        dry_run: bool = typer.Option(False, help="Show actions without changing files"),
        file_extension: str = typer.Option("MP4", help="extension to catalog"),
        format_card: bool = typer.Option(False, help="Format drive after import and move")
    ):
    """Import SD cards to the configured card store."""
    from sdcard.utils.cards import list_sdcards
    from sdcard.utils.cards import import_cards
    config = Config(config_path)

    # Auto-discover cards if none were provided
    if not card_path:
        card_path = list_sdcards(format_type, card_size)

    import_cards(
        config=config,
        card_path=card_path,
        copy=copy,
        move=move,
        find=find,
        allow_overwrite=allow_overwrite,
        check=check,
        dry_run=dry_run,
        file_extension=file_extension,
        format_card=format_card,
    )

@sdcard.command('turbo')
def import_all(config_path: str = typer.Option(None, help="Path to config directory."),
         max_processes:int = typer.Option(4, help="Number of concurrent transfers"),
         format_type:str = typer.Option('exfat', help="Card format type"),
         card_size:int = typer.Option(512, help="maximum card size"),
         clean:bool = typer.Option(False, help="move all the files to location"),
         debug:bool = typer.Option(False, help="Card format type"),
         max_sd_speed:float = typer.Option(140.0, help="Maximum realistic SD card read speed in MB/s"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    """
    This command orchestrates the import process for ALL detected SD cards,
    launching multiple processes to handle concurrent imports.
    It supports both Linux and Windows environments.
    """
    if platform.system() == "Linux":
        processes = set()
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
            reader_groups[reader_key].extend([card['mountpoint'] for card in group])
        for cards in reader_groups.values():
            # Launch process for this reader's cards
            venv_path = Path(sys.prefix)
            cards_str = " ".join(cards)
            clean_flag = " --clean" if clean else ""
            cmd = f'gnome-terminal -- bash --login -c "export PATH={venv_path}/bin:$PATH && sdcard import {cards_str}{clean_flag}"'
            proc = subprocess.Popen(shlex.split(cmd))
            processes.add(proc)
            print(f"   ✅ Launched import process for {len(cards)} cards")        
    else:
        # Windows version - use existing get_available_cards function
        processes = set()
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
        
        # Launch terminals with their assigned drives
        for terminal_key, assigned_drives in terminal_assignments.items():
            if not assigned_drives:
                continue

            if debug:
                # In debug mode, process sequentially
                for drive in assigned_drives:
                    runner.invoke(sdcard, ["import", drive] + (['--clean'] if clean else []))
            else:
                # Launch 64-bit PowerShell terminal with all assigned drives
                drives_str = " ".join(assigned_drives)
                clean_flag = " --clean" if clean else ""

                # Get current Python executable path and venv path
                python_exe = sys.executable
                venv_path = sys.prefix
                powershell_64 = r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"

                # Detect conda environment
                conda_env = os.environ.get('CONDA_DEFAULT_ENV')
                if conda_env:
                    # Use conda activate
                    python_cmd = f"python -m sdcard import {drives_str}{clean_flag}"
                    ps_command = f"conda activate {conda_env}; {python_cmd}"
                else:
                    # Use venv activation
                    activate_ps1 = os.path.join(venv_path, 'Scripts', 'Activate.ps1')
                    python_cmd = f"python -m sdcard import {drives_str}{clean_flag}"
                    ps_command = f"& '{activate_ps1}'; {python_cmd}"

                cmd = [
                    powershell_64,
                    "-NoExit",
                    "-Command",
                    ps_command
                ]
                print("[DEBUG] PowerShell command:", cmd)
                proc = subprocess.Popen(cmd)
                processes.add(proc)
                print(f"   ✅ Launched 64-bit PowerShell terminal for {len(assigned_drives)} drives")

        # Don't wait for PowerShell processes - they run independently for better performance
        if processes:
            print(f"🚀 Launched {len(processes)} PowerShell terminals")
            print("💡 Check the PowerShell windows to monitor import progress")
            print("⚡ Processes are running independently for maximum performance")
        else:
            print("❌ No import processes were launched")
        
        print("✅ Import orchestration completed!")


if __name__ == "__main__":
    sdcard()
