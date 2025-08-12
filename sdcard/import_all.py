from unicodedata import name
import pandas as pd
import subprocess
import shlex
from typer.testing import CliRunner
import typer
import platform
import psutil
import sys
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
from sdcard.utils.cards import get_available_cards
from collections import defaultdict

sdall = typer.Typer(
    name="Import all SD cards",
)

@sdall.command('probe')
def probe(config_path: str = typer.Option(None, help="Root path to MarImBA collection."),
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
def import_all(config_path: str = typer.Option(None, help="Root path to MarImBA collection."),
         max_processes:int = typer.Option(4, help="Number of concurrent transfers"),
         format_type:str = typer.Option('exfat', help="Card format type"),
         card_size:int = typer.Option(512, help="maximum card size"),
         clean:bool = typer.Option(False, help="move all the files to location"),
         debug:bool = typer.Option(False, help="Card format type"),
         max_sd_speed:float = typer.Option(140.0, help="Maximum realistic SD card read speed in MB/s"),
         dest_write_speed:float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")):
    
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
                # Launch PowerShell terminal with all assigned drives
                drives_str = " ".join(assigned_drives)
                clean_flag = " --clean" if clean else ""
                
                # Get current Python executable path
                python_exe = sys.executable
                
                cmd = [
                    "powershell",
                    "-Command",
                    f"Start-Process powershell -ArgumentList '-NoExit -Command \"& \\\"{python_exe}\\\" -m sdcard import {drives_str}{clean_flag}\"'"
                ]
                proc = subprocess.Popen(cmd)
                processes.add(proc)
                print(f"   ✅ Launched PowerShell terminal for {len(assigned_drives)} drives")

        # Don't wait for PowerShell processes - they run independently for better performance
        if processes:
            print(f"🚀 Launched {len(processes)} PowerShell terminals")
            print("💡 Check the PowerShell windows to monitor import progress")
            print("⚡ Processes are running independently for maximum performance")
        else:
            print("❌ No import processes were launched")
        
        print("✅ Import orchestration completed!")

# === AI DO NOT EDIT START ===
if __name__ == "__main__":
    sdall()
# === AI DO NOT EDIT END ===