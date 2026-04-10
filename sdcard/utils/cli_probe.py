import typer
import platform
from typer.testing import CliRunner
from pathlib import Path
from collections import defaultdict
from sdcard.config import Config
from sdcard.utils.cards_discovery import get_available_cards
from sdcard.utils.cli_defaults import (
    DEFAULT_CARD_SIZE,
    DEFAULT_FORMAT_TYPE,
    DEFAULT_MAX_SD_SPEED,
)
from sdcard.utils.usb import (
    get_complete_usb_card_info, scan_thunderbolt_ports, detect_thunderbolt_upstream_capacity,
    analyze_destination_capacity, get_usb_hierarchy_info, get_probe_destinations_info, get_probe_cards_info
)
import yaml

def probe(
    config_path: str = typer.Option(None, help="Path to config directory."),
    format_type: str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
    card_size: int = typer.Option(DEFAULT_CARD_SIZE, help="maximum card size"),
    max_sd_speed: float = typer.Option(DEFAULT_MAX_SD_SPEED, help="Maximum realistic SD card read speed in MB/s"),
    dest_write_speed: float = typer.Option(None, help="Override destination drive write speed in MB/s (auto-detect if not specified)")
):
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
    print("🔍 Port Analysis:")
    print("=" * 60)
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
    print("📱 SD Card Analysis:")
    print("=" * 60)
    cards_without_config = []
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
        if card.get('device_type') == 'destination':
            print(f"💽 🎯 {card['mountpoint']} ({card['size']}) - DESTINATION DRIVE")
            print(f"    💾 Device: {card['name']}")
            print(f"    📂 Filesystem: {card['fstype']}")
            print(f"    🎯 This is where SD card content will be stored")
            if card.get('reader_manufacturer'):
                print(f"    📖 Connection: {card['reader_manufacturer']} {card['reader_product']}")
                print(f"       USB Speed: {card['reader_speed']} ({card['reader_speed_mbps']} Mbps)")
                usb_transfer_rate = card.get('actual_transfer_rate', 0)
                print(f"    ⚡ USB Transfer Rate: {usb_transfer_rate:.1f} MB/s")
            else:
                print(f"    🔌 Connection: Internal/SATA (not USB)")
            if card.get('thunderbolt_connected', False):
                print(f"    ⚡ Thunderbolt: {card.get('thunderbolt_info', 'Connected')}")
            elif card.get('thunderbolt_info'):
                print(f"    🔌 Connection: {card.get('thunderbolt_info', 'Standard connection')}")
            print()
            continue
        mountpoint = card['mountpoint']
        has_import_yml = (Path(mountpoint) / "import.yml").exists()
        card['card_number'] = 0
        if has_import_yml:
            try:
                import_yml_path = Path(mountpoint) / "import.yml"
                raw_data = yaml.safe_load(import_yml_path.read_text(encoding='utf-8'))
                card['card_number'] = raw_data['card_number'] if 'card_number' in raw_data else 0
            except Exception:
                pass
        if not has_import_yml:
            cards_without_config.append(mountpoint)
        reader_key = f"{card.get('reader_manufacturer', 'Unknown')}_{card.get('reader_product', 'Unknown')}"
        hierarchy = get_usb_hierarchy_info(card['name'])
        if 'card_reader' in hierarchy:
            reader_path = hierarchy['card_reader'].get('usb_path', 'unknown')
            reader_key = f"{reader_key}_{reader_path}"
        cards_in_reader = reader_groups.get(reader_key, [card])
        is_dual_slot = len(cards_in_reader) > 1
        usb_limited_speed = card.get('actual_transfer_rate', 0)
        is_usb_optimal = usb_limited_speed <= max([p['reader_speed_mbps'] for p in cards_in_reader if 'reader_speed_mbps' in p], default=0) * 0.9 if cards_in_reader else True
        realistic_speed = min(usb_limited_speed, max_sd_speed)
        if is_dual_slot:
            dual_slot_factor = 0.6
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
