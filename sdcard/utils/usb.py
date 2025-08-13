import pandas as pd
import subprocess
import json
import numpy as np
import os
import shlex
from pathlib import Path
import re
import platform
from sdcard.config import Config
from sdcard.utils.cards import get_available_cards

def get_usb_device_info(usb_path):
    """Get detailed info including both negotiated and theoretical speeds"""
    try:
        usb_device_path = f"/sys/bus/usb/devices/{usb_path}"
        
        if not Path(usb_device_path).exists():
            return {'path': usb_path, 'error': 'Device not found'}
        
        info = {'usb_path': usb_path}
        
        # Read device attributes
        attr_files = {
            'vendor_id': 'idVendor',
            'product_id': 'idProduct', 
            'serial': 'serial',
            'manufacturer': 'manufacturer',
            'product': 'product',
            'version': 'version',
            'device_class': 'bDeviceClass',
            'device_subclass': 'bDeviceSubClass',
            'speed': 'speed',
            'max_power': 'bMaxPower',
            'configuration': 'bConfigurationValue'
        }
        
        for key, filename in attr_files.items():
            try:
                with open(f"{usb_device_path}/{filename}", 'r') as f:
                    info[key] = f.read().strip()
            except (FileNotFoundError, PermissionError):
                info[key] = None
        
        # Get negotiated speed
        speed_raw = info.get('speed')
        if speed_raw:
            speed_mapping = {
                '1.5': {'name': 'Low Speed', 'mbps': 1.5, 'usb_version': '1.0'},
                '12': {'name': 'Full Speed', 'mbps': 12, 'usb_version': '1.1'}, 
                '480': {'name': 'High Speed', 'mbps': 480, 'usb_version': '2.0'},
                '5000': {'name': 'SuperSpeed', 'mbps': 5000, 'usb_version': '3.0'},
                '10000': {'name': 'SuperSpeed+', 'mbps': 10000, 'usb_version': '3.1'},
                '20000': {'name': 'SuperSpeed+ Gen2', 'mbps': 20000, 'usb_version': '3.2'}
            }
            
            speed_info = speed_mapping.get(speed_raw, {
                'name': f'Unknown ({speed_raw} Mbps)',
                'mbps': float(speed_raw) if speed_raw.replace('.', '').isdigit() else 0,
                'usb_version': 'Unknown'
            })
            
            info['negotiated_speed_mbps'] = speed_info['mbps']
            info['negotiated_speed_name'] = speed_info['name']
            info['usb_version'] = speed_info['usb_version']
        else:
            info['negotiated_speed_mbps'] = 0
            info['negotiated_speed_name'] = 'Unknown'
            info['usb_version'] = 'Unknown'
        
        # Create unique identifier
        vendor_id = info.get('vendor_id', '')
        product_id = info.get('product_id', '')
        serial = info.get('serial', '')
        info['unique_id'] = f"{vendor_id}:{product_id}:{serial}" if vendor_id and product_id else usb_path
        
        return info
        
    except Exception as e:
        return {'usb_path': usb_path, 'error': str(e)}

def scan_thunderbolt_ports():
    """Scan for Thunderbolt ports and controllers using multiple methods"""
    try:
        thunderbolt_ports = []
        thunderbolt_controllers = []
        
        # Method 1: Check for Thunderbolt controllers via lspci
        result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
        
        for line in result.stdout.split('\n'):
            if 'thunderbolt' in line.lower() or 'usb4' in line.lower():
                thunderbolt_controllers.append(line.strip())
        
        # Method 2: Use boltctl to get detailed Thunderbolt device info
        try:
            result = subprocess.run(['boltctl', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                current_device = None
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('●'):
                        # New device entry - save previous device first
                        if current_device is not None:
                            thunderbolt_ports.append(current_device)
                        
                        # Initialize new device
                        device_name = line.replace('●', '').strip()
                        current_device = {
                            'path': device_name,
                            'device': device_name,
                            'vendor': 'Unknown',
                            'type': 'Thunderbolt',
                            'status': 'Unknown',
                            'generation': 'Unknown',
                            'speeds': {}
                        }
                    elif current_device is not None:  # Only process if we have a current device
                        if '├─ vendor:' in line or '└─ vendor:' in line:
                            vendor = line.split('vendor:', 1)[1].strip()
                            current_device['vendor'] = vendor
                        elif '├─ name:' in line or '└─ name:' in line:
                            name = line.split('name:', 1)[1].strip()
                            current_device['device'] = name
                        elif '├─ status:' in line or '└─ status:' in line:
                            status = line.split('status:', 1)[1].strip()
                            current_device['status'] = status
                        elif '├─ generation:' in line or '└─ generation:' in line:
                            generation = line.split('generation:', 1)[1].strip()
                            current_device['generation'] = generation
                        elif 'rx speed:' in line:
                            rx_speed = line.split('rx speed:', 1)[1].strip()
                            current_device['speeds']['rx'] = rx_speed
                        elif 'tx speed:' in line:
                            tx_speed = line.split('tx speed:', 1)[1].strip()
                            current_device['speeds']['tx'] = tx_speed
                
                # Add the last device if it exists
                if current_device is not None:
                    thunderbolt_ports.append(current_device)
                    
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"DEBUG: Error parsing boltctl output: {e}")
        
        return thunderbolt_ports, thunderbolt_controllers
        
    except Exception as e:
        print(f"Error scanning Thunderbolt ports: {e}")
        return [], []

def scan_available_usb_ports():
    """Scan all available USB ports and their speeds"""
    try:
        usb_devices_path = Path("/sys/bus/usb/devices")
        available_ports = []
        
        for device_path in usb_devices_path.iterdir():
            if device_path.is_dir() and re.match(r'\d+-[\d.]+$', device_path.name):
                device_info = get_usb_device_info(device_path.name)
                if not device_info.get('error'):
                    device_class = device_info.get('device_class', '00')
                    
                    # Only include hubs and root hubs
                    if device_class == '09':  # Hub
                        available_ports.append({
                            'usb_path': device_path.name,
                            'manufacturer': device_info.get('manufacturer', 'Unknown'),
                            'product': device_info.get('product', 'Unknown'),
                            'speed_name': device_info.get('negotiated_speed_name', 'Unknown'),
                            'speed_mbps': device_info.get('negotiated_speed_mbps', 0),
                            'device_class': device_class
                        })
        
        # Sort by speed (highest first)
        available_ports.sort(key=lambda x: x['speed_mbps'], reverse=True)
        return available_ports
        
    except Exception as e:
        print(f"Error scanning USB ports: {e}")
        return []

def get_usb_hierarchy_info(device_name):
    """Get simple USB hierarchy information"""
    try:
        # Get the full device path
        result = subprocess.run(['udevadm', 'info', '-q', 'path', '-n', f'/dev/{device_name}'], 
                               capture_output=True, text=True)
        device_path = result.stdout.strip()
        
        # Extract USB hierarchy from the path
        usb_pattern = r'(\d+-[\d.]+)'
        usb_matches = re.findall(usb_pattern, device_path)
        
        # Remove duplicates while preserving order
        unique_usb_paths = []
        seen_devices = set()
        
        for usb_path in usb_matches:
            if ':' in usb_path:  # Skip interface paths
                continue
                
            device_info = get_usb_device_info(usb_path)
            device_key = device_info.get('unique_id', usb_path)
            
            if device_key not in seen_devices and not device_info.get('error'):
                unique_usb_paths.append(usb_path)
                seen_devices.add(device_key)
        
        hierarchy = {}
        
        # Simple classification
        if len(unique_usb_paths) == 1:
            hierarchy['card_reader'] = get_usb_device_info(unique_usb_paths[0])
        elif len(unique_usb_paths) >= 2:
            # Last device is usually the card reader
            hierarchy['card_reader'] = get_usb_device_info(unique_usb_paths[-1])
            # First device is usually the hub
            hierarchy['hub'] = get_usb_device_info(unique_usb_paths[0])
        
        return hierarchy
        
    except Exception as e:
        print(f"Error getting USB hierarchy for {device_name}: {e}")
        return {}

def calculate_real_transfer_rate(speed_mbps):
    """Calculate realistic transfer rate accounting for USB overhead"""
    if speed_mbps == 0:
        return 0
    
    overhead_factors = {
        1.5: 0.8,    # USB 1.0 Low Speed
        12: 0.85,    # USB 1.1 Full Speed  
        480: 0.85,   # USB 2.0 High Speed
        5000: 0.9,   # USB 3.0 SuperSpeed
        10000: 0.92, # USB 3.1 SuperSpeed+
        20000: 0.94  # USB 3.2 SuperSpeed+ Gen2
    }
    
    overhead = 0.85  # Default
    for speed_tier, factor in overhead_factors.items():
        if speed_mbps <= speed_tier:
            overhead = factor
            break
    
    return (speed_mbps / 8) * overhead

def is_thunderbolt_connected(device_name):
    """Check if a device is connected through Thunderbolt using enhanced detection"""
    try:
        # Get the full device path
        result = subprocess.run(['udevadm', 'info', '-q', 'path', '-n', f'/dev/{device_name}'], 
                               capture_output=True, text=True)
        device_path = result.stdout.strip()
        
        # Check if path contains thunderbolt indicators
        if 'thunderbolt' in device_path.lower():
            return True, 'Direct Thunderbolt connection'
        
        # Extract USB controller info from the device path
        usb_pattern = r'(\d+)-'
        controller_match = re.search(usb_pattern, device_path)
        
        if controller_match:
            controller_number = controller_match.group(1)
            
            # Check if this USB controller is Thunderbolt-based
            try:
                # Look for USB controllers in lspci output
                pci_result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
                
                # Check if any USB controller mentions Thunderbolt/USB4
                for line in pci_result.stdout.split('\n'):
                    if 'usb controller' in line.lower():
                        if ('thunderbolt' in line.lower() or 
                            'usb4' in line.lower() or
                            ('intel' in line.lower() and ('tiger lake' in line.lower() or 'ice lake' in line.lower()))):
                            
                            # This is a Thunderbolt USB controller
                            # Now check if we can detect CalDigit specifically
                            hierarchy = get_usb_hierarchy_info(device_name)
                            
                            # Look for CalDigit in the hierarchy
                            for device_type in ['hub', 'card_reader']:
                                if device_type in hierarchy:
                                    device_info = hierarchy[device_type]
                                    manufacturer = device_info.get('manufacturer', '').lower()
                                    product = device_info.get('product', '').lower()
                                    
                                    if 'caldigit' in manufacturer or 'caldigit' in product:
                                        return True, f"Connected via CalDigit Thunderbolt Hub (USB{controller_number})"
                            
                            return True, f"Connected via Thunderbolt USB controller (USB{controller_number})"
                        
            except Exception as e:
                pass
        
        return False, 'Standard USB connection'
        
    except Exception as e:
        return False, f'Error checking Thunderbolt connection: {e}'

def detect_thunderbolt_upstream_capacity():
    """Detect Thunderbolt upstream capacity from system info"""
    try:
        # Check lspci for Thunderbolt controllers and their capabilities
        result = subprocess.run(['lspci', '-vv'], capture_output=True, text=True)
        
        thunderbolt_info = {}
        
        for line in result.stdout.split('\n'):
            if 'thunderbolt' in line.lower() or 'usb4' in line.lower():
                if 'usb controller' in line.lower():
                    current_controller = line.strip()
                    
                    # Determine generation from controller name
                    if 'tiger lake' in line.lower():
                        thunderbolt_info = {
                            'controller': current_controller,
                            'generation': 'Thunderbolt 4',
                            'max_speed_gbps': 40,
                            'practical_usb_capacity_mbps': 40 * 1000 * 0.7 / 8  # 40Gbps * 70% efficiency / 8 bits per byte
                        }
                    elif 'ice lake' in line.lower():
                        thunderbolt_info = {
                            'controller': current_controller,
                            'generation': 'Thunderbolt 3',
                            'max_speed_gbps': 40,
                            'practical_usb_capacity_mbps': 40 * 1000 * 0.7 / 8
                        }
                    elif 'usb4' in line.lower():
                        thunderbolt_info = {
                            'controller': current_controller,
                            'generation': 'USB4',
                            'max_speed_gbps': 40,
                            'practical_usb_capacity_mbps': 40 * 1000 * 0.7 / 8
                        }
        
        return thunderbolt_info
        
    except Exception as e:
        return {}

def analyze_destination_capacity(usb_cards, dest_write_speed_override=None, global_destination=None):
    """Analyze destination drive capacity and identify potential bottlenecks"""
    try:
        import yaml
        destinations = {}
        
        # Filter out destination path from USB cards if it exists
        filtered_usb_cards = usb_cards.copy()
        if global_destination:
            # Find the mount point for the global destination first
            result = subprocess.run(['df', global_destination], capture_output=True, text=True)
            destination_mount = None
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    destination_mount = lines[1].split()[-1]
            
            # Remove any cards that are mounted at the destination mount point
            if destination_mount:
                filtered_usb_cards = usb_cards[usb_cards['mountpoint'] != destination_mount].copy()
        
        for _, card in filtered_usb_cards.iterrows():
            # Skip destination drives in analysis
            if card.get('device_type') == 'destination':
                continue
                
            mountpoint = card['mountpoint']
            dest_path = None
            
            # Use global destination if provided, otherwise try to read from card's import.yml
            if global_destination:
                dest_path = global_destination
            else:
                # Try to find import.yml to get destination
                import_yml_path = Path(mountpoint) / "import.yml"
                if import_yml_path.exists():
                    try:
                        with open(import_yml_path, 'r') as f:
                            config = yaml.safe_load(f)
                        dest_path = config.get('destination_path', '/unknown')
                    except Exception:
                        dest_path = '/unknown'
                else:
                    dest_path = '/unknown'
            
            if dest_path:
                # Find the mount point for this destination path
                result = subprocess.run(['df', dest_path], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        dest_device = lines[1].split()[0]
                        dest_mount = lines[1].split()[-1]
                        
                        if dest_mount not in destinations:
                            destinations[dest_mount] = {
                                'device': dest_device,
                                'mount_point': dest_mount,
                                'destination_path': dest_path,
                                'cards': [],
                                'total_demand': 0,
                                'drive_type': 'unknown',
                                'estimated_write_speed': 0,
                                'speed_override': dest_write_speed_override is not None,  # Fixed typo
                                'using_global_config': global_destination is not None
                            }
                        
                        destinations[dest_mount]['cards'].append(card)
                        destinations[dest_mount]['total_demand'] += card.get('actual_transfer_rate', 0)
        
        # Analyze each destination drive
        for dest_mount, dest_info in destinations.items():
            try:
                # Use override speed if provided
                if dest_write_speed_override is not None:
                    dest_info['estimated_write_speed'] = dest_write_speed_override
                    dest_info['drive_type'] = f"User Override ({dest_write_speed_override} MB/s)"
                else:
                    # Detect drive type and estimate write speed with minimum 200 MB/s
                    device = dest_info['device']
                    
                    if '/dev/nvme' in device:
                        dest_info['drive_type'] = 'NVMe SSD'
                        dest_info['estimated_write_speed'] = max(2000, 200)  # At least 200 MB/s
                    elif any(x in device for x in ['/dev/sd', '/dev/hd']):
                        device_name = device.split('/')[-1].rstrip('0123456789')
                        try:
                            with open(f'/sys/block/{device_name}/queue/rotational', 'r') as f:
                                is_rotational = f.read().strip() == '1'
                            
                            if is_rotational:
                                dest_info['drive_type'] = 'HDD'
                                dest_info['estimated_write_speed'] = max(150, 200)  # At least 200 MB/s
                            else:
                                dest_info['drive_type'] = 'SATA SSD'
                                dest_info['estimated_write_speed'] = max(500, 200)  # At least 200 MB/s
                        except:
                            dest_info['drive_type'] = 'Unknown SATA'
                            dest_info['estimated_write_speed'] = 200  # Minimum write speed
                    else:
                        dest_info['drive_type'] = 'Unknown'
                        dest_info['estimated_write_speed'] = 200  # Minimum write speed
                
                # Check for potential saturation
                utilization = (dest_info['total_demand'] / dest_info['estimated_write_speed'] * 100) if dest_info['estimated_write_speed'] > 0 else 0
                dest_info['utilization_percentage'] = utilization
                dest_info['is_saturated'] = utilization > 80
                
            except Exception:
                pass
        
        return destinations
        
    except Exception:
        return {}

def get_complete_usb_card_info(destination_path=None, card_size=512, format_type='exfat'):
    """Get SD cards with USB hierarchy information, filtering by size and format, plus destination mount"""
    if platform.system() == "Windows":
        try:
            import wmi
            from pywinusb import hid
            c = wmi.WMI()
            # Get all removable drives (SD cards, USB sticks, etc.)
            drives = [d for d in c.Win32_DiskDrive() if d.MediaType and ("Removable" in d.MediaType or d.InterfaceType == "USB")]
            # Map physical drives to logical disks (mountpoints)
            partitions = {p.DeviceID: p for p in c.Win32_DiskPartition()}
            logical_disks = {ld.DeviceID: ld for ld in c.Win32_LogicalDisk()}
            # Build a lookup of USB devices by serial number (using pywinusb)
            usb_devices = hid.HidDeviceFilter().get_devices()
            pywinusb_lookup = {}
            for dev in usb_devices:
                try:
                    serial = dev.serial_number
                    vendor_id = dev.vendor_id
                    product_id = dev.product_id
                    pywinusb_lookup[(vendor_id, product_id, serial)] = dev
                except Exception:
                    continue
            usb_info = []
            for drive in drives:
                # Find partitions for this drive
                drive_partitions = [p for p in c.Win32_DiskPartition() if p.DiskIndex == drive.Index]
                for part in drive_partitions:
                    # Find logical disks for this partition
                    part_logical = [ld for ld in c.Win32_LogicalDiskToPartition() if ld.Antecedent.split('=')[1].strip('"') == part.DeviceID]
                    for mapping in part_logical:
                        logical_id = mapping.Dependent.split('=')[1].strip('"')
                        logical = logical_disks.get(logical_id)
                        if not logical:
                            continue
                        # Filter by filesystem type and size
                        if logical.FileSystem and logical.FileSystem.lower() == format_type.lower():
                            size_gb = float(logical.Size) / (1024 ** 3) if logical.Size else 0
                            if size_gb <= card_size:
                                # Try to get vendor_id, product_id, serial for pywinusb lookup
                                vendor_id = None
                                product_id = None
                                serial = None
                                try:
                                    vendor_id = int(drive.PNPDeviceID.split('VID_')[1][:4], 16)
                                    product_id = int(drive.PNPDeviceID.split('PID_')[1][:4], 16)
                                    serial = drive.SerialNumber if hasattr(drive, 'SerialNumber') else None
                                except Exception:
                                    pass
                                pywinusb_dev = pywinusb_lookup.get((vendor_id, product_id, serial))
                                # pywinusb does not provide negotiated speed, but we can at least show USB version if available
                                reader_speed = None
                                reader_speed_mbps = None
                                if pywinusb_dev:
                                    try:
                                        # pywinusb does not expose speed, but we can show USB version
                                        usb_version = pywinusb_dev.device_release_number
                                        if usb_version >= 0x0300:
                                            reader_speed = 'SuperSpeed'
                                            reader_speed_mbps = 5000
                                        elif usb_version >= 0x0200:
                                            reader_speed = 'High Speed'
                                            reader_speed_mbps = 480
                                        elif usb_version >= 0x0110:
                                            reader_speed = 'Full Speed'
                                            reader_speed_mbps = 12
                                        else:
                                            reader_speed = 'Low Speed'
                                            reader_speed_mbps = 1.5
                                    except Exception:
                                        pass
                                card_info = {
                                    'name': drive.DeviceID,
                                    'mountpoint': logical.DeviceID,
                                    'size': f"{size_gb:.1f}G",
                                    'fstype': logical.FileSystem,
                                    'device_type': 'sd_card',
                                    'reader_manufacturer': getattr(drive, 'Manufacturer', None),
                                    'reader_product': getattr(drive, 'Model', None),
                                    'reader_speed': reader_speed,
                                    'reader_speed_mbps': reader_speed_mbps,
                                    'actual_transfer_rate': None,
                                    'thunderbolt_connected': None,
                                    'thunderbolt_info': None,
                                }
                                usb_info.append(card_info)
            # Available ports: list USB hubs
            available_ports = []
            for hub in c.Win32_USBHub():
                available_ports.append({
                    'usb_path': hub.DeviceID,
                    'manufacturer': getattr(hub, 'Manufacturer', 'Unknown'),
                    'product': getattr(hub, 'Name', 'Unknown'),
                    'speed_name': None,
                    'speed_mbps': None,
                    'device_class': None
                })
            return pd.DataFrame(usb_info), available_ports
        except ImportError:
            print("wmi or pywinusb module not installed. Please install wmi and pywinusb for Windows support.")
            return pd.DataFrame([]), []
        except Exception as e:
            print(f"Error in Windows USB info: {e}")
            return pd.DataFrame([]), []
    # Get mounted devices
    mountpoints = pd.DataFrame(json.loads(subprocess.getoutput('lsblk -J -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT'))['blockdevices'])
    mountpoints = mountpoints[~mountpoints.children.isna()]
    mountpoints = pd.DataFrame(mountpoints.children.apply(lambda x: x[0]).to_list())[['name','mountpoint','fstype','size']]
    
    # Filter for SD cards: specific format and size limit
    sd_cards = mountpoints.loc[
        (mountpoints.fstype == format_type.lower()) & 
        (~mountpoints.mountpoint.isna())
    ]
    
    # Filter by card size (convert size string to GB and check)
    def parse_size_to_gb(size_str):
        """Convert size string like '64G', '512M' to GB"""
        if not size_str:
            return 0
        size_str = size_str.upper()
        if size_str.endswith('G'):
            return float(size_str[:-1])
        elif size_str.endswith('M'):
            return float(size_str[:-1]) / 1024
        elif size_str.endswith('T'):
            return float(size_str[:-1]) * 1024
        return 0
    
    # Filter SD cards by size
    sd_cards = sd_cards[sd_cards['size'].apply(lambda x: parse_size_to_gb(x) <= card_size)]
    
    # Add destination mount if provided
    destination_mount = None
    if destination_path:
        try:
            result = subprocess.run(['df', destination_path], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    dest_device = lines[1].split()[0]
                    dest_mount_point = lines[1].split()[-1]
                    
                    # Find the destination in mountpoints (any filesystem type)
                    dest_info = mountpoints[mountpoints['mountpoint'] == dest_mount_point]
                    if not dest_info.empty:
                        destination_mount = dest_info.iloc[0]
        except Exception as e:
            print(f"DEBUG: Error finding destination mount: {e}")
    
    # Scan available USB ports
    available_ports = scan_available_usb_ports()
    max_available_speed = max([port['speed_mbps'] for port in available_ports]) if available_ports else 0
    
    usb_info = []
    
    # Process SD cards
    for _, row in sd_cards.iterrows():
        device_name = row['name']
        hierarchy = get_usb_hierarchy_info(device_name)
        
        # Add Thunderbolt detection
        is_tb, tb_info = is_thunderbolt_connected(device_name)
        
        card_info = {
            'name': device_name,
            'mountpoint': row['mountpoint'],
            'size': row['size'],
            'fstype': row['fstype'],
            'device_type': 'sd_card',
            'thunderbolt_connected': is_tb,
            'thunderbolt_info': tb_info,
        }
        
        # Get reader info
        if 'card_reader' in hierarchy:
            reader = hierarchy['card_reader']
            card_info.update({
                'reader_manufacturer': reader.get('manufacturer'),
                'reader_product': reader.get('product'),
                'reader_speed': reader.get('negotiated_speed_name'),
                'reader_speed_mbps': reader.get('negotiated_speed_mbps', 0),
                'actual_transfer_rate': calculate_real_transfer_rate(reader.get('negotiated_speed_mbps', 0))
            })
        
        # Get hub info
        if 'hub' in hierarchy:
            hub = hierarchy['hub']
            card_info.update({
                'hub_manufacturer': hub.get('manufacturer'),
                'hub_product': hub.get('product'),
                'hub_speed': hub.get('negotiated_speed_name'),
                'hub_speed_mbps': hub.get('negotiated_speed_mbps', 0),
            })
        
        # Check if using optimal port
        current_speed = card_info.get('reader_speed_mbps', 0)
        if 'hub_speed_mbps' in card_info:
            current_speed = min(current_speed, card_info['hub_speed_mbps'])
        
        card_info['is_optimal_port'] = current_speed >= max_available_speed * 0.9  # Within 10% of max
        card_info['max_available_speed'] = max_available_speed
        card_info['speed_efficiency'] = (current_speed / max_available_speed * 100) if max_available_speed > 0 else 0
        
        usb_info.append(card_info)
    
    # Add destination mount info if found
    if destination_mount is not None:
        # Get USB hierarchy info for destination drive too
        dest_device_name = destination_mount['name']
        dest_hierarchy = get_usb_hierarchy_info(dest_device_name)
        
        # Add Thunderbolt detection for destination
        dest_is_tb, dest_tb_info = is_thunderbolt_connected(dest_device_name)
        
        dest_info = {
            'name': destination_mount['name'],
            'mountpoint': destination_mount['mountpoint'],
            'size': destination_mount['size'],
            'fstype': destination_mount['fstype'],
            'device_type': 'destination',
            'thunderbolt_connected': dest_is_tb,
            'thunderbolt_info': dest_tb_info,
        }
        
        # Get reader/connection info for destination drive
        if 'card_reader' in dest_hierarchy:
            reader = dest_hierarchy['card_reader']
            dest_info.update({
                'reader_manufacturer': reader.get('manufacturer', 'Unknown'),
                'reader_product': reader.get('product', 'External Drive'),
                'reader_speed': reader.get('negotiated_speed_name', 'Unknown'),
                'reader_speed_mbps': reader.get('negotiated_speed_mbps', 0),
                'actual_transfer_rate': calculate_real_transfer_rate(reader.get('negotiated_speed_mbps', 0))
            })
        else:
            # No USB hierarchy found - likely internal drive
            dest_info.update({
                'reader_manufacturer': None,
                'reader_product': 'Internal Drive',
                'reader_speed': 'Internal/SATA',
                'reader_speed_mbps': 0,
                'actual_transfer_rate': 0
            })
        
        # Get hub info for destination if applicable
        if 'hub' in dest_hierarchy:
            hub = dest_hierarchy['hub']
            dest_info.update({
                'hub_manufacturer': hub.get('manufacturer'),
                'hub_product': hub.get('product'),
                'hub_speed': hub.get('negotiated_speed_name'),
                'hub_speed_mbps': hub.get('negotiated_speed_mbps', 0),
            })
        
        # Check if using optimal port
        current_speed = dest_info.get('reader_speed_mbps', 0)
        if 'hub_speed_mbps' in dest_info:
            current_speed = min(current_speed, dest_info['hub_speed_mbps'])
        
        dest_info['is_optimal_port'] = current_speed >= max_available_speed * 0.9 if max_available_speed > 0 else True
        dest_info['max_available_speed'] = max_available_speed
        dest_info['speed_efficiency'] = (current_speed / max_available_speed * 100) if max_available_speed > 0 else 0
        
        usb_info.append(dest_info)
    
    return pd.DataFrame(usb_info), available_ports


def get_probe_cards_info(config_path, format_type, card_size, max_sd_speed):
    """
    Returns (usb_cards_df, available_ports, thunderbolt_bandwidth, destination_path)
    - usb_cards_df: DataFrame with all card info (same columns on both OS)
    - available_ports: list/dict of available USB ports (if needed)
    - thunderbolt_bandwidth: dict or None
    - destination_path: str or Path (Linux only, None on Windows)
    """
    config = Config(config_path)
    destination_path = config.get_path('card_store')
    usb_cards, available_ports = get_complete_usb_card_info(destination_path, card_size, format_type)
    thunderbolt_bandwidth = detect_thunderbolt_upstream_capacity()
    # Cap transfer rates
    for i, row in usb_cards.iterrows():
        usb_cards.at[i, 'actual_transfer_rate'] = min(row['actual_transfer_rate'], max_sd_speed)
        usb_cards.at[i, 'realistic_single_speed'] = min(row['actual_transfer_rate'], max_sd_speed)
    return usb_cards, available_ports, thunderbolt_bandwidth, destination_path

def get_probe_destinations_info(usb_cards, dest_write_speed, destination_path):
    """
    Returns a dict of destination drive analysis (same keys on both OS)
    """
    if platform.system() == "Linux":
        return analyze_destination_capacity(usb_cards, dest_write_speed, destination_path)
    else:
        # Windows: fake a similar structure for parity
        destinations = {}
        for _, card in usb_cards.iterrows():
            if card.get('device_type') == 'destination':
                destinations[card['mountpoint']] = {
                    'drive_type': 'destination',
                    'device': card.get('name', ''),
                    'destination_path': card.get('mountpoint', ''),
                    'using_global_config': True,
                    'speed_override': dest_write_speed is not None,
                    'estimated_write_speed': dest_write_speed or 200.0,
                    'total_demand': card.get('actual_transfer_rate', 0),
                    'utilization_percentage': 0.0,
                    'is_saturated': False,
                    'cards': [card],
                }
        return destinations