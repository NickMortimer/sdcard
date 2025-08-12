import ffmpeg
import struct
import io
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import subprocess
import re
import json


def extract_gopro_binary_gpmf(video_path, output_bin=None):
    """
    Extract GPMF binary data from GoPro video files
    
    Args:
        video_path (str/Path): Path to GoPro video file
        output_bin (str/Path, optional): Path to save binary output
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_bin = video_path.with_suffix('.bin') if output_bin is None else Path(output_bin)
    if not output_bin.exists():
        try:
            # Find GPMF stream
            probe = ffmpeg.probe(str(video_path))
            gpmf_stream = None

            for i, stream in enumerate(probe["streams"]):
                if stream.get("codec_tag_string") == "gpmd":
                    gpmf_stream = i
                    break

            if gpmf_stream is None:
                print("No GPMF stream found in video file")
                return False

            # Extract GPMF data
            (
                ffmpeg
                .input(str(video_path))
                .output(str(output_bin), map=f'0:{gpmf_stream}', codec='copy', format='rawvideo')
                .overwrite_output()
                .run(quiet=True)
            )

            return True

        except Exception as e:
            print(f"Error extracting GPMF data: {e}")
            return False
    else:
        print(f"Output file already exists: {output_bin}")
        return True

def extract_gopro_sensor_data(video_path, output_csv=None, sensor_types=['gps', 'accel', 'temp']):
    """
    Extract sensor data (GPS, accelerometer, temperature) from GoPro video files
    
    Args:
        video_path (str/Path): Path to GoPro video file
        output_csv (str/Path, optional): Path to save CSV output
        sensor_types (list): List of sensors to extract ['gps', 'accel', 'temp']
        
    Returns:
        dict: Dictionary with DataFrames for each sensor type
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # First extract the binary GPMF data
    bin_file = video_path.with_suffix('.bin')
    extract_gopro_binary_gpmf(video_path, bin_file)
    
    try:
        # Parse sensor data from binary
        sensor_data = parse_gpmf_sensor_data(bin_file, sensor_types)
        
        # Save to CSV files if specified
        if output_csv:
            output_path = Path(output_csv)
            for sensor_type, data in sensor_data.items():
                if not data.empty:
                    sensor_csv = output_path.with_stem(f"{output_path.stem}_{sensor_type}")
                    data.to_csv(sensor_csv, index=False)
                    print(f"{sensor_type.upper()} data saved to {sensor_csv}")
        
        return sensor_data
        
    except Exception as e:
        print(f"Error extracting sensor data: {e}")
        return {}

def parse_gpmf_sensor_data(bin_file, sensor_types=['gps', 'accel', 'temp']):
    """Parse GPMF binary data to extract multiple sensor types"""
    gps_data = [] if 'gps' in sensor_types else None
    accel_data = [] if 'accel' in sensor_types else None
    temp_data = [] if 'temp' in sensor_types else None
    gyro_data = [] if 'gyro' in sensor_types else None
    
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        
        stream = io.BytesIO(data)
        
        while True:
            # Read GPMF header
            header = stream.read(8)
            if len(header) < 8:
                break
            
            tag = header[:4].decode('ascii', errors='ignore')
            type_char = chr(header[4]) if header[4] < 128 else '?'
            size = header[5]
            repeat = struct.unpack('>H', header[6:8])[0]
            
            # Calculate total data size
            total_bytes = repeat * size
            data_bytes = stream.read(total_bytes)
            
            # DEVC is a container - we need to parse its contents
            if tag == 'DEVC':
                parse_devc_container(data_bytes, gps_data, accel_data, temp_data, gyro_data)
            
            # Also check for direct sensor tags (in case they're not in DEVC)
            elif tag == 'GPS5' and gps_data is not None:
                parse_gps5_data(data_bytes, size, repeat, gps_data)
            elif tag == 'GPSU' and gps_data is not None:
                parse_gps_timestamp(data_bytes, gps_data)
            elif tag == 'GPSF' and gps_data is not None:
                parse_gps_fix(data_bytes, gps_data)
            elif tag == 'GPSP' and gps_data is not None:
                parse_gps_precision(data_bytes, gps_data)
            elif tag == 'ACCL' and accel_data is not None:
                parse_accelerometer_data(data_bytes, size, repeat, accel_data)
            elif tag == 'GYRO' and gyro_data is not None:
                parse_gyroscope_data(data_bytes, size, repeat, gyro_data)
            elif tag == 'TMPC' and temp_data is not None:
                parse_temperature_data(data_bytes, size, repeat, temp_data)
            
            # Align to 4-byte boundary
            pos = stream.tell()
            if pos % 4 != 0:
                stream.seek(pos + (4 - pos % 4))
    
    except Exception as e:
        print(f"Error parsing GPMF sensor data: {e}")
        return {}
    
    # Convert to DataFrames (same as before)
    result = {}
    
    if gps_data is not None and gps_data:
        df = pd.DataFrame(gps_data)
        df = df.sort_values('sample_index').reset_index(drop=True)
        if len(df) > 1:
            df['relative_time'] = df.index / 10.0  # 10Hz sample rate
        else:
            df['relative_time'] = 0.0
        result['gps'] = df[['sample_index', 'relative_time', 'utc_time', 'latitude', 'longitude', 
                           'altitude', 'speed_2d', 'speed_3d', 'gps_fix', 'precision']]
    
    if accel_data is not None and accel_data:
        df = pd.DataFrame(accel_data)
        df = df.sort_values('sample_index').reset_index(drop=True)
        if len(df) > 1:
            df['relative_time'] = df.index / 200.0  # 200Hz sample rate for accelerometer
        else:
            df['relative_time'] = 0.0
        result['accel'] = df[['sample_index', 'relative_time', 'x_accel', 'y_accel', 'z_accel']]
    
    if gyro_data is not None and gyro_data:
        df = pd.DataFrame(gyro_data)
        df = df.sort_values('sample_index').reset_index(drop=True)
        if len(df) > 1:
            df['relative_time'] = df.index / 200.0  # 200Hz sample rate for gyroscope
        else:
            df['relative_time'] = 0.0
        result['gyro'] = df[['sample_index', 'relative_time', 'x_gyro', 'y_gyro', 'z_gyro']]
    
    if temp_data is not None and temp_data:
        df = pd.DataFrame(temp_data)
        df = df.sort_values('sample_index').reset_index(drop=True)
        if len(df) > 1:
            df['relative_time'] = df.index / 1.0  # 1Hz sample rate for temperature
        else:
            df['relative_time'] = 0.0
        result['temp'] = df[['sample_index', 'relative_time', 'temperature_c']]
    
    return result

def parse_devc_container(data_bytes, gps_data, accel_data, temp_data, gyro_data):
    """Parse DEVC container contents to find sensor data"""
    stream = io.BytesIO(data_bytes)
    
    while True:
        # Read nested GPMF header
        header = stream.read(8)
        if len(header) < 8:
            break
        
        tag = header[:4].decode('ascii', errors='ignore')
        type_char = chr(header[4]) if header[4] < 128 else '?'
        size = header[5]
        repeat = struct.unpack('>H', header[6:8])[0]
        
        # Calculate total data size
        total_bytes = repeat * size
        nested_data = stream.read(total_bytes)
        
        # Parse sensor data found inside DEVC
        if tag == 'STRM':
            # STRM is another container level - parse its contents
            parse_strm_container(nested_data, gps_data, accel_data, temp_data, gyro_data)
        elif tag == 'GPS5' and gps_data is not None:
            parse_gps5_data(nested_data, size, repeat, gps_data)
        elif tag == 'GPSU' and gps_data is not None:
            parse_gps_timestamp(nested_data, gps_data)
        elif tag == 'GPSF' and gps_data is not None:
            parse_gps_fix(nested_data, gps_data)
        elif tag == 'GPSP' and gps_data is not None:
            parse_gps_precision(nested_data, gps_data)
        elif tag == 'ACCL' and accel_data is not None:
            parse_accelerometer_data(nested_data, size, repeat, accel_data)
        elif tag == 'GYRO' and gyro_data is not None:
            parse_gyroscope_data(nested_data, size, repeat, gyro_data)
        elif tag == 'TMPC' and temp_data is not None:
            parse_temperature_data(nested_data, size, repeat, temp_data)
        
        # Align to 4-byte boundary
        pos = stream.tell()
        if pos % 4 != 0:
            stream.seek(pos + (4 - pos % 4))

def parse_strm_container(data_bytes, gps_data, accel_data, temp_data, gyro_data):
    """Parse STRM container contents (another level of nesting)"""
    stream = io.BytesIO(data_bytes)
    
    while True:
        # Read nested GPMF header
        header = stream.read(8)
        if len(header) < 8:
            break
        
        tag = header[:4].decode('ascii', errors='ignore')
        type_char = chr(header[4]) if header[4] < 128 else '?'
        size = header[5]
        repeat = struct.unpack('>H', header[6:8])[0]
        
        # Calculate total data size
        total_bytes = repeat * size
        nested_data = stream.read(total_bytes)
        
        # Parse actual sensor data
        if tag == 'GPS5' and gps_data is not None:
            parse_gps5_data(nested_data, size, repeat, gps_data)
        elif tag == 'GPSU' and gps_data is not None:
            parse_gps_timestamp(nested_data, gps_data)
        elif tag == 'GPSF' and gps_data is not None:
            parse_gps_fix(nested_data, gps_data)
        elif tag == 'GPSP' and gps_data is not None:
            parse_gps_precision(nested_data, gps_data)
        elif tag == 'ACCL' and accel_data is not None:
            parse_accelerometer_data(nested_data, size, repeat, accel_data)
        elif tag == 'GYRO' and gyro_data is not None:
            parse_gyroscope_data(nested_data, size, repeat, gyro_data)
        elif tag == 'TMPC' and temp_data is not None:
            parse_temperature_data(nested_data, size, repeat, temp_data)
        
        # Align to 4-byte boundary
        pos = stream.tell()
        if pos % 4 != 0:
            stream.seek(pos + (4 - pos % 4))

def parse_accelerometer_data(data_bytes, size, repeat, accel_data):
    """Parse accelerometer data (ACCL tag)"""
    sample_size = 6  # 3 * 2 bytes (3 int16 values: x, y, z)
    
    for i in range(0, len(data_bytes), sample_size):
        if i + sample_size <= len(data_bytes):
            # Unpack 3 signed 16-bit integers
            x_raw, y_raw, z_raw = struct.unpack('>hhh', data_bytes[i:i+sample_size])
            
            # Convert to g-force (scale factor varies by GoPro model)
            # Typical scale factors: Hero 8/9/10: 8192 LSB/g, Hero 7: 4096 LSB/g
            scale_factor = 8192.0  # Adjust based on your GoPro model
            
            x_accel = x_raw / scale_factor
            y_accel = y_raw / scale_factor
            z_accel = z_raw / scale_factor
            
            accel_data.append({
                'sample_index': len(accel_data),
                'x_accel': x_accel,
                'y_accel': y_accel,
                'z_accel': z_accel
            })

def parse_gyroscope_data(data_bytes, size, repeat, gyro_data):
    """Parse gyroscope data (GYRO tag)"""
    sample_size = 6  # 3 * 2 bytes (3 int16 values: x, y, z)
    
    for i in range(0, len(data_bytes), sample_size):
        if i + sample_size <= len(data_bytes):
            # Unpack 3 signed 16-bit integers
            x_raw, y_raw, z_raw = struct.unpack('>hhh', data_bytes[i:i+sample_size])
            
            # Convert to degrees per second (scale factor varies by GoPro model)
            # Typical scale factor: 16.4 LSB/(°/s) for ±2000°/s range
            scale_factor = 16.4
            
            x_gyro = x_raw / scale_factor
            y_gyro = y_raw / scale_factor
            z_gyro = z_raw / scale_factor
            
            gyro_data.append({
                'sample_index': len(gyro_data),
                'x_gyro': x_gyro,
                'y_gyro': y_gyro,
                'z_gyro': z_gyro
            })

def parse_temperature_data(data_bytes, size, repeat, temp_data):
    """Parse camera temperature data (TMPC tag)"""
    sample_size = 2  # 2 bytes (1 int16 value)
    
    for i in range(0, len(data_bytes), sample_size):
        if i + sample_size <= len(data_bytes):
            # Unpack signed 16-bit integer
            temp_raw = struct.unpack('>h', data_bytes[i:i+sample_size])[0]
            
            # Convert to Celsius (typical scale: raw value in 1/256 degrees C)
            temperature_c = temp_raw / 256.0
            
            temp_data.append({
                'sample_index': len(temp_data),
                'temperature_c': temperature_c
            })

def extract_gopro_accelerometer_data(video_path, output_csv=None):
    """Extract only accelerometer data from GoPro video"""
    sensor_data = extract_gopro_sensor_data(video_path, output_csv, ['accel'])
    return sensor_data.get('accel', pd.DataFrame())

def extract_gopro_temperature_data(video_path, output_csv=None):
    """Extract only temperature data from GoPro video"""
    sensor_data = extract_gopro_sensor_data(video_path, output_csv, ['temp'])
    return sensor_data.get('temp', pd.DataFrame())

def batch_extract_sensor_data(video_directory, pattern="*.MP4", sensor_types=['gps', 'accel', 'temp']):
    """
    Extract sensor data from all GoPro videos in a directory
    
    Args:
        video_directory (str/Path): Directory containing GoPro videos
        pattern (str): File pattern to match (default: "*.MP4")
        sensor_types (list): List of sensors to extract
    
    Returns:
        dict: Dictionary mapping video filenames to sensor data dictionaries
    """
    video_dir = Path(video_directory)
    sensor_results = {}
    
    for video_file in video_dir.glob(pattern):
        print(f"Processing sensor data from: {video_file.name}")
        
        try:
            csv_output = video_file.with_suffix('.csv')
            sensor_data = extract_gopro_sensor_data(video_file, csv_output, sensor_types)
            
            if sensor_data:
                sensor_results[video_file.name] = sensor_data
                for sensor_type, data in sensor_data.items():
                    print(f"  Extracted {len(data)} {sensor_type.upper()} samples")
            else:
                print(f"  No sensor data found")
                
        except Exception as e:
            print(f"  Error processing {video_file}: {e}")
    
    return sensor_results

# Update the existing extract_gopro_gps_data function to use the new system
def extract_gopro_gps_data(video_path, output_csv=None):
    """Extract only GPS data from GoPro video (backward compatibility)"""
    sensor_data = extract_gopro_sensor_data(video_path, output_csv, ['gps'])
    return sensor_data.get('gps', pd.DataFrame())

# Get SD card speed class and performance information
def get_sd_card_speed_info(device_name):
    """Get SD card speed class and performance information (READ-ONLY)"""
    try:
        # Method 1: Use hdparm for READ-ONLY speed testing
        # -t tests READ speed only (no writes)
        # -T tests cache reads (completely safe)
        result = subprocess.run(['hdparm', '-t', f'/dev/{device_name}'], 
                               capture_output=True, text=True, timeout=30)
        
        hdparm_info = {}
        for line in result.stdout.split('\n'):
            if 'Timing buffered disk reads:' in line:
                # Extract MB/sec from line like "Timing buffered disk reads: 284 MB in 3.01 seconds = 94.35 MB/sec"
                match = re.search(r'= ([\d.]+) MB/sec', line)
                if match:
                    hdparm_info['measured_read_speed'] = float(match.group(1))
        
        # Method 2: Check for SD card specific information using udevadm (READ-ONLY)
        result2 = subprocess.run(['udevadm', 'info', '-q', 'all', '-n', f'/dev/{device_name}'], 
                                capture_output=True, text=True)
        
        sd_info = {}
        for line in result2.stdout.split('\n'):
            if 'ID_SERIAL_SHORT=' in line:
                sd_info['serial'] = line.split('=', 1)[1]
            elif 'ID_MODEL=' in line:
                sd_info['model'] = line.split('=', 1)[1]
            elif 'ID_VENDOR=' in line:
                sd_info['vendor'] = line.split('=', 1)[1]
        
        # Method 3: Estimate speed class from model name patterns (READ-ONLY)
        model = sd_info.get('model', '').upper()
        speed_class = 'Unknown'
        estimated_min_write = 0
        
        # Common SD card speed class patterns
        if 'U3' in model or 'V30' in model:
            speed_class = 'U3/V30'
            estimated_min_write = 30  # MB/s minimum
        elif 'U1' in model or 'V10' in model or 'CLASS 10' in model:
            speed_class = 'U1/V10/Class 10'
            estimated_min_write = 10  # MB/s minimum
        elif 'V60' in model:
            speed_class = 'V60'
            estimated_min_write = 60  # MB/s minimum
        elif 'V90' in model:
            speed_class = 'V90'
            estimated_min_write = 90  # MB/s minimum
        elif 'EXTREME' in model or 'PRO' in model:
            speed_class = 'High Performance'
            estimated_min_write = 30  # Estimated
        elif 'ULTRA' in model:
            speed_class = 'Ultra'
            estimated_min_write = 10  # Estimated
        
        # Method 4: Get filesystem info for additional clues (READ-ONLY)
        result3 = subprocess.run(['lsblk', '-J', '-o', 'NAME,SIZE,FSTYPE,LABEL', f'/dev/{device_name}'], 
                                capture_output=True, text=True)
        
        fs_info = {}
        try:
            lsblk_data = json.loads(result3.stdout)
            if lsblk_data.get('blockdevices'):
                device = lsblk_data['blockdevices'][0]
                fs_info['label'] = device.get('label', '')
                fs_info['fstype'] = device.get('fstype', '')
                fs_info['size'] = device.get('size', '')
        except:
            pass
        
        # Method 5: Alternative safe read speed test using dd (READ-ONLY)
        # Read a small amount from the beginning of the device
        dd_speed = 0
        try:
            # Read 100MB from device (READ-ONLY, no writing)
            dd_result = subprocess.run([
                'dd', f'if=/dev/{device_name}', 'of=/dev/null', 
                'bs=1M', 'count=100', 'iflag=direct'
            ], capture_output=True, text=True, timeout=60)
            
            # Parse dd output for speed
            for line in dd_result.stderr.split('\n'):
                if 'MB/s' in line:
                    match = re.search(r'([\d.]+) MB/s', line)
                    if match:
                        dd_speed = float(match.group(1))
        except:
            pass
        
        return {
            'measured_read_speed': hdparm_info.get('measured_read_speed', dd_speed),
            'dd_read_speed': dd_speed,  # Alternative measurement
            'estimated_speed_class': speed_class,
            'estimated_min_write_speed': estimated_min_write,
            'card_model': sd_info.get('model', 'Unknown'),
            'card_vendor': sd_info.get('vendor', 'Unknown'),
            'card_serial': sd_info.get('serial', 'Unknown'),
            'card_label': fs_info.get('label', ''),
            'card_size': fs_info.get('size', ''),
            'test_method': 'READ-ONLY (hdparm -t)',
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'measured_read_speed': 0,
            'estimated_speed_class': 'Unknown',
            'estimated_min_write_speed': 0,
            'test_method': 'ERROR'
        }

def get_complete_usb_card_info():
    """Get SD cards with USB hierarchy information, port optimization warnings, and SD card speeds (SAFE)"""
    # Get mounted devices
    mountpoints = pd.DataFrame(json.loads(subprocess.getoutput('lsblk -J -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT'))['blockdevices'])
    mountpoints = mountpoints[~mountpoints.children.isna()]
    mountpoints = pd.DataFrame(mountpoints.children.apply(lambda x: x[0]).to_list())[['name','mountpoint','fstype','size']]
    mountpoints = mountpoints.loc[(mountpoints.fstype=='exfat') & (~mountpoints.mountpoint.isna())]
    
    # Scan available USB ports
    available_ports = scan_available_usb_ports()
    max_available_speed = max([port['speed_mbps'] for port in available_ports]) if available_ports else 0
    
    usb_info = []
    for _, row in mountpoints.iterrows():
        device_name = row['name']
        hierarchy = get_usb_hierarchy_info(device_name)
        
        # Get SD card speed information (READ-ONLY)
        print(f"🔍 Testing SD card READ speed for {device_name} (READ-ONLY)...")
        sd_speed_info = get_sd_card_speed_info(device_name)
        
        card_info = {
            'name': device_name,
            'mountpoint': row['mountpoint'],
            'size': row['size'],
        }
        
        # Add SD card speed info
        card_info.update({
            'sd_measured_read_speed': sd_speed_info.get('measured_read_speed', 0),
            'sd_speed_class': sd_speed_info.get('estimated_speed_class', 'Unknown'),
            'sd_min_write_speed': sd_speed_info.get('estimated_min_write_speed', 0),
            'sd_model': sd_speed_info.get('card_model', 'Unknown'),
            'sd_vendor': sd_speed_info.get('card_vendor', 'Unknown'),
            'sd_label': sd_speed_info.get('card_label', ''),
            'sd_test_method': sd_speed_info.get('test_method', 'Unknown'),
        })
        
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
        
        # Calculate actual bottleneck considering SD card speed
        usb_speed = card_info.get('reader_speed_mbps', 0)
        if 'hub_speed_mbps' in card_info:
            usb_speed = min(usb_speed, card_info['hub_speed_mbps'])
        
        usb_transfer_rate = calculate_real_transfer_rate(usb_speed)
        sd_read_speed = card_info.get('sd_measured_read_speed', 0)
        
        # The actual bottleneck is the slower of USB or SD card
        if sd_read_speed > 0:
            card_info['effective_transfer_rate'] = min(usb_transfer_rate, sd_read_speed)
            card_info['bottleneck_type'] = 'USB' if usb_transfer_rate < sd_read_speed else 'SD Card'
        else:
            card_info['effective_transfer_rate'] = usb_transfer_rate
            card_info['bottleneck_type'] = 'USB'
        
        # Check if using optimal port
        current_speed = usb_speed
        card_info['is_optimal_port'] = current_speed >= max_available_speed * 0.9
        card_info['max_available_speed'] = max_available_speed
        card_info['speed_efficiency'] = (current_speed / max_available_speed * 100) if max_available_speed > 0 else 0
        
        usb_info.append(card_info)
    
    return pd.DataFrame(usb_info), available_ports

# Example usage
if __name__ == "__main__":
    video_file = "GOPR0001.MP4"
    
    # Extract all sensor data
    all_sensors = extract_gopro_sensor_data(video_file, "sensor_data.csv")
    
    for sensor_type, data in all_sensors.items():
        if not data.empty:
            print(f"\n{sensor_type.upper()} Data:")
            print(f"  Samples: {len(data)}")
            print(f"  Duration: {data['relative_time'].max():.2f} seconds")
            print(data.head())
            
            if sensor_type == 'accel':
                print(f"  Max acceleration: {data[['x_accel', 'y_accel', 'z_accel']].abs().max().max():.2f} g")
            elif sensor_type == 'temp':
                print(f"  Temperature range: {data['temperature_c'].min():.1f}°C to {data['temperature_c'].max():.1f}°C")
        else:
            print(f"No {sensor_type.upper()} data found")
    
    # Get SD card info and analyze performance
    usb_cards, available_ports = get_complete_usb_card_info()
    
    # Show card analysis with warnings
    print("📱 SD Card Performance Analysis:")
    print("=" * 60)
    
    warnings = []
    for _, card in usb_cards.iterrows():
        icon = "✅" if card['is_optimal_port'] else "⚠️"
        print(f"{icon} {card['mountpoint']} ({card['size']})")
        
        # Show SD card info
        if card.get('sd_model'):
            print(f"    💾 SD Card: {card['sd_vendor']} {card['sd_model']}")
            print(f"       Speed Class: {card['sd_speed_class']}")
            if card.get('sd_measured_read_speed', 0) > 0:
                print(f"       Measured Read: {card['sd_measured_read_speed']:.1f} MB/s")
            if card.get('sd_min_write_speed', 0) > 0:
                print(f"       Min Write: {card['sd_min_write_speed']} MB/s")
        
        if card.get('reader_manufacturer'):
            print(f"    📖 Reader: {card['reader_manufacturer']} {card['reader_product']}")
            print(f"       Speed: {card['reader_speed']} ({card['reader_speed_mbps']} Mbps)")
        
        if card.get('hub_manufacturer'):
            print(f"    🔗 Hub: {card['hub_manufacturer']} {card['hub_product']}")
            print(f"       Speed: {card['hub_speed']} ({card['hub_speed_mbps']} Mbps)")
        
        print(f"    ⚡ Effective Transfer Rate: {card['effective_transfer_rate']:.1f} MB/s")
        print(f"    🚧 Bottleneck: {card['bottleneck_type']}")
        print(f"    📊 Port Efficiency: {card['speed_efficiency']:.1f}%")
        
        if not card['is_optimal_port']:
            max_possible = calculate_real_transfer_rate(card['max_available_speed'])
            speed_gain = max_possible - card['effective_transfer_rate']
            warnings.append({
                'card': card['mountpoint'],
                'current_speed': card['effective_transfer_rate'],
                'max_speed': max_possible,
                'speed_gain': speed_gain
            })
            print(f"    ⚠️  NOT OPTIMAL! Could be {speed_gain:.1f} MB/s faster on different port")
        
        print()