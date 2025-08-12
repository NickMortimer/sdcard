from html import parser
import doit
import os
import glob
from doit import get_var
import jinja2
import yaml
from doit.action import CmdAction
from doit.tools import title_with_actions
from doit.tools import run_once
import pandas as pd
from doit import create_after
import psutil
import platform
import subprocess
import json
from pathlib import Path
import numpy as np
from doit.task import clean_targets
from sdcard.config import Config
import shutil



DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
format_type = 'exfat'

# def task_create_bin():
#     """Extract binary GPMF data from MP4 files"""
#     from sdcard.bruv.gpmf import extract_gopro_binary_gpmf
    
#     def process_video(dependencies, targets):
#         if extract_gopro_binary_gpmf(dependencies[0], targets[0]):
#             return True
#         else:
#             print(f"Failed to extract GPMF from {dependencies[0]}")
#         return False

#     config = Config(get_var('config',None))
#     for path in config.get_path('card_store').rglob('100GOPRO'):
#         file_dep = list(path.glob("*.MP4"))
#         for video_path in file_dep:
#             output_bin = video_path.with_suffix('.bin')
#             yield {
#                 'name': str(video_path),
#                 'file_dep': [video_path],
#                 'actions': [process_video],
#                 'targets': [output_bin],
#                 'uptodate': [True],
#                 'clean': True,
#             }



# @create_after(executed='extract_telemetry', target_regex='.*\.json') 
# def task_create_telemetry_summary():
#     """Create summary CSV of all telemetry data"""
    
#     def create_summary(dependencies, targets):
#         """Summarize all telemetry JSON files"""
#         summary_data = []
        
#         for json_file in dependencies:
#             try:
#                 with open(json_file, 'r') as f:
#                     data = json.load(f)
                
#                 video_name = Path(json_file).stem.replace('_telemetry', '')
#                 camera_info = data.get('camera_info', {})
#                 sensors = data.get('sensors', {})
#                 metadata = data.get('metadata', {})
                
#                 summary_row = {
#                     'video_name': video_name,
#                     'json_file': str(json_file),
#                     'camera_manufacturer': camera_info.get('manufacturer', 'Unknown'),
#                     'camera_model': camera_info.get('model', 'Unknown'),
#                     'camera_firmware': camera_info.get('firmware', 'Unknown'),
#                     'total_sensors': metadata.get('total_sensors', 0),
#                     'has_accelerometer': 'accelerometer' in sensors,
#                     'has_gyroscope': 'gyroscope' in sensors,
#                     'has_gps': 'gps' in sensors,
#                     'has_temperature': 'temperature' in sensors,
#                     'accel_samples': sensors.get('accelerometer', {}).get('sample_count', 0),
#                     'gyro_samples': sensors.get('gyroscope', {}).get('sample_count', 0),
#                     'gps_samples': sensors.get('gps', {}).get('sample_count', 0),
#                     'temp_samples': sensors.get('temperature', {}).get('sample_count', 0),
#                     'extraction_method': data.get('extraction_method', 'telemetry-parser')
#                 }
                
#                 summary_data.append(summary_row)
                
#             except Exception as e:
#                 print(f"⚠️  Error reading {json_file}: {e}")
#                 continue
        
#         if summary_data:
#             df = pd.DataFrame(summary_data)
#             df.to_csv(targets[0], index=False)
            
#             # Print summary statistics
#             total = len(summary_data)
#             with_accel = sum(row['has_accelerometer'] for row in summary_data)
#             with_gyro = sum(row['has_gyroscope'] for row in summary_data)
#             with_gps = sum(row['has_gps'] for row in summary_data)
#             manufacturers = set(row['camera_manufacturer'] for row in summary_data)
            
#             print(f"📊 Telemetry Summary:")
#             print(f"   Total videos: {total}")
#             print(f"   With accelerometer: {with_accel}")
#             print(f"   With gyroscope: {with_gyro}")
#             print(f"   With GPS: {with_gps}")
#             print(f"   Camera manufacturers: {', '.join(manufacturers)}")
            
#             return True
#         else:
#             print("❌ No telemetry data found")
#             return False
    
#     config = Config(get_var('config',None))
    
#     # Find all telemetry JSON files
#     json_files = list(config.get_path('card_store').rglob("*_telemetry.json"))
    
#     for json_file in json_files:
#         summary_csv = config.get_path('card_store') / "telemetry_summary.csv"
#         yield {
#             'name': summary_csv,
#             'file_dep': [json_file],
#             'actions': [create_summary],
#             'targets': [summary_csv],
#             'uptodate': [True],
#             'clean': True
#         }

def run():
    import sys
    from doit.cmd_base import ModuleTaskLoader
    from doit.doit_cmd import DoitMain
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    DoitMain(ModuleTaskLoader(globals())).run(sys.argv[1:])

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    doit.run(globals())