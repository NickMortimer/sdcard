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


cfg = None
CATALOG_DIR = None
DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
format_type = 'exfat'




def task_create_json():
        config = Config(get_var('config',None))
        if platform.system() == 'Windows':
            exiftool_path = config.current_path / 'bin' /'exiftool.exe'
        else:
            exiftool_path = 'exiftool'
        for path in config.get_path('card_store').rglob('100GOPRO'):
            file_dep = list(path.glob("*.MP4"))
            if len(file_dep)>0:
                target  = path / config.data['exifname']
                command = f'{exiftool_path} -api largefilesupport=1 -m -u -q -q -n -CameraSerialNumber -CreateDate -TrackCreateDate -OffsetTimeOriginal -SourceFile -Duration -Rate -VideoFrameRate -Model -FileSize -FieldOfView -json -ext MP4 {path} > {target} || :'
                if file_dep:
                     yield {  
                        'name':path,
                        'file_dep':file_dep,
                        'actions':[command],
                        'targets':[target],
                        'uptodate':[run_once],
                        'clean':True,
                    }
                     
@create_after(executed='create_json', target_regex='.*\.json') 
def task_extract_telemetry():
    """Extract telemetry data using telemetry-parser"""
    
    def process_telemetry(dependencies, targets):
        """Process MP4 file to extract telemetry data"""
        try:         
            video_path = Path(dependencies[0])
            csv_output = Path(targets[0])
            
            print(f"📊 Processing telemetry from {video_path.name}")
            from telemetry_parser import Parser as TelemetryParser
            # Extract telemetry data directly from MP4
            extractor = TelemetryParser(str(video_path))
            
            # Get the normalized IMU data
            imu_data = extractor.normalized_imu()
            data = pd.DataFrame(imu_data)

            # Expand gyro tuple into separate columns
            if 'gyro' in data.columns:
                data[['gyro_x', 'gyro_y', 'gyro_z']] = pd.DataFrame(data['gyro'].tolist(), index=data.index)
                data = data.drop('gyro', axis=1)
            
            # Expand accelerometer tuple into separate columns  
            if 'accl' in data.columns:
                data[['accl_x', 'accl_y', 'accl_z']] = pd.DataFrame(data['accl'].tolist(), index=data.index)
                data = data.drop('accl', axis=1)
            
            data.to_csv(targets[0], index=False)
            return True
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            return False
    

    
    config = Config(get_var('config',None))
    for path in config.get_path('card_store').rglob('100GOPRO'):
        file_dep = list(path.glob("*.MP4"))
        for video_path in file_dep:
            json_output = video_path.with_name(f"{video_path.stem}_telemetry.csv")
            yield {
                'name': json_output,
                'file_dep': [video_path],
                'actions': [process_telemetry],
                'targets': [json_output],
                'uptodate': [True],
                'clean': True,
            }                     
                     
                    
@create_after(executed='create_json', target_regex='.*\.json') 
def task_make_yml():

        def concat(dependencies, targets,config):
            data = pd.read_json(dependencies[0])
            data['CreateDate'] = pd.to_datetime(data.CreateDate, format='%Y:%m:%d  %H:%M:%S')
            data['CreateDate'] = data['CreateDate'].dt.tz_localize('UTC')
            data['TrackCreateDate'] = pd.to_datetime(data.CreateDate, format='%Y:%m:%d  %H:%M:%S')
            data['TrackCreateDate'] = data['TrackCreateDate'].apply(lambda x:x.strftime('%Y%m%dT%H%M%S%z'))
            data['StartTimeLocal'] = data['CreateDate'].dt.tz_convert(config.data['time_zone'])
            data['CreateDate'] = data['CreateDate'].apply(lambda x:x.strftime('%Y%m%dT%H%M%S%z'))
            data['Duration'] = pd.to_timedelta(data['Duration'], unit='s')
            data['EndTimeLocal'] = data['StartTimeLocal'] + data['Duration']
            data['StartTimeLocal'] = data['StartTimeLocal'].apply(lambda x:x.strftime('%Y%m%dT%H%M%S%z'))
            data['EndTimeLocal'] = data['EndTimeLocal'].apply(lambda x:x.strftime('%Y%m%dT%H%M%S%z'))
            data['Duration'] = data['Duration'].astype(str)
            for index, row in data.iterrows():
                target = Path(row['SourceFile']).with_suffix('.yml')
                if target.exists():
                    video_data = yaml.safe_load(target.read_text(encoding='utf-8'))
                    update = row.to_dict()
                    for key in update.keys():
                        if key in video_data:
                            video_data[key] = update[key]
                else:
                    video_data = row.to_dict()
                with open(target, 'w') as f:
                    yaml.dump(video_data, f, default_flow_style=False)
                
                 
        config = Config(get_var('config',None))
        exiffiles = config.get_path('card_store').rglob(config.data['exifname'])
        for exif in exiffiles:
            targets =[file.with_suffix('.yml') for file in exif.parent.glob('*.MP4')]
            yield { 
                'name':exif,
                'file_dep':[exif],
                'actions':[(concat,[],{'config':config})],
                'targets':targets,
                'uptodate':[True],
                'clean':True,
            } 

@create_after(executed='make_yml', target_regex='.*\.json') 
def task_concat_yml():
    def concat(dependencies, targets):
        def calculatetimes(df):
            df =df.sort_values('ItemId')
            if len(df)>1:
                start = df.Duration.cumsum().shift(+1)
                start.iloc[0] = 0
                start =pd.to_timedelta(start,unit='s')
                df['CalculatedStartTime']=(df['StartTimeLocal']+start).dt.round('1s')
            else:                 
                df['CalculatedStartTime']=df['StartTimeLocal']
            return df        
        data = pd.DataFrame([yaml.safe_load(Path(source).read_text(encoding='utf-8')) for source in dependencies])
        data['Duration'] = pd.to_timedelta(data['Duration'])
        data['FileName']=data.SourceFile.apply(lambda x:Path(x).name)
        data['StartTimeLocal'] = pd.to_datetime(data['StartTimeLocal'], format='%Y%m%dT%H%M%S%z')
        data[['ItemId','GroupId']]=data.FileName.str.extract('(?P<item>\d\d)(?P<group>\d\d\d\d).MP4')
        data =data.groupby(['StartTimeLocal','CameraSerialNumber','GroupId'],group_keys=False).apply(calculatetimes).reset_index()
        data['CalculatedEndTime'] = data['CalculatedStartTime'] + data['Duration']
        data.sort_values(['CameraSerialNumber','CreateDate']).drop('index',axis=1).to_csv(targets[0],index=False)


        data.sort_values(['CameraSerialNumber','CreateDate']).to_csv(targets[0],index=False)
             
    config = Config(get_var('config',None))
    ymlfiles = list(config.get_path('card_store').rglob('G*.yml'))
    target = config.get_path('videolist')
    return { 
        'file_dep':ymlfiles,
        'actions':[concat],
        'targets':[target],
        'uptodate':[True],
        'clean':True,
    } 


@create_after(executed='concat_yml', target_regex='.*\.json') 
def task_camera_names():
    config = Config(get_var('config',None))
    def concat(dependencies, targets):
        bars = pd.read_csv(config.get_path('camera_bars'))
        videos = pd.read_csv(config.get_path('videolist'))
        videos['StartTimeLocal'] = pd.to_datetime(videos['StartTimeLocal'])
        stations = pd.read_csv(config.get_path('station_path'))
        stations['date_time'] = pd.to_datetime(stations['date_time']) 
        
        # Merge keeping all videos entries, even without matching cameras
        cams_names = pd.merge(
            videos, 
            bars, 
            on='CameraSerialNumber',  # common key to merge on
            how='left'                 # keep all entries from videos (left table)
        )          
        def find_nearest_station(row):
            # Find closest station time within tolerance
            systems = stations.loc[stations.system_number == row['SystemNumber']]
            if len(systems) == 0:
                # Return a Series with same columns as stations but filled with None
                return pd.Series({col: None for col in stations.columns})
                
            time_diff = abs(systems['date_time'] - row['StartTimeLocal'])
            mask = time_diff <= pd.Timedelta(minutes=5)
            if not mask.any():
                # Return a Series with same columns as stations but filled with None
                return pd.Series({col: None for col in stations.columns})
            
            nearest_idx = time_diff[mask].idxmin()
            # Return the entire row from stations DataFrame
            return stations.loc[nearest_idx]
        matches = cams_names.apply(find_nearest_station, axis=1)
        result = pd.concat([cams_names, matches.add_suffix('_station')], axis=1)
        result = result.drop(columns=['date_time_station'], errors='ignore')
        def calculate_destination_directory(group):
            """Calculate destination directory based on SystemNumber and GroupId"""
            dest = config.data['output_path']
            fields = {}
            
            # Get camera names starting with 'L'
            left_cams = group[group.CameraName.str.startswith('L')]['CameraName']
            fields['left_camera'] = left_cams.iloc[0] if not left_cams.empty else 'Lxx'
            
            # Similarly for right cameras
            right_cams = group[group.CameraName.str.startswith('R')]['CameraName']
            fields['right_camera'] = right_cams.iloc[0] if not right_cams.empty else 'Rxx'
            
            # Use first row for other fields since they should be same within group
            first_row = group.iloc[0]
            fields['frame_number'] = str(first_row.Frame)
            fields['deployment_date'] = pd.to_datetime(first_row.StartTimeLocal).strftime('%Y%m%d')
            fields['opcode'] = first_row.opcode_station
            
            return dest.format(**fields)

        directories = {}
        for name, group in result.groupby(['SystemNumber', 'opcode_station']):
            if pd.isna(name[0]) or pd.isna(name[1]):  # Skip if either key is None/NaN
                continue
            directory = calculate_destination_directory(group)
            directories[name] = directory
        
        # Then map back to original DataFrame with error handling
        def get_directory(row):
            key = (row['SystemNumber'], row['opcode_station'])
            return directories.get(key, '')  # Return 'unknown' if key not found
        
        result['destination_directory'] = result.apply(get_directory, axis=1)
        result['CalculatedStartTime'] = pd.to_datetime(result['CalculatedStartTime'])
        file_name = config.data['output_file_name']
        def make_name(row):
            fields = {}
            if row.opcode_station is None:
                return ''
            fields['opcode'] = row.opcode_station
            fields['frame_name'] =row.Frame
            fields['time_stamp'] =f"{row.CalculatedStartTime.strftime('%Y%m%dT%H%M%S')}"
            fields['camera_name'] =row.CameraName
            return config.data['output_file_name'].format(**fields)
        result['NewFileName'] =result.apply(make_name, axis=1)
        result['AccelerometerFile'] = result['NewFileName'].apply(lambda x: x.replace('.MP4', '_telemetry.csv'))
        result['AccelerometerSourceFile'] = result['SourceFile'].apply(lambda x: x.replace('.MP4', '_telemetry.csv'))
        result['CameraSide'] = result['AccelerometerFile'].str.extract(r'_([LR])[^_]*_telemetry\.csv$')[0]
        result.to_csv(targets[0],index=False)
             
    config = Config(get_var('config',None))
    file_dep = [config.get_path('videolist'),config.get_path('camera_bars'),config.get_path('station_path')]
                
    target = config.get_path('deployments')
    return { 
        'file_dep':file_dep,
        'actions':[concat],
        'targets':[target],
        'uptodate':[True],
        'clean':True,
    } 



@create_after(executed='camera_names', target_regex='.*\.json') 
def task_stage_data():
        def hardlink(dependencies, targets):
             stage = pd.read_csv(dependencies[0])
             for index, row in stage.iterrows():
                print(row.NewFileName)
                if (row.NewFileName is not None):
                    dir = row.destination_directory
                    try:
                        if len(dir)>0:
                            video_source = Path(row.SourceFile)
                            Path(dir).mkdir(parents=True, exist_ok=True)
                            video_target = Path(dir) / row.NewFileName
                            bin_source = Path(row.AccelerometerSourceFile)
                            bin_target = Path(dir) / row.AccelerometerFile
                            if not video_target.exists():
                                try:
                                    os.link(row.SourceFile, video_target)
                                except (OSError, NotImplementedError, PermissionError) as e:
                                    # Hard linking failed - fall back to copy
                                    print(f"Hard linking failed for {row.SourceFile} -> {video_target}: {e}")
                                    print("Falling back to copy operation...")
                                    shutil.copy2(row.SourceFile, video_target)
                            if not bin_target.exists() and bin_source.exists():
                                try:
                                    os.link(bin_source, bin_target)
                                except (OSError, NotImplementedError, PermissionError) as e:
                                    # Hard linking failed - fall back to copy
                                    print(f"Hard linking failed for {bin_source} -> {bin_target}: {e}")
                                    shutil.copy2(bin_source, bin_target)
                    except:
                        pass
        config = Config(get_var('config',None))      
        if config.get_path('deployments').exists():
            return { 
                'file_dep':[config.get_path('deployments')],
                'actions':[hardlink],
                'uptodate':[True]
            } 




@create_after(executed='stage_data', target_regex='.*\.json') 
def task_create_ass():
    def process_srt(dependencies, targets):
        stage = pd.read_csv(config.get_path('deployments'))
        details =stage.loc[stage.NewFileName==Path(dependencies[0]).name].iloc[0]
        def maketag(time):
            total = time.total_seconds()
            m, s = divmod(total, 60)
            h, m = divmod(m, 60)
            result = "%02d:%02d:%02d,%03d" % (h, m, s, int((total % 1) * 1000))
            return(result)
        starttime = pd.to_datetime(details.CalculatedStartTime)
        endtime = starttime + pd.Timedelta(details.Duration)
        currenttime = starttime
        isub=1
        FdSub = open(targets[0], 'w')
        while (currenttime < endtime):
            FdSub.write("%d\n" % (isub))
            FdSub.write(maketag(currenttime-starttime) + ' --> ' + maketag(currenttime-starttime+pd.Timedelta(1,'s')) + '\n')
            FdSub.write(f"{currenttime.isoformat()} Trip:{config.data['field_trip_id']} Stn:{details.opcode_station} Cam:{details.Frame}_{details.CameraName} Lat:{details.latitude_dd_station:.4f} Lon:{details.longitude_dd_station:.4f} Depth:{details.depth_m_station:.1f}\n\n")
            isub = isub +1
            currenttime = currenttime +pd.Timedelta(1,'s')
        FdSub.close()

    config = Config(get_var('config',None))  
    if os.path.exists(config.get_path('deployments')):
        data = pd.read_csv(config.get_path('deployments')).drop_duplicates()
        data = data.loc[~data.destination_directory.isna()]
        for index, row in data.iterrows():
            if len(row.NewFileName) == 0:
                continue
            source = Path(row.destination_directory) / row.NewFileName
            target =source.with_suffix('.srt')
            yield { 
                'name':source.name,
                'file_dep':[source],
                'targets': [target],
                'actions':[process_srt],
                'uptodate':[True]
            } 


@create_after(executed='stage_data', target_regex='.*\.csv')
def task_sync_analysis():
    """Analyze time synchronization between paired cameras"""
    
    def analyze_sync(dependencies, targets):
        """Analyze sync between accelerometer files from same GroupId"""
        # Interpolate both signals onto common time grid
        import pandas as pd
        import numpy as np
        from scipy.signal import correlate, correlation_lags
        import matplotlib.pyplot as plt
        
        if len(dependencies) < 2:
            print("Need at least 2 telemetry files for sync analysis")
            return False
            
        # Take first two files for comparison
        csv_left = list(filter(lambda x: '_L' in Path(x).name, dependencies))[0]
        csv_right = list(filter(lambda x: '_R' in Path(x).name, dependencies))[0]

        if not csv_left or not csv_right:
            print("Could not find both left and right camera files")
            return False
            
        print(f"Analyzing sync between:")
        print(f"  Left file: {csv_left}")
        print(f"  Right file: {csv_right}")
        try:
            # Read CSV files 
            left_data = pd.read_csv(csv_left)
            right_data = pd.read_csv(csv_right)

            # Convert timestamp_ms to datetime and set as index
            left_data['timestamp'] = pd.to_datetime(left_data['timestamp_ms'], unit='ms')
            right_data['timestamp'] = pd.to_datetime(right_data['timestamp_ms'], unit='ms')
            
            print(f"Original data shapes: Left={left_data.shape}, Right={right_data.shape}")
            print(f"Left timestamp range: {left_data['timestamp'].min()} to {left_data['timestamp'].max()}")
            print(f"Right timestamp range: {right_data['timestamp'].min()} to {right_data['timestamp'].max()}")
            
            # Check original sample rates
            left_diff = left_data['timestamp'].diff().median().total_seconds() * 1000
            right_diff = right_data['timestamp'].diff().median().total_seconds() * 1000
            print(f"Original sample intervals: Left={left_diff:.2f}ms, Right={right_diff:.2f}ms")
            
            def trim(df,time_length=10):
                """Trim dataframe to specified time range"""
                start_time = df.index[0]
                end_time = start_time + pd.Timedelta(minutes=time_length)
                result = df.loc[start_time:end_time].copy()
                result = result.resample('0.5ms').mean().interpolate("time").resample('1ms').first()
                return result
            # Set timestamp as index and sort
            left_data = trim(left_data.set_index('timestamp').sort_index())
            right_data = trim(right_data.set_index('timestamp').sort_index())
            
            # Use 1ms sampling interval as set by the trim function
            sampling_interval = pd.Timedelta(milliseconds=1).total_seconds()  # 0.001 seconds
            
            print("Using trimmed and resampled data directly...")
            # Both datasets are now trimmed and resampled to 1ms intervals
            # No need for concatenation since they're already on the same time grid
            
            print(f"Final processed shapes: Left={left_data.shape}, Right={right_data.shape}")
            print(f"Sampling interval: {sampling_interval*1000:.1f}ms")
            print(f"Left accl_x stats: mean={left_data['accl_x'].mean():.3f}, std={left_data['accl_x'].std():.3f}, range={left_data['accl_x'].min():.3f} to {left_data['accl_x'].max():.3f}")
            print(f"Right accl_x stats: mean={right_data['accl_x'].mean():.3f}, std={right_data['accl_x'].std():.3f}, range={right_data['accl_x'].min():.3f} to {right_data['accl_x'].max():.3f}")

            # Use the minimum length of both datasets (should be equal after trimming)
            min_length = min(len(left_data), len(right_data))
            left_data = left_data[:min_length]
            right_data = right_data[:min_length]

            # Normalize each axis separately for both cameras
            left_x_norm = (left_data['accl_x'] - left_data['accl_x'].mean()) / left_data['accl_x'].std()
            left_y_norm = (left_data['accl_y'] - left_data['accl_y'].mean()) / left_data['accl_y'].std()
            left_z_norm = (left_data['accl_z'] - left_data['accl_z'].mean()) / left_data['accl_z'].std()
            
            right_x_norm = (right_data['accl_x'] - right_data['accl_x'].mean()) / right_data['accl_x'].std()
            right_y_norm = (right_data['accl_y'] - right_data['accl_y'].mean()) / right_data['accl_y'].std()
            right_z_norm = (right_data['accl_z'] - right_data['accl_z'].mean()) / right_data['accl_z'].std()

            print(f"Signal lengths: Left={len(left_data)}, Right={len(right_data)}")
            print(f"Left axis stats: X(std={left_data['accl_x'].std():.3f}), Y(std={left_data['accl_y'].std():.3f}), Z(std={left_data['accl_z'].std():.3f})")
            print(f"Right axis stats: X(std={right_data['accl_x'].std():.3f}), Y(std={right_data['accl_y'].std():.3f}), Z(std={right_data['accl_z'].std():.3f})")

            # Cross-correlate each axis pair and find best lag for each
            from scipy.signal import correlate, correlation_lags
            
            time_lags = []
            correlations = []
            
            # Process each axis
            for axis_name, left_signal, right_signal in [('X', left_x_norm, right_x_norm), 
                                                         ('Y', left_y_norm, right_y_norm), 
                                                         ('Z', left_z_norm, right_z_norm)]:
                
                # Cross-correlation
                correlation = correlate(left_signal, right_signal, mode='full')
                lags = correlation_lags(len(left_signal), len(right_signal), mode='full')
                
                best_lag_index = np.argmax(correlation)
                best_lag = lags[best_lag_index]
                best_corr = correlation[best_lag_index]
                time_lag = best_lag * sampling_interval  # 5ms sampling interval
                
                time_lags.append(time_lag)
                correlations.append(best_corr)
                
                print(f"Axis {axis_name}: lag={best_lag} samples ({time_lag:.3f}s), correlation={best_corr:.3f}")
            
            # Calculate weighted average time lag (weight by correlation strength)
            total_weight = sum(abs(c) for c in correlations)
            if total_weight > 0:
                weighted_time_lag = sum(t * abs(c) for t, c in zip(time_lags, correlations)) / total_weight
                average_correlation = sum(correlations) / len(correlations)
            else:
                weighted_time_lag = sum(time_lags) / len(time_lags)  # Simple average if all correlations are zero
                average_correlation = 0.0
            
            print(f"Individual time lags: X={time_lags[0]:.3f}s, Y={time_lags[1]:.3f}s, Z={time_lags[2]:.3f}s")
            print(f"Weighted average time lag: {weighted_time_lag:.3f}s")
            print(f"Average correlation: {average_correlation:.3f}")

            # For plotting, use the axis with the highest correlation
            best_axis_idx = np.argmax([abs(c) for c in correlations])
            axis_names = ['X', 'Y', 'Z']
            signals = [(left_x_norm, right_x_norm), (left_y_norm, right_y_norm), (left_z_norm, right_z_norm)]
            
            plot_left, plot_right = signals[best_axis_idx]
            plot_correlation = correlate(plot_left, plot_right, mode='full')
            plot_lags = correlation_lags(len(plot_left), len(plot_right), mode='full')
            
            # Create a combined magnitude signal for JSON output (optional)
            left_data['magnitude'] = np.sqrt(left_data['accl_x']**2 + left_data['accl_y']**2 + left_data['accl_z']**2)
            right_data['magnitude'] = np.sqrt(right_data['accl_x']**2 + right_data['accl_y']**2 + right_data['accl_z']**2)
            
            # Use weighted average results
            best_lag = int(weighted_time_lag / sampling_interval)
            best_corr = average_correlation
            time_lag = weighted_time_lag

            print(f"Best lag: {best_lag} samples ({time_lag:.3f} seconds)")
            print(f"Max correlation: {best_corr:.3f}")



            # Write results as JSON
            import json
            analysis_duration = min_length * sampling_interval
            actual_sample_rate_hz = 1.0 / sampling_interval if sampling_interval > 0 else 0
            results = {
                "sync_analysis_results": {
                    "left_file": str(csv_left),
                    "right_file": str(csv_right),
                    "actual_sample_rate_hz": float(actual_sample_rate_hz),
                    "sampling_interval_seconds": float(sampling_interval),
                    "analysis_duration_seconds": float(analysis_duration),
                    "interpolated_samples": min_length,
                    "method": "multi_axis_weighted_average_merged_interpolation",
                    "individual_time_lags": {
                        "x_axis_seconds": float(time_lags[0]),
                        "y_axis_seconds": float(time_lags[1]), 
                        "z_axis_seconds": float(time_lags[2])
                    },
                    "individual_correlations": {
                        "x_axis": float(correlations[0]),
                        "y_axis": float(correlations[1]),
                        "z_axis": float(correlations[2])
                    },
                    "sample_lag": int(best_lag),
                    "time_lag_seconds": float(time_lag),
                    "correlation_coefficient": float(best_corr),
                    "best_axis_for_plot": axis_names[best_axis_idx],
                    "analysis_timestamp": pd.Timestamp.now().isoformat()
                }
            }
            
            with open(targets[0], 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"Time offset: {time_lag:.3f} seconds")
            print(f"Correlation: {best_corr:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error in sync analysis: {e}")
            return False
    
    config = Config(get_var('config', None))
    
    # Check if deployments file exists
    deployments_path = config.get_path('deployments')
    if not deployments_path.exists():
        print(f"Deployments file not found: {deployments_path}")
        return

    # Read deployments data and filter for valid entries
    data = pd.read_csv(deployments_path).drop_duplicates()
    data = data.loc[~data.destination_directory.isna()]
    data = data.loc[data.destination_directory != '']
    
    # Group by deployment location to find paired cameras
    grouped_data = data.groupby(['SystemNumber', 'opcode_station'])
    
    for (system_number, opcode_station), group in grouped_data:
        # Skip if either key is None/NaN
        if pd.isna(system_number) or pd.isna(opcode_station):
            continue
            
        # Find left and right cameras using the CameraSide column
        left_cameras = group.loc[group.CameraSide == 'L']
        right_cameras = group.loc[group.CameraSide == 'R']
        
        if not left_cameras.empty and not right_cameras.empty:
            # Get the accelerometer file paths
            left_accel_files = []
            right_accel_files = []
            
            for _, row in left_cameras.iterrows():
                accel_path = Path(row.destination_directory) / row.AccelerometerFile
                if accel_path.exists():
                    left_accel_files.append(str(accel_path))
                    
            for _, row in right_cameras.iterrows():
                accel_path = Path(row.destination_directory) / row.AccelerometerFile
                if accel_path.exists():
                    right_accel_files.append(str(accel_path))
            
            # Create sync analysis task if we have files from both sides
            if left_accel_files and right_accel_files:
                # Use the first left and right file for comparison
                file_deps = [left_accel_files[0], right_accel_files[0]]
                
                # Create output directory and filename
                output_dir = Path(left_cameras.iloc[0].destination_directory)
                sync_report = output_dir / f"sync_analysis_{system_number}_{opcode_station}.json"

                yield {
                    'name': f"sync_{system_number}_{opcode_station}",
                    'file_dep': file_deps,
                    'actions': [analyze_sync],
                    'targets': [sync_report],
                    'uptodate': [True],
                    'clean': True,
                }

# @create_after(executed='concat_json', target_regex='.*\.json') 
# def task_report_json():

#         def exifreport(dependencies, targets):
#             data = pd.read_csv(dependencies[0])
#             data['Bad'] =data['CreateDate'].isna()
#             data['SourceFile'] = data.apply(lambda x: f"{{CATALOG_DIR}}/{os.path.relpath(x['SourceFile'],CATALOG_DIR)}",axis=1)
#             data['Directory']=data['SourceFile'].apply(lambda x: os.path.split(x)[0])
#             data['FileName'] = data['SourceFile'].apply(os.path.basename)
#             data[['ItemId','GroupId']]=data.FileName.str.extract('(?P<item>\d\d)(?P<group>\d\d\d\d).MP4')
#             data =data.sort_values(['SourceFile'])
#             data['CreateDate'] =pd.to_datetime(data.CreateDate)
#             #ok lets try and fix missing data from bad videos
#             data['SpeedUp'] =pd.to_numeric(data.Rate.str.extract(r'(\d+)X')[0])
#             data['AdjustedDuration'] = data['SpeedUp']* data['Duration']
#             top =data.groupby(['CameraSerialNumber','GroupId']).first()
#             top['TotalRunTime'] = data.groupby(['CameraSerialNumber','GroupId'])['AdjustedDuration'].sum()
#             top['EndTime'] = pd.to_timedelta(top['TotalRunTime'],unit='s')+top['CreateDate']
#             top =top.sort_values(['SourceFile'])
#             top.to_csv(targets[0],index=False)
#         return { 

#             'file_dep':[geturl('exifstore')],
#             'actions':[exifreport],
#             'targets':[geturl('exifreport')],
#             'uptodate':[True],
#             'clean':True,
#         }        
    
# @create_after(executed='concat_json', target_regex='.*\.json') 
# def task_checkbars():
#         def checkbars(dependencies, targets):
#             data = pd.read_csv(dependencies[0],parse_dates=['CreateDate'])
#             data.Duration =pd.to_timedelta(data.Duration,unit='s')
#             data['CreateEnd'] =  data.CreateDate + data.Duration
#             data['CreateStart'] =  data.CreateDate
#             cstart =data.groupby('CameraSerialNumber')['CreateStart'].min()
#             cend = data.groupby('CameraSerialNumber')['CreateEnd'].max()
#             stats =pd.concat([cstart,cend],axis=1) 
#             if os.path.exists(targets[0]):
#                 barnumbers = pd.read_csv(targets[0],parse_dates=['BarStartDate','BarEndDate']).drop(['CreateStart','CreateEnd'],axis=1,errors='ignore')                      
#             else:
#                  barnumbers = pd.DataFrame(columns=['PlatformName','CameraName','CameraSerialNumber','BarStartDate','BarEndDate','HousingNumber','CalibrationDate'])
#             barnumbers = pd.merge(barnumbers,stats.reset_index(),how='outer')
#             barnumbers.to_csv(targets[0],index=False)

#         return { 

#             'file_dep':[geturl('exifstore')],
#             'actions':[checkbars],
#             'targets':[geturl('cameranames')],
#             'uptodate':[True],
#             'clean':True,
#         }         


# @create_after(executed='checkbars', target_regex='.*\.json') 
# def task_check_cam():

#         def checkcams(dependencies, targets):
#             if geturl('check').exists():
#                 shutil.rmtree(geturl('check'))
#             os.makedirs(geturl('check'),exist_ok=True)
#             data = pd.read_csv(geturl('exifstore'))
#             data = data.loc[data.ItemId==1]
#             barnumbers = pd.read_csv(geturl('cameranames'),parse_dates=['BarStartDate','BarEndDate']) 
#             result = matchbars(data,barnumbers)
#             result['CreateDate'] =pd.to_datetime(result['CreateDate'] ).dt.strftime('%Y%m%dT%H%M%S') 
#             result['target'] = result.CameraName +'_' + result.CameraSerialNumber + '_'+result['CreateDate']+'.MP4'
#             result['SourceFile'] = result['SourceFile'].apply(lambda x: x.format(**{'CATALOG_DIR':CATALOG_DIR}))
#             result.to_csv(targets[0])
            
#             for index,row in result.iterrows():
#                 if not os.path.exists(row.target):
#                     os.makedirs(geturl('check') / row.PlatformName,exist_ok=True)
#                     target =geturl('check') / row.PlatformName / row.target
#                     if not target.exists():
#                         os.link(row.SourceFile,geturl('check') / row.PlatformName / row.target)

#         return { 

#             'file_dep':[geturl('exifstore'),geturl('cameranames')],
#             'actions':[checkcams],
#             'targets':[geturl('checkstore')],
#             'uptodate':[True],
#             'clean':True,
#         } 



# @create_after(executed='check_cam', target_regex='.*\.json') 
# def task_make_autodeployments():

#         def deployments(dependencies, targets):
#             data = pd.read_csv(geturl('exifstore'),parse_dates=['CreateDate'])
#             totaltime =pd.to_datetime(data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['Duration'].sum(), unit='s').dt.strftime("%H:%M:%S").rename('TotalTime')
#             totalfilesize =(data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['FileSize'].sum()/1000000000).rename('TotalSize')
#             maxid =data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['ItemId'].max().rename('MaxId')
#             minid =data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['ItemId'].min().rename('MinId')
#             filecount = data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['ItemId'].count().rename('FileCount')
#             groups =data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])[['SourceFile','FieldOfView']].first()
#             output =groups.join(filecount).join(minid).join(maxid).join(totalfilesize).join(totaltime)

#             barnumbers = pd.read_csv(geturl('cameranames'),parse_dates=['BarStartDate','BarEndDate']) 
#             result = matchbars(output.reset_index(),barnumbers)
#             result['CreateDate'] = pd.to_datetime(result['CreateDate'] )
#             result['DeploymentId']=result.apply(lambda x: f"{x.CreateDate.strftime('%Y%m%dT%H%M%S')}_{x.PlatformName}_{x.CameraName}_{x.CameraSerialNumber}_{x.GroupId:02}", axis=1)
#             manualfile = geturl('timecorrection')
#             manual =result.loc[:, ['DeploymentId', 'TotalTime','CreateDate','SourceFile']]
#             manual =manual.set_index('DeploymentId')
#             if os.path.exists(manualfile):
#                  old = pd.read_csv(manualfile,index_col='DeploymentId')
#                  manual =manual.join(old['CorrectedTime'])
#                  manual.loc[manual.CorrectedTime.isnull(),'CorrectedTime']=manual.loc[manual.CorrectedTime.isnull(),'CreateDate']
#             else:
#                 manual['CorrectedTime'] = manual['CreateDate']
#             #manual['SourceFile'] = manual['SourceFile'].apply(lambda x: f'=HYPERLINK("file://{x}", "{os.path.basename(x)}")')
#             manual.sort_values('DeploymentId').to_csv(manualfile)
#             manual.sort_values('DeploymentId').to_excel(manualfile.with_suffix('.xlsx'))
#             result.to_csv(targets[0],index=False)



#         target = geturl('autodeployment')
#         return { 

#             'file_dep':[geturl('exifstore')],
#             'actions':[deployments],
#             'targets':[target,geturl('timecorrection')],
#             'uptodate':[True],
#             'clean':True,
#         } 

# def matchbars(deployments,barnumbers,datecolumn='CreateDate'):
#     conn = sqlite3.connect(':memory:')
#     #write the tables
#     # Drop columns from the DataFrame if they exist in the list
#     barnumbers.to_sql('bars', conn, index=False)
#     deployments[deployments.columns[~deployments.columns.isin(['PlatformName','HousingNumber','CameraName','BarStartDate','BarEndDate','CalibrationDate'])]].to_sql('deployments', conn, index=False)
#     qry = f'''
#         select  
#             deployments.*,
#             bars.PlatformName,
#             bars.HousingNumber,
#             bars.CameraName,
#             bars.BarStartDate,
#             bars.BarEndDate,
#             bars.CalibrationDate
#         from
#             deployments join bars on
#             (deployments.{datecolumn} between bars.BarStartDate and bars.BarEndDate) and
#             (deployments.CameraSerialNumber = bars.CameraSerialNumber)
#         '''
#     result =pd.read_sql_query(qry, conn)
#     result['CreateDate'] = pd.to_datetime(result['CreateDate'] )
#     result['DeploymentId']=result.apply(lambda x: f"{x.CreateDate.strftime('%Y%m%dT%H%M%S')}_{x.PlatformName}_{x.CameraName}_{x.CameraSerialNumber}", axis=1)
#     return result


# @create_after(executed='make_autodeployments', target_regex='.*\.json') 
# def task_make_matchbars():

#         def stagedeployments(dependencies, targets):
#             def calculatetimes(df):
#                 df =df.sort_values('ItemId')
#                 if len(df)>1:
#                     start = df.Duration.cumsum().shift(+1)
#                     start.iloc[0] = 0
#                     start =pd.to_timedelta(start,unit='s')
#                     df['CalculatedStartTime']=(df['CorrectedTime']+start).dt.round('1s')
#                 else:                 
#                     df['CalculatedStartTime']=df['CorrectedTime']
#                 return df

#             def makedeploymentkey(df):
#                 def makedirs(row):
#                     left = 0
#                     right = 0
#                     if leftcam in row.keys(): 
#                          left = row[leftcam]
#                     if rightcam in row.keys():
#                          right = row[rightcam]
#                     # if right==left:
#                     #     result = f"{row['StageId']}_{int(left):02}"
#                     # else:
#                     result = f"{row['StageId']}_{int(left):02}_{int(right):02}"
#                     return row['StageId'],result
#                 leftcam = df.CameraName[df.CameraName.str.startswith('L')].min()
#                 rightcam = df.CameraName[df.CameraName.str.startswith('R')].min()
#                 left = df[(df.CameraName==leftcam) & (df.Duration>600)].groupby('CorrectedTime').first().reset_index()[['CorrectedTime','CameraName','PlatformName']].add_suffix('_Left')
#                 left['MatchTime'] = left['CorrectedTime_Left']
#                 right = df[(df.CameraName==rightcam) & (df.Duration>600)].groupby('CorrectedTime').first().reset_index()[['CorrectedTime','CameraName','PlatformName']].add_suffix('_Right')
#                 right['MatchTime'] = right['CorrectedTime_Right']
#                 merged_df = pd.merge_asof(right, left, left_on='MatchTime', right_on='MatchTime', direction='nearest', tolerance=pd.Timedelta(minutes=30),suffixes=( '_right','_left'))
#                 merged_df =pd.concat([merged_df,left[~left.CorrectedTime_Left.isin(merged_df.CorrectedTime_Left.unique())]])
#                 merged_df.loc[merged_df.CorrectedTime_Right.isna(),'PlatformName_Right'] = merged_df.loc[merged_df.CorrectedTime_Right.isna(),'PlatformName_Left']
#                 merged_df.loc[merged_df.CorrectedTime_Left.isna(),'CorrectedTime_Left'] = merged_df.loc[merged_df.CorrectedTime_Left.isna(),'CorrectedTime_Right']
#                 starttime =merged_df.MatchTime.dt.strftime("%Y%m%dT%H%M%S")
#                 merged_df['StageId']=merged_df.PlatformName_Right+'_'+starttime
#                 stageId =pd.concat((merged_df[['CorrectedTime_Left','StageId']].rename(columns={'CorrectedTime_Left':'CorrectedTime'}),merged_df[['CorrectedTime_Right','StageId']].rename(columns={'CorrectedTime_Right':'CorrectedTime'}))).dropna()
#                 df =pd.merge(df,stageId)
#                 totals =df.groupby(['StageId','CameraName']).size().reset_index().pivot_table(index ='StageId',values=0,columns='CameraName').reset_index().fillna(0)
#                 totals =totals.apply(makedirs,axis=1).apply(pd.Series)
#                 totals.columns = ['StageId','StageDir']
#                 df = df.merge(totals)
#                 return df
#             dep = pd.read_csv(geturl('autodeployment'),parse_dates=['CreateDate'])
#             exifdata = pd.read_csv(geturl('exifstore'),parse_dates=['CreateDate']).set_index(['CreateDate','CameraSerialNumber','GroupId'])
#             correcttimes = pd.read_csv(geturl('timecorrection'),parse_dates=['CreateDate','CorrectedTime'])
#             dep =pd.merge(dep,correcttimes[['DeploymentId','CorrectedTime']],on='DeploymentId', how='left').set_index(['CreateDate','CameraSerialNumber','GroupId'])
#             combined = dep.join(exifdata,rsuffix='_exif').reset_index()
#             combined =combined.drop_duplicates(subset=['CameraSerialNumber','CreateDate','GroupId','ItemId'],keep='last')
#             combined = combined.sort_values(['CorrectedTime','GroupId','ItemId'])
#             combined =combined.groupby(['CreateDate','CameraSerialNumber','GroupId'],group_keys=False).apply(calculatetimes).reset_index()
#             barnumbers = pd.read_csv(geturl('cameranames'),parse_dates=['BarStartDate','BarEndDate']) 
#             result = matchbars(combined,barnumbers,datecolumn='CalculatedStartTime')
#             result['CalculatedStartTime'] = pd.to_datetime(result['CalculatedStartTime'])
#             result['CorrectedTime'] = pd.to_datetime(result['CorrectedTime'])
#             result['Bait'] = 'Bait'
#             result['StageName'] = result.apply(lambda x: f'{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4',axis=1)
#             result =result.drop_duplicates(subset=['CameraSerialNumber','CreateDate','GroupId','ItemId'],keep='last')
            
#             result=result.groupby('PlatformName').apply(makedeploymentkey)
#             result.to_csv(targets[0],index=False)
#         return { 

#             'file_dep':[geturl('autodeployment'),geturl('exifstore'),geturl('cameranames'),geturl('timecorrection')],
#             'actions':[stagedeployments],
#             'targets':[geturl('stage')],
#             'uptodate':[True],
#             'clean':True,
#         } 

# @create_after(executed='make_autodeployments', target_regex='.*\.json') 
# def task_stage_data():
#         def hardlink(dependencies, targets):
#              stage = pd.read_csv(geturl('stage'))
#              stage['target'] =stage.apply(lambda x: os.path.join(CATALOG_DIR,'stage',x.StageDir,x.StageName),axis=1)
#              stage['SourceFile_exif'] = stage['SourceFile_exif'].apply(lambda x: x.format(**{'CATALOG_DIR':CATALOG_DIR}))
#              for index, row in stage.iterrows():
#                   if not os.path.exists(row.target):
#                     dir =os.path.split(row.target)[0]
#                     os.makedirs(dir,exist_ok=True)
#                     os.link(row.SourceFile_exif,row.target)

#         def delete_empty_folders(dryrun):
#             for dirpath, dirnames, filenames in os.walk(os.path.join(CATALOG_DIR,'stage'), topdown=False):
#                 for dirname in dirnames:
#                     full_path = os.path.join(dirpath, dirname)
#                     if not os.listdir(full_path): 
#                         if dryrun:
#                              print(f'Remove dir {full_path}')
#                         else:
#                             os.rmdir(full_path)

               
#         if os.path.exists(geturl('stage')):
#             stage = pd.read_csv(geturl('stage'))
#             targets =stage.apply(lambda x: os.path.join(CATALOG_DIR,'stage',x.StageDir,x.StageName),axis=1).unique().tolist()
#             return { 

#                 'file_dep':[geturl('stage')],
#                 'actions':[hardlink],
#                 'targets':targets,
#                 'uptodate':[True],
#                 'clean':[clean_targets,delete_empty_folders],
#             } 
        
# @create_after(executed='stage_data', target_regex='.*\.json') 
# def task_update_stationinformation():
#         def finalnames(dependencies, targets):
#             stage = pd.read_csv(geturl('stage'),index_col='StageId')
#             stage['CorrectedTime'] = pd.to_datetime(stage.CorrectedTime)
#             if geturl('QGIS').exists():
#                 def findnearest(row):
#                     row['StageId'] = None
#                     df =stage.loc[(stage.PlatformName==row.PlatformName) & (stage.ItemId==1)]
#                     if len(df)>0 :
#                         times =((row.TimeStamp -df.CorrectedTime ).dt.total_seconds()).sort_values()
#                         times = times[times>0]
#                         if times[0]<1200:
#                             delta =times.reset_index().iloc[0].StageId
#                             row['StageId'] = times.reset_index().iloc[0].StageId
#                     return row
#             qgis = pd.read_csv(geturl('QGIS'))
#             qgis =qgis[~qgis['sample'].isna()]
#             qgis['PlatformName']='CSIRO-'+qgis.left_right_cam.str.split('/',expand=True)[0].str.replace('L','')
#             qgis['TimeStamp'] =pd.to_datetime(qgis['Time_date'])
#             qgis =qgis.apply(findnearest,axis=1).set_index('StageId')
#             stage = stage.groupby('StageId')['CalculatedStartTime'].agg([('VideoStart', 'min'), ('VideoEnd', 'max')]).reset_index()
#             if os.path.exists(geturl('stationinfo')):
#                 stations = pd.read_csv(geturl('stationinfo'),index_col='StageId')
#             else:
#                 stations = pd.DataFrame(columns=['StartTime','FinishTime','CollectionId','Station','Operation','Latitude','Longitude','Depth','VideoStart','VideoEnd'])
#             stations =stations.join(stage.groupby('StageId').first(),how='outer',rsuffix='_Stage')
#             stations['PlatformName'] = stations.index
#             stations[['PlatformName','CameraTime']] =stations.PlatformName.str.split('_',expand=True)
#             stations.loc[stations.VideoStart_Stage.isna(),'VideoStart_Stage'] = stations.loc[stations.VideoStart_Stage.isna(),'VideoStart']
#             stations.loc[stations.VideoStart_Stage.isna(),'VideoEnd_Stage'] = stations.loc[stations.VideoEnd_Stage.isna(),'VideoEnd']
#             stations[['VideoStart','VideoEnd']] = stations[['VideoStart_Stage','VideoEnd_Stage']]
#             # Get the columns of both dataframes
#             scols = set(stations.columns)
#             qgiscol = set(qgis.columns)
#             common = scols.intersection(qgiscol) - {'StageId'}
#             stations = stations.drop(columns=common)
#             stations =stations.join(qgis,rsuffix='_qgis')
#             stations.Latitude = stations.latitude_dd
#             stations.Longitude = stations.longitude_dd
#             stations['CollectionId'] = 'DR2024-02'
#             stations['Station'] = stations['sample']
#             stations['Bait'] = 'Bait'
#             stations['Depth'] = stations['depth_m']
#             stations.drop(['VideoStart_Stage','VideoEnd_Stage'],axis=1).sort_values('VideoStart').to_csv(geturl('stationinfo'))
#         return { 

#                 'file_dep':[geturl('stage'),geturl('QGIS')],
#                 'targets': [geturl('stationinfo')],
#                 'actions':[finalnames],
#                 'uptodate':[run_once],
#                 'clean':True,
#             } 

# @create_after(executed='update_stationinformation', target_regex='.*\.json') 
# def task_process_names():
#     def hardlink(dependencies, targets):
#         stage = pd.read_csv(geturl('stage'))
#         targets =stage.apply(lambda x: os.path.join(geturl('deployments').resolve(),x.StageDir,x.StageName),axis=1).unique().tolist()   
#         station = pd.read_csv(geturl('stationinfo'))
#         comb =pd.merge(stage,station,on='StageId',suffixes=('','_station')).dropna(subset=['CollectionId'])
#         comb.CalculatedStartTime = pd.to_datetime(comb.CalculatedStartTime) 
#         comb.CorrectedTime = pd.to_datetime(comb.CorrectedTime) 
#         comb['CamCount'] = comb['StageDir'].str.split('_',expand=True)[[2,3]].agg('_'.join, axis=1)
#         comb['target'] =comb.apply(lambda x: os.path.join(geturl('deployments').resolve(),f'{x.CorrectedTime.strftime("%Y%m%d")}',f'{x.CollectionId}_{x.Station}_{x.StageId}_{x.Bait}_{x.CamCount}',f'{x.CollectionId}_{x.Station}_{x.PlatformName}_{x.CameraName}_{x.Bait}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4'),axis=1)
#         comb['SourceFile_exif'] = comb['SourceFile_exif'].apply(lambda x: x.format(**{'CATALOG_DIR':CATALOG_DIR}))
#         comb.to_csv(geturl('renamed'),index=False)
#         for index, row in comb.iterrows():
#             if not os.path.exists(row.target):
#                 dir =os.path.split(row.target)[0]
#                 os.makedirs(dir,exist_ok=True)
#                 os.link(row.SourceFile_exif,row.target)

#     if os.path.exists(geturl('stage')) and os.path.exists(geturl('stationinfo')):
#         stage = pd.read_csv(geturl('stage'))  
#         station = pd.read_csv(geturl('stationinfo'))
#         comb =pd.merge(stage,station,on='StageId',suffixes=('','_qgis')).dropna(subset=['CollectionId'])
#         comb.CalculatedStartTime = pd.to_datetime(comb.CalculatedStartTime) 
#         comb.CorrectedTime = pd.to_datetime(comb.CorrectedTime) 
#         targets =comb.apply(lambda x: os.path.join(geturl('deployments').resolve(),f'{x.CorrectedTime.strftime("%Y%m%d")}',f'{x.CollectionId}_{x.Station}_{x.StageId}',f'{x.CollectionId}_{x.Station}_{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4'),axis=1)
#         if len(targets)>0:
#             targets = targets.unique().tolist()
#             geturl('stationinfo')
#             return { 

#                 'file_dep':[geturl('stationinfo'),geturl('stage')],
#                 'targets': targets,
#                 'actions':[hardlink],
#                 'uptodate':[run_once],
#                 'clean':True,
#             } 








# @create_after(executed='process_names', target_regex='.*\.json') 
# def task_data_report():
#     def makereport(dependencies, targets):
#         data = pd.read_csv(dependencies[0])
#         data['Side']=data['CameraName'].str[0]
#         df =data[['StageId','TotalTime','Side']].drop_duplicates().pivot(index='StageId',columns='Side',values='TotalTime').reset_index()
#         df2=data[['TimeStamp','StageId','Station','Latitude','Longitude']].drop_duplicates()
#         result = pd.merge(df,df2)
#         result['DeploymentKey'] =data.apply(lambda x:f'{x.CollectionId}_{x.Station}_{x.StageId}',axis=1)
#         l = (pd.to_timedelta(result['L'])>pd.to_timedelta('1:20:00')).astype(int)*50
#         r = (pd.to_timedelta(result['R'])>pd.to_timedelta('1:20:00')).astype(int)*50
#         result['Qflag'] = l+ r
#         result.sort_values(['TimeStamp','StageId']).to_csv(targets[0],index=False)      
#     return { 
#         'file_dep':[geturl('renamed')],
#         'targets': [geturl('videoreport')],
#         'actions':[makereport],
#         'uptodate':[run_once],
#         'clean':True,
#     } 
# # def delete_empty_folders(dryrun):
#     for dirpath, dirnames, filenames in os.walk(os.path.join(CATALOG_DIR,'stage'), topdown=False):
#         for dirname in dirnames:
#             full_path = os.path.join(dirpath, dirname)
#             if not os.listdir(full_path): 
#                 if dryrun:
#                         print(f'Remove dir {full_path}')
#                 else:
#                     os.rmdir(full_path)

def run():
    import sys
    from doit.cmd_base import ModuleTaskLoader, get_loader
    from doit.doit_cmd import DoitMain
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    #print(globals())
    DoitMain(ModuleTaskLoader(globals())).run(sys.argv[1:])

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    #print(globals())
    doit.run(globals())