import doit
import os
import glob
from doit import get_var
from flask import config
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
from sdcard.config import Config
import shutil


def task_extract_gpx_waypoints():
    """
    Extract all waypoints from GPX files listed in the config file to a CSV file.
    """

    def extract(dependencies, targets):
        import gpxpy
        import pandas as pd
        from pathlib import Path
        gpx_file = dependencies[0]
        output_waypoints = list(filter(lambda x: 'waypoint' in x,targets))[0]
        output_tracks = list(filter(lambda x: 'track' in x,targets))[0]
        waypoints = []
        tracks = []
        with open(gpx_file, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
            for wpt in gpx.waypoints:
                waypoints.append({
                    'gpx_file': str(gpx_file),
                    'waypoint_name': wpt.name or '',
                    'gps_latitude': wpt.latitude,
                    'gps_longitude': wpt.longitude,
                    'gps_elevation': wpt.elevation or '',
                    'gps_timestamp_utc': wpt.time.isoformat() if wpt.time else ''
                })
            for track in gpx.tracks:
                for i, segment in enumerate(track.segments):
                    for j, point in enumerate(segment.points):
                        tracks.append({
                            'gpx_file': str(gpx_file),
                            'track_name': track.name or '',
                            'segment_index': i,
                            'point_index': j,
                            'gps_latitude': point.latitude,
                            'gps_longitude': point.longitude,
                            'gps_elevation': point.elevation or '',
                            'gps_timestamp_utc': point.time.isoformat() if point.time else ''
                        })
        # Write waypoints
        if waypoints:
            dfw = pd.DataFrame(waypoints, columns=['gpx_file', 'waypoint_name', 'gps_latitude', 'gps_longitude', 'gps_elevation', 'gps_timestamp_utc'])
            dfw['gps_timestamp_utc'] = pd.to_datetime(dfw['gps_timestamp_utc'])
            dfw['gps_timestamp_local'] = dfw['gps_timestamp_utc'].dt.tz_convert(config.data['time_zone'])
            dfw.to_csv(output_waypoints, index=False)
        # Write tracks
        if tracks:
            dft = pd.DataFrame(tracks, columns=['gpx_file', 'track_name', 'segment_index', 'point_index', 'gps_latitude', 'gps_longitude', 'gps_elevation', 'gps_timestamp_utc'])
            dft['gps_timestamp_utc'] = pd.to_datetime(dft['gps_timestamp_utc'])
            dft['gps_timestamp_local'] = dft['gps_timestamp_utc'].dt.tz_convert(config.data['time_zone'])
            dft.to_csv(output_tracks, index=False)

    config = Config(get_var('config',None))      
    if config.get_path('gps_store').exists():
        import gpxpy
        for file in config.get_path('gps_store').glob("*.gpx"):
            with open(file, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
                targets = []
                if  bool(gpx.waypoints):
                    targets.append(file.parent / f"{file.stem}_waypoints.csv")
                if  bool(gpx.tracks):
                    targets.append(file.parent / f"{file.stem}_tracks.csv")
            if targets:
                yield {
                    'name': targets[0],
                    'file_dep': [file],
                    'actions': [extract],
                    'targets': targets,
                    'verbosity': 2,
                }

@create_after(executed='extract_gpx_waypoints', target_regex='.*\.csv') 
def task_all_gpx_waypoints():
    """
    Aggregate all waypoints and tracks from individual CSV files into combined CSV files.
    """
    def aggregate(dependencies, targets):
        import pandas as pd
        data = pd.concat([pd.read_csv(dep) for dep in dependencies], ignore_index=True)
        #remove duplicates based on waypoint_name, gps_timestamp_utc
        data = data.drop_duplicates(subset=['waypoint_name', 'gps_timestamp_utc'])
        data.to_csv(targets[0], index=False)

    config = Config(get_var('config',None))      
    if config.get_path('gps_store').exists():
        file_dep = list(config.get_path('gps_store').glob("*_waypoints.csv"))
        target = config.get_path('report_path') / "waypoints.csv"
        if file_dep:
            return {
                'file_dep': file_dep,
                'actions': [aggregate],
                'targets': [target],
                'verbosity': 2,
            }




def task_add_waypoints():
    # Add Waypoints to ops.csv
    def add_waypoints(dependencies, targets):
        import pandas as pd
        waypoints = pd.read_csv(dependencies[0])
        ops = pd.read_csv(targets[0])
        if 'waypoint_name' not in ops.columns:
            ops['waypoint_name'] = ""
        if 'gps_timestamp_utc' not in ops.columns:
            ops['gps_timestamp_utc'] = ""
        if 'gps_timestamp_local' not in ops.columns:
            ops['gps_timestamp_local'] = ""
        if 'gps_latitude' not in ops.columns:
            ops['gps_latitude'] = ""
        if 'gps_longitude' not in ops.columns:
            ops['gps_longitude'] = ""
        ops = pd.merge(ops, waypoints[['waypoint_name', 'gps_timestamp_utc', 'gps_timestamp_local','gps_latitude','gps_longitude']], how='outer', on=['gps_timestamp_utc'])  
        # Collapse duplicate columns from merge - use waypoint data to fill missing ops data       
        ops['gps_latitude'] = ops['gps_latitude_x'].fillna(ops['gps_latitude_y'])
        ops['gps_longitude'] = ops['gps_longitude_x'].fillna(ops['gps_longitude_y'])
        ops['gps_timestamp_local'] = ops['gps_timestamp_local_x'].fillna(ops['gps_timestamp_local_y'])

        ops = ops.drop(['gps_latitude_x', 'gps_latitude_y'], axis=1)
        ops = ops.drop(['gps_longitude_x', 'gps_longitude_y'], axis=1)

        ops = ops.drop(['gps_timestamp_local_x', 'gps_timestamp_local_y'], axis=1)
        #ok now fill missing values in ops from waypoints
        ops.loc[ops['latitude_dd'].isna(), 'latitude_dd'] = ops.loc[ops['latitude_dd'].isna(), 'gps_latitude']
        ops.loc[ops['longitude_dd'].isna(), 'longitude_dd'] = ops.loc[ops['longitude_dd'].isna(), 'gps_longitude']
        ops.loc[ops['time_date'].isna(), 'time_date'] = ops.loc[ops['time_date'].isna(), 'gps_timestamp_local']
        # Save the updated ops file
        ops.to_csv(targets[0], index=False)

    config = Config(get_var('config',None))
    file_dep = config.get_path('report_path') / "waypoints.csv"
    target = config.get_path('station_path')
    if file_dep.exists() and target.exists():
        return {
                    'file_dep': [file_dep],
                    'actions': [add_waypoints],
                    'targets': [target],
                    'verbosity': 2,
                    'uptodate':[False],
                }
  

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