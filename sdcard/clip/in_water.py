from doit import get_var
import yaml
from doit.action import CmdAction
from doit.tools import run_once
import pandas as pd
from doit import create_after
from pathlib import Path
import numpy as np
from doit.task import clean_targets
import cv2
from sdcard.config import cfg

# Task to extract images from video files at specified intervals
# @create_after(executed='scan_gopros', target_regex='.*\.json')                     
def task_make_gopro_stills():
     def is_directory_empty(directory):
        """Check if the specified directory is empty."""
        return len(list(output.glob('*.jpg'))) > 0
     frame_interval = cfg.data.get('frame_interval', 10)  # Default to 1 frame every 10 seconds if not specified
     for video in cfg.get_path('stage_path').rglob('*.MP4'):
            output = (video.parent / (video.stem+'_stills')).resolve()
            output.mkdir(exist_ok=True)
            #command = f'ffmpeg -i {video.resolve()} -vf fps=1/{frame_interval} -q:v 2  {output / video.stem}_%%04d.jpg'
            command = f'ffmpeg -skip_frame nokey -i {video.resolve()} -vsync vfr -q:v 2  -frame_pts 1 {output / video.stem}_%%010d.jpg'
            yield { 
                'name':output,
                'file_dep':[video],
                'actions':[command],
                'uptodate':[run_once,is_directory_empty(output)],
                'clean':True,
            }   
    
def task_make_gopro_stills_timing():
     def is_directory_empty(directory):
        """Check if the specified directory is empty."""
        return len(list(output.glob('*.jpg'))) > 0
     frame_interval = cfg.data.get('frame_interval', 10)  # Default to 1 frame every 10 seconds if not specified
     for video in cfg.get_path('stage_path').rglob('*.MP4'):
            output = (video.parent / (video.stem+'_stills')).resolve()
            output.mkdir(exist_ok=True)
            command = f'ffprobe -select_streams v:0 -show_frames -show_entries \
                        frame=pts_time,pkt_pts,pict_type,key_frame \
                        -of csv=p=0 {video.resolve()} > {output / video.stem}_frames.csv'

            yield { 
                'name':output,
                'file_dep':[video],
                'actions':[command],
                'targets':[output / f"{video.stem}_frames.csv"],
                'uptodate':[run_once],
                'clean':True,
            } 
# @create_after(executed='make_gopro_stills', target_regex='.*\.json') 
# def task_classify_images():
#     def calculate_image_his(dependencies, targets):
#         data = pd.read_json(dependencies[0])
#         cameras = pd.read_csv(data_path / "camera_summary.csv")
#         data = pd.merge(data, cameras[['CameraSerialNumber','CameraName']], on='CameraSerialNumber', how='left')
#         data = data[(data.CameraName.str.contains('DCAM')==True) & (data.FileType=='JPEG')].copy()
#         data['DateTimeOriginal'] = pd.to_datetime(data.DateTimeOriginal,format='%Y:%m:%d %H:%M:%S', errors='coerce')
#         data['Idx'] = 0
#         data['Burst'] = 0
#         data.loc[data.Model=='AC004',['Burst','Idx']]=data.loc[data.Model=='AC004','FileName'].str.extract(r'DJI_(?P<DateTime>\d{14})_(?P<Burst>\d{4})_D_(?P<Idx>\d{3})')[['Burst','Idx']]
#         data.loc[(data.Model.str.contains('HERO')==True) & (data.FileType=='JPEG'),['Burst','Idx']]=data.loc[(data.Model.str.contains('HERO')==True) & (data.FileType=='JPEG') ,'FileName'].str.extract(r'.(?P<Burst>([A-Za-z]{3}|\d{3}))(?P<Idx>.{4})')[['Burst','Idx']]
#         files = data.groupby('Burst').first()['SourceFile'].to_list()
#         from PIL import Image
#         from sdcard.clipClassifier import CLIPClassifier
#         if runtime_config['image_classifier'] is None:
#             runtime_config['image_classifier'] = image_classifier = CLIPClassifier({'clipboard':'/media/mor582/Timor2/TL-202501/training/clipboard.JPG',
#                                          'seagrass':'/media/mor582/Timor2/TL-202501/training/seagrass.JPG',
#                                          'mud':'/media/mor582/Timor2/TL-202501/training/mud.JPG',
#                                          'water':'/media/mor582/Timor2/TL-202501/training/water.JPG',
#                                          'coral':'/media/mor582/Timor2/TL-202501/training/coral.JPG'
#                                          ''})
#         else:
#             image_classifier = runtime_config['image_classifier']
#         results = image_classifier.classify_batch(files)
#         data =pd.DataFrame(results)
#         data.columns=['SourceFile','Class','Score']
#         data.to_csv(targets[0],index=False)

    

#     data_path = Path(runtime_config['data_path'])
#     for exif_file in data_path.rglob('.exif.json'):
#         if 'DCAM' in str(exif_file.parent):
#             target = exif_file.parent / '.classify.csv'
#             yield { 
#                 'name':exif_file,
#                 'file_dep':[exif_file],
#                 'actions':[calculate_image_his],
#                 'targets':[target],
#                 'uptodate':[run_once],
#                 'clean':True,
#             }
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    #print(globals())
    doit.run(globals())            