import doit
import os
import glob
from doit import get_var
import yaml
from doit.action import CmdAction
from doit.tools import title_with_actions
from doit.tools import run_once
import pandas as pd
from doit import create_after
import sqlite3
import psutil
import platform
import subprocess
import json
from pathlib import Path
import numpy as np
from doit.task import clean_targets
import shutil


cfg = None
CATALOG_DIR = None
DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
format_type = 'exfat'



def geturl(key):
    global cfg
    global CATALOG_DIR
    return(Path(cfg[key].format(CATALOG_DIR=CATALOG_DIR)))

def task_config():
        def loadconfig(config):
            global cfg
            global CATALOG_DIR
            global COLLECTION_DIR
            with open(config, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, yaml.SafeLoader)
            CATALOG_DIR = os.path.dirname(os.path.abspath(config))
            COLLECTION_DIR = Path(CATALOG_DIR).resolve().parents[2]
        config = {"config": get_var('config', f'{os.path.split(__file__)[0]}/config.yml')}
        loadconfig(config['config'])


def task_create_json():
        for path in geturl('cardstore').rglob('.'):
            file_dep = list(path.glob("*.MP4"))
            if len(file_dep)>0:
                target  = path / geturl('exifname')
                command = f'exiftool -api largefilesupport=1 -m -u -q -q -n -CameraSerialNumber -CreateDate -TrackCreateDate -SourceFile -Duration -Rate -FileSize -FieldOfView -json -ext MP4 {path} > {target} || :'
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
def task_concat_json():

        def concat(dependencies, targets):
            data = pd.concat([ pd.read_json(dep) for dep in dependencies])
            data['Bad'] =data['CreateDate'].isna()
            data['SourceFile'] = data.apply(lambda x: f"{{CATALOG_DIR}}/{os.path.relpath(x['SourceFile'],CATALOG_DIR)}",axis=1)
            data['Directory']=data['SourceFile'].apply(lambda x: os.path.split(x)[0])
            data['FileName'] = data['SourceFile'].apply(os.path.basename)
            data[['ItemId','GroupId']]=data.FileName.str.extract('(?P<item>\d\d)(?P<group>\d\d\d\d).MP4')
            data =data.sort_values(['SourceFile'])
            data['CreateDate'] =pd.to_datetime(data.CreateDate,format='%Y:%m:%d  %H:%M:%S')
            #ok lets try and fix missing data from bad videos
            data['SpeedUp'] =pd.to_numeric(data.Rate.str.extract(r'(\d+)X')[0])
            data['AdjustedDuration'] = data['SpeedUp']* data['Duration']
            data['RunTime'] = data.groupby(['CameraSerialNumber','GroupId'])['AdjustedDuration'].cumsum()
            data['EndTime'] = data['CreateDate']+pd.to_timedelta(data['RunTime'],unit='S')
            data =data.sort_values(['SourceFile'])
            data.to_csv(targets[0],index=False)
        exiffiles = list(geturl('cardstore').rglob(cfg['exifname']))
        if exiffiles:
            return { 

                'file_dep':exiffiles,
                'actions':[concat],
                'targets':[geturl('exifstore')],
                'uptodate':[True],
                'clean':True,
            } 
        
@create_after(executed='concat_json', target_regex='.*\.json') 
def task_report_json():

        def exifreport(dependencies, targets):
            data = pd.read_csv(dependencies[0])
            data['Bad'] =data['CreateDate'].isna()
            data['SourceFile'] = data.apply(lambda x: f"{{CATALOG_DIR}}/{os.path.relpath(x['SourceFile'],CATALOG_DIR)}",axis=1)
            data['Directory']=data['SourceFile'].apply(lambda x: os.path.split(x)[0])
            data['FileName'] = data['SourceFile'].apply(os.path.basename)
            data[['ItemId','GroupId']]=data.FileName.str.extract('(?P<item>\d\d)(?P<group>\d\d\d\d).MP4')
            data =data.sort_values(['SourceFile'])
            data['CreateDate'] =pd.to_datetime(data.CreateDate)
            #ok lets try and fix missing data from bad videos
            data['SpeedUp'] =pd.to_numeric(data.Rate.str.extract(r'(\d+)X')[0])
            data['AdjustedDuration'] = data['SpeedUp']* data['Duration']
            top =data.groupby(['CameraSerialNumber','GroupId']).first()
            top['TotalRunTime'] = data.groupby(['CameraSerialNumber','GroupId'])['AdjustedDuration'].sum()
            top['EndTime'] = pd.to_timedelta(top['TotalRunTime'],unit='s')+top['CreateDate']
            top =top.sort_values(['SourceFile'])
            top.to_csv(targets[0],index=False)
        return { 

            'file_dep':[geturl('exifstore')],
            'actions':[exifreport],
            'targets':[geturl('exifreport')],
            'uptodate':[True],
            'clean':True,
        }        
    
@create_after(executed='concat_json', target_regex='.*\.json') 
def task_checkbars():
        def checkbars(dependencies, targets):
            data = pd.read_csv(dependencies[0],parse_dates=['CreateDate'])
            data.Duration =pd.to_timedelta(data.Duration,unit='s')
            data['CreateEnd'] =  data.CreateDate + data.Duration
            data['CreateStart'] =  data.CreateDate
            cstart =data.groupby('CameraSerialNumber')['CreateStart'].min()
            cend = data.groupby('CameraSerialNumber')['CreateEnd'].max()
            stats =pd.concat([cstart,cend],axis=1) 
            if os.path.exists(targets[0]):
                barnumbers = pd.read_csv(targets[0],parse_dates=['StartDate','EndDate']).drop(['CreateStart','CreateEnd'],axis=1,errors='ignore')                      
            else:
                 barnumbers = pd.DataFrame(columns=['PlatformName','CameraName','CameraSerialNumber','StartDate','EndDate','HousingNumber','CalibrationDate'])
            barnumbers = pd.merge(barnumbers,stats.reset_index(),how='outer')
            barnumbers.to_csv(targets[0],index=False)

        return { 

            'file_dep':[geturl('exifstore')],
            'actions':[checkbars],
            'targets':[geturl('cameranames')],
            'uptodate':[True],
            'clean':True,
        }         

@create_after(executed='checkbars', target_regex='.*\.json') 
def task_make_autodeployments():

        def deployments(dependencies, targets):
            data = pd.read_csv(geturl('exifstore'),parse_dates=['CreateDate'])
            totaltime =pd.to_datetime(data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['Duration'].sum(), unit='s').dt.strftime("%H:%M:%S").rename('TotalTime')
            totalfilesize =(data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['FileSize'].sum()/1000000000).rename('TotalSize')
            maxid =data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['ItemId'].max().rename('MaxId')
            minid =data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['ItemId'].min().rename('MinId')
            filecount = data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])['ItemId'].count().rename('FileCount')
            groups =data.groupby(['Directory','CreateDate','CameraSerialNumber','GroupId'])[['SourceFile','FieldOfView']].first()
            output =groups.join(filecount).join(minid).join(maxid).join(totalfilesize).join(totaltime)

            barnumbers = pd.read_csv(geturl('cameranames'),parse_dates=['StartDate','EndDate']) 
            result = matchbars(output.reset_index(),barnumbers)
            result['CreateDate'] = pd.to_datetime(result['CreateDate'] )
            result['DeploymentId']=result.apply(lambda x: f"{x.CreateDate.strftime('%Y%m%dT%H%M%S')}_{x.PlatformName}_{x.CameraName}_{x.CameraSerialNumber}_{x.GroupId:02}", axis=1)
            manualfile = geturl('timecorrection')
            manual =result.loc[:, ['DeploymentId', 'TotalTime','CreateDate','SourceFile']]
            manual =manual.set_index('DeploymentId')
            if os.path.exists(manualfile):
                 old = pd.read_csv(manualfile,index_col='DeploymentId')
                 manual =manual.join(old['CorrectedTime'])
                 manual.loc[manual.CorrectedTime.isnull(),'CorrectedTime']=manual.loc[manual.CorrectedTime.isnull(),'CreateDate']
            else:
                manual['CorrectedTime'] = manual['CreateDate']
            manual['SourceFile'] = manual['SourceFile'].apply(lambda x: f'=HYPERLINK("file://{x}", "{os.path.basename(x)}")')
            manual.sort_values('DeploymentId').to_csv(manualfile)
            manual.sort_values('DeploymentId').to_excel(manualfile.with_suffix('.xlsx'))
            result.to_csv(targets[0],index=False)



        target = geturl('autodeployment')
        return { 

            'file_dep':[geturl('exifstore')],
            'actions':[deployments],
            'targets':[target,geturl('timecorrection')],
            'uptodate':[True],
            'clean':True,
        } 

def matchbars(deployments,barnumbers,datecolumn='CreateDate'):
    conn = sqlite3.connect(':memory:')
    #write the tables
    # Drop columns from the DataFrame if they exist in the list
    barnumbers.to_sql('bars', conn, index=False)
    deployments[deployments.columns[~deployments.columns.isin(['PlatformName','HousingNumber','CameraName','StartDate','EndDate','CalibrationDate'])]].to_sql('deployments', conn, index=False)
    qry = f'''
        select  
            deployments.*,
            bars.PlatformName,
            bars.HousingNumber,
            bars.CameraName,
            bars.StartDate,
            bars.EndDate,
            bars.CalibrationDate
        from
            deployments join bars on
            (deployments.{datecolumn} between bars.StartDate and bars.EndDate) and
            (deployments.CameraSerialNumber = bars.CameraSerialNumber)
        '''
    result =pd.read_sql_query(qry, conn)
    result['CreateDate'] = pd.to_datetime(result['CreateDate'] )
    result['DeploymentId']=result.apply(lambda x: f"{x.CreateDate.strftime('%Y%m%dT%H%M%S')}_{x.PlatformName}_{x.CameraName}_{x.CameraSerialNumber}", axis=1)
    return result


@create_after(executed='make_autodeployments', target_regex='.*\.json') 
def task_make_matchbars():

        def stagedeployments(dependencies, targets):
            def calculatetimes(df):
                df =df.sort_values('ItemId')
                if len(df)>1:
                    start = df.Duration.cumsum().shift(+1)
                    start.iloc[0] = 0
                    start =pd.to_timedelta(start,unit='S')
                    df['CalculatedStartTime']=(df['CorrectedTime']+start).dt.round('1S')
                else:                 
                    df['CalculatedStartTime']=df['CorrectedTime']
                return df

            def makedeploymentkey(df):
                def makedirs(row):
                    left = 0
                    right = 0
                    if leftcam in row.keys():
                         left = row[leftcam]
                    if rightcam in row.keys():
                         right = row[rightcam]
                    if right==left:
                        result = f"{row['StageId']}_{int(left):02}"
                    else:
                        result = f"{row['StageId']}_{int(left):02}_{int(right):02}"
                    return row['StageId'],result
                leftcam = df.CameraName[df.CameraName.str.startswith('L')].min()
                rightcam = df.CameraName[df.CameraName.str.startswith('R')].min()
                left = df[df.CameraName==leftcam].groupby('CorrectedTime').first().reset_index()[['CorrectedTime','CameraName','PlatformName']].add_suffix('_Left')
                left['MatchTime'] = left['CorrectedTime_Left']
                right = df[df.CameraName==rightcam].groupby('CorrectedTime').first().reset_index()[['CorrectedTime','CameraName','PlatformName']].add_suffix('_Right')
                right['MatchTime'] = right['CorrectedTime_Right']
                merged_df = pd.merge_asof(right, left, left_on='MatchTime', right_on='MatchTime', direction='nearest', tolerance=pd.Timedelta(minutes=30),suffixes=( '_right','_left'))
                merged_df =pd.concat([merged_df,left[~left.CorrectedTime_Left.isin(merged_df.CorrectedTime_Left.unique())]])
                merged_df.loc[merged_df.CorrectedTime_Right.isna(),'PlatformName_Right'] = merged_df.loc[merged_df.CorrectedTime_Right.isna(),'PlatformName_Left']
                merged_df.loc[merged_df.CorrectedTime_Left.isna(),'CorrectedTime_Left'] = merged_df.loc[merged_df.CorrectedTime_Left.isna(),'CorrectedTime_Right']
                starttime =merged_df.MatchTime.dt.strftime("%Y%m%dT%H%M%S")
                merged_df['StageId']=merged_df.PlatformName_Right+'_'+starttime
                stageId =pd.concat((merged_df[['CorrectedTime_Left','StageId']].rename(columns={'CorrectedTime_Left':'CorrectedTime'}),merged_df[['CorrectedTime_Right','StageId']].rename(columns={'CorrectedTime_Right':'CorrectedTime'}))).dropna()
                df =pd.merge(df,stageId)
                totals =df.groupby(['StageId','CameraName']).size().reset_index().pivot_table(index ='StageId',values=0,columns='CameraName').reset_index().fillna(0)
                totals =totals.apply(makedirs,axis=1).apply(pd.Series)
                totals.columns = ['StageId','StageDir']
                df = df.merge(totals)
                return df
            dep = pd.read_csv(geturl('autodeployment'),parse_dates=['CreateDate'])
            exifdata = pd.read_csv(geturl('exifstore'),parse_dates=['CreateDate']).set_index(['CreateDate','CameraSerialNumber','GroupId'])
            correcttimes = pd.read_csv(geturl('timecorrection'),parse_dates=['CreateDate','CorrectedTime'])
            dep =pd.merge(dep,correcttimes[['DeploymentId','CorrectedTime']],on='DeploymentId', how='left').set_index(['CreateDate','CameraSerialNumber','GroupId'])
            combined = dep.join(exifdata,rsuffix='_exif').reset_index()
            combined =combined.drop_duplicates(subset=['CameraSerialNumber','CreateDate','GroupId','ItemId'],keep='last')
            combined = combined.sort_values(['CorrectedTime','GroupId','ItemId'])
            combined =combined.groupby(['CreateDate','CameraSerialNumber','GroupId'],group_keys=False).apply(calculatetimes).reset_index()
            barnumbers = pd.read_csv(geturl('cameranames'),parse_dates=['StartDate','EndDate']) 
            result = matchbars(combined,barnumbers,datecolumn='CalculatedStartTime')
            result['CalculatedStartTime'] = pd.to_datetime(result['CalculatedStartTime'])
            result['CorrectedTime'] = pd.to_datetime(result['CorrectedTime'])
            result['StageName'] = result.apply(lambda x: f'{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4',axis=1)
            result =result.drop_duplicates(subset=['CameraSerialNumber','CreateDate','GroupId','ItemId'],keep='last')
            result=result.groupby('PlatformName').apply(makedeploymentkey)
            result.to_csv(targets[0],index=False)
        return { 

            'file_dep':[geturl('autodeployment'),geturl('exifstore'),geturl('cameranames')],
            'actions':[stagedeployments],
            'targets':[geturl('stage')],
            'uptodate':[True],
            'clean':True,
        } 

@create_after(executed='make_autodeployments', target_regex='.*\.json') 
def task_stage_data():
        def hardlink(dependencies, targets):
             stage = pd.read_csv(geturl('stage'))
             stage['target'] =stage.apply(lambda x: os.path.join(CATALOG_DIR,'stage',x.StageDir,x.StageName),axis=1)
             stage['SourceFile'] = stage['SourceFile'].apply(lambda x: x.format(**{'CATALOG_DIR':CATALOG_DIR}))
             for index, row in stage.iterrows():
                  if not os.path.exists(row.target):
                    dir =os.path.split(row.target)[0]
                    os.makedirs(dir,exist_ok=True)
                    os.link(row.SourceFile,row.target)

        def delete_empty_folders(dryrun):
            for dirpath, dirnames, filenames in os.walk(os.path.join(CATALOG_DIR,'stage'), topdown=False):
                for dirname in dirnames:
                    full_path = os.path.join(dirpath, dirname)
                    if not os.listdir(full_path): 
                        if dryrun:
                             print(f'Remove dir {full_path}')
                        else:
                            os.rmdir(full_path)

               
        if os.path.exists(geturl('stage')):
            stage = pd.read_csv(geturl('stage'))
            targets =stage.apply(lambda x: os.path.join(CATALOG_DIR,'stage',x.StageDir,x.StageName),axis=1).unique().tolist()
            return { 

                'file_dep':[geturl('stage')],
                'actions':[hardlink],
                'targets':targets,
                'uptodate':[True],
                'clean':[clean_targets,delete_empty_folders],
            } 
        
@create_after(executed='stage_data', target_regex='.*\.json') 
def task_update_stationinformation():
        def finalnames(dependencies, targets):
             stage = pd.read_csv(geturl('stage'),index_col='StageId')
             stage = stage.groupby('StageId')['CalculatedStartTime'].agg([('VideoStart', 'min'), ('VideoEnd', 'max')]).reset_index()
             if os.path.exists(geturl('stationinfo')):
                stations = pd.read_csv(geturl('stationinfo'),index_col='StageId')
             else:
                stations = pd.DataFrame(columns=['StartTime','FinishTime','CollectionId','Station','Operation','Latitude','Longitude','Depth','VideoStart','VideoEnd'])
             stations =stations.join(stage.groupby('StageId').first(),how='outer',rsuffix='_Stage')
             stations['PlatformName'] = stations.index
             stations[['PlatformName','CameraTime']] =stations.PlatformName.str.split('_',expand=True)
             stations.loc[stations.VideoStart_Stage.isna(),'VideoStart_Stage'] = stations.loc[stations.VideoStart_Stage.isna(),'VideoStart']
             stations.loc[stations.VideoStart_Stage.isna(),'VideoEnd_Stage'] = stations.loc[stations.VideoEnd_Stage.isna(),'VideoEnd']
             stations[['VideoStart','VideoEnd']] = stations[['VideoStart_Stage','VideoEnd_Stage']]
             stations.drop(['VideoStart_Stage','VideoEnd_Stage'],axis=1).sort_values('VideoStart').to_csv(geturl('stationinfo'))          
        return { 

            'file_dep':[geturl('stage')],
            'targets': [geturl('stationinfo')],
            'actions':[finalnames],
            'uptodate':[run_once],
            'clean':True,
        } 

@create_after(executed='update_stationinformation', target_regex='.*\.json') 
def task_process_names():
    def hardlink(dependencies, targets):
        stage = pd.read_csv(geturl('stage'))
        targets =stage.apply(lambda x: os.path.join(geturl('deployments').resolve(),x.StageDir,x.StageName),axis=1).unique().tolist()   
        station = pd.read_csv(geturl('stationinfo'))
        comb =pd.merge(stage,station).dropna(subset=['CollectionId'])
        comb.CalculatedStartTime = pd.to_datetime(comb.CalculatedStartTime) 
        comb.CorrectedTime = pd.to_datetime(comb.CorrectedTime) 
        comb['target'] =comb.apply(lambda x: os.path.join(geturl('deployments').resolve(),f'{x.CorrectedTime.strftime("%Y%m%d")}',f'{x.CollectionId}_{x.Station}_{x.StageId}',f'{x.CollectionId}_{x.Station}_{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4'),axis=1)
        comb['SourceFile_exif'] = comb['SourceFile_exif'].apply(lambda x: x.format(**{'CATALOG_DIR':CATALOG_DIR}))
        comb.to_csv(geturl('renamed'),index=False)
        for index, row in comb.iterrows():
            if not os.path.exists(row.target):
                dir =os.path.split(row.target)[0]
                os.makedirs(dir,exist_ok=True)
                os.link(row.SourceFile_exif,row.target)

    if os.path.exists(geturl('stage')) and os.path.exists(geturl('stationinfo')):
        stage = pd.read_csv(geturl('stage'))  
        station = pd.read_csv(geturl('stationinfo'))
        comb =pd.merge(stage,station).dropna(subset=['CollectionId'])
        comb.CalculatedStartTime = pd.to_datetime(comb.CalculatedStartTime) 
        comb.CorrectedTime = pd.to_datetime(comb.CorrectedTime) 
        targets =comb.apply(lambda x: os.path.join(geturl('deployments').resolve(),f'{x.CorrectedTime.strftime("%Y%m%d")}',f'{x.CollectionId}_{x.Station}_{x.StageId}',f'{x.CollectionId}_{x.Station}_{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4'),axis=1)
        if len(targets)>0:
            targets = targets.unique().tolist()
            geturl('stationinfo')
            return { 

                'file_dep':[geturl('stationinfo'),geturl('stage')],
                'targets': targets,
                'actions':[hardlink],
                'uptodate':[run_once],
                'clean':True,
            } 



# def task_generate_cameraqr():
#     import qrcode
#     from PIL import Image, ImageDraw, ImageFont

#     def generate_qr_codes_from_csv(csv_file, output_folder):
#         with open(csv_file, newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             for i, row in enumerate(reader):
#                 # Assuming each line contains one data point
#                 data = row[0]
                
#                 # Generate QR code
#                 qr = qrcode.QRCode(
#                     version=1,
#                     error_correction=qrcode.constants.ERROR_CORRECT_L,
#                     box_size=10,
#                     border=4,
#                 )
#                 qr.add_data(data)
#                 qr.make(fit=True)
#                 img = qr.make_image(fill_color="black", back_color="white")
                
#                 # Add human-readable text
#                 draw = ImageDraw.Draw(img)
#                 font = ImageFont.load_default()
#                 text = data
#                 text_width, text_height = draw.textsize(text, font)
#                 draw.text(((img.width - text_width) // 2, img.height - text_height - 10), text, fill="black", font=font)
                
#                 # Save QR code image
#                 img.save(f"{output_folder}/qrcode_{i+1}.png")

# # Example usage
# csv_file = "data.csv"
# output_folder = "qrcodes_with_text"
# generate_qr_codes_from_csv(csv_file, output_folder)


@create_after(executed='update_stationinformation', target_regex='.*\.json') 
def task_make_srt():
    def makesubtitle(dependencies, targets):
        stage = pd.read_csv(geturl('stage'))
        targets =stage.apply(lambda x: os.path.join(geturl('deployments').resolve(),x.StageDir,x.StageName),axis=1).unique().tolist()   
        station = pd.read_csv(geturl('stationinfo'))
        comb =pd.merge(stage,station).dropna(subset=['CollectionId'])
        comb.CalculatedStartTime = pd.to_datetime(comb.CalculatedStartTime) 
        comb.CorrectedTime = pd.to_datetime(comb.CorrectedTime) 
        comb['target'] =comb.apply(lambda x: os.path.join(geturl('deployments').resolve(),f'{x.CorrectedTime.strftime("%Y%m%d")}',f'{x.CollectionId}_{x.Station}_{x.StageId}',f'{x.CollectionId}_{x.Station}_{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4'),axis=1)
        comb['SourceFile_exif'] = comb['SourceFile_exif'].apply(lambda x: x.format(**{'CATALOG_DIR':CATALOG_DIR}))
        comb.to_csv(geturl('renamed'),index=False)
        for index, row in comb.iterrows():
            if not os.path.exists(row.target):
                dir =os.path.split(row.target)[0]
                os.makedirs(dir,exist_ok=True)
                os.link(row.SourceFile_exif,row.target)

    if geturl('renamed').exists():
        names = pd.read_csv()
        stage = pd.read_csv(geturl('stage'))  
        station = pd.read_csv(geturl('stationinfo'))
        comb =pd.merge(stage,station).dropna(subset=['CollectionId'])
        comb.CalculatedStartTime = pd.to_datetime(comb.CalculatedStartTime) 
        comb.CorrectedTime = pd.to_datetime(comb.CorrectedTime) 
        targets =comb.apply(lambda x: os.path.join(geturl('deployments').resolve(),f'{x.CorrectedTime.strftime("%Y%m%d")}',f'{x.CollectionId}_{x.Station}_{x.StageId}',f'{x.CollectionId}_{x.Station}_{x.PlatformName}_{x.CameraName}_{x.CalculatedStartTime.strftime("%Y%m%dT%H%M%S")}_{x.CameraSerialNumber}_{int(x.GroupId):02d}_{int(x.ItemId):02d}.MP4'),axis=1)
        if len(targets)>0:
            targets = targets.unique().tolist()
            geturl('stationinfo')
            return { 

                'file_dep':[geturl('stationinfo'),geturl('stage')],
                'targets': targets,
                'actions':[hardlink],
                'uptodate':[run_once],
                'clean':True,
            } 

# @create_after(executed='process_names', target_regex='.*\.json') 
# def task_make_calfiles():
#     def cals(dependencies, targets):
#         for file in targets:
#              if not os.path.exists(file):
#                   shutil.copy(os.path.join(geturl('calstore'),'GoPro9WideWater_default.CamCAL'),file)

#     if os.path.exists(geturl('barstore')):
#         bars = pd.read_csv(geturl('barstore'),parse_dates=['CalibrationDate'])
#         targets=bars.apply(lambda x: f"{geturl('calstore')}/{x.CameraName}_{x.CalibrationDate.strftime('%Y%m%d')}_{x.CameraSerialNumber}.CamCAL",axis=1).tolist()
#         return { 

#             'file_dep':[geturl('barstore')],
#             'targets': targets,
#             'actions':[cals],
#             'uptodate':[run_once],
#             'clean':True,
#         }    

# @create_after(executed='make_calfiles', target_regex='.*\.json') 
# def task_move_calfiles():
#     def movecals(dependencies, targets):
#         renamed = pd.read_csv(geturl('renamed'),parse_dates=['CalibrationDate'])
#         renamed['DeploymentPath']=renamed.target.apply(lambda x: os.path.split(x)[0])
#         renamed = renamed.groupby(['DeploymentPath','CameraName']).first().reset_index()
#         renamed['CalibrationSouce']=renamed.apply(lambda x: f"{geturl('calstore')}/{x.CameraName}_{x.CalibrationDate.strftime('%Y%m%d')}_{x.CameraSerialNumber}.CamCAL",axis=1).tolist()
#         for index,row in renamed.iterrows():
#                   shutil.copy(row.CalibrationSouce,os.path.join(row.DeploymentPath,os.path.basename(row.CalibrationSouce)))
#     if os.path.exists(geturl('renamed')):
#         return { 

#             'file_dep':[geturl('barstore'),geturl('renamed')],
#             'actions':[movecals],
#             'uptodate':[run_once],
#             'clean':True,
#         } 
def delete_empty_folders(dryrun):
    for dirpath, dirnames, filenames in os.walk(os.path.join(CATALOG_DIR,'stage'), topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path): 
                if dryrun:
                        print(f'Remove dir {full_path}')
                else:
                    os.rmdir(full_path)

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