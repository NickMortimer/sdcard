
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime,  timedelta
import uuid
import psutil
from math import  ceil
import typer
import yaml
import glob
import pandas as pd
import subprocess
import shlex
import logging
import os
from io import StringIO
import platform
import re

def get_card_number_from_import_yml(file_path: Path):
    """
    Extract card number from import.yml file.
    
    Args:
        file_path (Path): Path to the import.yml file.
    
    Returns:
        int: Card number extracted from the file, or 0 if not found.
    """
    if file_path.exists():
        try:
            with open(file_path, 'r') as stream:
                data = yaml.safe_load(stream)
                return data.get('card_number', 0)
        except yaml.YAMLError as exc:
            logging.error(f"Error reading {file_path}: {exc}")
            return 0
    return 0

def get_available_cards(format_type='exfat',maxcardsize=512):
    """Get list of available SD cards"""
    drives = []
    for part in psutil.disk_partitions():
        if part.fstype.lower() == format_type.lower():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                if usage.total < maxcardsize * 1024**3:  # Less than maxcardsize GB
                    usb_host = get_usb_info(part.device)
                    if usb_host is not None:
                        drives.append({
                            'mountpoint': part.mountpoint,
                            'host': f"usb{usb_host}",
                            'size_gb': round(usage.total / (1024**3), 2),
                            'free_gb': round(usage.free / (1024**3), 2),
                            'card_number': get_card_number_from_import_yml(Path(part.mountpoint) / "import.yml")
                        })
            except Exception as e:
                print(f"Error processing {part.device}: {e}")
    return drives


def get_usb_info(device_path):
    """Get USB controller info based on platform"""
    if platform.system() == "Linux":
        cmd = f"udevadm info -q path -n {device_path}"
        dev_path = subprocess.getoutput(cmd)
        if 'usb' in dev_path:
            # Extract the last host number from the path
            host_match = re.search(r'host(\d+)', dev_path)
            if host_match:
                return host_match.group(1)
    else:
        try:
            import wmi
            c = wmi.WMI()
            # Convert drive letter to physical disk
            for disk in c.Win32_DiskDrive():
                for partition in disk.associators("Win32_DiskDriveToDiskPartition"):
                    for logical_disk in partition.associators("Win32_LogicalDiskToPartition"):
                        if logical_disk.DeviceID.replace(":", "") == device_path[0]:
                            # Get USB controller info from host controller
                            # SCSIPort is valid (can be 0) - it's the SCSI adapter number
                            return disk.SCSIPort
            # If no matching disk found
            return None
        except Exception as e:
            # If WMI fails (permissions, missing module, etc.)
            return None
                    
def list_sdcards(format_type,maxcardsize=512):
    """
    Scan for SD cards.

    Args:
        format_type : type of format on the sdcard (exfat preffered)
        maxcardsize : select drives with less than the max in Gb
    """
    result =[]
    for i in psutil.disk_partitions():
        if i.fstype.lower()==format_type:
            p =psutil.disk_usage(i.mountpoint)
            if ceil(p.total/1000000000)<=maxcardsize:            
                result.append(i.mountpoint)
    return result

def make_yml(file_path, config, dry_run, card_number=0, overwrite=False):
    if file_path.exists():
        if (overwrite):
            if card_number==0:
                raw_data = yaml.safe_load(file_path.read_text(encoding='utf-8'))
                card_number = raw_data['card_number'] if 'card_number' in raw_data else 0
        else:
            typer.echo(f"Error SDCard already initialise {file_path}")
            return
    if card_number==0:
        card_number = typer.prompt(f"Card number [{str(file_path)}]", type=str, default='1')    
    env = Environment(loader = FileSystemLoader(config.get_path('CATALOG_DIR')),   trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(config.get_path('import_template_path').name)
    fill = {"instrumentPath" : Path.cwd(), "instrument" : 'gopro_bruv',
            "import_date" : f"{datetime.now():%Y-%m-%d}",
            "import_token" : str(uuid.uuid4())[0:8],
            "card_number" : card_number}
    #self.logger.info(f'{dry_run_log_string}Making import file "{file_path}"')
    if not dry_run:
        with open(file_path, "w") as file:
            file.write(template.render(fill))

def register_cards(config,card_path,card_number,overwrite,dry_run: bool):
    """
    Implementation of the MarImBA initialise command for the BRUVS
    """


    # Set dry run log string to prepend to logging
    dry_run_log_string = "DRY_RUN - " if dry_run else ""    
    if isinstance(card_path,list):
        [make_yml(Path(path)/ "import.yml", config, dry_run, cardno, overwrite) for path,cardno in zip(card_path,card_number)]
    else:
        make_yml(Path(card_path) /"import.yml", config, dry_run, card_number, overwrite)

def import_cards(config,card_path:Path,copy,move,find,file_extension,dry_run: bool,format_card=False):
    """
    Implementation of the MarImBA initalise command for the BRUVS
    """
    for card in card_path:
        dry_run_log_string = "DRY_RUN - " if dry_run else ""
        importyml =f"{card}/import.yml"
        if os.path.exists(importyml):
            with open(importyml, 'r') as stream:
                try:
                    importdetails=yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise typer.Abort(f"Error possible corrupt yaml {importyml}")
        else:
            typer.echo(f"Error {importyml} not found")
        typer.echo(f"💾 Reading {importdetails['import_token']} from {card}")
        importdetails['card_store'] = config.data['card_store']  
        importdetails["import_date"] = f"{datetime.now():%Y-%m-%d}"
        # Find all matching paths
        matches = list(Path(config.get_path('card_store')).rglob(f"*{importdetails['import_token']}*"))
        if platform.system() == "Windows":
            rclone_path = config.catalog_dir / 'bin' / 'rclone.exe'
        else:
            rclone_path = 'rclone'

        # Get destination path - either newest match or create new from template
        destination = (max(matches) if matches 
                    else Path(config.data['import_path_template'].format(**importdetails)))
        if copy:
            logging.info(f'{dry_run_log_string}  Copy  {card} --> {destination}')
            command =f"{rclone_path} copy {Path(card).resolve()} {destination.resolve()} --progress --low-level-retries 1 "
            logging.info(f'running {command}')
            command = command.replace('\\','/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True,parents=True)
                process = subprocess.Popen(shlex.split(command))
                process.wait()
        if move:
            command =f"{rclone_path} move {card} {destination} --progress --delete-empty-src-dirs"
            command = command.replace('\\','/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True,parents=True)
                process = subprocess.Popen(shlex.split(command))
                process.wait()
                ## check if drive is empty less than 1Gb

                if format_card and (psutil.disk_usage(card).used < 1 * 1024**3):  # Less than 1 GB used
                    #format the drive on windows and linux
                    if platform.system() == "Windows":
                        command = f"format {card} /FS:exFAT /Q /Y"
                    else:
                        command = f"mkfs.exfat {card}"
                    command = command.replace('\\','/')
                    logging.info(f'{dry_run_log_string}  Deleting empty drive {card}')
                    command = f"rmdir {card}"
                    command = command.replace('\\','/')
                    logging.info(f'{dry_run_log_string}  {command}')
                    if not dry_run:
                        process = subprocess.Popen(shlex.split(command))
                        process.wait()
                make_yml(Path(card) / "import.yml", config, dry_run, importdetails['card_number'], overwrite=True)

