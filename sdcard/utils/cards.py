
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

def register_cards(config,card_path,card_number,overwrite,dry_run: bool):
    """
    Implementation of the MarImBA initialise command for the BRUVS
    """

    def make_yml(file_path,card_number=0):
        if file_path.exists():
            if (overwrite):
                if card_number==0:
                    raw_data = yaml.safe_load(file_path.read_text(encoding='utf-8'))
                    card_number = raw_data['card_number'] if 'card_number' in raw_data else 0
                    card_number = int(card_number)
            else:
                typer.echo(f"Error SDCard already initialise {file_path}")
                return
        if card_number==0:
            card_number = typer.prompt(f"Card number [{str(file_path)}]", type=int, default=1)    
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
    # Set dry run log string to prepend to logging
    dry_run_log_string = "DRY_RUN - " if dry_run else ""    
    if isinstance(card_path,list):
    
        [make_yml(Path(path)/ "import.yml",cardno) for path,cardno in zip(card_path,card_number)]
    else:
        make_yml(Path(card_path) /"import.yml",card_number)

def import_cards(config,card_path:Path,copy,move,find,file_extension,dry_run: bool):
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
        importdetails['card_store'] = config.data['card_store']  
        importdetails["import_date"] = f"{datetime.now():%Y-%m-%d}"
        # Find all matching paths
        matches = list(Path(config.get_path('card_store')).rglob(f"*{importdetails['import_token']}*"))

        # Get destination path - either newest match or create new from template
        destination = (max(matches) if matches 
                    else Path(config.data['import_path_template'].format(**importdetails)))
        if copy:
            logging.info(f'{dry_run_log_string}  Copy  {card} --> {destination}')
            command =f"rclone copy {Path(card).resolve()} {destination.resolve()} --progress --low-level-retries 1 "
            logging.info(f'running {command}')
            command = command.replace('\\','/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True,parents=True)
                process = subprocess.Popen(shlex.split(command))
                process.wait()
        if move:
            command =f"rclone move {card} {destination} --progress --delete-empty-src-dirs"
            command = command.replace('\\','/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True,parents=True)
                process = subprocess.Popen(shlex.split(command))
                process.wait()

