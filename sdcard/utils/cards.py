
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime, timedelta
import uuid
import psutil
from math import ceil
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
from sdcard.config import DEFAULT_IMPORT_TEMPLATE

DEFAULT_IMPORT_YML_TEMPLATE = """card_number: {{ card_number }}\nimport_token: {{ import_token }}\nregister_date: {{ register_date }}\nimport_date: {{ import_date }}\ndestination_path: {{ destination_path }}\n"""

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

def _render_destination_path(config, import_context: dict) -> str:
    """Render the destination path using config template and context."""
    template_string = config.data.get('import_path_template', DEFAULT_IMPORT_TEMPLATE)
    environment = Environment()
    template = environment.from_string(template_string)
    return template.render(**import_context)


def make_yml(file_path, config, dry_run, card_number=0, overwrite=False):
    """Create or refresh an import.yml for a card."""
    existing = {}
    if file_path.exists():
        if not overwrite:
            typer.echo(f"Error SDCard already initialise {file_path}")
            return
        try:
            existing = yaml.safe_load(file_path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError:
            existing = {}
        if card_number == 0:
            card_number = existing.get('card_number', 0)

    if card_number == 0:
        card_number = typer.prompt(f"Card number [{str(file_path)}]", type=str, default='1')

    register_date = existing.get("register_date") or f"{datetime.now():%Y-%m-%d}"
    import_date = existing.get("import_date") or register_date
    import_token = existing.get("import_token") or str(uuid.uuid4())[0:8]
    destination_path = existing.get("destination_path")

    import_context = {
        "import_date": import_date,
        "import_token": import_token,
        "card_number": card_number,
        "card_store": str(config.get_path('card_store')),
        "CATALOG_DIR": str(config.catalog_dir),
        "register_date": register_date,
    }
    if not destination_path:
        destination_path = _render_destination_path(config, import_context)
    import_context["destination_path"] = destination_path

    template_path = config.data.get('import_template_path')
    if template_path:
        env = Environment(loader=FileSystemLoader(Path(template_path).parent), trim_blocks=True, lstrip_blocks=True)
        template = env.get_template(Path(template_path).name)
        rendered = template.render(import_context)
    else:
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(DEFAULT_IMPORT_YML_TEMPLATE)
        rendered = template.render(import_context)

    if not dry_run:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(rendered)

def register_cards(config,card_path,card_number,overwrite,dry_run: bool):
    """
    Register SD cards by creating an import.yml on each card
    """
    # Set dry run log string to prepend to logging
    dry_run_log_string = "DRY_RUN - " if dry_run else ""

    if card_number is None:
        card_number = ['0'] * len(card_path) if isinstance(card_path, list) else 0

    if isinstance(card_path, list):
        for path, cardno in zip(card_path, card_number):
            make_yml(Path(path) / "import.yml", config, dry_run, cardno, overwrite)
    else:
        make_yml(Path(card_path) / "import.yml", config, dry_run, card_number, overwrite)

def import_cards(config, card_path, copy, move, find, file_extension, dry_run: bool, format_card=False):
    """Copy or move SD card contents to the configured card store."""
    for card in card_path:
        dry_run_log_string = "DRY_RUN - " if dry_run else ""
        importyml = Path(card) / "import.yml"
        if importyml.exists():
            try:
                importdetails = yaml.safe_load(importyml.read_text(encoding='utf-8'))
            except yaml.YAMLError as exc:
                raise typer.Abort(f"Error possible corrupt yaml {importyml}: {exc}")
        else:
            typer.echo(f"Error {importyml} not found")
            continue

        typer.echo(f"💾 Reading {importdetails['import_token']} from {card}")
        card_store = Path(config.data.get('card_store'))
        importdetails.setdefault("register_date", f"{datetime.now():%Y-%m-%d}")
        if not importdetails.get("import_date"):
            importdetails["import_date"] = f"{datetime.now():%Y-%m-%d}"

        # Resolve destination: prefer explicit destination_path from import.yml, otherwise render from template
        destination_path = importdetails.get('destination_path')
        if not destination_path:
            import_context = {
                "import_date": importdetails.get("import_date"),
                "import_token": importdetails.get("import_token"),
                "card_number": importdetails.get("card_number", 0),
                "card_store": str(card_store),
                "CATALOG_DIR": str(config.catalog_dir),
                "register_date": importdetails.get("register_date"),
            }
            destination_path = _render_destination_path(config, import_context)

        destination = Path(destination_path)

        # Allow reuse of an existing destination when find flag is on
        if find:
            matches = list(card_store.rglob(f"*{importdetails.get('import_token')}*"))
            if matches:
                destination = max(matches)

        # Choose rclone binary per platform
        if platform.system() == "Windows":
            rclone_path = config.catalog_dir / 'bin' / 'rclone.exe'
        else:
            rclone_path = 'rclone'

        if copy:
            logging.info(f'{dry_run_log_string}  Copy  {card} --> {destination}')
            command = f"{rclone_path} copy {Path(card).resolve()} {destination.resolve()} --progress --low-level-retries 1 "
            logging.info(f'running {command}')
            command = command.replace('\\', '/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True, parents=True)
                process = subprocess.Popen(shlex.split(command))
                process.wait()

        if move:
            command = f"{rclone_path} move {card} {destination} --progress --delete-empty-src-dirs"
            command = command.replace('\\', '/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True, parents=True)
                process = subprocess.Popen(shlex.split(command))
                process.wait()

                if format_card and (psutil.disk_usage(card).used < 1 * 1024**3):
                    if platform.system() == "Windows":
                        command = f"format {card} /FS:exFAT /Q /Y"
                    else:
                        command = f"mkfs.exfat {card}"
                    command = command.replace('\\', '/')
                    logging.info(f'{dry_run_log_string}  Deleting empty drive {card}')
                    command = f"rmdir {card}"
                    command = command.replace('\\', '/')
                    logging.info(f'{dry_run_log_string}  {command}')
                    if not dry_run:
                        process = subprocess.Popen(shlex.split(command))
                        process.wait()

                # Re-create import.yml so the card stays registered
                make_yml(importyml, config, dry_run, importdetails.get('card_number', 0), overwrite=True)

        # Persist updated import_date/register_date when a transfer actually occurred
        if not dry_run and (copy or move):
            persisted = {
                "card_number": importdetails.get("card_number"),
                "import_token": importdetails.get("import_token"),
                "register_date": importdetails.get("register_date"),
                "import_date": importdetails.get("import_date"),
                "destination_path": importdetails.get("destination_path", destination_path),
            }
            importyml.write_text(yaml.safe_dump(persisted), encoding="utf-8")

