from pathlib import Path
from math import ceil
import logging
import psutil
import yaml

from sdcard.utils.platform_adapters import get_usb_info


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


def get_available_cards(format_type='exfat', maxcardsize=512):
    """Get list of available SD cards"""
    drives = []
    for part in psutil.disk_partitions():
        if part.fstype.lower() == format_type.lower():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                if usage.total < maxcardsize * 1024**3:
                    usb_host = get_usb_info(part.device)
                    if usb_host is not None:
                        drives.append({
                            'mountpoint': part.mountpoint,
                            'host': f"usb{usb_host}",
                            'size_gb': round(usage.total / (1024**3), 2),
                            'free_gb': round(usage.free / (1024**3), 2),
                            'card_number': get_card_number_from_import_yml(Path(part.mountpoint) / "import.yml")
                        })
            except Exception as error:
                print(f"Error processing {part.device}: {error}")
    return drives


def list_sdcards(format_type, maxcardsize=512):
    """
    Scan for SD cards.

    Args:
        format_type : type of format on the sdcard (exfat preffered)
        maxcardsize : select drives with less than the max in Gb
    """
    result = []
    for part in psutil.disk_partitions():
        if part.fstype.lower() == format_type:
            usage = psutil.disk_usage(part.mountpoint)
            if ceil(usage.total / 1000000000) <= maxcardsize:
                result.append(part.mountpoint)
    return result
