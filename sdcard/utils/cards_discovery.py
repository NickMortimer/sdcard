from pathlib import Path
from math import ceil
import logging
import psutil
import yaml

from sdcard.utils.platform_adapters import get_usb_info
import platform
import subprocess
import ctypes


def _get_volume_label(mountpoint: str, device: str | None = None) -> str | None:
    """Return the volume label for a mountpoint or device if available."""
    try:
        if platform.system() == 'Windows':
            try:
                kernel32 = ctypes.windll.kernel32
                volume_name_buf = ctypes.create_unicode_buffer(1024)
                fs_name_buf = ctypes.create_unicode_buffer(1024)
                serial_number = ctypes.c_uint()
                max_comp_len = ctypes.c_uint()
                file_sys_flags = ctypes.c_uint()
                res = kernel32.GetVolumeInformationW(
                    ctypes.c_wchar_p(mountpoint),
                    volume_name_buf,
                    ctypes.sizeof(volume_name_buf),
                    ctypes.byref(serial_number),
                    ctypes.byref(max_comp_len),
                    ctypes.byref(file_sys_flags),
                    fs_name_buf,
                    ctypes.sizeof(fs_name_buf),
                )
                if res:
                    return volume_name_buf.value
            except Exception:
                return None
        else:
            if device:
                try:
                    result = subprocess.run(['blkid', '-s', 'LABEL', '-o', 'value', device], capture_output=True, text=True)
                    if result.returncode == 0:
                        label = result.stdout.strip()
                        return label if label else None
                except Exception:
                    return None
    except Exception:
        return None
    return None


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
                        label = _get_volume_label(part.mountpoint, part.device)
                        drives.append({
                            'mountpoint': part.mountpoint,
                            'partition': part.device,
                            'label': label,
                            'host': f"usb{usb_host}",
                            'size_gb': round(usage.total / (1024**3), 2),
                            'free_gb': round(usage.free / (1024**3), 2),
                            'card_number': get_card_number_from_import_yml(Path(part.mountpoint) / "import.yml")
                        })
            except Exception as error:
                print(f"Error processing {part.device}: {error}")
    return drives


def _read_ignore_drives(cwd_path: Path, config_dir: Path | None) -> set[str]:
    """Read .ignore-drives.yml from cwd and/or config dir and return a set of mountpoints to ignore."""
    ignore_set: set[str] = set()
    candidates = []
    candidates.append(cwd_path / ".ignore-drives.yml")
    if config_dir is not None:
        candidates.append(Path(config_dir) / ".ignore-drives.yml")

    for p in candidates:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                    # support either a top-level list or dict with key 'ignore_drives'
                    if isinstance(data, list):
                        for item in data:
                            ignore_set.add(str(item))
                    elif isinstance(data, dict):
                        entries = data.get("ignore_drives", [])
                        for item in entries:
                            ignore_set.add(str(item))
        except Exception:
            # ignore malformed files
            continue
    return ignore_set


def list_sdcards(format_type, maxcardsize=512, config=None):
    """
    Scan for SD cards.

    Args:
        format_type : type of format on the sdcard (exfat preffered)
        maxcardsize : select drives with less than the max in Gb
    """
    result = []
    cwd = Path.cwd()
    config_dir = None
    try:
        if config is not None:
            # Config may be either a path or a Config instance
            if hasattr(config, 'catalog_dir'):
                config_dir = Path(config.catalog_dir)
            else:
                config_dir = Path(config).parent
    except Exception:
        config_dir = None

    ignores = _read_ignore_drives(cwd, config_dir)

    format_type_normalized = str(format_type).lower()
    for part in psutil.disk_partitions():
        if part.fstype.lower() == format_type_normalized:
            usage = psutil.disk_usage(part.mountpoint)
            if ceil(usage.total / 1000000000) <= maxcardsize:
                mp = part.mountpoint
                if mp in ignores:
                    continue
                result.append(mp)
    return result


def card_has_payload(card_path: str) -> bool:
    """Return True when a card has at least one file other than import.yml."""
    root = Path(card_path)
    if not root.exists():
        return False
    for entry in root.rglob("*"):
        if entry.is_file() and entry.name.lower() != "import.yml":
            return True
    return False


def filter_empty_cards(card_paths: list[str]) -> tuple[list[str], list[str]]:
    """Split card_paths into (has_payload, empty_only_import_yml).

    Prints a warning for each card that contains only import.yml.
    Returns (ready, skipped).
    """
    ready: list[str] = []
    skipped: list[str] = []
    for path in card_paths:
        if card_has_payload(path):
            ready.append(path)
        else:
            skipped.append(path)
            print(f"⚠️  Skipping {path} — only import.yml found, nothing to import")
    return ready, skipped
