from importlib import import_module
from functools import lru_cache
import platform
import re
import shutil
import subprocess

import typer


def require_linux(command_name: str) -> None:
    """Exit early when a Linux-only command is called on non-Linux hosts."""
    if platform.system() != "Linux":
        typer.echo(f"❌ The '{command_name}' command is only supported on Linux.")
        raise typer.Exit(1)


def require_command(binary_name: str, install_hint: str) -> None:
    """Exit early when an external command dependency is not available."""
    if not shutil.which(binary_name):
        typer.echo(f"❌ {binary_name} not found. {install_hint}")
        raise typer.Exit(1)


def get_usb_info(device_path: str):
    """Get USB host/controller information for a block device path."""
    if platform.system() == "Linux":
        cmd = f"udevadm info -q path -n {device_path}"
        dev_path = subprocess.getoutput(cmd)
        if 'usb' in dev_path:
            host_match = re.search(r'host(\d+)', dev_path)
            if host_match:
                return host_match.group(1)
        return None

    drive_letter = str(device_path)[0:1].upper()
    if not drive_letter:
        return None

    host_map = get_windows_drive_host_map()
    return host_map.get(drive_letter)


def _normalize_windows_reader_id(pnp_device_id: str) -> str:
    """Normalize WMI PNP id so multi-slot readers share a stable key."""
    text = (pnp_device_id or "").upper()
    if not text:
        return ""
    # LUN and trailing instance fragments often differ per slot/card.
    text = re.sub(r"&LUN_\d+", "", text)
    text = re.sub(r"\\[^\\]*$", "", text)
    return text


@lru_cache(maxsize=1)
def get_windows_drive_reader_map() -> dict[str, str]:
    """Return cached mapping of drive letter to a reader-level grouping key."""
    mapping: dict[str, str] = {}
    if platform.system() != "Windows":
        return mapping

    try:
        wmi = import_module("wmi")
        controller = wmi.WMI()
        for disk in controller.Win32_DiskDrive():
            pnp_id = str(getattr(disk, "PNPDeviceID", ""))
            normalized_pnp = _normalize_windows_reader_id(pnp_id)
            host = str(getattr(disk, "SCSIPort", ""))
            bus = str(getattr(disk, "SCSIBus", ""))
            target = str(getattr(disk, "SCSITargetId", ""))
            # Prefer stable reader identity from PNP id, then fallback.
            reader_key = normalized_pnp or f"{host}:{bus}:{target}"
            if not reader_key:
                continue
            for partition in disk.associators("Win32_DiskDriveToDiskPartition"):
                for logical_disk in partition.associators(
                    "Win32_LogicalDiskToPartition"
                ):
                    device_id = str(getattr(logical_disk, "DeviceID", ""))
                    drive_letter = device_id.replace(":", "").upper()
                    if drive_letter:
                        mapping[drive_letter] = reader_key
    except Exception:
        return {}

    return mapping


@lru_cache(maxsize=1)
def get_windows_drive_host_map() -> dict[str, str]:
    """Return a cached mapping of drive letter to Windows storage host id."""
    mapping: dict[str, str] = {}
    if platform.system() != "Windows":
        return mapping

    try:
        wmi = import_module("wmi")
        controller = wmi.WMI()
        for disk in controller.Win32_DiskDrive():
            host = str(getattr(disk, "SCSIPort", ""))
            if not host:
                continue
            for partition in disk.associators("Win32_DiskDriveToDiskPartition"):
                for logical_disk in partition.associators(
                    "Win32_LogicalDiskToPartition"
                ):
                    device_id = str(getattr(logical_disk, "DeviceID", ""))
                    drive_letter = device_id.replace(":", "").upper()
                    if drive_letter:
                        mapping[drive_letter] = host
    except Exception:
        return {}

    return mapping
