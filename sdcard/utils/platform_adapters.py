from importlib import import_module
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

    try:
        wmi = import_module("wmi")
        controller = wmi.WMI()
        for disk in controller.Win32_DiskDrive():
            for partition in disk.associators("Win32_DiskDriveToDiskPartition"):
                for logical_disk in partition.associators("Win32_LogicalDiskToPartition"):
                    if logical_disk.DeviceID.replace(":", "") == device_path[0]:
                        return disk.SCSIPort
        return None
    except Exception:
        return None
