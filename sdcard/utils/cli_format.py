import typer
import shutil
import subprocess
import platform
from pathlib import Path
import psutil
from typing import Optional

app = typer.Typer()


def _find_device_for_mount(mount: str) -> Optional[str]:
    # Normalize mount
    for part in psutil.disk_partitions():
        if part.mountpoint.rstrip('\\/') == str(mount).rstrip('\\/'):
            return part.device
    return None


def _run(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def format_command(
    target: str = typer.Argument(..., help="Mount point (e.g. K:\\ or /mnt/card) or device (/dev/sdb1)"),
    instrument: str = typer.Option(..., help="Instrument name to set as the volume label"),
    format_type: str = typer.Option("exfat", help="Filesystem to create (exfat, vfat)"),
    yes: bool = typer.Option(False, "--yes", help="Actually perform the format (destructive). Default: dry-run only"),
    set_label_only: bool = typer.Option(False, help="Attempt to set the volume label without formatting (non-destructive when supported)."),
):
    """Format a drive and set its label to the instrument name.

    This command is destructive. By default it runs in dry-run mode and prints the command it would run.
    Use --yes to actually perform the format. On Windows administrative privileges may be required.
    """
    # Resolve device on Linux/macOS
    system = platform.system()
    device = target
    mountpoint = None
    if system == "Windows":
        # Accept drive letter forms like K:\ or K:
        if len(target) == 2 and target.endswith(":"):
            drive_letter = target[0]
        elif len(target) == 3 and target[1] == ':' and (target[2] == '\\' or target[2] == '/'):
            drive_letter = target[0]
        else:
            # Might accept a mount like K:\
            drive_letter = None
        if drive_letter:
            # Use drive letter for Windows PowerShell commands
            drive = drive_letter.upper()
            if set_label_only:
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Set-Volume -DriveLetter {drive} -NewFileSystemLabel '{instrument}'"
                ]
            else:
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Format-Volume -DriveLetter {drive} -FileSystem {format_type} -NewFileSystemLabel '{instrument}' -Confirm:$false"
                ]
            typer.echo(f"Command: {' '.join(cmd)}")
            if not yes:
                typer.echo("Dry-run: use --yes to actually perform formatting")
                raise typer.Exit()
            code, output = _run(cmd)
            typer.echo(output)
            raise typer.Exit(code=code)
    else:
        # On Unix-like systems, if target is a mountpoint try to find device
        if Path(target).exists() and Path(target).is_dir():
            dev = _find_device_for_mount(target)
            if dev:
                device = dev
                mountpoint = target
        # If set_label_only, prefer exfatlabel or fatlabel
        if set_label_only:
            if shutil.which('exfatlabel'):
                cmd = ['exfatlabel', device, instrument]
            elif shutil.which('fatlabel'):
                cmd = ['fatlabel', device, instrument]
            else:
                typer.echo('No label utility found (exfatlabel/fatlabel). Cannot set label without formatting.')
                raise typer.Exit(1)
            typer.echo(f"Command: {' '.join(cmd)}")
            if not yes:
                typer.echo('Dry-run: use --yes to actually set label')
                raise typer.Exit()
            code, output = _run(cmd)
            typer.echo(output)
            raise typer.Exit(code=code)

        # Formatting path
        # prefer mkfs.exfat, fall back to mkfs.vfat for FAT32
        if format_type.lower() in ('exfat', 'exfatex') and shutil.which('mkfs.exfat'):
            cmd = ['mkfs.exfat', '-n', instrument, device]
        elif format_type.lower() in ('vfat', 'fat32') and shutil.which('mkfs.vfat'):
            cmd = ['mkfs.vfat', '-F', '32', '-n', instrument, device]
        else:
            typer.echo('No suitable mkfs tool found for requested filesystem. Install mkfs.exfat or mkfs.vfat.')
            raise typer.Exit(1)

        typer.echo(f"Will format device: {device} as {format_type} with label '{instrument}'")
        if mountpoint:
            typer.echo(f"(detected mountpoint: {mountpoint})")
        typer.echo(f"Command: {' '.join(cmd)}")
        if not yes:
            typer.echo('Dry-run: use --yes to actually perform formatting')
            raise typer.Exit()
        code, output = _run(cmd)
        typer.echo(output)
        raise typer.Exit(code=code)


format = format_command
