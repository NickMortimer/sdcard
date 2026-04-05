import platform
import subprocess
import shutil
import typer
import psutil
import re

# --- Linux-only mount/unmount commands ---


def parse_size(size_str):
    # e.g. '59.5G', '32G', '512M'
    units = {'G': 1024**3, 'M': 1024**2, 'K': 1024}
    m = re.match(r"([\d.]+)([GMK])", size_str)
    if not m:
        return 0
    return float(m.group(1)) * units[m.group(2)]


def mount_cards(
    format_type: str = 'exfat',
    card_size: int = 512,
    all: bool = False,
    card_path: list[str] = None,
):
    """Mount all exFAT SD cards under the specified size using udisksctl (Linux only)."""
    if platform.system() != "Linux":
        typer.echo("❌ The 'mount' command is only supported on Linux.")
        raise typer.Exit(1)
    if not shutil.which("udisksctl"):
        typer.echo("❌ udisksctl not found. Please install udisks2.")
        raise typer.Exit(1)

    mounted_any = False
    if not all and not card_path:
        typer.echo("❌ You must specify either --all or one or more device paths to mount.")
        raise typer.Exit(1)
    if all:
        # Get all block devices and mount all eligible partitions
        try:
            lsblk = subprocess.check_output(["lsblk", "-o", "NAME,FSTYPE,SIZE,TYPE,MOUNTPOINT", "-J"], text=True)
            import json
            blk = json.loads(lsblk)
        except Exception as e:
            typer.echo(f"❌ Failed to list block devices: {e}")
            raise typer.Exit(1)

        mounted = set(p.device for p in psutil.disk_partitions())
        def process_partition(part):
            nonlocal mounted_any
            if part['type'] != 'part':
                return
            if part['fstype'] and part['fstype'].lower() == format_type.lower():
                dev_path = f"/dev/{part['name']}"
                if dev_path in mounted:
                    typer.echo(f"🟢 Already mounted: {dev_path}")
                    return
                # Try to mount
                typer.echo(f"🔄 Mounting {dev_path}...")
                try:
                    out = subprocess.check_output(["udisksctl", "mount", "-b", dev_path], stderr=subprocess.STDOUT, text=True)
                    typer.echo(f"✅ {out.strip()}")
                    mounted_any = True
                except subprocess.CalledProcessError as e:
                    typer.echo(f"❌ Failed to mount {dev_path}: {e.output.strip()}")
        for dev in blk.get('blockdevices', []):
            if dev['type'] == 'disk' and 'children' in dev:
                for part in dev['children']:
                    process_partition(part)
    else:
        for path in card_path:
            try:
                out = subprocess.check_output(["udisksctl", "mount", "-b", str(path)], stderr=subprocess.STDOUT, text=True)
                typer.echo(f"✅ Mounted {path}: {out.strip()}")
                mounted_any = True
            except subprocess.CalledProcessError as e:
                typer.echo(f"❌ Failed to mount {path}: {e.output.strip()}")

    if not mounted_any:
        typer.echo("ℹ️  No exFAT SD cards needed mounting.")


def unmount_cards(
    format_type: str = 'exfat',
    card_size: int = 512,
    power_off: bool = typer.Option(False, "--power-off", help="Also power off the device (LED goes out, device disappears until replugged)")
):
    """Unmount and eject all mounted exFAT SD cards (Linux only): unmount and call system 'eject' command."""
    if platform.system() != "Linux":
        typer.echo("❌ The 'unmount' command is only supported on Linux.")
        raise typer.Exit(1)
    if not shutil.which("eject"):
        typer.echo("❌ eject command not found. Please install the 'eject' utility.")
        raise typer.Exit(1)

    ejected_any = False

    for part in psutil.disk_partitions():
        if part.fstype.lower() != format_type.lower():
            continue
        dev_path = part.device
        typer.echo(f"🔄 Unmounting and ejecting {dev_path}...")
        try:
            out = subprocess.check_output(["udisksctl", "unmount", "-b", dev_path], stderr=subprocess.STDOUT, text=True)
            typer.echo(f"✅ Unmounted: {out.strip()}")
            try:
                out2 = subprocess.check_output(["eject", dev_path], stderr=subprocess.STDOUT, text=True)
                typer.echo(f"💡 Ejected: {out2.strip()}")
            except subprocess.CalledProcessError as e:
                typer.echo(f"⚠️  Eject failed: {e.output.strip()}")
            ejected_any = True
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to unmount {dev_path}: {e.output.strip()}")

    if not ejected_any:
        typer.echo("ℹ️  No exFAT SD cards needed unmounting.")


def mount_command(
    card_path: list[str] = typer.Argument(None, help="One or more SD card device paths (e.g. /dev/sdXn)"),
    all: bool = typer.Option(False, help="Mount all detected cards of the given format/type."),
    card_size: int = typer.Option(512, help="Maximum card size to auto-detect"),
    format_type: str = typer.Option('exfat', help="Card format type"),
):
    """Mount SD cards by device path or all detected cards."""
    mounted_any = False
    if card_path is None:
        card_path = []
    if not all and not card_path:
        typer.echo("❌ You must specify either --all or one or more device paths to mount.")
        raise typer.Exit(1)
    if all:
        # Auto-discover eligible partitions using lsblk (like original mount_cards)
        try:
            lsblk = subprocess.check_output([
                "lsblk", "-o", "NAME,FSTYPE,SIZE,TYPE,MOUNTPOINT", "-J"
            ], text=True)
            import json
            blk = json.loads(lsblk)
        except Exception as e:
            typer.echo(f"❌ Failed to list block devices: {e}")
            raise typer.Exit(1)

        mounted = set(p.device for p in psutil.disk_partitions())
        card_path = []
        def process_partition(part):
            if part['type'] != 'part':
                return
            if part['fstype'] and part['fstype'].lower() == format_type.lower():
                dev_path = f"/dev/{part['name']}"
                if dev_path not in mounted:
                    card_path.append(dev_path)
        for dev in blk.get('blockdevices', []):
            if dev['type'] == 'disk' and 'children' in dev:
                for part in dev['children']:
                    process_partition(part)
    for path in card_path:
        try:
            out = subprocess.check_output(["udisksctl", "mount", "-b", str(path)], stderr=subprocess.STDOUT, text=True)
            typer.echo(f"✅ Mounted {path}: {out.strip()}")
            mounted_any = True
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to mount {path}: {e.output.strip()}")
    if not mounted_any:
        typer.echo("ℹ️  No SD cards needed mounting.")


def eject_cards_command(
    card_path: list[str] = typer.Argument(None, help="One or more SD card device paths (e.g. /dev/sdXn)"),
    all: bool = typer.Option(False, help="Eject all detected cards of the given format/type."),
    card_size: int = typer.Option(512, help="Maximum card size to auto-detect"),
    format_type: str = typer.Option('exfat', help="Card format type"),
):
    """Eject SD cards by device path or all detected cards."""
    if platform.system() != "Linux":
        typer.echo("❌ The 'eject' command is only supported on Linux.")
        raise typer.Exit(1)
    if not shutil.which("eject"):
        typer.echo("❌ eject command not found. Please install the 'eject' utility.")
        raise typer.Exit(1)
    if card_path is None:
        card_path = []
    if all and not card_path:
        # Find all mounted exFAT partitions and use their device paths
        card_path = [p.device for p in psutil.disk_partitions() if p.fstype.lower() == format_type.lower()]
    ejected_any = False
    for path in card_path:
        typer.echo(f"🔄 Unmounting and ejecting {path}...")
        try:
            out = subprocess.check_output(["udisksctl", "unmount", "-b", str(path)], stderr=subprocess.STDOUT, text=True)
            typer.echo(f"✅ Unmounted: {out.strip()}")
            try:
                out2 = subprocess.check_output(["eject", str(path)], stderr=subprocess.STDOUT, text=True)
                typer.echo(f"💡 Ejected: {out2.strip()}")
            except subprocess.CalledProcessError as e:
                typer.echo(f"⚠️  Eject failed: {e.output.strip()}")
            ejected_any = True
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to unmount {path}: {e.output.strip()}")
    if not ejected_any:
        typer.echo("ℹ️  No SD cards needed ejecting.")


