import typer
from pathlib import Path
import yaml
from sdcard.utils.cards_discovery import list_sdcards
from sdcard.config import Config
from sdcard.utils.config_path_cache import resolve_config_path

app = typer.Typer()


def _write_ignore_file(target_path: Path, mounts: list[str]):
    data = {"ignore_drives": mounts}
    target_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def scan_command(
    config_path: Path = typer.Option(None, help="Path to config file"),
    format_type: str = typer.Option("exfat", help="Card filesystem type to scan"),
    card_size: int = typer.Option(512, help="Maximum card size in GB"),
    to_config_dir: bool = typer.Option(False, help="Write .ignore-drives.yml to the config file directory instead of the current directory"),
):
    """Scan current drives and write a .ignore-drives.yml containing their mountpoints.

    The file will be written to the current directory by default. If --config-path is provided
    and --to-config-dir is set, the file will be written to the directory containing the config file.
    """
    config_path = resolve_config_path(config_path)
    config = Config(config_path)

    # Remove any existing ignore files (cwd and config dir) so the scan sees all drives
    cwd = Path.cwd()
    cwd_ignore = cwd / ".ignore-drives.yml"
    try:
        if cwd_ignore.exists():
            cwd_ignore.unlink()
    except Exception:
        pass

    config_dir = Path(config.config_path).parent
    config_ignore = config_dir / ".ignore-drives.yml"
    try:
        if config_ignore.exists():
            config_ignore.unlink()
    except Exception:
        pass

    # Now scan without any ignores in place
    mounts = list_sdcards(format_type, card_size, config)
    if not mounts:
        typer.echo("No matching drives found to ignore.")
        raise typer.Exit()

    # By default, prefer the config directory when an explicit config path was provided.
    if config_path is not None or to_config_dir:
        target = config_dir / ".ignore-drives.yml"
    else:
        target = cwd / ".ignore-drives.yml"

    _write_ignore_file(target, mounts)
    typer.echo(f"Wrote ignore file with {len(mounts)} mounts: {target}")


# expose function name for import
scan = scan_command
