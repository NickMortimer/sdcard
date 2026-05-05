import typer
import psutil
from pathlib import Path
from sdcard.config import Config
from sdcard.utils.cli_defaults import (
    DEFAULT_ALL,
    DEFAULT_CARD_SIZE,
    DEFAULT_DRY_RUN,
    DEFAULT_FORMAT_TYPE,
    DEFAULT_OVERWRITE,
    DEFAULT_PROMPT_CARD_DETAILS,
    DEFAULT_FORMAT_CARD,
    DEFAULT_REFRESH,
    DEFAULT_VERBOSE,
)
from sdcard.utils.cards_discovery import get_card_number_from_import_yml, get_available_cards, list_sdcards
from sdcard.utils.import_metadata import register_cards
from sdcard.utils.config_path_cache import resolve_config_path


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{int(num_bytes)}B"


def _mount_used_bytes(mountpoint: str) -> int:
    try:
        return int(psutil.disk_usage(mountpoint).used)
    except Exception:
        return 0

def register_command(
    card_path: list[str] = typer.Argument(None, help="One or more SD card mount points"),
    card_number: list[str] = typer.Option(None, help="Set card number metadata on the card"),
    config_path: str = typer.Option(None, help="Path to config directory."),
    all: bool = typer.Option(DEFAULT_ALL, help="Execute the command and print logging to the terminal, but do not change any files."),
    dry_run: bool = typer.Option(DEFAULT_DRY_RUN, help="Execute the command and print logging to the terminal, but do not change any files."),
    overwrite: bool = typer.Option(DEFAULT_OVERWRITE, help="Overwrite import.yaml"),
    refresh: bool = typer.Option(DEFAULT_REFRESH, "--refresh", help="Refresh existing import.yml using config defaults while keeping card number and chosen instrument, and generating a new import token"),
    instrument: str = typer.Option(None, "--instrument", "-i", help="Instrument code/name to store in import.yml"),
    prompt_card_details: bool = typer.Option(DEFAULT_PROMPT_CARD_DETAILS, "--card-details", help="Prompt for optional card manufacturer and rated UHS metadata"),
    set_label: bool = typer.Option(False, "--set-label", help="Set the drive volume label to the chosen instrument after registering."),
    format_card: bool = typer.Option(DEFAULT_FORMAT_CARD, "--format-card", "--format", help="Format the card before writing import.yml (destructive)."),
    format_yes: bool = typer.Option(False, "--format-yes", "-y", help="Actually perform the format (required to make destructive changes)."),
    format_full: bool = typer.Option(False, "--format-full", help="Use a full (slow) format when formatting a card. Default: quick format."),
    card_size: int = typer.Option(
        DEFAULT_CARD_SIZE,
        "--card-size",
        "--cardsize",
        help="Maximum card size",
    ),
    format_type: str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
):
    config_path = resolve_config_path(config_path)
    config = Config(config_path)
    if config.uses_implicit_defaults:
        typer.echo(
            "⚠️  No config file provided; using default parameters for import.yml"
        )
    # --format-card is destructive and only supported when doing a turbo-style
    # refresh; enforce the coupling here to avoid accidental formatting.
    if format_card and not refresh:
        typer.echo("Error: --format-card may only be used together with --refresh")
        raise typer.Exit(code=1)
    if all and (not card_path ):
        card_path = list_sdcards(format_type, card_size, config)
        if card_number is None:
            card_number = ['0'] * len(card_path)
    else:
        normalized_paths: list[Path] = []
        for path in card_path:
            # On Windows, "F:" is a valid argument but Path("F:") is NOT the drive
            # root directory. Ensure we treat drive-only inputs as "F:\\".
            raw = str(path)
            if len(raw) == 2 and raw[1] == ':':
                raw = raw + "\\"
            normalized_paths.append(Path(raw))
        card_path = normalized_paths
    register_cards(
        config,
        card_number=card_number,
        card_path=card_path,
        dry_run=dry_run,
        overwrite=overwrite,
        refresh=refresh,
        instrument=instrument,
        prompt_card_details=prompt_card_details,
        set_label=set_label,
        format_card=format_card,
        format_yes=format_yes,
        format_full=format_full,
    )

def list_cards(
    card_size: int = typer.Option(DEFAULT_CARD_SIZE, help="Maximum card size to list"),
    format_type: str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
    verbose: bool = typer.Option(DEFAULT_VERBOSE, "--verbose", "-v", help="Show reader and speed details"),
    config_path: str = typer.Option(None, help="Path to config file (used to resolve .ignore-drives.yml)."),
):
    from sdcard.config import Config
    config_path = resolve_config_path(config_path)
    config = Config(config_path)
    if not verbose:
        allowed = set(list_sdcards(format_type, card_size, config))
        cards = [c for c in get_available_cards(format_type, card_size) if c['mountpoint'] in allowed]
        for card in cards:
            mountpoint = card['mountpoint']
            label = card.get('label', '') or ''
            import_tick = "✅" if (Path(mountpoint) / "import.yml").exists() else "❌"
            used = _format_bytes(_mount_used_bytes(mountpoint))
            typer.echo(
                f"Reader Card: {card['card_number']} {card['host']} {label} : {mountpoint} "
                f"({card['size_gb']}GB, used={used}, import.yml={import_tick})"
            )
        return cards

    from sdcard.utils.usb import get_complete_usb_card_info
    usb_cards, _ = get_complete_usb_card_info(
        destination_path=None,
        card_size=card_size,
        format_type=format_type,
    )
    listed_cards = []

    for _, card in usb_cards.iterrows():
        if card.get('device_type') == 'destination':
            continue

        mountpoint = card['mountpoint']
        partition = card.get('name') or card.get('partition') or ''
        label = card.get('label') or ''
        import_yml_path = Path(mountpoint) / "import.yml"
        card_number = get_card_number_from_import_yml(import_yml_path)
        has_import = import_yml_path.exists()
        used = _format_bytes(_mount_used_bytes(mountpoint))

        listed_card = {
            'card_number': card_number,
            'mountpoint': mountpoint,
            'partition': partition,
            'label': label,
            'size': card.get('size'),
            'fstype': card.get('fstype'),
            'used': used,
            'has_import_yml': has_import,
            'reader_manufacturer': card.get('reader_manufacturer') or 'Unknown',
            'reader_product': card.get('reader_product') or 'Unknown',
            'reader_speed': card.get('reader_speed') or 'Unknown',
            'reader_speed_mbps': card.get('reader_speed_mbps') or 0,
            'actual_transfer_rate': card.get('actual_transfer_rate') or 0,
            'thunderbolt_connected': bool(card.get('thunderbolt_connected', False)),
            'thunderbolt_info': card.get('thunderbolt_info') or 'Standard USB',
        }
        listed_cards.append(listed_card)

        typer.echo(
            f"Reader Card: {card_number} {label} {mountpoint} "
            f"({listed_card['size']}, {listed_card['fstype']}, used={used}, import.yml={'✅' if has_import else '❌'})"
        )
        typer.echo(
            f"  Reader: {listed_card['reader_manufacturer']} {listed_card['reader_product']}"
        )
        typer.echo(
            f"  USB Speed: {listed_card['reader_speed']} ({listed_card['reader_speed_mbps']} Mbps)"
        )
        typer.echo(
            f"  Estimated Transfer: {listed_card['actual_transfer_rate']:.1f} MB/s"
        )
        typer.echo(
            f"  Connection: {listed_card['thunderbolt_info']}"
        )

    return listed_cards
