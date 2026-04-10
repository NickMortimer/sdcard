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
    DEFAULT_REFRESH,
    DEFAULT_VERBOSE,
)
from sdcard.utils.cards_discovery import get_card_number_from_import_yml, get_available_cards, list_sdcards
from sdcard.utils.import_metadata import register_cards
from sdcard.utils.import_transfer import import_cards
from sdcard.utils.usb import get_complete_usb_card_info


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
    card_size: int = typer.Option(
        DEFAULT_CARD_SIZE,
        "--card-size",
        "--cardsize",
        help="Maximum card size",
    ),
    format_type: str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
):
    config = Config(config_path)
    if config.uses_implicit_defaults:
        typer.echo(
            "⚠️  No config file provided; using default parameters for import.yml"
        )
    if all and (not card_path ):
        card_path = list_sdcards(format_type, card_size)
        if card_number is None:
            card_number = ['0'] * len(card_path)
    else:
        card_path = [Path(path) for path in card_path]
    register_cards(
        config,
        card_number=card_number,
        card_path=card_path,
        dry_run=dry_run,
        overwrite=overwrite,
        refresh=refresh,
        instrument=instrument,
        prompt_card_details=prompt_card_details,
    )

def list_cards(
    card_size: int = typer.Option(DEFAULT_CARD_SIZE, help="Maximum card size to list"),
    format_type: str = typer.Option(DEFAULT_FORMAT_TYPE, help="Card format type"),
    verbose: bool = typer.Option(DEFAULT_VERBOSE, "--verbose", "-v", help="Show reader and speed details"),
):
    if not verbose:
        cards = get_available_cards(format_type, card_size)
        for card in cards:
            mountpoint = card['mountpoint']
            import_tick = "✅" if (Path(mountpoint) / "import.yml").exists() else "❌"
            used = _format_bytes(_mount_used_bytes(mountpoint))
            typer.echo(
                f"Reader Card: {card['card_number']} {card['host']}: {mountpoint} "
                f"({card['size_gb']}GB, used={used}, import.yml={import_tick})"
            )
        return cards

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
        import_yml_path = Path(mountpoint) / "import.yml"
        card_number = get_card_number_from_import_yml(import_yml_path)
        has_import = import_yml_path.exists()
        used = _format_bytes(_mount_used_bytes(mountpoint))

        listed_card = {
            'card_number': card_number,
            'mountpoint': mountpoint,
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
            f"Reader Card: {card_number} {mountpoint} "
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
