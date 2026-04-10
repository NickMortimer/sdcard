#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import typer


__author__ = "SDCard Maintainers"
__license__ = "MIT"
__version__ = "0.2"
__status__ = "Development"


# CLI command implementations are now in utils/cli_basic, cli_import, cli_probe, cli_turbo
from sdcard.utils.mount_tools import mount_cards, eject_cards_command
from sdcard.utils.cli_basic import register_command, list_cards
from sdcard.utils.cli_import import import_command
from sdcard.utils.cli_probe import probe
from sdcard.utils.cli_thumbnail import thumbnail
from sdcard.utils.cli_turbo import turbo
from sdcard.utils.cli_xif import xif


sdcard = typer.Typer(
    name="SDCard manager",
    help="""SD card manager \nA Python CLI for managing sdcards""",
    short_help="SD Card Manager",
    no_args_is_help=True,
)


sdcard.command()(mount_cards)
sdcard.command('eject-cards')(eject_cards_command)
sdcard.command('register')(register_command)
sdcard.command('list')(list_cards)
sdcard.command('import')(import_command)
sdcard.command('probe')(probe)
sdcard.command('turbo')(turbo)
sdcard.command('xif')(xif)
sdcard.command('thumbnail')(thumbnail)



if __name__ == "__main__":
    sdcard()
