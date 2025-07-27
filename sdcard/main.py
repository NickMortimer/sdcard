#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import typer
from sdcard.utils.file_system import list_sdcards
from typer import Typer
import sys
from pathlib import Path
import os
import yaml
from sdcard.config import Config



__author__ = "GoPro BRUV Development Team"
__copyright__ = "Copyright 2023                                                                                                                                                                                                                                                                                                                                                                                                                               , CSIRO"
__credits__ = [
    "Nick Mortimer <nick.mortimer@csiro.au>",
]
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Nick Mortimer"
__email__ = "nick.mortimer@csiro.au"
__status__ = "Development"

sdcard = typer.Typer(
    name="SDCard manager",
    help="""SD card manager \n
        A Python CLI for managing sdcards""",
    short_help="SD Card Manager",
    no_args_is_help=True,
)


@sdcard.command('register')
def register_command(
        config_path: str = typer.Argument(None, help="Root path to MarImBA collection."),
        card_path: list[str] = typer.Argument(None, help="MarImBA instrument ID.",),
        card_number: list[int] = typer.Option(None, help="set card number metadata on the card"),
        all: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        dry_run: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        overwrite:bool = typer.Option(False, help="Overwrite import.yaml"),
        cardsize:int = typer.Option(512, help="maximum card size"),
        format_type:str = typer.Option('exfat', help="Card format type"),
):
    """
    register sd cards
    """
    from sdcard.utils.cards import list_sdcards,register_cards  
    config = Config(config_path)
    if all and (not card_path ):
        card_path = list_sdcards(format_type,cardsize)
        if len(card_number) == 0:
            card_number = [0] * len(card_path)
    register_cards(config,card_number=card_number,card_path=card_path,dry_run=dry_run,overwrite=overwrite)

@sdcard.command('import')
def import_command(
        card_path: list[str] = typer.Argument(None, help="MarImBA instrument ID."),
        config_path: Path = typer.Option(None,help="Path to config file"),
        copy: bool = typer.Option(True, help="Clean source"),
        move: bool = typer.Option(False, help="move source"),
        find: bool = typer.Option(True, help="import to the same hash"),
        cardsize:int = typer.Option(512, help="maximum card size"),
        format_type:str = typer.Option('exfat', help="Card format type"),
        extra: list[str] = typer.Option([], help="Extra key-value pass-through arguments."),
        dry_run: bool = typer.Option(False, help="Execute the command and print logging to the terminal, but do not change any files."),
        file_extension: str = typer.Option("MP4", help="extension to catalog"),
):
    """
    Import SD cards to working directorypip
    """ 
    from sdcard.utils.cards import list_sdcards
    from sdcard.utils.cards import import_cards
    config = Config(config_path)
    if all and (not card_path ):
        card_path = list_sdcards(format_type,cardsize)
    import_cards(config=config,card_path=card_path,copy=copy,move=move,find=find,dry_run=dry_run,file_extension=file_extension)







if __name__ == "__main__":
    sdcard()
