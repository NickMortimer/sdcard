import logging
import typer

import typer
from typer import Typer
import sys
from pathlib import Path



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

goprobruv = typer.Typer(
    name="GoPro BRUV manager",
    help="""GoPro BRUV manager \n
        A Python CLI for managing GoPro Based BRUV systems""",
    short_help="GoPro BRUV Manager",
    no_args_is_help=True,
)


@goprobruv.command('process')
def process(config_path : str= typer.Argument(None, help="path to config file")):
    """
     process drone data if no path to config file is specified all drones processed in sequence
    """
    from sdcard.config import Config
    config = Config(config_path)
    sys.argv.append(f'config={config.get_path("CATALOG_DIR") / "config.yml"}')
    import sdcard.bruv.stage as dodo
    dodo.run()

if __name__ == "__main__":
    goprobruv()                