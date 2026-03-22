import typer
from pathlib import Path
import gpxpy
import csv
import sys

app = typer.Typer()

@app.command()
def waypoint(
    config_path: str = typer.Option(None, help="Root path to MarImBA collection."),
):
    """Extract waypoints from GPX files in the gps folder or in the config file.

    Args:
        config_path (str, optional): _description_. Defaults to typer.Option(None, help="Root path to MarImBA collection.").
    """
    from sdcard.config import Config
    config = Config(config_path)
    sys.argv.append(f'config={config.get_path("CATALOG_DIR") / "config.yml"}')
    import sdcard.gps.gps_dodo as dodo
    dodo.run()

if __name__ == "__main__":
    app()
