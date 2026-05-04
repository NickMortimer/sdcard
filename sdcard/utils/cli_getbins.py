from __future__ import annotations

import platform
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import typer

from sdcard.config import Config
from sdcard.utils.config_path_cache import resolve_config_path

DEFAULT_RCLONE_URL = "https://downloads.rclone.org/rclone-current-windows-amd64.zip"
DEFAULT_EXIFTOOL_URL = "https://sourceforge.net/projects/exiftool/files/exiftool-13.57_64.zip/download"


def _download_zip(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "sdcard-getbins/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def _resolve_latest_exiftool_url() -> str:
    request = urllib.request.Request(
        "https://exiftool.org/",
        headers={"User-Agent": "sdcard-getbins/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", "ignore")

    matches = re.findall(r"exiftool-([0-9]+\.[0-9]+)_64\.zip", html)
    if not matches:
        raise RuntimeError("Unable to determine latest ExifTool Windows zip from exiftool.org")

    latest_version = sorted(set(matches), key=lambda value: tuple(int(x) for x in value.split(".")))[-1]
    return f"https://sourceforge.net/projects/exiftool/files/exiftool-{latest_version}_64.zip/download"


def _extract_executable_from_zip(
    zip_path: Path,
    target_path: Path,
    candidate_names: list[str],
) -> None:
    candidates = {name.lower() for name in candidate_names}
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_name = Path(member.filename).name.lower()
            if member_name not in candidates:
                continue
            payload = archive.read(member)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(payload)
            return

    raise RuntimeError(
        f"Expected one of {candidate_names} in archive {zip_path.name}, but none were found"
    )


def _install_rclone(bin_dir: Path, url: str, force: bool) -> str:
    destination = bin_dir / "rclone.exe"
    if destination.exists() and not force:
        return "kept"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        temp_zip = Path(tmp.name)

    try:
        _download_zip(url, temp_zip)
        _extract_executable_from_zip(temp_zip, destination, ["rclone.exe"])
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

    return "installed"


def _install_exiftool(bin_dir: Path, url: str, force: bool) -> str:
    destination = bin_dir / "exiftool.exe"
    support_dir = bin_dir / "exiftool_files"
    if destination.exists() and support_dir.exists() and not force:
        return "kept"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        temp_zip = Path(tmp.name)

    try:
        try:
            _download_zip(url, temp_zip)
        except Exception:
            if url == DEFAULT_EXIFTOOL_URL:
                latest_url = _resolve_latest_exiftool_url()
                _download_zip(latest_url, temp_zip)
            else:
                raise

        with zipfile.ZipFile(temp_zip) as archive:
            exe_members = [
                member
                for member in archive.infolist()
                if Path(member.filename).name.lower() in {"exiftool.exe", "exiftool(-k).exe"}
            ]
            if not exe_members:
                raise RuntimeError("ExifTool executable not found in downloaded archive")

            payload = archive.read(exe_members[0])
            destination.write_bytes(payload)

            for member in archive.infolist():
                parts = Path(member.filename).parts
                if "exiftool_files" not in parts:
                    continue
                relative_parts = parts[parts.index("exiftool_files"):]
                relative_path = Path(*relative_parts)
                output_path = bin_dir / relative_path
                if member.is_dir():
                    output_path.mkdir(parents=True, exist_ok=True)
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(archive.read(member))
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

    return "installed"


def getbins(
    config_path: Path | None = typer.Option(None, "--config-path", help="Path to config file"),
    force: bool = typer.Option(False, "--force", help="Replace existing binaries"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without downloading"),
    rclone_url: str = typer.Option(
        DEFAULT_RCLONE_URL,
        "--rclone-url",
        help="URL to a Windows rclone zip archive",
    ),
    exiftool_url: str = typer.Option(
        DEFAULT_EXIFTOOL_URL,
        "--exiftool-url",
        help="URL to a Windows ExifTool zip archive",
    ),
) -> None:
    """Download rclone.exe and exiftool.exe into {CATALOG_DIR}/bin on Windows."""
    if platform.system() != "Windows":
        typer.echo("⛔ getbins is currently supported on Windows only")
        raise typer.Exit(code=1)

    config_path = resolve_config_path(config_path)
    config = Config(config_path)
    bin_dir = config.catalog_dir / "bin"

    typer.echo(f"📦 Target bin directory: {bin_dir}")
    typer.echo(f"   rclone url: {rclone_url}")
    typer.echo(f"   exiftool url: {exiftool_url}")

    if dry_run:
        return

    bin_dir.mkdir(parents=True, exist_ok=True)

    try:
        rclone_status = _install_rclone(bin_dir, rclone_url, force)
        typer.echo(f"✅ rclone.exe {rclone_status} at {bin_dir / 'rclone.exe'}")

        exiftool_status = _install_exiftool(bin_dir, exiftool_url, force)
        typer.echo(f"✅ exiftool.exe {exiftool_status} at {bin_dir / 'exiftool.exe'}")
    except Exception as exc:
        typer.echo(f"⛔ getbins failed: {exc}")
        raise typer.Exit(code=1)
