
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, DebugUndefined
from datetime import datetime, timedelta
import uuid
import psutil
from math import ceil
import typer
import yaml
import glob
import pandas as pd
import subprocess
import shlex
import logging
import os
import sys
from io import StringIO
import platform
import re
from sdcard.config import DEFAULT_IMPORT_TEMPLATE

DEFAULT_IMPORT_YML_TEMPLATE = """card_number: {{ card_number }}\nimport_token: {{ import_token }}\nregister_date: {{ register_date }}\n{% if import_date %}import_date: {{ import_date }}\n{% endif %}destination_path: {{ destination_path }}\npartition_label: {{ partition_label }}\ncard_size_gb: {{ card_size_gb }}\ncard_format: {{ card_format }}\n{% if card_manufacturer %}card_manufacturer: {{ card_manufacturer }}\n{% endif %}{% if rated_uhs %}rated_uhs: {{ rated_uhs }}\n{% endif %}{% if max_read_speed_mb_s is not none %}max_read_speed_mb_s: {{ max_read_speed_mb_s }}\n{% endif %}{% if max_write_speed_mb_s is not none %}max_write_speed_mb_s: {{ max_write_speed_mb_s }}\n{% endif %}"""
DEFAULT_DESTINATION_README_TEMPLATE = """# {{ project_name or 'Unknown project' }}

## Data Custody
- Custodian: {{ custodian or 'Unknown custodian' }}
{% if email %}- Contact: {{ email }}
{% endif %}
## Import Details
{% if field_trip_id %}- Project / Trip ID: {{ field_trip_id }}
{% endif %}
{% if card_number is not none %}- Card Number: {{ card_number }}
{% endif %}
{% if import_token %}- Import Token: {{ import_token }}
{% endif %}
{% if register_date %}- Registration Date: {{ register_date }}
{% endif %}
{% if import_date %}- Import Date: {{ import_date }}
{% endif %}
"""


def get_card_media_details(mount_path: Path) -> dict:
    """Get detected size and filesystem type for a mounted card."""
    details = {
        'card_size_gb': 0.0,
        'card_format': 'unknown',
        'partition_label': 'unknown',
    }
    device_path = None

    try:
        usage = psutil.disk_usage(str(mount_path))
        details['card_size_gb'] = round(usage.total / (1024 ** 3), 2)
    except Exception:
        pass

    for partition in psutil.disk_partitions(all=True):
        if Path(partition.mountpoint) == mount_path:
            details['card_format'] = partition.fstype or 'unknown'
            device_path = partition.device
            break

    if device_path:
        details['partition_label'] = Path(device_path).name
        try:
            result = subprocess.run(['lsblk', '-no', 'LABEL', device_path], capture_output=True, text=True)
            if result.returncode == 0:
                label = result.stdout.strip()
                if label:
                    details['partition_label'] = label
        except Exception:
            pass

    if details['partition_label'] == 'unknown':
        details['partition_label'] = mount_path.name or str(mount_path)

    return details

def get_card_number_from_import_yml(file_path: Path):
    """
    Extract card number from import.yml file.
    
    Args:
        file_path (Path): Path to the import.yml file.
    
    Returns:
        int: Card number extracted from the file, or 0 if not found.
    """
    if file_path.exists():
        try:
            with open(file_path, 'r') as stream:
                data = yaml.safe_load(stream)
                return data.get('card_number', 0)
        except yaml.YAMLError as exc:
            logging.error(f"Error reading {file_path}: {exc}")
            return 0
    return 0

def get_available_cards(format_type='exfat',maxcardsize=512):
    """Get list of available SD cards"""
    drives = []
    for part in psutil.disk_partitions():
        if part.fstype.lower() == format_type.lower():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                if usage.total < maxcardsize * 1024**3:  # Less than maxcardsize GB
                    usb_host = get_usb_info(part.device)
                    if usb_host is not None:
                        drives.append({
                            'mountpoint': part.mountpoint,
                            'host': f"usb{usb_host}",
                            'size_gb': round(usage.total / (1024**3), 2),
                            'free_gb': round(usage.free / (1024**3), 2),
                            'card_number': get_card_number_from_import_yml(Path(part.mountpoint) / "import.yml")
                        })
            except Exception as e:
                print(f"Error processing {part.device}: {e}")
    return drives


def get_usb_info(device_path):
    """Get USB controller info based on platform"""
    if platform.system() == "Linux":
        cmd = f"udevadm info -q path -n {device_path}"
        dev_path = subprocess.getoutput(cmd)
        if 'usb' in dev_path:
            # Extract the last host number from the path
            host_match = re.search(r'host(\d+)', dev_path)
            if host_match:
                return host_match.group(1)
    else:
        try:
            import wmi
            c = wmi.WMI()
            # Convert drive letter to physical disk
            for disk in c.Win32_DiskDrive():
                for partition in disk.associators("Win32_DiskDriveToDiskPartition"):
                    for logical_disk in partition.associators("Win32_LogicalDiskToPartition"):
                        if logical_disk.DeviceID.replace(":", "") == device_path[0]:
                            # Get USB controller info from host controller
                            # SCSIPort is valid (can be 0) - it's the SCSI adapter number
                            return disk.SCSIPort
            # If no matching disk found
            return None
        except Exception as e:
            # If WMI fails (permissions, missing module, etc.)
            return None
                    
def list_sdcards(format_type,maxcardsize=512):
    """
    Scan for SD cards.

    Args:
        format_type : type of format on the sdcard (exfat preffered)
        maxcardsize : select drives with less than the max in Gb
    """
    result =[]
    for i in psutil.disk_partitions():
        if i.fstype.lower()==format_type:
            p =psutil.disk_usage(i.mountpoint)
            if ceil(p.total/1000000000)<=maxcardsize:            
                result.append(i.mountpoint)
    return result

def _render_destination_path(config, import_context: dict, preserve_unknown: bool = False) -> str:
    """Render the destination path using config template and context."""
    template_string = (
        config.data.get('import_path_template')
        or config.data.get('destination_path')
        or DEFAULT_IMPORT_TEMPLATE
    )
    environment = Environment(undefined=DebugUndefined) if preserve_unknown else Environment()
    template = environment.from_string(template_string)
    return template.render(**import_context)


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def _resolve_destination_path_template(template_string: str, context: dict) -> str:
    """Resolve destination template using Jinja and single-brace placeholders."""
    resolved = Environment(undefined=DebugUndefined).from_string(template_string).render(**context)
    return resolved.format_map(_SafeFormatDict(context))


def _list_file_metadata(path: Path | str) -> dict[str, dict[str, object]]:
    """Return a mapping of relative file path to metadata using native Python stat calls."""
    root = Path(path)
    file_metadata = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            abs_path = Path(dirpath) / name
            try:
                st = abs_path.stat()
            except OSError:
                continue
            rel_path = str(abs_path.relative_to(root))
            file_metadata[rel_path] = {
                "mod_time": st.st_mtime,
                "size": st.st_size,
            }
    return file_metadata


def _is_comparison_ignored(rel_path: str) -> bool:
    """Exclude metadata and trash paths from overwrite safety comparison."""
    normalized = rel_path.replace('\\', '/').strip()
    basename = Path(normalized).name.lower()
    if normalized.startswith('.Trash-'):
        return True
    if normalized.lower().endswith('.trashinfo'):
        return True
    if basename in {"import.yml", "readme.md"}:
        return True
    return False


def _modtimes_differ(src_mod: object, dst_mod: object, tolerance_seconds: float = 1.0) -> bool:
    """Compare modtimes (float epoch seconds) with tolerance to avoid filesystem precision false positives."""
    if src_mod is None and dst_mod is None:
        return False
    if src_mod is None or dst_mod is None:
        return True
    try:
        return abs(float(src_mod) - float(dst_mod)) > tolerance_seconds
    except Exception:
        return src_mod != dst_mod


def _collect_destination_conflicts(source: Path, destination: Path) -> dict[str, object]:
    """Collect overlap and conflict metadata between source and destination trees."""
    if not destination.exists():
        return {
            "source_metadata": {},
            "destination_metadata": {},
            "overlapping_files": [],
            "differing_files": [],
            "likely_partial_files": [],
        }

    source_metadata = _list_file_metadata(source)
    destination_metadata = _list_file_metadata(destination)

    overlapping_files = sorted(
        rel_path
        for rel_path in (set(source_metadata) & set(destination_metadata))
        if not _is_comparison_ignored(rel_path)
    )

    differing_files = [
        rel_path
        for rel_path in overlapping_files
        if (
            _modtimes_differ(
                source_metadata.get(rel_path, {}).get("mod_time"),
                destination_metadata.get(rel_path, {}).get("mod_time"),
            )
            or source_metadata.get(rel_path, {}).get("size") != destination_metadata.get(rel_path, {}).get("size")
        )
    ]

    likely_partial_files = [
        rel_path
        for rel_path in differing_files
        if isinstance(source_metadata.get(rel_path, {}).get("size"), int)
        and isinstance(destination_metadata.get(rel_path, {}).get("size"), int)
        and destination_metadata.get(rel_path, {}).get("size") < source_metadata.get(rel_path, {}).get("size")
    ]

    return {
        "source_metadata": source_metadata,
        "destination_metadata": destination_metadata,
        "overlapping_files": overlapping_files,
        "differing_files": differing_files,
        "likely_partial_files": likely_partial_files,
    }


def _write_conflict_report(
    report_dir: Path,
    card: Path,
    destination: Path,
    import_token: str,
    conflict_data: dict[str, object],
) -> Path:
    """Write a detailed conflict report and return its path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_card = str(card).strip("/").replace("/", "_") or "card"
    report_path = report_dir / f"{timestamp}_{import_token}_{safe_card}.txt"

    source_metadata = conflict_data.get("source_metadata", {})
    destination_metadata = conflict_data.get("destination_metadata", {})
    differing_files = conflict_data.get("differing_files", [])
    likely_partial_files = set(conflict_data.get("likely_partial_files", []))

    lines = [
        "SDCard import conflict report",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"card: {card}",
        f"destination: {destination}",
        f"import_token: {import_token}",
        f"conflict_count: {len(differing_files)}",
        "",
        "files:",
    ]

    for rel_path in differing_files:
        src = source_metadata.get(rel_path, {})
        dst = destination_metadata.get(rel_path, {})
        partial_tag = " [likely_partial]" if rel_path in likely_partial_files else ""
        lines.append(
            f"- {rel_path}{partial_tag}\n"
            f"  source:      mod={src.get('mod_time')} size={src.get('size')}\n"
            f"  destination: mod={dst.get('mod_time')} size={dst.get('size')}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _prompt_for_new_token_if_destination_changed(
    source: Path,
    destination: Path,
    importdetails: dict,
    rclone_path: Path | str,
    allow_overwrite: bool = False,
    prompt_overwrite: bool = True,
) -> bool:
    """Abort import and report differences when an existing destination has changed files."""
    try:
        conflict_data = _collect_destination_conflicts(source, destination)
    except Exception as exc:
        logging.warning(f"Unable to compare source and destination: {exc}")
        return False

    source_metadata = conflict_data["source_metadata"]
    destination_metadata = conflict_data["destination_metadata"]
    overlapping_files = conflict_data["overlapping_files"]
    differing_files = conflict_data["differing_files"]
    likely_partial_files = conflict_data["likely_partial_files"]

    if not overlapping_files:
        return False
    if not differing_files:
        return False

    token = importdetails.get("import_token", "unknown")
    typer.echo("\n⛔ Import stopped to prevent overwriting existing data.")
    typer.echo(f"Destination: {destination}")
    typer.echo(f"Import token: {token}")
    typer.echo(
        f"Detected {len(differing_files)} changed files out of {len(overlapping_files)} overlapping files "
        f"(modification time and/or size)."
    )

    if likely_partial_files:
        typer.echo(
            f"Likely leftovers from a failed previous copy: {len(likely_partial_files)} files "
            f"have smaller destination size than source."
        )
        typer.echo("\nLikely partial files (first 10):")
        for rel_path in likely_partial_files[:10]:
            typer.echo(
                f"- {rel_path}\n"
                f"  source size:      {source_metadata.get(rel_path, {}).get('size')}\n"
                f"  destination size: {destination_metadata.get(rel_path, {}).get('size')}"
            )
        if len(likely_partial_files) > 10:
            typer.echo(f"... and {len(likely_partial_files) - 10} more likely partial files.")

    typer.echo("\nChanged files (first 10):")
    for rel_path in differing_files[:10]:
        typer.echo(
            f"- {rel_path}\n"
            f"  source:      mod={source_metadata.get(rel_path, {}).get('mod_time')} size={source_metadata.get(rel_path, {}).get('size')}\n"
            f"  destination: mod={destination_metadata.get(rel_path, {}).get('mod_time')} size={destination_metadata.get(rel_path, {}).get('size')}"
        )
    if len(differing_files) > 10:
        typer.echo(f"... and {len(differing_files) - 10} more changed files.")

    if allow_overwrite:
        typer.echo("\n⚠️ --allow-overwrite is set; continuing with overwrite enabled.")
        return True

    stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if prompt_overwrite and stdin_is_tty:
        overwrite_confirmed = typer.confirm(
            "\nOverwrite changed files in destination? This is unsafe and may replace existing data.",
            default=False,
        )
        if overwrite_confirmed:
            typer.echo("⚠️ User confirmed overwrite for this import.")
            return True

    raise typer.Abort(
        "Destination already contains different data. Create a new import token before retrying."
    )


def _write_destination_readme(destination: Path, importdetails: dict, config) -> None:
    template_path = importdetails.get("destination_readme_template_path")
    template_string = importdetails.get("destination_readme_template")

    if template_path:
        template_file = Path(template_path)
        if not template_file.is_absolute():
            template_file = config.catalog_dir / template_file
        if template_file.exists():
            template_string = template_file.read_text(encoding="utf-8")
        else:
            sibling_match = None
            try:
                for sibling in template_file.parent.iterdir():
                    if sibling.name.lower() == template_file.name.lower():
                        sibling_match = sibling
                        break
            except Exception:
                sibling_match = None

            if sibling_match and sibling_match.exists():
                template_string = sibling_match.read_text(encoding="utf-8")
            else:
                logging.warning(f"README template not found at {template_file}; using default template")

    if not template_string:
        template_string = DEFAULT_DESTINATION_README_TEMPLATE

    environment = Environment(trim_blocks=True, lstrip_blocks=True)
    template = environment.from_string(template_string)
    rendered = template.render(**importdetails).rstrip() + "\n"
    destination.joinpath("readme.md").write_text(rendered, encoding="utf-8")


def _get_pass_through_config_values(config) -> dict:
    """Return raw key/value pairs from the user-provided config file."""
    config_path = Path(getattr(config, 'config_path', ''))
    if not config_path.exists() or config_path.is_dir():
        return {}

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}

    return raw_config if isinstance(raw_config, dict) else {}


def make_yml(file_path, config, dry_run, card_number=0, overwrite=False, prompt_card_details=False):
    """Create or refresh an import.yml for a card."""
    existing = {}
    if file_path.exists():
        if not overwrite:
            typer.echo(f"Error SDCard already initialise {file_path}")
            return
        try:
            existing = yaml.safe_load(file_path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError:
            existing = {}
        if card_number == 0:
            card_number = existing.get('card_number', 0)

    if card_number == 0:
        card_number = typer.prompt(f"Card number [{str(file_path)}]", type=str, default='1')

    register_date = existing.get("register_date") or f"{datetime.now():%Y-%m-%d}"
    import_date = existing.get("import_date")
    import_token = existing.get("import_token") or str(uuid.uuid4())[0:8]
    destination_path = existing.get("destination_path")
    media_details = get_card_media_details(file_path.parent)
    card_manufacturer = existing.get("card_manufacturer", "")
    rated_uhs = existing.get("rated_uhs", "")
    max_read_speed = existing.get("max_read_speed_mb_s")
    max_write_speed = existing.get("max_write_speed_mb_s")

    if prompt_card_details:
        if typer.confirm(f"Add optional metadata for {file_path.parent}?", default=bool(card_manufacturer or rated_uhs)):
            card_manufacturer = typer.prompt(
                "Card manufacturer",
                default=card_manufacturer,
                show_default=bool(card_manufacturer),
            ).strip()
            rated_uhs = typer.prompt(
                "Rated UHS",
                default=rated_uhs,
                show_default=bool(rated_uhs),
            ).strip().upper()
            max_read_speed = typer.prompt(
                "Stated max read speed (MB/s)",
                default=max_read_speed if max_read_speed is not None else 0.0,
                show_default=max_read_speed is not None,
                type=float,
            )
            max_write_speed = typer.prompt(
                "Stated max write speed (MB/s)",
                default=max_write_speed if max_write_speed is not None else 0.0,
                show_default=max_write_speed is not None,
                type=float,
            )

            if max_read_speed <= 0:
                max_read_speed = None
            if max_write_speed <= 0:
                max_write_speed = None

    base_data = {
        "import_date": import_date,
        "import_token": import_token,
        "card_number": card_number,
        "register_date": register_date,
        "partition_label": media_details['partition_label'],
        "card_size_gb": media_details['card_size_gb'],
        "card_format": media_details['card_format'],
        "card_manufacturer": card_manufacturer,
        "rated_uhs": rated_uhs,
        "max_read_speed_mb_s": max_read_speed,
        "max_write_speed_mb_s": max_write_speed,
    }

    pass_through_config = _get_pass_through_config_values(config)
    merged_data = {**pass_through_config, **existing, **base_data}

    render_context = {
        **merged_data,
        "card_store": str(config.get_path('card_store')),
        "CATALOG_DIR": str(config.catalog_dir),
    }
    render_context = {key: value for key, value in render_context.items() if value is not None}

    destination_path = merged_data.get("destination_path")
    if not destination_path:
        destination_path = (
            pass_through_config.get('destination_path')
            or config.data.get('import_path_template')
            or config.data.get('destination_path')
            or DEFAULT_IMPORT_TEMPLATE
        )

    merged_data["destination_path"] = destination_path
    render_context["destination_path"] = destination_path

    template_path = config.data.get('import_template_path')
    if template_path:
        env = Environment(loader=FileSystemLoader(Path(template_path).parent), trim_blocks=True, lstrip_blocks=True)
        template = env.get_template(Path(template_path).name)
        rendered = template.render(render_context)
    else:
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(DEFAULT_IMPORT_YML_TEMPLATE)
        rendered = template.render(render_context)

    try:
        rendered_data = yaml.safe_load(rendered) or {}
        if not isinstance(rendered_data, dict):
            rendered_data = {}
    except Exception:
        rendered_data = {}

    final_data = {**merged_data, **rendered_data}
    final_data = {key: value for key, value in final_data.items() if value is not None}

    if not dry_run:
        file_path.write_text(yaml.safe_dump(final_data, sort_keys=False), encoding="utf-8")
        _write_destination_readme(file_path.parent, final_data, config)

def register_cards(config,card_path,card_number,overwrite,dry_run: bool, prompt_card_details: bool = False):
    """
    Register SD cards by creating an import.yml on each card
    """
    # Set dry run log string to prepend to logging
    dry_run_log_string = "DRY_RUN - " if dry_run else ""

    if card_number is None:
        card_number = ['0'] * len(card_path) if isinstance(card_path, list) else 0

    if isinstance(card_path, list):
        for path, cardno in zip(card_path, card_number):
            make_yml(Path(path) / "import.yml", config, dry_run, cardno, overwrite, prompt_card_details)
    else:
        make_yml(Path(card_path) / "import.yml", config, dry_run, card_number, overwrite, prompt_card_details)

def import_cards(config, card_path, copy, move, find, file_extension, dry_run: bool, format_card=False, allow_overwrite: bool = False, check: bool = False):
    """Copy or move SD card contents to the configured card store."""
    check_failures: list[str] = []
    check_checked = 0

    if check:
        copy = False
        move = False
        allow_overwrite = False

    for card in card_path:
        check_checked += 1
        dry_run_log_string = "DRY_RUN - " if dry_run else ""
        importyml = Path(card) / "import.yml"
        if importyml.exists():
            try:
                importdetails = yaml.safe_load(importyml.read_text(encoding='utf-8'))
            except yaml.YAMLError as exc:
                if check:
                    check_failures.append(f"{card}: corrupt import.yml ({exc})")
                    typer.echo(f"⛔ {card}: corrupt import.yml")
                    continue
                raise typer.Abort(f"Error possible corrupt yaml {importyml}: {exc}")
        else:
            typer.echo(f"Error {importyml} not found")
            if check:
                check_failures.append(f"{card}: missing import.yml")
            continue

        import_metadata_changed = False
        if not importdetails.get("import_token"):
            importdetails["import_token"] = str(uuid.uuid4())[0:8]
            import_metadata_changed = True

        typer.echo(f"💾 Reading {importdetails['import_token']} from {card}")
        card_store = Path(config.data.get('card_store'))
        configured_destination_template = (
            config.data.get('import_path_template')
            or config.data.get('destination_path')
            or DEFAULT_IMPORT_TEMPLATE
        )
        importdetails.setdefault("register_date", f"{datetime.now():%Y-%m-%d}")
        if not importdetails.get("import_date"):
            importdetails["import_date"] = f"{datetime.now():%Y-%m-%d}"
            import_metadata_changed = True

        # Persist import metadata before transfer so future imports resolve to the same directory.
        if not dry_run and import_metadata_changed:
            importyml.write_text(yaml.safe_dump(importdetails, sort_keys=False), encoding="utf-8")

        # Resolve destination from stored template (or fallback template) at import time
        import_context = {
            **importdetails,
            "import_date": importdetails.get("import_date"),
            "import_token": importdetails.get("import_token"),
            "card_number": importdetails.get("card_number", 0),
            "card_store": str(card_store),
            "CATALOG_DIR": str(config.catalog_dir),
            "register_date": importdetails.get("register_date"),
        }
        destination_template = (
            importdetails.get('destination_path')
            or configured_destination_template
        )
        destination_path = _resolve_destination_path_template(destination_template, import_context)
        importdetails['destination_path'] = destination_path

        destination = Path(destination_path)

        # Allow reuse of an existing destination when find flag is on
        if find:
            matches = list(card_store.rglob(f"*{importdetails.get('import_token')}*"))
            if matches:
                destination = max(matches)

        # Choose rclone binary per platform
        if platform.system() == "Windows":
            rclone_path = config.catalog_dir / 'bin' / 'rclone.exe'
        else:
            rclone_path = 'rclone'

        if check:
            try:
                conflict_data = _collect_destination_conflicts(Path(card), destination)
            except Exception as exc:
                check_failures.append(f"{card}: unable to compare source and destination ({exc})")
                typer.echo(f"⛔ {card}: unable to compare source and destination")
                continue

            differing_files = conflict_data["differing_files"]
            if differing_files:
                import_token = str(importdetails.get("import_token", "unknown"))
                report_path = _write_conflict_report(
                    Path(config.catalog_dir) / "conflicts",
                    Path(card),
                    destination,
                    import_token,
                    conflict_data,
                )
                check_failures.append(f"{card}: {len(differing_files)} changed file(s) -> {report_path}")
                typer.echo(f"⛔ {card}: {len(differing_files)} changed file(s)")
                typer.echo(f"   Report: {report_path}")
                typer.echo("   Problem files (first 20):")
                for rel_path in differing_files[:20]:
                    typer.echo(f"   - {rel_path}")
                if len(differing_files) > 20:
                    typer.echo(f"   ... and {len(differing_files) - 20} more")
            else:
                typer.echo(f"✅ {card}: no overwrite conflicts detected")
            continue

        try:
            overwrite_allowed_for_card = _prompt_for_new_token_if_destination_changed(
                Path(card),
                destination,
                importdetails,
                rclone_path,
                allow_overwrite=allow_overwrite,
                prompt_overwrite=not dry_run,
            )
        except typer.Abort:
            raise

        if copy:
            logging.info(f'{dry_run_log_string}  Copy  {card} --> {destination}')
            immutable_flag = "" if (allow_overwrite or overwrite_allowed_for_card) else " --immutable"
            command = f"{rclone_path} copy {Path(card).resolve()} {destination.resolve()} --progress --low-level-retries 1 --modify-window=2s{immutable_flag} "
            logging.info(f'running {command}')
            command = command.replace('\\', '/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True, parents=True)
                try:
                    subprocess.run(shlex.split(command), check=True)
                except subprocess.CalledProcessError as exc:
                    raise typer.Abort(f"rclone copy failed (exit code {exc.returncode}). Import aborted before overwrite.")

        if move:
            command = f"{rclone_path} move {card} {destination} --progress --delete-empty-src-dirs"
            command = command.replace('\\', '/')
            logging.info(f'{dry_run_log_string}  {command}')
            if not dry_run:
                destination.mkdir(exist_ok=True, parents=True)
                try:
                    subprocess.run(shlex.split(command), check=True)
                except subprocess.CalledProcessError as exc:
                    raise typer.Abort(f"rclone move failed (exit code {exc.returncode}). Move aborted.")

                if format_card and (psutil.disk_usage(card).used < 1 * 1024**3):
                    if platform.system() == "Windows":
                        command = f"format {card} /FS:exFAT /Q /Y"
                    else:
                        command = f"mkfs.exfat {card}"
                    command = command.replace('\\', '/')
                    logging.info(f'{dry_run_log_string}  Deleting empty drive {card}')
                    command = f"rmdir {card}"
                    command = command.replace('\\', '/')
                    logging.info(f'{dry_run_log_string}  {command}')
                    if not dry_run:
                        process = subprocess.Popen(shlex.split(command))
                        process.wait()

                # Re-create import.yml so the card stays registered
                make_yml(importyml, config, dry_run, importdetails.get('card_number', 0), overwrite=True)

        # Persist updated import_date/register_date when a transfer actually occurred
        if not dry_run and (copy or move):
            persisted = dict(importdetails)
            persisted.update({
                "card_number": importdetails.get("card_number"),
                "import_token": importdetails.get("import_token"),
                "register_date": importdetails.get("register_date"),
                "import_date": importdetails.get("import_date"),
                "destination_path": importdetails.get("destination_path", destination_path),
            })
            importyml.write_text(yaml.safe_dump(persisted), encoding="utf-8")

    if check:
        typer.echo(f"\nChecked {check_checked} card(s).")
        if check_failures:
            typer.echo(f"Found {len(check_failures)} issue(s):")
            for issue in check_failures:
                typer.echo(f"- {issue}")
            raise typer.Exit(code=1)
        typer.echo("No overwrite issues detected.")

