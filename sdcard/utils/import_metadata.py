from datetime import datetime
from pathlib import Path
import logging
import uuid
from typing import Any
import typer
import yaml
import psutil
import subprocess
import shutil
from jinja2 import Environment, FileSystemLoader, DebugUndefined
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
    import platform as _platform
    import ctypes

    details = {
        'card_size_gb': 0.0,
        'card_format': 'unknown',
        'partition_label': 'unknown',
    }

    try:
        usage = psutil.disk_usage(str(mount_path))
        details['card_size_gb'] = round(usage.total / (1024 ** 3), 2)
    except Exception:
        pass

    if _platform.system() == 'Windows':
        try:
            kernel32 = ctypes.windll.kernel32
            volume_name_buf = ctypes.create_unicode_buffer(1024)
            fs_name_buf = ctypes.create_unicode_buffer(1024)
            serial_number = ctypes.c_uint()
            max_comp_len = ctypes.c_uint()
            file_sys_flags = ctypes.c_uint()
            mount_str = str(mount_path)
            if not mount_str.endswith('\\'):
                mount_str += '\\'
            res = kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(mount_str),
                volume_name_buf,
                ctypes.sizeof(volume_name_buf),
                ctypes.byref(serial_number),
                ctypes.byref(max_comp_len),
                ctypes.byref(file_sys_flags),
                fs_name_buf,
                ctypes.sizeof(fs_name_buf),
            )
            if res:
                details['card_format'] = fs_name_buf.value or 'unknown'
                details['partition_label'] = volume_name_buf.value or mount_path.name or str(mount_path)
        except Exception:
            pass
    else:
        device_path = None
        for partition in psutil.disk_partitions():
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


def _set_volume_label(mount_path: Path, label: str, dry_run: bool) -> bool:
    """Set the volume label for the given mount path. Returns True on success."""
    try:
        system = subprocess.getoutput('uname') if False else None
        import platform
        system = platform.system()
        if system == 'Windows':
            # Use PowerShell Set-Volume
            drive = str(mount_path)
            if drive.endswith('\\') or drive.endswith('/'):
                drive = drive[:-1]
            # drive should be like 'K:' or 'K:\\'
            drive_letter = drive[0] if len(drive) >= 2 and drive[1] == ':' else None
            if not drive_letter:
                return False
            cmd = [
                'powershell', '-NoProfile', '-Command',
                f"Set-Volume -DriveLetter {drive_letter} -NewFileSystemLabel '{label}'"
            ]
            if dry_run:
                typer.echo(f"Would run: {' '.join(cmd)}")
                return True
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                typer.echo(proc.stdout + proc.stderr)
            return proc.returncode == 0
        else:
            # Unix-like: try exfatlabel or fatlabel
            device = None
            for part in psutil.disk_partitions():
                if Path(part.mountpoint) == mount_path:
                    device = part.device
                    break
            if not device:
                return False
            if shutil.which('exfatlabel'):
                cmd = ['exfatlabel', device, label]
            elif shutil.which('fatlabel'):
                cmd = ['fatlabel', device, label]
            else:
                typer.echo('No label utility found (exfatlabel/fatlabel).')
                return False
            if dry_run:
                typer.echo(f"Would run: {' '.join(cmd)}")
                return True
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                typer.echo(proc.stdout + proc.stderr)
            return proc.returncode == 0
    except Exception as e:
        typer.echo(f"Error setting volume label: {e}")
        return False


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


def _write_destination_readme_for_registration(
    import_yml_path: Path,
    importdetails: dict,
    config,
) -> None:
    """Write readme.md alongside the card's import.yml.

    This matches the operator expectation that the README lands on the card next
    to the freshly-written import.yml.
    """
    try:
        destination = import_yml_path.parent
        destination.mkdir(parents=True, exist_ok=True)

        typer.echo(f"📝 Writing readme.md to {destination / 'readme.md'}")
        _write_destination_readme(destination, importdetails, config)
    except Exception as exc:
        typer.echo(f"📝 Failed to write readme.md: {exc}")
        logging.warning(f"Failed to write destination readme for {import_yml_path}: {exc}")


def _format_instrument_summary(instrument_value: Any) -> str:
    """Format the configured instrument value for registration output."""
    if isinstance(instrument_value, dict):
        keys = [str(key).strip() for key in instrument_value if str(key).strip()]
        return ", ".join(keys) if keys else "-"

    if isinstance(instrument_value, (list, tuple, set)):
        values = [str(item).strip() for item in instrument_value if str(item).strip()]
        return ", ".join(values) if values else "-"

    if isinstance(instrument_value, str) and instrument_value.strip():
        return instrument_value.strip()

    return "-"


def _report_registration_result(
    file_path: Path,
    instrument: Any,
    destination_path: str,
    dry_run: bool,
    selected_choices: dict[str, Any] | None = None,
) -> None:
    """Print the registration outcome for one card."""
    action = "Would write" if dry_run else "Wrote"
    typer.echo(f"✏️  {action} {file_path}")
    typer.echo(f"   Instrument: {_format_instrument_summary(instrument)}")
    typer.echo(f"   Destination: {destination_path}")
    try:
        media = get_card_media_details(file_path.parent)
        partition_label = media.get('partition_label') if isinstance(media, dict) else None
    except Exception:
        partition_label = None
    drive = str(file_path.parent)
    if partition_label:
        typer.echo(f"   Drive: {partition_label} : {drive}")
    else:
        typer.echo(f"   Drive: {drive}")
    if selected_choices:
        for key in sorted(selected_choices):
            if key == "instrument":
                continue
            typer.echo(f"   {key}: {selected_choices[key]}")


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


def _extract_choice_options(raw_value: Any) -> list[tuple[str, str]]:
    """Build selectable options from config values."""
    if isinstance(raw_value, dict):
        return [
            (str(key).strip(), str(value).strip() or str(key).strip())
            for key, value in raw_value.items()
            if str(key).strip()
        ]

    if isinstance(raw_value, (list, tuple, set)):
        return [
            (str(item).strip(), str(item).strip())
            for item in raw_value
            if str(item).strip()
        ]

    if isinstance(raw_value, str) and raw_value.strip():
        value = raw_value.strip()
        return [(value, value)]

    return []


def _match_choice_value(value: str, options: list[tuple[str, str]]) -> str | None:
    """Return canonical option key if value matches key or label."""
    requested = value.strip().lower()
    for key, label in options:
        if requested in {key.lower(), label.lower()}:
            return key
    return None


def _prompt_for_choice(field_name: str, options: list[tuple[str, str]], mount_path: Path) -> str:
    """Prompt operator to select one value from configured options.

    The prompt now includes the partition label (if available) before the mount path, e.g.
    "Select <field> for P4RTK K:\\:".
    """
    try:
        media = get_card_media_details(mount_path)
        partition_label = media.get('partition_label') if isinstance(media, dict) else None
    except Exception:
        partition_label = None

    # On Windows get_card_media_details may not return the volume label; try Win32 API as a fallback
    if (not partition_label or str(partition_label).lower() == str(mount_path.name).lower()):
        try:
            import platform
            if platform.system() == 'Windows':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                volume_name_buf = ctypes.create_unicode_buffer(1024)
                fs_name_buf = ctypes.create_unicode_buffer(1024)
                serial_number = ctypes.c_uint()
                max_comp_len = ctypes.c_uint()
                file_sys_flags = ctypes.c_uint()
                res = kernel32.GetVolumeInformationW(
                    ctypes.c_wchar_p(str(mount_path)),
                    volume_name_buf,
                    ctypes.sizeof(volume_name_buf),
                    ctypes.byref(serial_number),
                    ctypes.byref(max_comp_len),
                    ctypes.byref(file_sys_flags),
                    fs_name_buf,
                    ctypes.sizeof(fs_name_buf),
                )
                if res and volume_name_buf.value:
                    partition_label = volume_name_buf.value
        except Exception:
            pass

    # Match the list command display: show label followed by a colon and the mount path
    if partition_label:
        target_display = f"{partition_label} : {mount_path}"
    else:
        target_display = str(mount_path)
    typer.echo(f"Select {field_name} for {target_display}:")
    for index, (key, label) in enumerate(options, start=1):
        typer.echo(f"  {index}. {key} ({label})")

    while True:
        selection = typer.prompt(
            f"{field_name} number",
            type=int,
        )
        if 1 <= selection <= len(options):
            return options[selection - 1][0]
        typer.echo(f"Please enter a value between 1 and {len(options)}")

def _extract_platform_instrument_options(raw_platforms: Any) -> list[tuple[str, str, str, str]]:
    """Return flattened options as (key, label, platform, instrument)."""
    if not isinstance(raw_platforms, dict):
        return []

    options: list[tuple[str, str, str, str]] = []
    for platform_key, platform_data in raw_platforms.items():
        platform = str(platform_key).strip()
        if not platform:
            continue

        instruments = platform_data.get("instruments", {}) if isinstance(platform_data, dict) else {}
        if not isinstance(instruments, dict):
            continue

        for instrument_key, instrument_label in instruments.items():
            instrument = str(instrument_key).strip()
            if not instrument:
                continue
            key = f"{platform}_{instrument}"
            label = str(instrument_label).strip() or instrument
            options.append((key, label, platform, instrument))

    return options


def _resolve_config_choice_value(
    field_name: str,
    raw_value: Any,
    existing_value: Any,
    mount_path: Path,
) -> tuple[Any, bool]:
    """Resolve config values that offer multiple choices."""
    options = _extract_choice_options(raw_value)
    if not options:
        return raw_value, False

    if isinstance(existing_value, str) and existing_value.strip():
        matched = _match_choice_value(existing_value, options)
        if matched is not None:
            return matched, False

    if len(options) == 1:
        return options[0][0], False

    return _prompt_for_choice(field_name, options, mount_path), True


def _resolve_config_choice_value_noninteractive(
    raw_value: Any,
    existing_value: Any,
) -> Any:
    """Resolve config choices without prompting.

    Preference order:
    1) existing matching value
    2) single available option
    3) first available option (deterministic)
    """
    options = _extract_choice_options(raw_value)
    if not options:
        return raw_value

    if isinstance(existing_value, str) and existing_value.strip():
        matched = _match_choice_value(existing_value, options)
        if matched is not None:
            return matched

    if len(options) == 1:
        return options[0][0]

    return options[0][0]


def _resolve_config_choices(
    pass_through_config: dict,
    existing: dict,
    mount_path: Path,
    skip_keys: set[str] | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Resolve top-level config values that define multiple choices."""
    skip_keys = skip_keys or set()
    resolved = pass_through_config.copy()
    prompted_choices: dict[str, Any] = {}
    for key, raw_value in pass_through_config.items():
        if key in skip_keys:
            continue
        if isinstance(raw_value, (dict, list, tuple, set)):
            resolved_value, prompted = _resolve_config_choice_value(
                key,
                raw_value,
                existing.get(key),
                mount_path,
            )
            resolved[key] = resolved_value
            if prompted:
                prompted_choices[key] = resolved_value
    return resolved, prompted_choices


def _resolve_config_choices_noninteractive(
    pass_through_config: dict,
    existing: dict,
    skip_keys: set[str] | None = None,
) -> dict:
    """Resolve top-level config choices without prompting the operator."""
    skip_keys = skip_keys or set()
    resolved = pass_through_config.copy()
    for key, raw_value in pass_through_config.items():
        if key in skip_keys:
            continue
        if isinstance(raw_value, (dict, list, tuple, set)):
            resolved[key] = _resolve_config_choice_value_noninteractive(
                raw_value,
                existing.get(key),
            )
    return resolved


def _resolve_instrument(
    cli_instrument: str | None,
    existing_instrument: Any,
    options: list[tuple[str, str]],
    mount_path: Path,
) -> str | None:
    """Resolve instrument from CLI, existing metadata, or operator prompt."""
    if cli_instrument:
        if not options:
            return cli_instrument.strip()

        matched = _match_choice_value(cli_instrument, options)
        if matched is not None:
            return matched

        valid = ", ".join([key for key, _ in options])
        raise typer.BadParameter(f"Invalid instrument '{cli_instrument}'. Choose one of: {valid}")

    if isinstance(existing_instrument, str) and existing_instrument.strip():
        existing_value = existing_instrument.strip()
        if not options:
            return existing_value
        matched = _match_choice_value(existing_value, options)
        if matched is not None:
            return matched

    if len(options) == 1:
        return options[0][0]

    if len(options) > 1:
        return _prompt_for_choice("instrument", options, mount_path)

    return None


def _get_destination_value_to_write(config, pass_through_config: dict) -> str:
    """Return the destination string that should be written to import.yml."""
    return (
        pass_through_config.get('destination_path')
        or pass_through_config.get('import_path_template')
        or config.data.get('import_path_template')
        or config.data.get('destination_path')
        or DEFAULT_IMPORT_TEMPLATE
    )


def refresh_import_yml_after_clean(
    file_path: Path,
    existing_data_pre_clean: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Refresh import.yml fields after a successful clean operation.

    Only updates import_token and register_date, and clears import_date.
    All other values are preserved from the pre-clean import.yml snapshot.
    """
    refreshed = dict(existing_data_pre_clean or {})
    refreshed["import_token"] = str(uuid.uuid4())[0:8]
    refreshed["register_date"] = f"{datetime.now():%Y-%m-%d}"
    refreshed.pop("import_date", None)

    if not dry_run:
        file_path.write_text(yaml.safe_dump(refreshed, sort_keys=False), encoding="utf-8")

    return refreshed


def make_yml(
    file_path,
    config,
    dry_run,
    card_number=0,
    overwrite=False,
    refresh=False,
    prompt_card_details=False,
    instrument: str | None = None,
    set_label: bool = False,
    format_card: bool = False,
    format_yes: bool = False,
    format_full: bool = False,
):
    """Create or refresh an import.yml for a card."""

    def _is_auto_card_number(value: Any) -> bool:
        """Return True when card number means "use existing/prompt"."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in {"", "0"}
        return value == 0

    def _format_card_before_write(
        mount_path: Path,
        label: str,
        dry_run: bool,
        format_yes: bool,
    format_full: bool,
    ) -> bool:
        """Format the card volume and set the filesystem label.

        Returns True if formatting was performed successfully, False if skipped or failed.
        """
        try:
            try:
                media = get_card_media_details(mount_path)
                partition_label = media.get('partition_label') if isinstance(media, dict) else None
            except Exception:
                partition_label = None

            drive_display = f"{partition_label} : {mount_path}" if partition_label else str(mount_path)
            desired_label = label or ""

            import platform as _platform
            system = _platform.system()
            cmd: list[str] | None

            if system == 'Windows':
                drive = str(mount_path)
                if drive.endswith('\\') or drive.endswith('/'):
                    drive = drive[:-1]
                drive_letter = drive[0] if len(drive) >= 2 and drive[1] == ':' else None
                if not drive_letter:
                    typer.echo(f"💾 Formatting not supported on {drive_display} (no drive letter); skipping format")
                    return False

                # Default to quick format. Full format is available via --format-full.
                # Surface failures by exiting non-zero.
                full_flag = " -Full" if format_full else ""
                cmd = [
                    'powershell',
                    '-NoProfile',
                    '-Command',
                    (
                        "$ErrorActionPreference='Stop'; "
                        f"Format-Volume -DriveLetter {drive_letter} -FileSystem exFAT "
                        f"-NewFileSystemLabel '{desired_label}' -Confirm:$false{full_flag} | Out-Null"
                    ),
                ]
            else:
                device = None
                for part in psutil.disk_partitions():
                    if Path(part.mountpoint) == mount_path:
                        device = part.device
                        break
                if not device:
                    typer.echo(f"💾 Formatting not supported on {drive_display} (no device); skipping format")
                    return False

                if shutil.which('mkfs.exfat'):
                    cmd = ['mkfs.exfat', '-n', desired_label, device]
                elif shutil.which('mkfs.vfat'):
                    cmd = ['mkfs.vfat', '-F', '32', '-n', desired_label, device]
                else:
                    typer.echo(f"💾 Formatting not supported on {drive_display} (no mkfs tool); skipping format")
                    return False

            typer.echo(f"💾 Command: {' '.join(cmd)}")
            if dry_run:
                typer.echo(f"💾 Would format {drive_display} with label '{desired_label}'")
                return True

            if not format_yes:
                if not typer.confirm(
                    f"Format {drive_display} (current label) as exFAT with new label '{desired_label}'?",
                    default=False,
                ):
                    typer.echo(f"💾 Skipping format for {drive_display}")
                    return False
                format_yes = True

            typer.echo(f"💾 Formatting {drive_display} with label '{desired_label}'")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout.strip() or proc.stderr.strip():
                typer.echo(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                typer.echo(f"💾 Format failed for {drive_display} (exit {proc.returncode})")
                return False
            return True
        except Exception as e:
            typer.echo(f"Formatting failed: {e}")
            return False
    existing = {}
    preserved_existing = {}
    if file_path.exists():
        if not overwrite and not refresh:
            typer.echo(f"Error SDCard already initialise {file_path}")
            return
        try:
            existing = yaml.safe_load(file_path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError:
            existing = {}
        # If caller requested turbo-style refresh, perform it using the current
        # import.yml contents as the pre-clean snapshot. This mirrors how
        # turbo/import uses refresh after a clean transfer.
        if refresh and existing:
            # Refresh mode mirrors turbo's "after clean" refresh. If formatting is
            # requested, show it first (and execute only when format_yes is set).
            if format_card:
                desired_label = str(existing.get("instrument") or "")
                _format_card_before_write(
                    mount_path=file_path.parent,
                    label=desired_label,
                    dry_run=dry_run,
                    format_yes=format_yes,
                    format_full=format_full,
                )
            refreshed = refresh_import_yml_after_clean(file_path, existing, dry_run)
            if not dry_run:
                _write_destination_readme_for_registration(file_path, refreshed, config)
            _report_registration_result(
                file_path,
                refreshed.get("instrument"),
                refreshed.get("destination_path", "-"),
                dry_run,
                None,
            )
            return refreshed
        if refresh:
            preserved_existing = {
                "instrument": existing.get("instrument"),
                "card_number": existing.get("card_number", 0),
            }
            existing = {key: value for key, value in preserved_existing.items() if value not in {None, ""}}
        if _is_auto_card_number(card_number):
            card_number = existing.get('card_number', 0)

    if _is_auto_card_number(card_number):
        card_number = typer.prompt(f"Card number [{str(file_path)}]", type=str, default='1')

    register_date = (
        f"{datetime.now():%Y-%m-%d}"
        if refresh else existing.get("register_date") or f"{datetime.now():%Y-%m-%d}"
    )
    import_date = None if refresh else existing.get("import_date")
    import_token = str(uuid.uuid4())[0:8] if refresh else existing.get("import_token") or str(uuid.uuid4())[0:8]
    media_details = get_card_media_details(file_path.parent)
    card_manufacturer = "" if refresh else existing.get("card_manufacturer", "")
    rated_uhs = "" if refresh else existing.get("rated_uhs", "")
    max_read_speed = None if refresh else existing.get("max_read_speed_mb_s")
    max_write_speed = None if refresh else existing.get("max_write_speed_mb_s")

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
    if refresh:
        # Refresh should preserve existing metadata choices on-card.
        pass_through_config = _resolve_config_choices_noninteractive(
            pass_through_config,
            existing,
            skip_keys={"instrument", "platforms"},
        )
        prompted_choices = {}
    else:
        # Non-refresh registration can prompt for config choices.
        pass_through_config, prompted_choices = _resolve_config_choices(
            pass_through_config,
            existing,
            file_path.parent,
            skip_keys={"instrument", "platforms"},
        )
    platforms_dict = pass_through_config.get("platforms") or config.data.get("platforms")
    # Behavior contract:
    # - refresh=True: preserve existing instrument metadata from import.yml.
    # - refresh=False and set_label=True: prompt operator to choose instrument.
    if set_label and not refresh and not instrument:
        existing_instrument_value = None
    else:
        existing_instrument_value = existing.get("instrument")
    if isinstance(platforms_dict, dict) and platforms_dict:
        option_rows = _extract_platform_instrument_options(platforms_dict)

        if not option_rows:
            typer.echo("No instruments defined under 'platforms'. Instrument will be set to '-'.")
            base_data["instrument"] = "-"
        else:
            options = [(key, label) for key, label, _platform, _instrument in option_rows]
            if instrument:
                matched_key = _match_choice_value(instrument, options)
                if matched_key is None:
                    valid = ", ".join([key for key, _ in options])
                    raise typer.BadParameter(
                        f"Invalid instrument '{instrument}'. Choose one of: {valid}"
                    )
                selected_key = matched_key
            elif isinstance(existing_instrument_value, str) and existing_instrument_value.strip():
                existing_instrument = existing_instrument_value.strip()
                matched_key = _match_choice_value(existing_instrument, options)
                selected_key = matched_key if matched_key else None
            elif len(options) == 1:
                selected_key = options[0][0]
            else:
                selected_key = _prompt_for_choice("instrument", options, file_path.parent)

            if selected_key is not None:
                selected_row = next(
                    row for row in option_rows if row[0].lower() == selected_key.lower()
                )
                base_data["instrument"] = selected_row[0]
                base_data["platform"] = selected_row[2]
                if len(options) > 1:
                    prompted_choices["instrument"] = selected_row[0]
    else:
        # fallback to old logic
        raw_instrument_value = pass_through_config.get("instrument", config.data.get("instrument"))
        instrument_options = _extract_choice_options(raw_instrument_value)
        selected_instrument = _resolve_instrument(
            cli_instrument=instrument,
            existing_instrument=existing_instrument_value,
            options=instrument_options,
            mount_path=file_path.parent,
        )
        platform = pass_through_config.get("platform") or config.data.get("platform")
        if not platform:
            # Try to get from new platforms structure
            platforms_dict = pass_through_config.get("platforms") or config.data.get("platforms")
            if isinstance(platforms_dict, dict) and len(platforms_dict) == 1:
                platform = list(platforms_dict.keys())[0]
        if selected_instrument is not None:
            if platform and selected_instrument:
                base_data["instrument"] = f"{platform}_{selected_instrument}"
            else:
                base_data["instrument"] = selected_instrument
            if len(instrument_options) > 1:
                prompted_choices["instrument"] = base_data["instrument"]

    merged_data = {**pass_through_config, **existing, **base_data}
    destination_path = _get_destination_value_to_write(config, pass_through_config)
    merged_data["destination_path"] = destination_path

    render_context = {
        **merged_data,
        "card_store": str(config.get_path('card_store')),
        "CATALOG_DIR": str(config.catalog_dir),
    }
    render_context = {key: value for key, value in render_context.items() if value is not None}
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
    # Remove config keys not meant for import.yml
    for unwanted in ("platforms", "instruments"):
        final_data.pop(unwanted, None)
    final_data = {key: value for key, value in final_data.items() if value is not None}

    # Optionally format the card before writing import.yml.
    if format_card:
        desired_label = str(final_data.get("instrument")) if final_data.get("instrument") else ""
        _format_card_before_write(
            mount_path=file_path.parent,
            label=desired_label,
            dry_run=dry_run,
            format_yes=format_yes,
            format_full=format_full,
        )

    if dry_run:
        typer.echo(f"✏️  Would write import.yml to {file_path}")
    else:
        typer.echo(f"✏️  Writing import.yml to {file_path}")
        file_path.write_text(yaml.safe_dump(final_data, sort_keys=False), encoding="utf-8")
    _write_destination_readme_for_registration(file_path, final_data, config)

    _report_registration_result(
        file_path,
        final_data.get("instrument"),
        final_data.get("destination_path", destination_path),
        dry_run,
        prompted_choices,
    )
    # Optionally set the volume label to the instrument name
    try:
        if set_label and final_data.get("instrument"):
            desired_label = str(final_data.get("instrument"))
            ok = _set_volume_label(file_path.parent, desired_label, dry_run)
            if not ok and not dry_run:
                typer.echo(f"Failed to set volume label to '{desired_label}' on {file_path.parent}")
    except Exception:
        # Don't raise on label-set failure; registration should still succeed
        pass
    return final_data


def register_cards(
    config,
    card_path,
    card_number,
    overwrite,
    dry_run: bool,
    refresh: bool = False,
    prompt_card_details: bool = False,
    instrument: str | None = None,
    set_label: bool = False,
    format_card: bool = False,
    format_yes: bool = False,
    format_full: bool = False,
):
    """
    Register SD cards by creating an import.yml on each card
    """
    if card_number is None:
        card_number = ['0'] * len(card_path) if isinstance(card_path, list) else 0

    results = []

    if isinstance(card_path, list):
        for path, cardno in zip(card_path, card_number):
            result = make_yml(
                Path(path) / "import.yml",
                config,
                dry_run,
                cardno,
                overwrite,
                refresh,
                prompt_card_details,
                instrument,
                set_label=set_label,
                format_card=format_card,
                format_yes=format_yes,
                format_full=format_full,
            )
            if result is not None:
                results.append(result)
    else:
        result = make_yml(
            Path(card_path) / "import.yml",
            config,
            dry_run,
            card_number,
            overwrite,
            refresh,
            prompt_card_details,
            instrument,
            set_label=set_label,
            format_card=format_card,
            format_yes=format_yes,
            format_full=format_full,
        )
        if result is not None:
            results.append(result)

    return results
