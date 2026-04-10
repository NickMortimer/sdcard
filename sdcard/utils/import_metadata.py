from datetime import datetime
from pathlib import Path
import logging
import uuid
from typing import Any
import typer
import yaml
import psutil
import subprocess
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
    """Prompt operator to select one value from configured options."""
    typer.echo(f"Select {field_name} for {mount_path}:")
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


def make_yml(
    file_path,
    config,
    dry_run,
    card_number=0,
    overwrite=False,
    refresh=False,
    prompt_card_details=False,
    instrument: str | None = None,
):
    """Create or refresh an import.yml for a card."""
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
        if refresh:
            preserved_existing = {
                "instrument": existing.get("instrument"),
                "card_number": existing.get("card_number", 0),
            }
            existing = {key: value for key, value in preserved_existing.items() if value not in {None, ""}}
        if card_number == 0:
            card_number = existing.get('card_number', 0)

    if card_number == 0:
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
    pass_through_config, prompted_choices = _resolve_config_choices(
        pass_through_config,
        existing,
        file_path.parent,
        skip_keys={"instrument"},
    )
    raw_instrument_value = pass_through_config.get("instrument", config.data.get("instrument"))
    instrument_options = _extract_choice_options(raw_instrument_value)
    selected_instrument = _resolve_instrument(
        cli_instrument=instrument,
        existing_instrument=existing.get("instrument"),
        options=instrument_options,
        mount_path=file_path.parent,
    )
    if selected_instrument is not None:
        base_data["instrument"] = selected_instrument
        if len(instrument_options) > 1:
            prompted_choices["instrument"] = selected_instrument

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
    final_data = {key: value for key, value in final_data.items() if value is not None}

    if not dry_run:
        file_path.write_text(yaml.safe_dump(final_data, sort_keys=False), encoding="utf-8")
        _write_destination_readme(file_path.parent, final_data, config)

    _report_registration_result(
        file_path,
        final_data.get("instrument"),
        final_data.get("destination_path", destination_path),
        dry_run,
        prompted_choices,
    )
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
        )
        if result is not None:
            results.append(result)

    return results
