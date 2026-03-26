from datetime import datetime
from pathlib import Path
import logging
import os
import platform
import shlex
import subprocess
import typer
import uuid
import yaml
import psutil
from sdcard.config import DEFAULT_IMPORT_TEMPLATE
from sdcard.utils.import_conflicts import (
    _collect_destination_conflicts,
    _prompt_for_new_token_if_destination_changed,
    _write_conflict_report,
)
from sdcard.utils.import_metadata import _resolve_destination_path_template, make_yml


def import_cards(config, card_path, copy, move, find, file_extension, dry_run: bool, format_card=False, allow_overwrite: bool = False, check: bool = False, precheck: bool = False):
    """Copy or move SD card contents to the configured card store."""
    _ = file_extension
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

        if not dry_run and import_metadata_changed:
            importyml.write_text(yaml.safe_dump(importdetails, sort_keys=False), encoding="utf-8")

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

        if find:
            matches = list(card_store.rglob(f"*{importdetails.get('import_token')}*"))
            if matches:
                destination = max(matches)

        typer.echo(f"💾 Reading {importdetails['import_token']} from {card} to {destination}")

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

        overwrite_allowed_for_card = False
        if precheck:
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
            immutable_flag = ""
            if precheck and not (allow_overwrite or overwrite_allowed_for_card):
                immutable_flag = " --immutable"
            check_first_flag = " --check-first" if precheck else ""
            progress_flags = "--stats-one-line --stats=1s -v" if os.environ.get("SDCARD_RCLONE_TABLE") else "--progress"
            command = f"{rclone_path} copy {Path(card).resolve()} {destination.resolve()} {progress_flags} --update{check_first_flag} --low-level-retries 1 --modify-window=2s{immutable_flag} "
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
            progress_flags = "--stats-one-line --stats=1s -v" if os.environ.get("SDCARD_RCLONE_TABLE") else "--progress"
            check_first_flag = " --check-first" if precheck else ""
            command = f"{rclone_path} move {card} {destination} {progress_flags} --update{check_first_flag} --delete-empty-src-dirs"
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

                make_yml(importyml, config, dry_run, importdetails.get('card_number', 0), overwrite=True)

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
