from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from sdcard.utils.config_path_cache import resolve_config_path


def _load_workflows(file_path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate workflow definitions from a YAML file."""
    if not file_path.exists():
        raise typer.BadParameter(
            f"Workflow file not found: {file_path}",
            param_hint="--file",
        )

    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise typer.BadParameter(
            f"Failed to parse workflow file {file_path}: {exc}",
            param_hint="--file",
        ) from exc

    if not isinstance(payload, dict):
        raise typer.BadParameter(
            "Workflow file must contain a top-level mapping.",
            param_hint="--file",
        )

    workflows = payload.get("workflows")
    if not isinstance(workflows, dict):
        raise typer.BadParameter(
            "Workflow file must contain a top-level 'workflows' mapping.",
            param_hint="--file",
        )

    normalized: dict[str, dict[str, Any]] = {}
    for workflow_name, workflow_data in workflows.items():
        if not isinstance(workflow_name, str) or not workflow_name.strip():
            raise typer.BadParameter(
                "Workflow names must be non-empty strings.",
                param_hint="--file",
            )

        if not isinstance(workflow_data, dict):
            raise typer.BadParameter(
                f"Workflow '{workflow_name}' must be a mapping.",
                param_hint="--file",
            )

        commands = workflow_data.get("commands")
        if not isinstance(commands, list) or not commands:
            raise typer.BadParameter(
                f"Workflow '{workflow_name}' must define a non-empty commands list.",
                param_hint="--file",
            )

        description = workflow_data.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise typer.BadParameter(
                f"Workflow '{workflow_name}' description must be a string.",
                param_hint="--file",
            )

        normalized[workflow_name] = {
            "description": description,
            "commands": commands,
        }

    return normalized


def _normalize_step(
    step_value: str | list[str],
    workflow_name: str,
    step_index: int,
) -> list[str]:
    """Normalize one workflow step into argv form."""
    if isinstance(step_value, str):
        args = shlex.split(step_value)
    elif isinstance(step_value, list):
        if not all(isinstance(item, str) and item for item in step_value):
            raise typer.BadParameter(
                f"Workflow '{workflow_name}' step {step_index} list entries must be non-empty strings."
            )
        args = step_value
    else:
        raise typer.BadParameter(
            f"Workflow '{workflow_name}' step {step_index} must be a string or list[str]."
        )

    if not args:
        raise typer.BadParameter(
            f"Workflow '{workflow_name}' step {step_index} resolves to no arguments."
        )

    return args


def _steps_for_workflow(
    workflow_name: str,
    workflow_data: dict[str, Any],
) -> list[list[str]]:
    """Return validated command steps for a workflow."""
    raw_commands = workflow_data.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        raise typer.BadParameter(
            f"Workflow '{workflow_name}' must define a non-empty commands list."
        )

    return [
        _normalize_step(step_value, workflow_name, index)
        for index, step_value in enumerate(raw_commands, start=1)
    ]


def _substitute_config_path(
    steps: list[list[str]],
    config_path: Path | None,
) -> list[list[str]]:
    """Replace {config_path} placeholders in workflow args."""
    needs_config = any("{config_path}" in arg for step in steps for arg in step)
    if needs_config and config_path is None:
        raise typer.BadParameter(
            "Workflow uses {config_path} but no config path is available. "
            "Pass --config-path or set a cached/default config path."
        )

    if config_path is None:
        return steps

    rendered: list[list[str]] = []
    config_text = str(config_path)
    for step in steps:
        rendered.append([arg.replace("{config_path}", config_text) for arg in step])
    return rendered


def _select_workflow(
    workflows: dict[str, dict[str, Any]],
) -> str | None:
    """Prompt user to select a workflow by number or exit."""
    names = list(workflows.keys())

    typer.echo("Available workflows:")
    for index, name in enumerate(names, start=1):
        description = workflows[name].get("description", "")
        suffix = f" - {description}" if description else ""
        typer.echo(f"  {index}. {name}{suffix}")
    typer.echo(f"  {len(names) + 1}. exit")

    selection = typer.prompt("Select workflow number", type=int)
    if selection == len(names) + 1:
        return None
    if selection < 1 or selection > len(names):
        raise typer.BadParameter(
            f"Selection {selection} is out of range 1..{len(names) + 1}."
        )
    return names[selection - 1]


def _run_selected_workflow(
    selected_name: str,
    workflows: dict[str, dict[str, Any]],
    resolved_config_path: Path | None,
    continue_on_error: bool,
) -> int:
    """Run one selected workflow and return 0 or a failing exit code."""
    steps = _steps_for_workflow(selected_name, workflows[selected_name])
    steps = _substitute_config_path(steps, resolved_config_path)

    had_failure = False
    for index, args in enumerate(steps, start=1):
        command = [sys.executable, "-m", "sdcard.main", *args]
        typer.echo(f"Step {index}/{len(steps)}: {shlex.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            had_failure = True
            typer.echo(f"Step {index} failed with exit code {result.returncode}")
            if not continue_on_error:
                return result.returncode

    return 1 if had_failure else 0


def workflow(
    file: Path = typer.Option(
        Path("workflows.yml"),
        "--file",
        help="Path to workflow YAML file.",
    ),
    list_workflows: bool = typer.Option(
        False,
        "--list",
        help="List workflows and exit.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Workflow name to run.",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        help="Path to config file for {config_path} substitution.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="Continue running later steps after a failure.",
    ),
) -> None:
    """Run reusable command workflows loaded from YAML."""
    workflows = _load_workflows(file)
    workflow_names = list(workflows.keys())

    if not workflow_names:
        typer.echo(f"No workflows found in {file}")
        raise typer.Exit(code=1)

    if list_workflows:
        for workflow_name in workflow_names:
            description = workflows[workflow_name].get("description", "")
            if description:
                typer.echo(f"{workflow_name}: {description}")
            else:
                typer.echo(workflow_name)
        return

    resolved_config_path = resolve_config_path(config_path)
    if name is not None:
        if name not in workflows:
            typer.echo(f"Unknown workflow: {name}")
            raise typer.Exit(code=1)
        exit_code = _run_selected_workflow(
            name,
            workflows,
            resolved_config_path,
            continue_on_error,
        )
        if exit_code != 0:
            raise typer.Exit(code=exit_code)
        return

    while True:
        selected_name = _select_workflow(workflows)
        if selected_name is None:
            return
        exit_code = _run_selected_workflow(
            selected_name,
            workflows,
            resolved_config_path,
            continue_on_error,
        )
        if exit_code != 0:
            typer.echo(f"Workflow '{selected_name}' finished with exit code {exit_code}")
