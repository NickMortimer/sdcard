import subprocess

import pytest
from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import cli_workflow


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_config_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDCARD_STATE_PATH", str(tmp_path / "state.json"))


@pytest.fixture(autouse=True)
def _fixed_python(monkeypatch) -> None:
    monkeypatch.setattr(cli_workflow.sys, "executable", "C:/mock/python.exe")


def test_workflow_list_prints_names(tmp_path) -> None:
    workflow_file = tmp_path / "workflows.yml"
    workflow_file.write_text(
        "\n".join(
            [
                "workflows:",
                "  quick_xif:",
                "    description: Quick EXIF run",
                "    commands:",
                "      - xif",
                "  setup_tools:",
                "    commands:",
                "      - getbins",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(sdcard, ["workflow", "--file", str(workflow_file), "--list"])

    assert result.exit_code == 0
    assert "quick_xif" in result.output
    assert "setup_tools" in result.output


def test_workflow_name_executes_with_config_placeholder(tmp_path, monkeypatch) -> None:
    workflow_file = tmp_path / "workflows.yml"
    workflow_file.write_text(
        "\n".join(
            [
                "workflows:",
                "  quick_xif:",
                "    commands:",
                "      - xif --config-path {config_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_file = tmp_path / "config.yml"
    config_file.write_text("project_name: test\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(command, check=False):  # noqa: ANN001, ARG001
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_workflow.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        [
            "workflow",
            "--file",
            str(workflow_file),
            "--name",
            "quick_xif",
            "--config-path",
            str(config_file),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][:3] == ["C:/mock/python.exe", "-m", "sdcard.main"]
    assert calls[0][3:] == ["xif", "--config-path", str(config_file.resolve())]


def test_workflow_placeholder_requires_config(tmp_path, monkeypatch) -> None:
    workflow_file = tmp_path / "workflows.yml"
    workflow_file.write_text(
        "\n".join(
            [
                "workflows:",
                "  quick_xif:",
                "    commands:",
                "      - xif --config-path {config_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    called = {"ran": False}

    def fake_run(command, check=False):  # noqa: ANN001, ARG001
        called["ran"] = True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_workflow.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        ["workflow", "--file", str(workflow_file), "--name", "quick_xif"],
    )

    assert result.exit_code != 0
    assert "{config_path}" in result.output
    assert called["ran"] is False


def test_workflow_interactive_selection_runs_selected(tmp_path, monkeypatch) -> None:
    workflow_file = tmp_path / "workflows.yml"
    workflow_file.write_text(
        "\n".join(
            [
                "workflows:",
                "  first:",
                "    commands:",
                "      - list",
                "  second:",
                "    commands:",
                "      - probe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(command, check=False):  # noqa: ANN001, ARG001
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_workflow.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        ["workflow", "--file", str(workflow_file)],
        input="2\n3\n",
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][3:] == ["probe"]


def test_workflow_interactive_mode_repeats_until_exit(tmp_path, monkeypatch) -> None:
    workflow_file = tmp_path / "workflows.yml"
    workflow_file.write_text(
        "\n".join(
            [
                "workflows:",
                "  first:",
                "    commands:",
                "      - list",
                "  second:",
                "    commands:",
                "      - probe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(command, check=False):  # noqa: ANN001, ARG001
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_workflow.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        ["workflow", "--file", str(workflow_file)],
        input="1\n2\n3\n",
    )

    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0][3:] == ["list"]
    assert calls[1][3:] == ["probe"]


def test_workflow_continue_on_error_runs_later_steps(tmp_path, monkeypatch) -> None:
    workflow_file = tmp_path / "workflows.yml"
    workflow_file.write_text(
        "\n".join(
            [
                "workflows:",
                "  mixed:",
                "    commands:",
                "      - list",
                "      - probe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(command, check=False):  # noqa: ANN001, ARG001
        calls.append(list(command))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 2)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_workflow.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        [
            "workflow",
            "--file",
            str(workflow_file),
            "--name",
            "mixed",
            "--continue-on-error",
        ],
    )

    assert result.exit_code != 0
    assert len(calls) == 2
