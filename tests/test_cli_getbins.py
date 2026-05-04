from pathlib import Path

from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import cli_getbins


runner = CliRunner()


def test_getbins_dry_run_prints_target(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("project_name: test\n", encoding="utf-8")

    monkeypatch.setattr(cli_getbins.platform, "system", lambda: "Windows")

    result = runner.invoke(sdcard, ["getbins", "--config-path", str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert "Target bin directory" in result.output
    assert not (tmp_path / "bin").exists()


def test_getbins_installs_binaries(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("project_name: test\n", encoding="utf-8")

    monkeypatch.setattr(cli_getbins.platform, "system", lambda: "Windows")

    def fake_install_rclone(bin_dir: Path, url: str, force: bool) -> str:
        (bin_dir / "rclone.exe").write_bytes(b"rclone")
        return "installed"

    def fake_install_exiftool(bin_dir: Path, url: str, force: bool) -> str:
        (bin_dir / "exiftool.exe").write_bytes(b"exiftool")
        return "installed"

    monkeypatch.setattr(cli_getbins, "_install_rclone", fake_install_rclone)
    monkeypatch.setattr(cli_getbins, "_install_exiftool", fake_install_exiftool)

    result = runner.invoke(sdcard, ["getbins", "--config-path", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "bin" / "rclone.exe").exists()
    assert (tmp_path / "bin" / "exiftool.exe").exists()


def test_getbins_windows_only(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("project_name: test\n", encoding="utf-8")

    monkeypatch.setattr(cli_getbins.platform, "system", lambda: "Linux")

    result = runner.invoke(sdcard, ["getbins", "--config-path", str(config_path)])

    assert result.exit_code != 0
    assert "Windows only" in result.output


def test_getbins_surfaces_install_error_message(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("project_name: test\n", encoding="utf-8")

    monkeypatch.setattr(cli_getbins.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cli_getbins, "_install_rclone", lambda *args, **kwargs: "installed")

    def failing_install_exiftool(*_args, **_kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(cli_getbins, "_install_exiftool", failing_install_exiftool)

    result = runner.invoke(sdcard, ["getbins", "--config-path", str(config_path)])

    assert result.exit_code != 0
    assert "getbins failed: download failed" in result.output


def test_getbins_reuses_cached_config_path(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("SDCARD_STATE_PATH", str(state_path))

    config_path = tmp_path / "config.yml"
    config_path.write_text("project_name: test\n", encoding="utf-8")

    monkeypatch.setattr(cli_getbins.platform, "system", lambda: "Windows")

    def fake_install_rclone(bin_dir: Path, url: str, force: bool) -> str:
        (bin_dir / "rclone.exe").write_bytes(b"rclone")
        return "installed"

    def fake_install_exiftool(bin_dir: Path, url: str, force: bool) -> str:
        (bin_dir / "exiftool.exe").write_bytes(b"exiftool")
        return "installed"

    monkeypatch.setattr(cli_getbins, "_install_rclone", fake_install_rclone)
    monkeypatch.setattr(cli_getbins, "_install_exiftool", fake_install_exiftool)

    first = runner.invoke(sdcard, ["getbins", "--config-path", str(config_path)])
    second = runner.invoke(sdcard, ["getbins"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert (tmp_path / "bin" / "rclone.exe").exists()
