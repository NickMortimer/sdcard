import uuid

import yaml
from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import import_metadata


runner = CliRunner()


def _patch_registration_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        import_metadata,
        "get_card_media_details",
        lambda _mount_path: {
            "card_size_gb": 128.0,
            "card_format": "exfat",
            "partition_label": "CARD_A",
        },
    )
    monkeypatch.setattr(
        import_metadata.uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )


def test_register_warns_when_using_implicit_default_config(
    tmp_path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "card-a"
    card_path.mkdir()
    monkeypatch.chdir(tmp_path)
    _patch_registration_dependencies(monkeypatch)

    result = runner.invoke(
        sdcard,
        ["register", str(card_path), "--card-number", "7"],
    )

    written = yaml.safe_load((card_path / "import.yml").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert (
        "⚠️  No config file provided; using default parameters for import.yml"
        in result.output
    )
    assert written["card_number"] == "7"
    assert written["destination_path"] == "{{card_store}}/{import_date}/{import_token}"


def test_register_fails_when_explicit_config_path_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "card-b"
    card_path.mkdir()
    monkeypatch.chdir(tmp_path)
    _patch_registration_dependencies(monkeypatch)
    missing_config = tmp_path / "missing.yml"

    result = runner.invoke(
        sdcard,
        [
            "register",
            str(card_path),
            "--card-number",
            "8",
            "--config-path",
            str(missing_config),
        ],
    )

    assert result.exit_code != 0
    assert not (card_path / "import.yml").exists()
    assert "Invalid value for '--config-path'" in result.output
    assert f"Config file not found: {missing_config}" in result.output


def test_register_uses_explicit_config_without_default_warning(
    tmp_path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "card-c"
    card_path.mkdir()
    monkeypatch.chdir(tmp_path)
    _patch_registration_dependencies(monkeypatch)
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                f"card_store: {tmp_path / 'store'}",
                'import_path_template: "custom/{card_number}/{import_token}"',
                "instrument: camera_a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        sdcard,
        [
            "register",
            str(card_path),
            "--card-number",
            "9",
            "--config-path",
            str(config_path),
        ],
    )

    written = yaml.safe_load((card_path / "import.yml").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "No config file provided" not in result.output
    assert written["destination_path"] == "custom/{card_number}/{import_token}"
