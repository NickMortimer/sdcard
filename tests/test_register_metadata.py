from datetime import datetime
import uuid

import yaml

from sdcard.config import Config
from sdcard.utils import import_metadata


def _build_config(tmp_path, import_path_template: str) -> Config:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                f"card_store: {tmp_path / 'store'}",
                f'import_path_template: "{import_path_template}"',
                "instrument: camera_a",
                "project_name: Survey Alpha",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return Config(config_path)


def _build_mapping_config(tmp_path) -> Config:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                f"card_store: {tmp_path / 'store'}",
                'destination_path: "{CATALOG_DIR}/raw/{import_date}/{instrument}_{partition_label}_{import_token}"',
                "instrument:",
                '  LR1: "Sony LR1 Camera"',
                '  GHADRON: "Gremsy Hadron 640R"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return Config(config_path)


def _build_multi_option_config(tmp_path) -> Config:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                f"card_store: {tmp_path / 'store'}",
                'destination_path: "{CATALOG_DIR}/raw/{import_date}/{instrument}_{partition_label}_{import_token}"',
                "project_name: Survey Alpha",
                "instrument:",
                '  LR1: "Sony LR1 Camera"',
                '  GHADRON: "Gremsy Hadron 640R"',
                "platform:",
                '  DODO: "DODO vessel"',
                '  SHORE: "Shore station"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return Config(config_path)


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


def test_make_yml_writes_raw_destination_from_config(tmp_path, monkeypatch, capsys) -> None:
    config = _build_config(
        tmp_path,
        "{{card_store}}/{instrument}/{card_number}_{import_token}",
    )
    card_path = tmp_path / "card-a"
    card_path.mkdir()
    _patch_registration_dependencies(monkeypatch)

    result = import_metadata.make_yml(
        card_path / "import.yml",
        config,
        dry_run=False,
        card_number="7",
    )

    output = capsys.readouterr().out
    written = yaml.safe_load((card_path / "import.yml").read_text(encoding="utf-8"))
    expected_destination = "{{card_store}}/{instrument}/{card_number}_{import_token}"

    assert result is not None
    assert written["destination_path"] == expected_destination
    assert result["destination_path"] == expected_destination
    assert "Instrument: camera_a" in output
    assert f"Destination: {expected_destination}" in output


def test_make_yml_prompts_for_multi_option_instrument(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = _build_mapping_config(tmp_path)
    card_path = tmp_path / "card-b"
    card_path.mkdir()
    _patch_registration_dependencies(monkeypatch)
    prompts: list[str] = []

    def _select_first_option(message: str, **kwargs):
        prompts.append(message)
        return 1

    monkeypatch.setattr(import_metadata.typer, "prompt", _select_first_option)

    result = import_metadata.make_yml(
        card_path / "import.yml",
        config,
        dry_run=False,
        card_number="9",
    )

    output = capsys.readouterr().out
    written = yaml.safe_load((card_path / "import.yml").read_text(encoding="utf-8"))
    expected_destination = "{CATALOG_DIR}/raw/{import_date}/{instrument}_{partition_label}_{import_token}"

    assert result is not None
    assert prompts == ["instrument number"]
    assert written["instrument"] == "LR1"
    assert written["destination_path"] == expected_destination
    assert result["destination_path"] == expected_destination
    assert f"✏️  Wrote {card_path / 'import.yml'}" in output
    assert "Instrument: LR1" in output
    assert f"Destination: {expected_destination}" in output


def test_make_yml_prompts_for_other_multi_option_config_values(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = _build_multi_option_config(tmp_path)
    card_path = tmp_path / "card-c"
    card_path.mkdir()
    _patch_registration_dependencies(monkeypatch)
    prompts: list[str] = []
    selections = iter([2, 1])

    def _select_options(message: str, **kwargs):
        prompts.append(message)
        return next(selections)

    monkeypatch.setattr(import_metadata.typer, "prompt", _select_options)

    result = import_metadata.make_yml(
        card_path / "import.yml",
        config,
        dry_run=False,
        card_number="11",
    )

    output = capsys.readouterr().out
    written = yaml.safe_load((card_path / "import.yml").read_text(encoding="utf-8"))

    assert result is not None
    assert prompts == ["platform number", "instrument number"]
    assert written["platform"] == "SHORE"
    assert written["instrument"] == "LR1"
    assert "Instrument: LR1" in output
    assert "platform: SHORE" in output


def test_make_yml_refresh_preserves_card_number_and_instrument(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = _build_multi_option_config(tmp_path)
    card_path = tmp_path / "card-d"
    card_path.mkdir()
    _patch_registration_dependencies(monkeypatch)
    import_yml = card_path / "import.yml"
    import_yml.write_text(
        yaml.safe_dump(
            {
                "card_number": "42",
                "instrument": "GHADRON",
                "register_date": "2026-03-01",
                "import_date": "2026-03-15",
                "import_token": "oldtoken",
                "platform": "DODO",
                "project_name": "Old Name",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    prompts: list[str] = []

    def _select_platform(message: str, **kwargs):
        prompts.append(message)
        return 2

    class _FrozenDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 7)

    monkeypatch.setattr(import_metadata, "datetime", _FrozenDateTime)
    monkeypatch.setattr(import_metadata.typer, "prompt", _select_platform)

    result = import_metadata.make_yml(
        import_yml,
        config,
        dry_run=False,
        card_number=0,
        overwrite=True,
        refresh=True,
    )

    output = capsys.readouterr().out
    written = yaml.safe_load(import_yml.read_text(encoding="utf-8"))

    assert result is not None
    assert prompts == ["platform number"]
    assert written["card_number"] == "42"
    assert written["instrument"] == "GHADRON"
    assert written["register_date"] == "2026-04-07"
    assert written["import_token"] == "12345678"
    assert written["platform"] == "SHORE"
    assert written["project_name"] == "Survey Alpha"
    assert written["destination_path"] == "{CATALOG_DIR}/raw/{import_date}/{instrument}_{partition_label}_{import_token}"
    assert "import_date" not in written
    assert "Instrument: GHADRON" in output
    assert "platform: SHORE" in output
