import os

from typer.testing import CliRunner

from sdcard.main import sdcard


runner = CliRunner()


def test_import_normalizes_windows_drive_roots(monkeypatch) -> None:
    if os.name != "nt":
        return

    captured = {}

    def fake_import_cards(*, config, card_path, **kwargs):  # noqa: ARG001
        captured["card_path"] = list(card_path)

    import sdcard.utils.cli_import as cli_import

    monkeypatch.setattr(cli_import, "import_cards", fake_import_cards)

    result = runner.invoke(sdcard, ["import", "F:", "G:"])

    assert result.exit_code == 0
    assert captured["card_path"] == ["F:\\", "G:\\"]
