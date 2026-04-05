from sdcard.utils.import_conflicts import _collect_destination_conflicts


def test_collect_destination_conflicts_includes_import_yml(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    (source / "import.yml").write_text("card_number: 1\n", encoding="utf-8")
    (destination / "import.yml").write_text("card_number: 200\n", encoding="utf-8")

    conflicts = _collect_destination_conflicts(source, destination)

    assert "import.yml" in conflicts["overlapping_files"]
    assert "import.yml" in conflicts["differing_files"]
