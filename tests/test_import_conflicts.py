from sdcard.utils.import_conflicts import _collect_destination_conflicts


def test_collect_destination_conflicts_ignores_internal_metadata_files(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    (source / "import.yml").write_text("card_number: 1\n", encoding="utf-8")
    (destination / "import.yml").write_text("card_number: 200\n", encoding="utf-8")

    (source / "file_times.txt.zst").write_text("source", encoding="utf-8")
    (destination / "file_times.txt.zst").write_text("destination", encoding="utf-8")

    conflicts = _collect_destination_conflicts(source, destination)

    assert "import.yml" not in conflicts["overlapping_files"]
    assert "import.yml" not in conflicts["differing_files"]
    assert "file_times.txt.zst" not in conflicts["overlapping_files"]
    assert "file_times.txt.zst" not in conflicts["differing_files"]
