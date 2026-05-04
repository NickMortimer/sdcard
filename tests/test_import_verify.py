from sdcard.utils.import_transfer import (
    _build_transfer_candidate_manifest,
    _verify_destination_contains_manifest,
)


def test_build_transfer_candidate_manifest_excludes_import_yml(tmp_path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "import.yml").write_text("card_number: 1\n", encoding="utf-8")
    (source / "video.mp4").write_text("video", encoding="utf-8")

    manifest = _build_transfer_candidate_manifest(source)

    assert "import.yml" not in manifest
    assert "video.mp4" in manifest


def test_verify_destination_contains_manifest_reports_missing_files(tmp_path) -> None:
    destination = tmp_path / "dst"
    destination.mkdir()

    manifest = {
        "video.mp4": {"size": 10, "mod_time": 0.0},
        "nested/photo.jpg": {"size": 20, "mod_time": 0.0},
    }
    (destination / "video.mp4").write_text("x", encoding="utf-8")

    result = _verify_destination_contains_manifest(destination, manifest)

    assert result["expected_count"] == 2
    assert result["missing_files"] == ["nested/photo.jpg"]


def test_verify_destination_contains_manifest_passes_when_all_present(tmp_path) -> None:
    destination = tmp_path / "dst"
    destination.mkdir()
    nested = destination / "nested"
    nested.mkdir()
    (destination / "video.mp4").write_text("x", encoding="utf-8")
    (nested / "photo.jpg").write_text("x", encoding="utf-8")

    manifest = {
        "video.mp4": {"size": 10, "mod_time": 0.0},
        "nested/photo.jpg": {"size": 20, "mod_time": 0.0},
    }

    result = _verify_destination_contains_manifest(destination, manifest)

    assert result["expected_count"] == 2
    assert result["missing_files"] == []
