import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import cli_thumbnail, cli_xif


runner = CliRunner()


def _write_fake_raw_with_preview(path, preview_bytes: bytes, offset: int = 64) -> None:
    """Write a file containing arbitrary bytes before an embedded preview."""
    path.write_bytes(b"0" * offset + preview_bytes + b"tail")


def _write_extracted_metadata(path, payload: dict[str, dict[str, object]]) -> None:
    """Write extracted EXIF metadata using the xif zstd JSON format."""
    path.write_bytes(cli_xif._compress_json_zstd(payload))


def test_thumbnail_writes_files_into_sibling_directory(tmp_path) -> None:
    raw_path = tmp_path / "photo.nef"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    _write_fake_raw_with_preview(raw_path, thumbnail_bytes)
    _write_extracted_metadata(
        tmp_path / "exif.json.zst",
        {
            "photo.nef": {
                "SourceFile": str(raw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            }
        },
    )

    result = runner.invoke(sdcard, ["thumbnail", str(tmp_path), "--workers", "1"])

    assert result.exit_code == 0
    output_path = tmp_path / "thumbnails" / "photo.thumbnail.jpg"
    assert output_path.read_bytes() == thumbnail_bytes
    assert "wrote 1" in result.output


def test_thumbnail_skips_directory_without_extracted_metadata(tmp_path) -> None:
    raw_path = tmp_path / "photo.nef"
    _write_fake_raw_with_preview(raw_path, b"\xff\xd8\xff\xe0thumb-data\xff\xd9")

    result = runner.invoke(sdcard, ["thumbnail", str(tmp_path), "--workers", "1"])

    assert result.exit_code == 0
    assert "no-metadata-dirs 1" in result.output
    assert not (tmp_path / "thumbnails" / "photo.thumbnail.jpg").exists()


def test_thumbnail_skips_existing_output(tmp_path) -> None:
    raw_path = tmp_path / "photo.nef"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    _write_fake_raw_with_preview(raw_path, thumbnail_bytes)
    _write_extracted_metadata(
        tmp_path / "exif.json.zst",
        {
            "photo.nef": {
                "SourceFile": str(raw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            }
        },
    )
    output_path = tmp_path / "thumbnails" / "photo.thumbnail.jpg"
    output_path.parent.mkdir()
    output_path.write_bytes(b"existing")

    result = runner.invoke(sdcard, ["thumbnail", str(tmp_path), "--workers", "1"])

    assert result.exit_code == 0
    assert output_path.read_bytes() == b"existing"
    assert "skipped 1" in result.output


def test_thumbnail_copy_meta_uses_extracted_json_import(tmp_path, monkeypatch) -> None:
    raw_path = tmp_path / "photo.nef"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    _write_fake_raw_with_preview(raw_path, thumbnail_bytes)
    _write_extracted_metadata(
        tmp_path / "exif.json.zst",
        {
            "photo.nef": {
                "SourceFile": str(raw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
                "Make": "CameraCo",
                "Model": "FieldCam",
                "DateTimeOriginal": "2026:04:06 10:11:12",
            }
        },
    )

    commands: list[list[str]] = []
    imported_payloads: list[list[dict[str, object]]] = []

    def fake_run(command, check, capture_output, text):
        commands.append(command)
        json_path = command[3].split("=", 1)[1]
        imported_payloads.append(json.loads(Path(json_path).read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_thumbnail.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_thumbnail.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        ["thumbnail", str(tmp_path), "--workers", "1", "--copy-meta"],
    )

    assert result.exit_code == 0
    assert commands[0][0:3] == ["exiftool", "-overwrite_original", "-m"]
    imported = imported_payloads[0][0]
    assert imported["Make"] == "CameraCo"
    assert imported["Model"] == "FieldCam"
    assert imported["DateTimeOriginal"] == "2026:04:06 10:11:12"
    assert imported["SourceFile"].endswith("thumbnails/photo.thumbnail.jpg")
    assert "PreviewImageStart" not in imported


def test_thumbnail_ignores_sibling_output_directory_on_rerun(tmp_path) -> None:
    raw_path = tmp_path / "photo.nef"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    _write_fake_raw_with_preview(raw_path, thumbnail_bytes)
    _write_extracted_metadata(
        tmp_path / "exif.json.zst",
        {
            "photo.nef": {
                "SourceFile": str(raw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            }
        },
    )

    first_result = runner.invoke(sdcard, ["thumbnail", str(tmp_path), "--workers", "1"])
    second_result = runner.invoke(sdcard, ["thumbnail", str(tmp_path), "--workers", "1"])

    assert first_result.exit_code == 0
    assert second_result.exit_code == 0
    assert "no-metadata-dirs 0" in second_result.output
    assert "skipped 1" in second_result.output


def test_thumbnail_command_is_listed_in_help() -> None:
    result = runner.invoke(sdcard, ["--help"])

    assert result.exit_code == 0
    assert "thumbnail" in result.output


def test_thumbnail_help_lists_output_dir_name_option() -> None:
    result = runner.invoke(sdcard, ["thumbnail", "--help"])

    assert result.exit_code == 0
    assert "--output-dir-name" in result.output


def test_thumbnail_filters_by_requested_extension(tmp_path) -> None:
    arw_path = tmp_path / "a.arw"
    nef_path = tmp_path / "b.nef"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    _write_fake_raw_with_preview(arw_path, thumbnail_bytes)
    _write_fake_raw_with_preview(nef_path, thumbnail_bytes)
    _write_extracted_metadata(
        tmp_path / "exif.json.zst",
        {
            "a.arw": {
                "SourceFile": str(arw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            },
            "b.nef": {
                "SourceFile": str(nef_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            },
        },
    )

    result = runner.invoke(
        sdcard,
        ["thumbnail", str(tmp_path), "--workers", "1", "--ext", "arw"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "thumbnails" / "a.thumbnail.jpg").exists()
    assert not (tmp_path / "thumbnails" / "b.thumbnail.jpg").exists()


def test_thumbnail_mirrors_directory_tree_into_output_root(tmp_path) -> None:
    nested_dir = tmp_path / "cards" / "trip1"
    nested_dir.mkdir(parents=True)
    raw_path = nested_dir / "photo.arw"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    output_root = tmp_path / "thumb-root"
    _write_fake_raw_with_preview(raw_path, thumbnail_bytes)
    _write_extracted_metadata(
        nested_dir / "exif.json.zst",
        {
            "photo.arw": {
                "SourceFile": str(raw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            }
        },
    )

    result = runner.invoke(
        sdcard,
        [
            "thumbnail",
            str(tmp_path / "cards"),
            "--workers",
            "1",
            "--output-dir-name",
            str(output_root),
        ],
    )

    assert result.exit_code == 0
    output_path = output_root / "trip1" / "photo.thumbnail.jpg"
    assert output_path.read_bytes() == thumbnail_bytes


def test_thumbnail_mirrored_output_root_is_ignored_on_rerun(tmp_path) -> None:
    raw_root = tmp_path / "cards"
    raw_root.mkdir()
    raw_path = raw_root / "photo.arw"
    thumbnail_bytes = b"\xff\xd8\xff\xe0thumb-data\xff\xd9"
    output_root = raw_root / "derived" / "thumbs"
    _write_fake_raw_with_preview(raw_path, thumbnail_bytes)
    _write_extracted_metadata(
        raw_root / "exif.json.zst",
        {
            "photo.arw": {
                "SourceFile": str(raw_path),
                "PreviewImageStart": 64,
                "PreviewImageLength": len(thumbnail_bytes),
            }
        },
    )

    first_result = runner.invoke(
        sdcard,
        [
            "thumbnail",
            str(raw_root),
            "--workers",
            "1",
            "--output-dir-name",
            str(output_root),
        ],
    )
    second_result = runner.invoke(
        sdcard,
        [
            "thumbnail",
            str(raw_root),
            "--workers",
            "1",
            "--output-dir-name",
            str(output_root),
        ],
    )

    assert first_result.exit_code == 0
    assert second_result.exit_code == 0
    assert "skipped 1" in second_result.output