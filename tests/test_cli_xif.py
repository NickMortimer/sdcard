import json
import subprocess

from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import cli_xif


runner = CliRunner()


def _decompress_json_zstd(path) -> dict:
    payload = path.read_bytes()

    try:
        import compression.zstd as std_zstd

        raw = std_zstd.decompress(payload)
    except ImportError:
        import zstandard

        raw = zstandard.ZstdDecompressor().decompress(payload)

    return json.loads(raw.decode("utf-8"))


def _make_fake_image(path) -> None:
    """Create a file with an image-like suffix for discovery tests."""
    path.write_bytes(b"fake")


def test_xif_writes_json_for_each_image_directory(tmp_path, monkeypatch) -> None:
    root_image = tmp_path / "root.jpg"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_image = nested_dir / "nested.jpg"

    _make_fake_image(root_image)
    _make_fake_image(nested_image)

    def fake_run(command, check, capture_output, text):
        file_names = [f for f in command[2:]]
        payload = [
            {
                "SourceFile": image_path,
                "Make": "RootCam" if image_path.endswith("root.jpg") else "NestedCam",
                "Model": "R1" if image_path.endswith("root.jpg") else "N1",
            }
            for image_path in file_names
        ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path)])

    assert result.exit_code == 0

    root_output = _decompress_json_zstd(tmp_path / "exif.json.zst")
    nested_output = _decompress_json_zstd(nested_dir / "exif.json.zst")

    assert root_output["root.jpg"]["Make"] == "RootCam"
    assert root_output["root.jpg"]["Model"] == "R1"
    assert nested_output["nested.jpg"]["Make"] == "NestedCam"
    assert nested_output["nested.jpg"]["Model"] == "N1"


def test_xif_skips_directory_when_output_exists(tmp_path, monkeypatch) -> None:
    skipped_dir = tmp_path / "skipped"
    fresh_dir = tmp_path / "fresh"
    skipped_dir.mkdir()
    fresh_dir.mkdir()

    _make_fake_image(skipped_dir / "skip.jpg")
    _make_fake_image(fresh_dir / "fresh.jpg")

    def fake_run(command, check, capture_output, text):
        payload = [{"SourceFile": command[2], "Make": "FreshCam", "Model": "F1"}]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    existing_output = skipped_dir / "exif.json.zst"
    existing_output.write_text('{"status": "existing"}', encoding="utf-8")

    result = runner.invoke(sdcard, ["xif", str(tmp_path)])

    assert result.exit_code == 0
    assert json.loads(existing_output.read_text(encoding="utf-8")) == {
        "status": "existing"
    }

    fresh_output = _decompress_json_zstd(fresh_dir / "exif.json.zst")
    assert fresh_output["fresh.jpg"]["Make"] == "FreshCam"


def test_xif_requires_exiftool(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: None)

    result = runner.invoke(sdcard, ["xif", str(tmp_path)])

    assert result.exit_code != 0
    assert "exiftool is required" in result.output


def test_xif_uses_batch_size_switch(tmp_path, monkeypatch) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_fake_image(photos_dir / "a.jpg")
    _make_fake_image(photos_dir / "b.jpg")
    _make_fake_image(photos_dir / "c.jpg")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        payload = [{"SourceFile": image_path, "Model": "BatchCam"} for image_path in command[2:]]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path), "--batch-size", "2"])

    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0][0:2] == ["exiftool", "-j"]
    assert len(calls[0][2:]) == 2
    assert len(calls[1][2:]) == 1


def test_xif_writes_good_files_when_batch_has_bad_file(tmp_path, monkeypatch) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    good_a = photos_dir / "a.jpg"
    bad_b = photos_dir / "b.jpg"
    good_c = photos_dir / "c.jpg"
    _make_fake_image(good_a)
    _make_fake_image(bad_b)
    _make_fake_image(good_c)

    def fake_run(command, check, capture_output, text):
        files = command[2:]
        if len(files) > 1 and any(path.endswith("b.jpg") for path in files):
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad file")
        if files[0].endswith("b.jpg"):
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad file")
        payload = [{"SourceFile": files[0], "Model": "OK"}]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path), "--batch-size", "3", "--workers", "1"])

    assert result.exit_code == 0
    output = _decompress_json_zstd(photos_dir / "exif.json.zst")
    assert "a.jpg" in output
    assert "c.jpg" in output
    assert "b.jpg" not in output
    assert "failed-files 1" in result.output


def test_batch_failed_status_style_is_red() -> None:
    assert cli_xif._status_style("batch-failed") == "red"


def test_xif_retries_with_smaller_split_batches(tmp_path, monkeypatch) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]:
        _make_fake_image(photos_dir / name)

    batch_sizes: list[int] = []

    def fake_run(command, check, capture_output, text):
        files = command[2:]
        batch_sizes.append(len(files))

        # Force first full batch to fail so logic must split.
        if len(files) == 4:
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="split")

        payload = [{"SourceFile": image_path, "Model": "OK"} for image_path in files]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path), "--batch-size", "4", "--workers", "1"])

    assert result.exit_code == 0
    assert batch_sizes[0] == 4
    assert 2 in batch_sizes