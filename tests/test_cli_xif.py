import json
import subprocess

import pytest
from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import cli_xif


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_config_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDCARD_STATE_PATH", str(tmp_path / "state.json"))


def _exiftool_file_args(command: list[str]) -> list[str]:
    return command[len(cli_xif.EXIFTOOL_BASE_ARGS):]


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
        file_names = _exiftool_file_args(command)
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
        payload = [
            {
                "SourceFile": _exiftool_file_args(command)[0],
                "Make": "FreshCam",
                "Model": "F1",
            }
        ]
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
    assert "getbins" in result.output


def test_xif_uses_batch_size_switch(tmp_path, monkeypatch) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _make_fake_image(photos_dir / "a.jpg")
    _make_fake_image(photos_dir / "b.jpg")
    _make_fake_image(photos_dir / "c.jpg")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        payload = [
            {"SourceFile": image_path, "Model": "BatchCam"}
            for image_path in _exiftool_file_args(command)
        ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path), "--batch-size", "2"])

    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0][0:6] == [
        "exiftool",
        "-api",
        "LargeFileSupport=1",
        "-api",
        "RequestAll=3",
        "-j",
    ]
    assert len(_exiftool_file_args(calls[0])) == 2
    assert len(_exiftool_file_args(calls[1])) == 1


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
        files = _exiftool_file_args(command)
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
        files = _exiftool_file_args(command)
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


def test_xif_discovers_raw_file_extensions(tmp_path, monkeypatch) -> None:
    raw_path = tmp_path / "photo.nef"
    _make_fake_image(raw_path)

    def fake_run(command, check, capture_output, text):
        payload = [{"SourceFile": _exiftool_file_args(command)[0], "Model": "RawCam"}]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path)])

    assert result.exit_code == 0
    output = _decompress_json_zstd(tmp_path / "exif.json.zst")
    assert output["photo.nef"]["Model"] == "RawCam"


def test_xif_filters_by_requested_extension(tmp_path, monkeypatch) -> None:
    arw_path = tmp_path / "a.arw"
    nef_path = tmp_path / "b.nef"
    _make_fake_image(arw_path)
    _make_fake_image(nef_path)

    def fake_run(command, check, capture_output, text):
        payload = [
            {"SourceFile": image_path, "Model": "FilteredCam"}
            for image_path in _exiftool_file_args(command)
        ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path), "--ext", "ARW"])

    assert result.exit_code == 0
    output = _decompress_json_zstd(tmp_path / "exif.json.zst")
    assert "a.arw" in output
    assert "b.nef" not in output


def test_xif_uses_card_store_from_config(tmp_path, monkeypatch) -> None:
    card_store = tmp_path / "card_store"
    card_store.mkdir()
    _make_fake_image(card_store / "img.jpg")
    config_path = tmp_path / "config.yml"
    config_path.write_text(f"card_store: {card_store}\n", encoding="utf-8")

    def fake_run(command, check, capture_output, text):
        payload = [{"SourceFile": _exiftool_file_args(command)[0], "Model": "CfgCam"}]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(
        sdcard,
        ["xif", "--card-store", "--config-path", str(config_path)],
    )

    assert result.exit_code == 0
    output = _decompress_json_zstd(card_store / "exif.json.zst")
    assert output["img.jpg"]["Model"] == "CfgCam"


def test_xif_uses_xif_path_from_config_without_card_store(tmp_path, monkeypatch) -> None:
    xif_path = tmp_path / "xif_root"
    xif_path.mkdir()
    _make_fake_image(xif_path / "img.jpg")
    config_path = tmp_path / "config.yml"
    config_path.write_text(f"xif_path: {xif_path}\n", encoding="utf-8")

    def fake_run(command, check, capture_output, text):
        payload = [{"SourceFile": _exiftool_file_args(command)[0], "Model": "XifPathCam"}]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", "--config-path", str(config_path)])

    assert result.exit_code == 0
    output = _decompress_json_zstd(xif_path / "exif.json.zst")
    assert output["img.jpg"]["Model"] == "XifPathCam"


def test_xif_resolves_catalog_dir_placeholder_in_xif_path(tmp_path, monkeypatch) -> None:
    xif_root = tmp_path / "raw" / "sdcards"
    xif_root.mkdir(parents=True)
    _make_fake_image(xif_root / "img.jpg")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "xif_path: \"{CATALOG_DIR}/raw/sdcards\"\n",
        encoding="utf-8",
    )

    def fake_run(command, check, capture_output, text):
        payload = [{"SourceFile": _exiftool_file_args(command)[0], "Model": "TemplateCam"}]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", "--config-path", str(config_path)])

    assert result.exit_code == 0
    output = _decompress_json_zstd(xif_root / "exif.json.zst")
    assert output["img.jpg"]["Model"] == "TemplateCam"


def test_xif_requires_directory_or_config_or_card_store(monkeypatch) -> None:
    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")

    result = runner.invoke(sdcard, ["xif"])

    assert result.exit_code != 0
    assert "Missing HEAD_DIRECTORY" in result.output


def test_xif_config_path_requires_xif_path_when_no_directory(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("card_store: C:/tmp/card_store\n", encoding="utf-8")
    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")

    result = runner.invoke(sdcard, ["xif", "--config-path", str(config_path)])

    assert result.exit_code != 0
    assert "xif_path" in result.output


def test_xif_rejects_directory_with_card_store_switch(tmp_path, monkeypatch) -> None:
    head_directory = tmp_path / "photos"
    head_directory.mkdir()
    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")

    result = runner.invoke(sdcard, ["xif", str(head_directory), "--card-store"])

    assert result.exit_code != 0
    assert "--card-store" in result.output


def test_xif_discovers_mp4_files(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_fake_image(video_path)

    def fake_run(command, check, capture_output, text):
        payload = [
            {"SourceFile": _exiftool_file_args(command)[0], "Model": "VideoCam"}
        ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(cli_xif.shutil, "which", lambda _: "/usr/bin/exiftool")
    monkeypatch.setattr(cli_xif.subprocess, "run", fake_run)

    result = runner.invoke(sdcard, ["xif", str(tmp_path)])

    assert result.exit_code == 0
    output = _decompress_json_zstd(tmp_path / "exif.json.zst")
    assert output["clip.mp4"]["Model"] == "VideoCam"