import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sdcard.main import sdcard
from sdcard.utils import cli_frames
from sdcard.utils.cli_xif import _compress_json_zstd


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_config_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDCARD_STATE_PATH", str(tmp_path / "state.json"))


def test_default_frames_output_root_maps_raw_sdcards(tmp_path: Path) -> None:
    card_store = tmp_path / "data" / "raw" / "sdcards"
    card_store.mkdir(parents=True)
    assert cli_frames._default_frames_output_root(card_store) == (
        tmp_path / "data" / "interim" / "frames"
    )


def test_frames_output_dir_includes_fps(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "sdcards"
    video = source / "2026-04-05" / "card" / "DCIM" / "clip.MP4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake")
    output_root = tmp_path / "interim" / "frames"
    out = cli_frames._frames_output_dir(video, source, output_root, 1.0)
    assert out == output_root / "2026-04-05" / "card" / "DCIM" / "clip.1fps.frames"
    out_half = cli_frames._frames_output_dir(video, source, output_root, 0.5)
    assert out_half == (
        output_root / "2026-04-05" / "card" / "DCIM" / "clip.0.5fps.frames"
    )


def test_merge_identity_tags_cli_overrides_win() -> None:
    discovered = {"Make": "Sony", "Model": "A7", "SerialNumber": "1"}
    merged = cli_frames._merge_identity_tags(
        discovered,
        ["Make=DJI", "Model=FC6310R"],
        None,
    )
    assert merged["Make"] == "DJI"
    assert merged["Model"] == "FC6310R"
    assert merged["SerialNumber"] == "1"


def test_merge_identity_tags_json_and_pairs() -> None:
    merged = cli_frames._merge_identity_tags(
        {},
        ["Model=FromPair"],
        '{"Make":"DJI","Model":"FromJson"}',
    )
    assert merged["Make"] == "DJI"
    assert merged["Model"] == "FromPair"


def test_identity_tags_from_metadata_filters_noise() -> None:
    tags = cli_frames._identity_tags_from_metadata(
        {
            "Make": "DJI",
            "Model": "FC6310R",
            "SourceFile": "/x/clip.mp4",
            "ThumbnailImage": "(Binary data 123 bytes)",
            "SerialNumber": "abc",
        }
    )
    assert tags == {"Make": "DJI", "Model": "FC6310R", "SerialNumber": "abc"}


def test_frame_time_math() -> None:
    start = datetime(2026, 8, 3, 13, 0, 31)
    assert cli_frames._format_exif_datetime(start) == "2026:08:03 13:00:31"
    half = start.replace(microsecond=500_000)
    assert cli_frames._format_exif_datetime(half) == "2026:08:03 13:00:31.5"
    assert cli_frames._subsec_time_tag(half) == "5"
    parsed = cli_frames._parse_exif_datetime("2026:08:03 13:00:31+08:00")
    assert parsed == datetime(2026, 8, 3, 13, 0, 31)
    assert cli_frames._parse_exif_datetime("2026:08:03 13:00:31.500") == half


def test_mirror_serial_tags_copies_both_names() -> None:
    assert cli_frames._mirror_serial_tags({"SerialNumber": "abc"}) == {
        "SerialNumber": "abc",
        "CameraSerialNumber": "abc",
    }
    assert cli_frames._mirror_serial_tags({"CameraSerialNumber": "xyz"}) == {
        "CameraSerialNumber": "xyz",
        "SerialNumber": "xyz",
    }


def test_resolve_identity_tags_samples_sibling_still(tmp_path: Path) -> None:
    video = tmp_path / "clip.MOV"
    video.write_bytes(b"fake")
    sidecar = {
        "clip.MOV": {
            "Make": "DJI",
            "Model": "FC6310R",
            "MediaCreateDate": "2026:08:03 13:00:31",
        },
        "DJI_0001.JPG": {
            "FileType": "JPEG",
            "Make": "DJI",
            "Model": "FC6310R",
            "SerialNumber": "serial-from-still",
        },
    }
    (tmp_path / "exif.json.zst").write_bytes(_compress_json_zstd(sidecar))
    tags = cli_frames._resolve_identity_tags(
        video,
        sidecar["clip.MOV"],
        {},
    )
    assert tags["SerialNumber"] == "serial-from-still"
    assert tags["CameraSerialNumber"] == "serial-from-still"
    assert tags["Make"] == "DJI"
    assert tags["Model"] == "FC6310R"


def test_resolve_identity_tags_samples_still_from_other_dcim_folder(
    tmp_path: Path,
) -> None:
    """Video in 100MEDIA; survey stills in DCIM/SURVEY/... share serial."""
    dcim = tmp_path / "card" / "DCIM"
    media = dcim / "100MEDIA"
    survey = dcim / "SURVEY" / "100_0001"
    media.mkdir(parents=True)
    survey.mkdir(parents=True)
    video = media / "DJI_0008.MOV"
    video.write_bytes(b"fake")
    (media / "exif.json.zst").write_bytes(
        _compress_json_zstd(
            {
                "DJI_0008.MOV": {
                    "FileType": "MOV",
                    "Make": "DJI",
                    "Model": "FC6310R",
                    "MediaCreateDate": "2026:08:03 13:00:31",
                }
            }
        )
    )
    (survey / "exif.json.zst").write_bytes(
        _compress_json_zstd(
            {
                "DJI_0001.JPG": {
                    "FileType": "JPEG",
                    "Make": "DJI",
                    "Model": "FC6310R",
                    "SerialNumber": "serial-from-survey",
                }
            }
        )
    )
    tags = cli_frames._resolve_identity_tags(
        video,
        {
            "Make": "DJI",
            "Model": "FC6310R",
            "MediaCreateDate": "2026:08:03 13:00:31",
        },
        {},
    )
    assert tags["SerialNumber"] == "serial-from-survey"
    assert tags["CameraSerialNumber"] == "serial-from-survey"


def test_frames_cli_extracts_stamps_and_inherits(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "sdcards" / "day" / "card"
    source.mkdir(parents=True)
    video = source / "clip.mp4"
    video.write_bytes(b"fake-video")
    sidecar_payload = {
        "clip.mp4": {
            "SourceFile": str(video),
            "Make": "DJI",
            "Model": "FC6310R",
            "SerialNumber": "serial-1",
            "MediaCreateDate": "2026:08:03 13:00:31",
        }
    }
    (source / "exif.json.zst").write_bytes(_compress_json_zstd(sidecar_payload))

    stamped: list[dict] = []

    def fake_ffmpeg(video_path, output_dir, fps, ffmpeg_executable):
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(1, 3):
            path = output_dir / f"frame_{index:06d}.jpg"
            path.write_bytes(b"jpeg")
            frames.append(path)
        return frames

    def fake_stamp(frame_path, frame_time, identity_tags, exiftool_executable):
        stamped.append(
            {
                "file": frame_path.name,
                "time": cli_frames._format_exif_datetime(frame_time),
                "tags": dict(identity_tags),
            }
        )

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli_frames, "_stamp_frame_exif", fake_stamp)

    result = runner.invoke(
        sdcard,
        [
            "frames",
            str(tmp_path / "raw" / "sdcards"),
            "--output-dir",
            str(tmp_path / "interim" / "frames"),
            "--fps",
            "1",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "extracted 1" in result.output
    assert len(stamped) == 2
    assert stamped[0]["time"] == "2026:08:03 13:00:31"
    assert stamped[1]["time"] == "2026:08:03 13:00:32"
    assert stamped[0]["tags"]["Make"] == "DJI"
    assert stamped[0]["tags"]["Model"] == "FC6310R"
    assert stamped[0]["tags"]["SerialNumber"] == "serial-1"
    assert stamped[0]["tags"]["CameraSerialNumber"] == "serial-1"

    out_dir = tmp_path / "interim" / "frames" / "day" / "card" / "clip.1fps.frames"
    assert (out_dir / "frame_000001.jpg").is_file()
    assert (out_dir / "frame_000002.jpg").is_file()
    manifest = out_dir / "frames.csv"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "frame_number,file,elapsed_s,frame_time" in text
    assert "1,frame_000001.jpg,0,2026-08-03 13:00:31" in text
    assert "2,frame_000002.jpg,1,2026-08-03 13:00:32" in text


def test_frames_cli_stamps_subsecond_at_2fps(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw" / "sdcards" / "day" / "card"
    source.mkdir(parents=True)
    video = source / "clip.mp4"
    video.write_bytes(b"fake-video")
    (source / "exif.json.zst").write_bytes(
        _compress_json_zstd(
            {
                "clip.mp4": {
                    "Make": "DJI",
                    "Model": "FC6310R",
                    "SerialNumber": "s1",
                    "MediaCreateDate": "2026:08:03 13:00:31",
                }
            }
        )
    )
    stamped: list[dict] = []

    def fake_ffmpeg(video_path, output_dir, fps, ffmpeg_executable):
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(1, 3):
            path = output_dir / f"frame_{index:06d}.jpg"
            path.write_bytes(b"jpeg")
            frames.append(path)
        return frames

    def fake_stamp(frame_path, frame_time, identity_tags, exiftool_executable):
        stamped.append(
            {
                "time": cli_frames._format_exif_datetime(frame_time),
                "subsec": cli_frames._subsec_time_tag(frame_time),
            }
        )

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli_frames, "_stamp_frame_exif", fake_stamp)

    result = runner.invoke(
        sdcard,
        [
            "frames",
            str(tmp_path / "raw" / "sdcards"),
            "--output-dir",
            str(tmp_path / "interim" / "frames"),
            "--fps",
            "2",
            "--workers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert stamped[0]["time"] == "2026:08:03 13:00:31"
    assert stamped[0]["subsec"] is None
    assert stamped[1]["time"] == "2026:08:03 13:00:31.5"
    assert stamped[1]["subsec"] == "5"


def test_frames_cli_tag_overrides_win(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    video = source / "clip.mp4"
    video.write_bytes(b"fake")
    (source / "exif.json.zst").write_bytes(
        _compress_json_zstd(
            {
                "clip.mp4": {
                    "Make": "Sony",
                    "Model": "A7",
                    "CreateDate": "2026:01:01 00:00:00",
                }
            }
        )
    )

    stamped: list[dict] = []

    def fake_ffmpeg(video_path, output_dir, fps, ffmpeg_executable):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "frame_000001.jpg"
        path.write_bytes(b"jpeg")
        return [path]

    def fake_stamp(frame_path, frame_time, identity_tags, exiftool_executable):
        stamped.append(dict(identity_tags))

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli_frames, "_stamp_frame_exif", fake_stamp)

    result = runner.invoke(
        sdcard,
        [
            "frames",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--tag",
            "Make=DJI",
            "--tag",
            "Model=FC6310R",
            "--workers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert stamped[0]["Make"] == "DJI"
    assert stamped[0]["Model"] == "FC6310R"


def test_frames_retries_incomplete_dir_without_manifest(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    video = source / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out" / "clip.1fps.frames"
    out.mkdir(parents=True)
    (out / "frame_000001.jpg").write_bytes(b"old")
    # Incomplete prior run: JPEGs present but no frames.csv → must re-extract

    called = {"ffmpeg": 0}

    def fake_ffmpeg(video_path, output_dir, fps, ffmpeg_executable):
        called["ffmpeg"] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "frame_000001.jpg"
        path.write_bytes(b"new")
        return [path]

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli_frames, "_stamp_frame_exif", lambda *_a, **_k: None)

    result = runner.invoke(
        sdcard,
        ["frames", str(source), "--output-dir", str(tmp_path / "out"), "--workers", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "extracted 1" in result.output
    assert called["ffmpeg"] == 1
    assert (out / "frames.csv").is_file()


def test_frames_skips_when_manifest_complete(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    video = source / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out" / "clip.1fps.frames"
    out.mkdir(parents=True)
    (out / "frame_000001.jpg").write_bytes(b"old")
    (out / "frames.csv").write_text(
        "frame_number,file,elapsed_s,frame_time\n"
        "1,frame_000001.jpg,0,2026-08-03 13:00:31\n",
        encoding="utf-8",
    )

    called = {"ffmpeg": 0}

    def fake_ffmpeg(*_args, **_kwargs):
        called["ffmpeg"] += 1
        return []

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)

    result = runner.invoke(
        sdcard,
        ["frames", str(source), "--output-dir", str(tmp_path / "out"), "--workers", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "skipped 1" in result.output
    assert called["ffmpeg"] == 0


def test_frames_clean_removes_existing_directory(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    video = source / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out" / "clip.1fps.frames"
    out.mkdir(parents=True)
    (out / "frame_000001.jpg").write_bytes(b"old")
    (out / "exif.json.zst").write_bytes(b"old-sidecar")

    def fake_ffmpeg(video_path, output_dir, fps, ffmpeg_executable):
        assert not (output_dir / "exif.json.zst").exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "frame_000001.jpg"
        path.write_bytes(b"new")
        return [path]

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli_frames, "_stamp_frame_exif", lambda *_a, **_k: None)

    result = runner.invoke(
        sdcard,
        ["frames", str(source), "--output-dir", str(tmp_path / "out"), "--clean", "--workers", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "extracted 1" in result.output
    assert (out / "frame_000001.jpg").read_bytes() == b"new"
    assert not (out / "exif.json.zst").exists()


def test_frames_parallel_workers_extract_two_videos(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (source / name).write_bytes(b"fake")
    (source / "exif.json.zst").write_bytes(
        _compress_json_zstd(
            {
                "a.mp4": {
                    "Make": "DJI",
                    "Model": "FC6310R",
                    "SerialNumber": "s",
                    "MediaCreateDate": "2026:08:03 13:00:31",
                },
                "b.mp4": {
                    "Make": "DJI",
                    "Model": "FC6310R",
                    "SerialNumber": "s",
                    "MediaCreateDate": "2026:08:03 13:00:31",
                },
            }
        )
    )

    seen: set[str] = set()

    def fake_ffmpeg(video_path, output_dir, fps, ffmpeg_executable):
        seen.add(video_path.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "frame_000001.jpg"
        path.write_bytes(b"jpeg")
        return [path]

    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    monkeypatch.setattr(cli_frames, "_extract_frames_with_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(cli_frames, "_stamp_frame_exif", lambda *_a, **_k: None)

    result = runner.invoke(
        sdcard,
        [
            "frames",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "extracted 2" in result.output
    assert seen == {"a.mp4", "b.mp4"}
    assert (tmp_path / "out" / "a.1fps.frames" / "frames.csv").is_file()
    assert (tmp_path / "out" / "b.1fps.frames" / "frames.csv").is_file()


def test_frames_requires_ffmpeg(tmp_path, monkeypatch) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    monkeypatch.setattr(cli_frames.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli_frames, "_resolve_exiftool_executable", lambda _: "exiftool")
    result = runner.invoke(sdcard, ["frames", str(source)])
    assert result.exit_code != 0
    assert "ffmpeg" in result.output.lower()
