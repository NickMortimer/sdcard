import os
from pathlib import Path
from types import SimpleNamespace

from sdcard.utils.import_transfer import (
    _build_rclone_command,
    _build_rsync_command,
    _delete_source_files_matching_destination,
    _resolve_transfer_tool_path,
    _write_file_times_manifest,
)


def _decompress_zstd_bytes(payload: bytes) -> bytes:
    try:
        import compression.zstd as std_zstd
        return std_zstd.decompress(payload)
    except ImportError:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(payload)


def test_write_file_times_manifest_creates_compressed_manifest(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.txt").write_text("a", encoding="utf-8")
    nested = source_root / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")

    output_path = tmp_path / "manifest" / "file_times.txt.zst"
    _write_file_times_manifest(source_root, output_path)

    assert output_path.exists()

    content = _decompress_zstd_bytes(output_path.read_bytes()).decode("utf-8")
    assert str(source_root / "a.txt") in content
    assert str(nested / "b.txt") in content


def test_write_file_times_manifest_handles_empty_tree(tmp_path) -> None:
    source_root = tmp_path / "empty"
    source_root.mkdir()

    output_path = tmp_path / "manifest" / "file_times.txt.zst"
    _write_file_times_manifest(source_root, output_path)

    assert output_path.exists()
    content = _decompress_zstd_bytes(output_path.read_bytes()).decode("utf-8")
    assert content == "\n"


def test_build_rsync_command_uses_no_overwrite_flags(tmp_path) -> None:
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    source.mkdir()
    destination.mkdir()

    command = _build_rsync_command(
        rsync_path="rsync",
        source=source,
        destination=destination,
        clean=True,
    )

    assert "--update" in command
    assert "--ignore-existing" in command
    assert "--remove-source-files" in command


def test_build_rsync_command_update_allows_newer_overwrite(tmp_path) -> None:
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    source.mkdir()
    destination.mkdir()

    command = _build_rsync_command(
        rsync_path="rsync",
        source=source,
        destination=destination,
        update=True,
    )

    assert "--update" in command
    assert "--ignore-existing" not in command


def test_build_rclone_command_uses_no_overwrite_flags(tmp_path) -> None:
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    source.mkdir()
    destination.mkdir()

    command = _build_rclone_command(
        rclone_path="rclone",
        source=source,
        destination=destination,
        clean=True,
    )

    assert command[1] == "move"
    assert "--update" in command
    assert "--ignore-existing" in command


def test_resolve_transfer_tool_path_windows_uses_rclone(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(catalog_dir=tmp_path)

    monkeypatch.setattr("sdcard.utils.import_transfer.platform.system", lambda: "Windows")
    monkeypatch.setattr("sdcard.utils.import_transfer.shutil.which", lambda name: "C:/Tools/rclone.exe" if name == "rclone" else None)

    resolved = _resolve_transfer_tool_path(config)

    assert resolved == "C:/Tools/rclone.exe"


def test_resolve_transfer_tool_path_linux_uses_rsync(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(catalog_dir=tmp_path)

    monkeypatch.setattr("sdcard.utils.import_transfer.platform.system", lambda: "Linux")
    monkeypatch.setattr("sdcard.utils.import_transfer.shutil.which", lambda name: "/usr/bin/rsync" if name == "rsync" else None)

    resolved = _resolve_transfer_tool_path(config)

    assert resolved == "/usr/bin/rsync"


def test_delete_source_files_matching_destination_by_mtime_and_size(tmp_path) -> None:
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    source.mkdir()
    destination.mkdir()

    matching_src = source / "matching.txt"
    matching_dst = destination / "matching.txt"
    matching_src.write_text("same", encoding="utf-8")
    matching_dst.write_text("same", encoding="utf-8")
    match_mtime = 1_700_000_000
    os.utime(matching_src, (match_mtime, match_mtime))
    os.utime(matching_dst, (match_mtime, match_mtime))

    newer_src = source / "newer_source.txt"
    newer_dst = destination / "newer_source.txt"
    newer_src.write_text("same", encoding="utf-8")
    newer_dst.write_text("same", encoding="utf-8")
    os.utime(newer_src, (match_mtime + 10, match_mtime + 10))
    os.utime(newer_dst, (match_mtime, match_mtime))

    different_size_src = source / "different_size.txt"
    different_size_dst = destination / "different_size.txt"
    different_size_src.write_text("small", encoding="utf-8")
    different_size_dst.write_text("bigger-file", encoding="utf-8")

    deleted_count = _delete_source_files_matching_destination(source, destination)

    assert deleted_count == 1
    assert not matching_src.exists()
    assert newer_src.exists()
    assert different_size_src.exists()