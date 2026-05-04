from sdcard.utils.cli_turbo import _parse_transfer_progress_line


def test_parse_transfer_progress_line_rsync() -> None:
    line = "123,456,789  53%   12.34MB/s    0:01:23"
    parsed = _parse_transfer_progress_line(line)
    assert parsed == (53, "12.34MB/s")


def test_parse_transfer_progress_line_rclone() -> None:
    line = "Transferred:   1.234 GiB / 2.000 GiB, 61%, 12.3 MiB/s, ETA 52s"
    parsed = _parse_transfer_progress_line(line)
    assert parsed == (61, "12.3 MiB/s")


def test_parse_transfer_progress_line_rclone_ansi() -> None:
    line = "\x1b[2KTransferred:   2.000 GiB / 2.000 GiB, 100%, 10.0 MiB/s, ETA 0s\r"
    parsed = _parse_transfer_progress_line(line)
    assert parsed == (100, "10.0 MiB/s")


def test_parse_transfer_progress_line_non_progress() -> None:
    assert _parse_transfer_progress_line("worker started") is None
