from sdcard.import_all import _parse_transfer_progress


def test_parse_transfer_progress_rsync_line() -> None:
    line = "123,456,789  53%   12.34MB/s    0:01:23"
    parsed = _parse_transfer_progress(line)
    assert parsed == (53, "12.34MB/s")


def test_parse_transfer_progress_rclone_line() -> None:
    line = "Transferred:   1.234 GiB / 2.000 GiB, 61%, 12.3 MiB/s, ETA 52s"
    parsed = _parse_transfer_progress(line)
    assert parsed == (61, "12.3 MiB/s")


def test_parse_transfer_progress_rclone_ansi_line() -> None:
    line = "\x1b[2KTransferred:   2.000 GiB / 2.000 GiB, 100%, 10.0 MiB/s, ETA 0s\r"
    parsed = _parse_transfer_progress(line)
    assert parsed == (100, "10.0 MiB/s")


def test_parse_transfer_progress_non_progress_line() -> None:
    assert _parse_transfer_progress("worker started") is None
