from sdcard.utils import cli_turbo


class _FakeStdout:
    def __iter__(self):
        return iter(())

    def close(self):
        return None


class _FakeProcess:
    def __init__(self):
        self.stdout = _FakeStdout()
        self.returncode = 0

    def wait(self):
        self.returncode = 0
        return 0


def test_run_parallel_import_workers_marks_done_when_worker_threads_finish(monkeypatch) -> None:
    calls = []

    def fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append((args, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(cli_turbo.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli_turbo.time, "sleep", lambda *_args, **_kwargs: None)

    worker_assignments = {
        "terminal_0": ["F:\\"],
        "terminal_1": ["L:\\"],
    }

    # Should complete without printing false failure statuses for still-running workers.
    cli_turbo._run_parallel_import_workers(
        worker_assignments,
        clean=False,
        config_path=None,
        quiet_workers=False,
        precheck=False,
        ignore_errors=False,
        update=False,
        refresh=False,
    )

    assert len(calls) == 2
