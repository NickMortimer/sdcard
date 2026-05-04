import json
from pathlib import Path

from sdcard.utils.config_path_cache import resolve_config_path


def test_resolve_config_path_persists_and_reuses(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("SDCARD_STATE_PATH", str(state_path))

    config_path = tmp_path / "config.yml"
    config_path.write_text("project_name: test\n", encoding="utf-8")

    resolved = resolve_config_path(config_path)
    assert resolved == config_path.resolve()

    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_data["last_config_path"] == str(config_path.resolve())

    reused = resolve_config_path(None)
    assert reused == config_path.resolve()


def test_resolve_config_path_ignores_stale_cache(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("SDCARD_STATE_PATH", str(state_path))

    stale_path = tmp_path / "missing.yml"
    state_path.write_text(
        json.dumps({"last_config_path": str(stale_path)}),
        encoding="utf-8",
    )

    assert resolve_config_path(None) is None