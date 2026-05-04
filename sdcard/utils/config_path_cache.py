from __future__ import annotations

import json
import os
from pathlib import Path


_ENV_CONFIG_PATH = "SDCARD_CONFIG_PATH"
_ENV_STATE_PATH = "SDCARD_STATE_PATH"
_STATE_KEY = "last_config_path"


def _state_path() -> Path:
    override = os.environ.get(_ENV_STATE_PATH)
    if override:
        return Path(override)

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "sdcard" / "state.json"

    return Path.home() / ".sdcard" / "state.json"


def _load_state() -> dict[str, str]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_state(data: dict[str, str]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def remember_config_path(config_path: str | Path) -> Path:
    resolved = Path(config_path).expanduser().resolve()
    state = _load_state()
    state[_STATE_KEY] = str(resolved)
    _save_state(state)
    return resolved


def _cached_config_path() -> Path | None:
    state = _load_state()
    raw = state.get(_STATE_KEY)
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    state.pop(_STATE_KEY, None)
    _save_state(state)
    return None


def resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path:
        return remember_config_path(config_path)

    env_path = os.environ.get(_ENV_CONFIG_PATH)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return _cached_config_path()
