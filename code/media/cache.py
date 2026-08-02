"""Dedup + cache for media descriptions, keyed by media_id (not message row) since several
messages can reference the same image/voice note."""

from __future__ import annotations

import json
import threading
from typing import Callable

import config

_lock = threading.Lock()
_CACHE_PATH = config.CACHE_DIR / "media_cache.json"


def _load() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _CACHE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(_CACHE_PATH)


def get_or_describe(media_id: str, fetch_fn: Callable[[], str]) -> str:
    with _lock:
        data = _load()
        if media_id in data:
            return data[media_id]

    description = fetch_fn()

    with _lock:
        data = _load()
        data[media_id] = description
        _save(data)

    return description
