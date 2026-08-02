"""JSON response cache keyed by message_id + payload hash, so reruns are free and byte-identical."""

from __future__ import annotations

import hashlib
import json
import threading

import config

_lock = threading.Lock()
_CACHE_PATH = config.CACHE_DIR / "llm_responses.json"


def _cache_key(message_id: str, payload: dict, system_prompt: str) -> str:
    digest_input = json.dumps(payload, sort_keys=True, default=str) + "\n" + system_prompt
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    return f"{message_id}:{digest}"


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


def get_cached(message_id: str, payload: dict, system_prompt: str) -> dict | None:
    key = _cache_key(message_id, payload, system_prompt)
    with _lock:
        return _load().get(key)


def set_cached(message_id: str, payload: dict, system_prompt: str, result: dict) -> None:
    key = _cache_key(message_id, payload, system_prompt)
    with _lock:
        data = _load()
        data[key] = result
        _save(data)
