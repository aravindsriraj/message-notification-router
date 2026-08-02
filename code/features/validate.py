"""Pre-write sanity checks on the final output rows — fail loudly rather than ship a malformed file."""

from __future__ import annotations

import config
from output_writer import OutputRow


def validate_output_rows(rows: list[OutputRow], expected_message_ids: list[str]) -> None:
    ids = [r.message_id for r in rows]

    if len(ids) != len(expected_message_ids):
        raise ValueError(f"row count mismatch: got {len(ids)}, expected {len(expected_message_ids)}")
    if set(ids) != set(expected_message_ids):
        missing = set(expected_message_ids) - set(ids)
        extra = set(ids) - set(expected_message_ids)
        raise ValueError(f"message_id set mismatch: missing={missing}, extra={extra}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate message_id in output rows")

    for r in rows:
        if r.action not in config.ACTIONS:
            raise ValueError(f"invalid action {r.action!r} for {r.message_id}")
        if r.message_type not in config.MESSAGE_TYPES:
            raise ValueError(f"invalid message_type {r.message_type!r} for {r.message_id}")
        if not (0.0 <= r.confidence <= 1.0):
            raise ValueError(f"confidence out of range for {r.message_id}: {r.confidence}")
