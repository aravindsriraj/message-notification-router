"""Shared routing pipeline used by both code/main.py and code/evaluation/main.py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import config
from data.schema import DataBundle, MessageRow
from features.signals import compute_signal_bundle
from llm.client import LLMClient
from output_writer import OutputRow


def _classify_one(
    message: MessageRow,
    bundle: DataBundle,
    llm_client: LLMClient,
    media_descriptions: dict[str, str],
) -> OutputRow:
    try:
        media_desc = media_descriptions.get(message.media_id) if message.media_id else None
        signals = compute_signal_bundle(message, bundle, media_description=media_desc)
        result = llm_client.classify(signals)

        confidence = max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, result.confidence))
        candidate_ids = {c.message_id for c in signals.evidence_candidates}
        evidence_ids = [eid for eid in result.evidence_message_ids if eid in candidate_ids]

        return OutputRow(
            message_id=message.message_id,
            action=result.action,
            message_type=result.message_type,
            reason=result.reason,
            confidence=confidence,
            evidence_message_ids=evidence_ids,
        )
    except Exception as exc:  # noqa: BLE001 - a single bad row must never crash the batch
        return OutputRow(
            message_id=message.message_id,
            action="digest",
            message_type="unknown",
            reason=f"Fallback due to a processing error: {exc}",
            confidence=0.3,
            evidence_message_ids=[],
        )


def route_messages(
    messages: list[MessageRow],
    bundle: DataBundle,
    llm_client: LLMClient,
    media_descriptions: dict[str, str] | None = None,
) -> list[OutputRow]:
    media_descriptions = media_descriptions or {}

    # ThreadPoolExecutor.map preserves input order in its output regardless of completion
    # order, so no extra bookkeeping is needed to keep rows aligned with `messages`.
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_WORKERS) as pool:
        rows = list(
            pool.map(
                lambda m: _classify_one(m, bundle, llm_client, media_descriptions),
                messages,
            )
        )
    return rows
