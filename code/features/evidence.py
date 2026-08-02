"""Shortlists real candidate evidence messages from message_history.csv for a given message.

The LLM is only ever allowed to cite message_ids returned by this module — never freeform IDs —
which prevents hallucinated evidence_message_ids in the final output.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime

import config
from data.schema import DataBundle, MessageRow

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _text_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


@dataclass
class EvidenceCandidate:
    message_id: str
    created_at: datetime
    sender_user_id: str
    group_id: str
    business_id: str
    text_snippet: str
    score: float
    relation_tags: list[str] = field(default_factory=list)
    past_opened: bool = False
    past_replied: bool = False
    past_dismissed: bool = False
    past_muted_after: bool = False
    past_reported: bool = False


def shortlist_evidence(
    message: MessageRow,
    bundle: DataBundle,
    effective_text: str,
    top_k: int = config.EVIDENCE_TOP_K,
) -> list[EvidenceCandidate]:
    pool = bundle.history_by_user.get(message.user_id, [])
    scored: list[EvidenceCandidate] = []

    for h in pool:
        if h.created_at > message.created_at:
            continue

        tags: list[str] = []
        score = 0.0

        if message.sender_user_id and h.sender_user_id == message.sender_user_id:
            score += config.EVIDENCE_SAME_SENDER_WEIGHT
            tags.append("same_sender")
        if message.group_id and h.group_id == message.group_id:
            score += config.EVIDENCE_SAME_GROUP_WEIGHT
            tags.append("same_group")
        if message.business_id and h.business_id == message.business_id:
            score += config.EVIDENCE_SAME_BUSINESS_WEIGHT
            tags.append("same_business")

        overlap = _text_overlap(effective_text, h.message_text)
        if overlap > 0:
            score += config.EVIDENCE_TEXT_OVERLAP_WEIGHT * overlap
            if overlap >= 0.5:
                tags.append("text_similar")

        if score <= 0:
            continue

        days_between = max((message.created_at - h.created_at).total_seconds() / 86400.0, 0.0)
        recency_factor = 0.5 + 0.5 * math.exp(-days_between / config.EVIDENCE_RECENCY_HALF_LIFE_DAYS)
        score *= recency_factor

        event = bundle.event_by_message_id.get(h.message_id)
        scored.append(
            EvidenceCandidate(
                message_id=h.message_id,
                created_at=h.created_at,
                sender_user_id=h.sender_user_id,
                group_id=h.group_id,
                business_id=h.business_id,
                text_snippet=(h.message_text or "")[:200],
                score=score,
                relation_tags=tags,
                past_opened=bool(event and event.message_opened),
                past_replied=bool(event and event.message_replied),
                past_dismissed=bool(event and event.notification_dismissed),
                past_muted_after=bool(event and event.muted_after_message),
                past_reported=bool(event and event.message_reported),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
