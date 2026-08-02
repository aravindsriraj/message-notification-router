"""Pure rule-based classifier: used as the M1 stand-in LLM, and as the safety net in
pipeline.py if the real Gemini call fails after retries, so every message_id always gets a row."""

from __future__ import annotations

from features.signals import SignalBundle
from llm.schema import ClassificationResult


def rule_based_fallback(signals: SignalBundle) -> ClassificationResult:
    ctx = signals.conversation_context
    evidence_ids = [c.message_id for c in signals.evidence_candidates[:3]]

    # Hard safety rule: an embedded instruction attempting to manipulate the router is
    # itself treated as a strong scam signal, regardless of any other context.
    if signals.injection_attempt_detected:
        return ClassificationResult(
            action="mute",
            message_type="scam",
            reason="Message contains an embedded instruction attempting to manipulate the routing decision.",
            confidence=0.85,
            evidence_message_ids=evidence_ids,
        )

    if ctx.conversation_type == "business":
        return _classify_business(ctx, evidence_ids)
    if ctx.conversation_type == "group":
        return _classify_group(ctx, signals, evidence_ids)
    return _classify_personal(ctx, signals, evidence_ids)


def _classify_business(ctx, evidence_ids: list[str]) -> ClassificationResult:
    if ctx.domain_mismatch or ctx.young_sending_domain:
        return ClassificationResult(
            action="mute",
            message_type="scam",
            reason="Sending domain does not match the business's official domain, or is newly registered.",
            confidence=0.75,
            evidence_message_ids=evidence_ids,
        )
    if ctx.promo_opted_out:
        return ClassificationResult(
            action="mute",
            message_type="promotion",
            reason="The user has opted out of, or has no recorded relationship with, this business.",
            confidence=0.7,
            evidence_message_ids=evidence_ids,
        )
    if ctx.verified and ctx.activity_count_180d > 0:
        return ClassificationResult(
            action="digest",
            message_type="business_update",
            reason="Verified business with an existing user relationship; not time-critical.",
            confidence=0.65,
            evidence_message_ids=evidence_ids,
        )
    return ClassificationResult(
        action="digest",
        message_type="unknown",
        reason="Business message with no strong trust or risk signal either way.",
        confidence=0.5,
        evidence_message_ids=evidence_ids,
    )


def _classify_group(ctx, signals: SignalBundle, evidence_ids: list[str]) -> ClassificationResult:
    if ctx.is_direct_mention or (ctx.sender_is_admin and signals.contains_urgency_keyword):
        return ClassificationResult(
            action="notify",
            message_type="urgent",
            reason="Direct mention or urgent admin update in the group.",
            confidence=0.7,
            evidence_message_ids=evidence_ids,
        )
    if (
        signals.is_near_duplicate_of_recent_history
        and signals.evidence_candidates
        and signals.evidence_candidates[0].past_dismissed
    ):
        return ClassificationResult(
            action="mute",
            message_type="forward",
            reason="Similar messages in this group were previously dismissed by the user.",
            confidence=0.6,
            evidence_message_ids=evidence_ids,
        )
    if ctx.group_muted_by_user:
        return ClassificationResult(
            action="mute",
            message_type="unknown",
            reason="Group is muted by the user and this message has no urgent signal.",
            confidence=0.55,
            evidence_message_ids=evidence_ids,
        )
    return ClassificationResult(
        action="digest",
        message_type="personal",
        reason="Ordinary group message with no urgent or risk signal.",
        confidence=0.55,
        evidence_message_ids=evidence_ids,
    )


def _classify_personal(ctx, signals: SignalBundle, evidence_ids: list[str]) -> ClassificationResult:
    if ctx.is_first_contact and signals.contains_urgency_keyword:
        return ClassificationResult(
            action="mute",
            message_type="scam",
            reason="First contact from this sender combined with urgency pressure is a common scam pattern.",
            confidence=0.6,
            evidence_message_ids=evidence_ids,
        )
    if signals.contains_urgency_keyword:
        return ClassificationResult(
            action="notify",
            message_type="urgent",
            reason="Urgency keywords detected in a personal message from a known contact.",
            confidence=0.65,
            evidence_message_ids=evidence_ids,
        )
    return ClassificationResult(
        action="digest",
        message_type="personal",
        reason="Ordinary personal message with no urgent or risk signal.",
        confidence=0.55,
        evidence_message_ids=evidence_ids,
    )
