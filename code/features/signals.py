"""Aggregates the deterministic per-message signal bundle passed to the LLM (and to the fallback rules)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from data.schema import DataBundle, MessageRow
from features.conversation_context import ConversationContext, build_conversation_context
from features.evidence import EvidenceCandidate, shortlist_evidence
from features.quiet_hours import is_within_quiet_hours
from features.user_profile import UserProfile, build_user_profile

_URGENCY_KEYWORDS = re.compile(
    r"\b(urgent|asap|immediately|emergency|deadline|escalat\w*|right away|expires? today|final notice)\b",
    re.IGNORECASE,
)

# Defense-in-depth signal only — the hard guardrail lives in the LLM system prompt (llm/prompt.py).
# A scam need not literally contain these phrases, so this is a hint, never a sole determinant.
_INJECTION_PHRASES = re.compile(
    r"(ignore (all )?(previous|prior|the above|sender)|disregard (the )?(above|previous)"
    r"|system prompt|you are now|new instructions?:|override (the )?routing|routing override"
    r"|assistant instruction|set (the )?action\s*[:=]|classify (this |it )?as (notify|urgent|safe)"
    r"|confidence\s*[:=]\s*1(\.0)?\b|mark (this|it) (as|notify))",
    re.IGNORECASE,
)


@dataclass
class SignalBundle:
    message: MessageRow
    user_profile: UserProfile
    conversation_context: ConversationContext
    evidence_candidates: list[EvidenceCandidate]
    effective_text: str
    is_quiet_hours: bool
    contains_urgency_keyword: bool
    injection_attempt_detected: bool
    is_near_duplicate_of_recent_history: bool
    rule_signal_strength: float


def compute_signal_bundle(
    message: MessageRow,
    bundle: DataBundle,
    media_description: str | None = None,
) -> SignalBundle:
    profile = build_user_profile(message.user_id, bundle)
    ctx = build_conversation_context(message, bundle)
    effective_text = message.message_text or media_description or ""
    evidence = shortlist_evidence(message, bundle, effective_text)

    is_quiet = is_within_quiet_hours(message.created_at, profile.dnd_window)
    has_urgency_kw = bool(_URGENCY_KEYWORDS.search(effective_text))
    has_injection = bool(_INJECTION_PHRASES.search(effective_text))
    is_near_dup = bool(
        evidence and evidence[0].score >= 3.0 and "text_similar" in evidence[0].relation_tags
    )

    rule_signal_strength = _compute_rule_signal_strength(ctx, has_injection, is_near_dup, evidence)

    return SignalBundle(
        message=message,
        user_profile=profile,
        conversation_context=ctx,
        evidence_candidates=evidence,
        effective_text=effective_text,
        is_quiet_hours=is_quiet,
        contains_urgency_keyword=has_urgency_kw,
        injection_attempt_detected=has_injection,
        is_near_duplicate_of_recent_history=is_near_dup,
        rule_signal_strength=rule_signal_strength,
    )


def _compute_rule_signal_strength(
    ctx: ConversationContext,
    has_injection: bool,
    is_near_dup: bool,
    evidence: list[EvidenceCandidate],
) -> float:
    """Rough signal in [-1, 1]: positive corroborates a confident notify/legit call,
    negative corroborates a confident mute/risk call. Used only as a light confidence nudge
    (see llm/client.py), never to override the LLM's action/message_type decision."""
    score = 0.0
    if has_injection:
        score -= 0.6
    if ctx.conversation_type == "business":
        if ctx.domain_mismatch or ctx.young_sending_domain:
            score -= 0.4
        if ctx.verified and not ctx.promo_opted_out:
            score += 0.3
        if ctx.promo_opted_out:
            score -= 0.2
    if ctx.conversation_type == "group":
        if ctx.sender_is_admin:
            score += 0.2
        if ctx.is_direct_mention:
            score += 0.3
    if is_near_dup and evidence:
        if evidence[0].past_dismissed:
            score -= 0.3
        if evidence[0].past_replied:
            score += 0.2
    return max(-1.0, min(1.0, score))
