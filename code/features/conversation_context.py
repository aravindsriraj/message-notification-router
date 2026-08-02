"""Per-message conversation context: group trust/mute state, business trust, or first-contact status."""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from data.schema import DataBundle, MessageRow


@dataclass
class ConversationContext:
    conversation_type: str

    # group fields
    group_name: str | None = None
    group_type: str | None = None
    member_count: int | None = None
    recipient_role: str | None = None
    sender_is_admin: bool = False
    group_muted_by_user: bool = False
    is_direct_mention: bool = False

    # business fields
    business_display_name: str | None = None
    business_category: str | None = None
    verified: bool = False
    domain_mismatch: bool = False
    young_sending_domain: bool = False
    business_report_rate: float = 0.0
    why_user_knows_account: str | None = None
    allows_promotions: bool = True
    promo_opted_out: bool = False
    activity_count_180d: int = 0
    business_dismiss_rate: float = 0.0

    # personal fields
    is_first_contact: bool = False


def build_conversation_context(message: MessageRow, bundle: DataBundle) -> ConversationContext:
    ctx = ConversationContext(conversation_type=message.conversation_type)

    if message.conversation_type == "group" and message.group_id:
        group = bundle.groups_by_id.get(message.group_id)
        if group:
            ctx.group_name = group.group_name
            ctx.group_type = group.group_type
            ctx.member_count = group.member_count

        member = bundle.group_member_by_key.get((message.group_id, message.user_id))
        if member:
            ctx.recipient_role = member.role
            ctx.group_muted_by_user = member.group_muted_by_user

        sender_member = bundle.group_member_by_key.get((message.group_id, message.sender_user_id))
        ctx.sender_is_admin = bool(sender_member and sender_member.role == "admin")

        ctx.is_direct_mention = bool(
            message.message_text
            and re.search(r"@" + re.escape(message.user_id) + r"\b", message.message_text)
        )

    elif message.conversation_type == "business" and message.business_id:
        business = bundle.business_by_id.get(message.business_id)
        if business:
            ctx.business_display_name = business.display_name
            ctx.business_category = business.category
            ctx.verified = business.verified
            ctx.domain_mismatch = (
                business.official_domain.strip().lower() != business.domain_used_by_sender.strip().lower()
            )
            ctx.young_sending_domain = (
                business.domain_used_by_sender_age_days < config.DOMAIN_AGE_YOUNG_THRESHOLD_DAYS
            )
            ctx.business_report_rate = (
                business.user_reports_30d / business.messages_sent_30d
                if business.messages_sent_30d > 0
                else 0.0
            )

        ubh = bundle.ubh_by_key.get((message.user_id, message.business_id))
        if ubh:
            ctx.why_user_knows_account = ubh.why_user_knows_account
            ctx.allows_promotions = ubh.allows_promotions
            ctx.promo_opted_out = (not ubh.allows_promotions) or (ubh.promotions_opted_out_at is not None)
            ctx.activity_count_180d = ubh.activity_count_180d
            ctx.business_dismiss_rate = (
                ubh.messages_dismissed_30d / ubh.messages_opened_30d if ubh.messages_opened_30d > 0 else 0.0
            )
        else:
            # No recorded relationship with this business at all.
            ctx.promo_opted_out = True

    elif message.conversation_type == "personal":
        history = bundle.history_by_user.get(message.user_id, [])
        ctx.is_first_contact = not any(
            h.sender_user_id == message.sender_user_id and h.created_at < message.created_at
            for h in history
        )

    return ctx
