"""Typed row representations for every dataset/*.csv file."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class MessageRow:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: datetime
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int


@dataclass
class UserRow:
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass
class GroupRow:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: date
    messages_30d: int


@dataclass
class GroupMemberRow:
    group_id: str
    user_id: str
    role: str
    joined_at: date
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass
class BusinessRow:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass
class UserBusinessRow:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: datetime | None
    allows_promotions: bool
    promotions_opted_out_at: datetime | None
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: datetime | None


@dataclass
class EventRow:
    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: int | None
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass
class DailyNotifRow:
    user_id: str
    date: date
    notifications_sent: int
    notifications_dismissed: int


@dataclass
class ImageRow:
    image_id: str
    file_path: str


@dataclass
class VoiceRow:
    voice_note_id: str
    file_path: str


@dataclass
class DataBundle:
    """All parsed rows plus O(1) lookup indices, built once by data/loaders.py."""

    users_by_id: dict[str, UserRow] = field(default_factory=dict)
    groups_by_id: dict[str, GroupRow] = field(default_factory=dict)
    business_by_id: dict[str, BusinessRow] = field(default_factory=dict)
    group_member_by_key: dict[tuple[str, str], GroupMemberRow] = field(default_factory=dict)
    ubh_by_key: dict[tuple[str, str], UserBusinessRow] = field(default_factory=dict)
    history_by_user: dict[str, list[MessageRow]] = field(default_factory=dict)
    event_by_message_id: dict[str, EventRow] = field(default_factory=dict)
    daily_notif_by_user: dict[str, list[DailyNotifRow]] = field(default_factory=dict)
    images_by_id: dict[str, ImageRow] = field(default_factory=dict)
    voice_by_id: dict[str, VoiceRow] = field(default_factory=dict)
