"""Reads dataset/*.csv into the typed rows in data/schema.py and builds a DataBundle."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from data.schema import (
    BusinessRow,
    DailyNotifRow,
    DataBundle,
    EventRow,
    GroupMemberRow,
    GroupRow,
    ImageRow,
    MessageRow,
    UserBusinessRow,
    UserRow,
    VoiceRow,
)

_DATETIME_FMT = "%Y-%m-%d %H:%M"
_DATE_FMT = "%Y-%m-%d"


def _parse_int(value: str, default: int = 0) -> int:
    value = (value or "").strip()
    return int(value) if value else default


def _parse_bool01(value: str) -> bool:
    return (value or "").strip() == "1"


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), _DATE_FMT).date()


def _parse_datetime(value: str) -> datetime:
    return datetime.strptime(value.strip(), _DATETIME_FMT)


def _parse_optional_datetime(value: str) -> datetime | None:
    value = (value or "").strip()
    return _parse_datetime(value) if value else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_messages(path: Path) -> list[MessageRow]:
    return [
        MessageRow(
            message_id=r["message_id"],
            user_id=r["user_id"],
            conversation_type=r["conversation_type"],
            group_id=r["group_id"],
            business_id=r["business_id"],
            sender_user_id=r["sender_user_id"],
            created_at=_parse_datetime(r["created_at"]),
            message_text=r["message_text"],
            media_type=r["media_type"],
            media_id=r["media_id"],
            forwarded_count=_parse_int(r["forwarded_count"]),
        )
        for r in _read_rows(path)
    ]


def load_all(dataset_dir: Path) -> DataBundle:
    bundle = DataBundle()

    for r in _read_rows(dataset_dir / "users.csv"):
        bundle.users_by_id[r["user_id"]] = UserRow(
            user_id=r["user_id"],
            do_not_disturb_window=r["do_not_disturb_window"],
            messages_opened_30d=_parse_int(r["messages_opened_30d"]),
            messages_replied_30d=_parse_int(r["messages_replied_30d"]),
            notifications_dismissed_30d=_parse_int(r["notifications_dismissed_30d"]),
            messages_reported_30d=_parse_int(r["messages_reported_30d"]),
        )

    for r in _read_rows(dataset_dir / "groups.csv"):
        bundle.groups_by_id[r["group_id"]] = GroupRow(
            group_id=r["group_id"],
            group_name=r["group_name"],
            group_type=r["group_type"],
            member_count=_parse_int(r["member_count"]),
            admin_count=_parse_int(r["admin_count"]),
            created_at=_parse_date(r["created_at"]),
            messages_30d=_parse_int(r["messages_30d"]),
        )

    for r in _read_rows(dataset_dir / "group_members.csv"):
        key = (r["group_id"], r["user_id"])
        bundle.group_member_by_key[key] = GroupMemberRow(
            group_id=r["group_id"],
            user_id=r["user_id"],
            role=r["role"],
            joined_at=_parse_date(r["joined_at"]),
            messages_sent_30d=_parse_int(r["messages_sent_30d"]),
            messages_read_30d=_parse_int(r["messages_read_30d"]),
            replies_sent_30d=_parse_int(r["replies_sent_30d"]),
            notifications_dismissed_30d=_parse_int(r["notifications_dismissed_30d"]),
            group_muted_by_user=_parse_bool01(r["group_muted_by_user"]),
        )

    for r in _read_rows(dataset_dir / "business_accounts.csv"):
        bundle.business_by_id[r["business_id"]] = BusinessRow(
            business_id=r["business_id"],
            display_name=r["display_name"],
            brand_name=r["brand_name"],
            category=r["category"],
            verified=_parse_bool01(r["verified"]),
            official_domain=r["official_domain"],
            domain_used_by_sender=r["domain_used_by_sender"],
            account_age_days=_parse_int(r["account_age_days"]),
            messages_sent_30d=_parse_int(r["messages_sent_30d"]),
            user_reports_30d=_parse_int(r["user_reports_30d"]),
            domain_used_by_sender_age_days=_parse_int(r["domain_used_by_sender_age_days"]),
        )

    for r in _read_rows(dataset_dir / "user_business_history.csv"):
        key = (r["user_id"], r["business_id"])
        bundle.ubh_by_key[key] = UserBusinessRow(
            user_id=r["user_id"],
            business_id=r["business_id"],
            why_user_knows_account=r["why_user_knows_account"],
            last_activity_at=_parse_optional_datetime(r["last_activity_at"]),
            allows_promotions=_parse_bool01(r["allows_promotions"]),
            promotions_opted_out_at=_parse_optional_datetime(r["promotions_opted_out_at"]),
            activity_count_180d=_parse_int(r["activity_count_180d"]),
            messages_opened_30d=_parse_int(r["messages_opened_30d"]),
            messages_dismissed_30d=_parse_int(r["messages_dismissed_30d"]),
            messages_replied_30d=_parse_int(r["messages_replied_30d"]),
            last_reply_at=_parse_optional_datetime(r["last_reply_at"]),
        )

    history_by_user: dict[str, list[MessageRow]] = defaultdict(list)
    for row in _load_messages(dataset_dir / "message_history.csv"):
        history_by_user[row.user_id].append(row)
    for user_id, rows in history_by_user.items():
        rows.sort(key=lambda r: r.created_at, reverse=True)
    bundle.history_by_user = dict(history_by_user)

    for r in _read_rows(dataset_dir / "message_events.csv"):
        bundle.event_by_message_id[r["message_id"]] = EventRow(
            user_id=r["user_id"],
            message_id=r["message_id"],
            message_opened=_parse_bool01(r["message_opened"]),
            message_replied=_parse_bool01(r["message_replied"]),
            reaction_time_minutes=(
                int(r["reaction_time_minutes"]) if r["reaction_time_minutes"].strip() else None
            ),
            notification_dismissed=_parse_bool01(r["notification_dismissed"]),
            muted_after_message=_parse_bool01(r["muted_after_message"]),
            message_reported=_parse_bool01(r["message_reported"]),
        )

    daily_notif_by_user: dict[str, list[DailyNotifRow]] = defaultdict(list)
    for r in _read_rows(dataset_dir / "daily_notification_summary.csv"):
        daily_notif_by_user[r["user_id"]].append(
            DailyNotifRow(
                user_id=r["user_id"],
                date=_parse_date(r["date"]),
                notifications_sent=_parse_int(r["notifications_sent"]),
                notifications_dismissed=_parse_int(r["notifications_dismissed"]),
            )
        )
    bundle.daily_notif_by_user = dict(daily_notif_by_user)

    for r in _read_rows(dataset_dir / "images.csv"):
        bundle.images_by_id[r["image_id"]] = ImageRow(image_id=r["image_id"], file_path=r["file_path"])

    for r in _read_rows(dataset_dir / "voice_notes.csv"):
        bundle.voice_by_id[r["voice_note_id"]] = VoiceRow(
            voice_note_id=r["voice_note_id"], file_path=r["file_path"]
        )

    return bundle


def load_messages_csv(path: Path) -> list[MessageRow]:
    """Loads dataset/messages.csv (or any file sharing its schema, e.g. sample_messages.csv's input columns)."""
    return _load_messages(path)
