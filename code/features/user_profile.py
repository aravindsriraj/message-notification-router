"""Per-user notification behavior profile, derived from users.csv + daily_notification_summary.csv."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from data.schema import DataBundle
from features.quiet_hours import parse_dnd_window


@dataclass
class UserProfile:
    user_id: str
    dnd_window: tuple[time, time] | None
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int
    engagement_ratio: float
    dismissal_ratio: float
    report_rate: float
    notification_fatigue: str  # "low" | "medium" | "high"


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _fatigue_bucket(bundle: DataBundle, user_id: str) -> str:
    rows = bundle.daily_notif_by_user.get(user_id, [])
    if not rows:
        return "low"
    avg_sent = sum(r.notifications_sent for r in rows) / len(rows)
    avg_dismissed = sum(r.notifications_dismissed for r in rows) / len(rows)
    dismiss_rate = avg_dismissed / avg_sent if avg_sent > 0 else 0.0
    if avg_sent >= 8 or dismiss_rate >= 0.5:
        return "high"
    if avg_sent >= 4 or dismiss_rate >= 0.25:
        return "medium"
    return "low"


def build_user_profile(user_id: str, bundle: DataBundle) -> UserProfile:
    user = bundle.users_by_id.get(user_id)
    if user is None:
        return UserProfile(
            user_id=user_id,
            dnd_window=None,
            messages_opened_30d=0,
            messages_replied_30d=0,
            notifications_dismissed_30d=0,
            messages_reported_30d=0,
            engagement_ratio=0.0,
            dismissal_ratio=0.0,
            report_rate=0.0,
            notification_fatigue="low",
        )
    return UserProfile(
        user_id=user_id,
        dnd_window=parse_dnd_window(user.do_not_disturb_window),
        messages_opened_30d=user.messages_opened_30d,
        messages_replied_30d=user.messages_replied_30d,
        notifications_dismissed_30d=user.notifications_dismissed_30d,
        messages_reported_30d=user.messages_reported_30d,
        engagement_ratio=_safe_ratio(user.messages_replied_30d, user.messages_opened_30d),
        dismissal_ratio=_safe_ratio(user.notifications_dismissed_30d, user.messages_opened_30d),
        report_rate=_safe_ratio(user.messages_reported_30d, user.messages_opened_30d),
        notification_fatigue=_fatigue_bucket(bundle, user_id),
    )
