"""Do-not-disturb window parsing, overnight-wrap aware."""

from __future__ import annotations

from datetime import datetime, time


def parse_dnd_window(window: str) -> tuple[time, time] | None:
    window = (window or "").strip()
    if not window or "-" not in window:
        return None
    start_s, end_s = window.split("-", 1)
    try:
        start = datetime.strptime(start_s.strip(), "%H:%M").time()
        end = datetime.strptime(end_s.strip(), "%H:%M").time()
    except ValueError:
        return None
    return start, end


def is_within_quiet_hours(moment: datetime, window: tuple[time, time] | None) -> bool:
    if window is None:
        return False
    start, end = window
    t = moment.time()
    if start <= end:
        return start <= t < end
    # Overnight window, e.g. 22:00-07:00
    return t >= start or t < end
