"""Paths, model names, and tunable thresholds for the message router."""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
CACHE_DIR = CODE_DIR / ".cache"

MESSAGES_CSV = DATASET_DIR / "messages.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"
SAMPLE_MESSAGES_CSV = DATASET_DIR / "sample_messages.csv"
USERS_CSV = DATASET_DIR / "users.csv"
GROUPS_CSV = DATASET_DIR / "groups.csv"
GROUP_MEMBERS_CSV = DATASET_DIR / "group_members.csv"
BUSINESS_ACCOUNTS_CSV = DATASET_DIR / "business_accounts.csv"
USER_BUSINESS_HISTORY_CSV = DATASET_DIR / "user_business_history.csv"
MESSAGE_HISTORY_CSV = DATASET_DIR / "message_history.csv"
MESSAGE_EVENTS_CSV = DATASET_DIR / "message_events.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
VOICE_NOTES_CSV = DATASET_DIR / "voice_notes.csv"
DAILY_NOTIFICATION_SUMMARY_CSV = DATASET_DIR / "daily_notification_summary.csv"

LLM_MODEL = os.environ.get("ROUTER_LLM_MODEL", "gemini-3.6-flash")
MEDIA_MODEL = os.environ.get("ROUTER_MEDIA_MODEL", "gemini-3.6-flash")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_SECONDS = (1, 2, 4)

# Classification and media calls are I/O-bound HTTP requests, so a thread pool parallelizes
# them safely without needing an async rewrite. Kept modest to avoid tripping rate limits.
LLM_MAX_WORKERS = 8
MEDIA_MAX_WORKERS = 5

# Evidence shortlist
EVIDENCE_TOP_K = 6
EVIDENCE_RECENCY_HALF_LIFE_DAYS = 30
EVIDENCE_SAME_SENDER_WEIGHT = 3.0
EVIDENCE_SAME_GROUP_WEIGHT = 2.0
EVIDENCE_SAME_BUSINESS_WEIGHT = 2.0
EVIDENCE_TEXT_OVERLAP_WEIGHT = 1.5

# Business trust
DOMAIN_AGE_YOUNG_THRESHOLD_DAYS = 180

# Confidence calibration
CONFIDENCE_MIN = 0.05
CONFIDENCE_MAX = 0.97
CONFIDENCE_RULE_BLEND_WEIGHT = 0.15

ACTIONS = ("notify", "digest", "mute")
MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)
