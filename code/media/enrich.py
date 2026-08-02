"""Pre-pass: describes every unique image/voice-note media_id referenced by a batch of
messages, once each (regardless of how many messages reuse the same media_id), and returns
a media_id -> description lookup for pipeline.route_messages(). Runs concurrently across
unique media items since each is an independent I/O-bound API call."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai

import config
from data.schema import DataBundle, MessageRow
from media.cache import get_or_describe
from media.gemini_media import describe_image, transcribe_voice


def _describe_one(media_id: str, media_type: str, path: Path, client: genai.Client) -> tuple[str, str]:
    if media_type == "image":
        description = get_or_describe(media_id, lambda: describe_image(client, path))
    else:
        description = get_or_describe(media_id, lambda: transcribe_voice(client, path))
    return media_id, description


def enrich_media(
    messages: list[MessageRow],
    bundle: DataBundle,
    client: genai.Client,
) -> dict[str, str]:
    unique_items: dict[str, tuple[str, Path]] = {}
    for message in messages:
        if not message.media_id or message.media_id in unique_items:
            continue
        if message.media_type == "image":
            image = bundle.images_by_id.get(message.media_id)
            if image is None:
                continue
            unique_items[message.media_id] = ("image", config.DATASET_DIR / image.file_path)
        elif message.media_type == "voice":
            voice = bundle.voice_by_id.get(message.media_id)
            if voice is None:
                continue
            unique_items[message.media_id] = ("voice", config.DATASET_DIR / voice.file_path)

    descriptions: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=config.MEDIA_MAX_WORKERS) as pool:
        futures = [
            pool.submit(_describe_one, media_id, kind, path, client)
            for media_id, (kind, path) in unique_items.items()
        ]
        for future in futures:
            media_id, description = future.result()
            descriptions[media_id] = description

    return descriptions
