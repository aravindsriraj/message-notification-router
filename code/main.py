"""Entry point: python code/main.py -> writes dataset/output.csv."""

from __future__ import annotations

import os

from google import genai

import config
from data.loaders import load_all, load_messages_csv
from features.validate import validate_output_rows
from llm.client import get_llm_client
from media.enrich import enrich_media
from output_writer import write_output_csv
from pipeline import route_messages


def main() -> None:
    bundle = load_all(config.DATASET_DIR)
    messages = load_messages_csv(config.MESSAGES_CSV)
    llm_client = get_llm_client(bundle)

    media_descriptions: dict[str, str] = {}
    if os.environ.get(config.GEMINI_API_KEY_ENV):
        media_client = genai.Client()
        media_descriptions = enrich_media(messages, bundle, media_client)
        print(f"Described/transcribed {len(media_descriptions)} unique media items")

    rows = route_messages(messages, bundle, llm_client, media_descriptions)
    validate_output_rows(rows, [m.message_id for m in messages])
    write_output_csv(rows, config.OUTPUT_CSV)

    print(f"Wrote {len(rows)} rows to {config.OUTPUT_CSV}")


if __name__ == "__main__":
    main()
