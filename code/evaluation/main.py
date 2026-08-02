"""Evaluates pipeline.route_messages() against the solved dataset/sample_messages.csv rows.

Run: python -m code.evaluation.main   (from the repo root)
 or: python code/evaluation/main.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from google import genai  # noqa: E402

import config  # noqa: E402
from data.loaders import load_all, load_messages_csv  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402
from llm.client import get_llm_client  # noqa: E402
from llm.few_shot import _EXAMPLES as FEW_SHOT_EXAMPLES  # noqa: E402
from media.enrich import enrich_media  # noqa: E402
from pipeline import route_messages  # noqa: E402


def _load_gold(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        r["message_id"]: {
            "action": r["action"],
            "message_type": r["message_type"],
            "confidence": float(r["confidence"]),
            "evidence_message_ids": (
                [] if r["evidence_message_ids"] == "none" else r["evidence_message_ids"].split(";")
            ),
        }
        for r in rows
    }


def main() -> None:
    bundle = load_all(config.DATASET_DIR)
    messages = load_messages_csv(config.SAMPLE_MESSAGES_CSV)
    gold = _load_gold(config.SAMPLE_MESSAGES_CSV)

    llm_client = get_llm_client(bundle)

    media_descriptions: dict[str, str] = {}
    if os.environ.get(config.GEMINI_API_KEY_ENV):
        media_client = genai.Client()
        media_descriptions = enrich_media(messages, bundle, media_client)

    predictions = route_messages(messages, bundle, llm_client, media_descriptions)

    few_shot_ids = {example[0] for example in FEW_SHOT_EXAMPLES}
    evaluate(predictions, gold, few_shot_ids)


if __name__ == "__main__":
    main()
