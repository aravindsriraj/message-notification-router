"""LLM client protocol. FallbackOnlyClient is the M1 stand-in; GeminiLLMClient (M2) is the real client."""

from __future__ import annotations

import json
import os
import time
from typing import Protocol

from google import genai

import config
from data.schema import DataBundle
from features.signals import SignalBundle
from llm import cache as llm_cache
from llm.fallback import rule_based_fallback
from llm.few_shot import build_few_shot_block
from llm.prompt import SYSTEM_PROMPT, build_user_payload
from llm.schema import ClassificationResult, build_classification_schema


class LLMClient(Protocol):
    def classify(self, signals: SignalBundle) -> ClassificationResult: ...


class FallbackOnlyClient:
    """Stand-in LLM used before the real Gemini client is wired in (M1 milestone)."""

    def classify(self, signals: SignalBundle) -> ClassificationResult:
        return rule_based_fallback(signals)


class GeminiLLMClient:
    """Hybrid classifier: deterministic signals + a shortlisted evidence candidate list
    (built in features/evidence.py) are sent to Gemini via the Interactions API, which
    returns a schema-validated ClassificationResult. Falls back to rule_based_fallback if
    the API call fails after retries, so a single bad row never crashes the batch."""

    def __init__(self, bundle: DataBundle, model: str = config.LLM_MODEL):
        self._client = genai.Client()
        self._model = model
        self._bundle = bundle
        self._system_prompt: str | None = None

    def _full_system_prompt(self) -> str:
        if self._system_prompt is None:
            few_shot = build_few_shot_block(self._bundle)
            self._system_prompt = SYSTEM_PROMPT + "\n\nWORKED EXAMPLES:\n\n" + few_shot
        return self._system_prompt

    def classify(self, signals: SignalBundle) -> ClassificationResult:
        system_prompt = self._full_system_prompt()
        payload = build_user_payload(signals)
        candidate_ids = [c.message_id for c in signals.evidence_candidates]
        schema_cls = build_classification_schema(candidate_ids)

        cached = llm_cache.get_cached(signals.message.message_id, payload, system_prompt)
        if cached is not None:
            return schema_cls.model_validate(cached)

        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                result = self._call_gemini(system_prompt, payload, schema_cls)
                candidate_set = set(candidate_ids)
                filtered_evidence = [e for e in result.evidence_message_ids if e in candidate_set]
                result = result.model_copy(update={"evidence_message_ids": filtered_evidence})
                llm_cache.set_cached(
                    signals.message.message_id, payload, system_prompt, result.model_dump()
                )
                return result
            except Exception:  # noqa: BLE001 - retry on any transient failure
                if attempt < config.LLM_MAX_RETRIES - 1:
                    time.sleep(config.LLM_RETRY_BACKOFF_SECONDS[attempt])

        # Retries exhausted: never let one bad LLM call take down the batch.
        return rule_based_fallback(signals)

    def _call_gemini(self, system_prompt: str, payload: dict, schema_cls: type[ClassificationResult]):
        interaction = self._client.interactions.create(
            model=self._model,
            system_instruction=system_prompt,
            input="Classify this message:\n\n" + json.dumps(payload, indent=2, default=str),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_cls.model_json_schema(),
            },
            generation_config={"temperature": 0},
        )
        return schema_cls.model_validate_json(interaction.output_text)


def get_llm_client(bundle: DataBundle | None = None) -> LLMClient:
    if bundle is not None and os.environ.get(config.GEMINI_API_KEY_ENV):
        return GeminiLLMClient(bundle)
    return FallbackOnlyClient()
