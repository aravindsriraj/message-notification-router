"""Structured output contract for a single message classification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

import config

ActionLiteral = Literal["notify", "digest", "mute"]
MessageTypeLiteral = Literal[
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
]


class ClassificationResult(BaseModel):
    action: ActionLiteral
    message_type: MessageTypeLiteral
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: list[str] = Field(default_factory=list)


assert set(ClassificationResult.model_fields["action"].annotation.__args__) == set(config.ACTIONS)
assert set(ClassificationResult.model_fields["message_type"].annotation.__args__) == set(
    config.MESSAGE_TYPES
)


def build_classification_schema(candidate_ids: list[str]) -> type[BaseModel]:
    """Returns a ClassificationResult variant whose evidence_message_ids is restricted to
    candidate_ids via a dynamic Literal, so a hallucinated ID is a schema violation rather
    than just a prompting request."""
    if not candidate_ids:
        return ClassificationResult
    evidence_id_type = Literal[tuple(candidate_ids)]  # type: ignore[valid-type]
    return create_model(
        "ClassificationResultWithEvidence",
        __base__=ClassificationResult,
        evidence_message_ids=(list[evidence_id_type], Field(default_factory=list)),
    )
