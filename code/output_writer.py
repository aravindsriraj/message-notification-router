"""Writes the required output.csv contract: message_id,action,message_type,reason,confidence,evidence_message_ids."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

FIELDNAMES = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


@dataclass
class OutputRow:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]


def write_output_csv(rows: list[OutputRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "message_id": r.message_id,
                    "action": r.action,
                    "message_type": r.message_type,
                    "reason": " ".join(r.reason.split()),
                    "confidence": f"{r.confidence:.2f}",
                    "evidence_message_ids": (
                        ";".join(r.evidence_message_ids) if r.evidence_message_ids else "none"
                    ),
                }
            )
