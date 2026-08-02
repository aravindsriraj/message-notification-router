"""Scoring: action/message_type accuracy, confusion table, evidence overlap, confidence bands."""

from __future__ import annotations

from collections import Counter

from output_writer import OutputRow

# Observed in dataset/sample_messages.csv's solved rows - used only as a drift signal, never
# to override the LLM's own confidence.
_REFERENCE_CONFIDENCE_BANDS = {
    "notify": (0.85, 0.91),
    "mute": (0.81, 0.87),
    "digest": (0.78, 0.84),
}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate(predictions: list[OutputRow], gold: dict[str, dict], few_shot_ids: set[str]) -> None:
    subsets = {
        "ALL sample_messages.csv rows": [p for p in predictions if p.message_id in gold],
        "HELD-OUT (excludes rows used as few-shot prompt examples)": [
            p for p in predictions if p.message_id in gold and p.message_id not in few_shot_ids
        ],
    }

    for label, rows in subsets.items():
        print(f"\n=== {label}: n={len(rows)} ===")
        _report_subset(rows, gold)

    print("\n=== Per-row predicted vs gold ===")
    for p in predictions:
        if p.message_id not in gold:
            continue
        g = gold[p.message_id]
        is_match = p.action == g["action"] and p.message_type == g["message_type"]
        marker = "OK " if is_match else "XX "
        tag = " [few-shot example]" if p.message_id in few_shot_ids else ""
        print(f"{marker}{p.message_id}{tag}")
        print(f"    gold: action={g['action']:7s} type={g['message_type']}")
        print(f"    pred: action={p.action:7s} type={p.message_type} conf={p.confidence:.2f}")
        print(f"    pred reason: {p.reason}")


def _report_subset(rows: list[OutputRow], gold: dict[str, dict]) -> None:
    if not rows:
        print("  (no rows)")
        return

    action_correct = sum(1 for p in rows if p.action == gold[p.message_id]["action"])
    type_correct = sum(1 for p in rows if p.message_type == gold[p.message_id]["message_type"])
    print(f"  action accuracy:       {action_correct}/{len(rows)} = {action_correct / len(rows):.2%}")
    print(f"  message_type accuracy: {type_correct}/{len(rows)} = {type_correct / len(rows):.2%}")

    confusion = Counter((gold[p.message_id]["action"], p.action) for p in rows)
    print("  action confusion (gold -> pred):")
    for (g_action, p_action), count in sorted(confusion.items()):
        marker = "" if g_action == p_action else "  <-- mismatch"
        print(f"    {g_action:7s} -> {p_action:7s}: {count}{marker}")

    overlaps = [
        _jaccard(set(gold[p.message_id]["evidence_message_ids"]), set(p.evidence_message_ids))
        for p in rows
    ]
    print(f"  evidence Jaccard overlap (avg): {sum(overlaps) / len(overlaps):.2f}")

    by_action: dict[str, list[float]] = {}
    for p in rows:
        by_action.setdefault(p.action, []).append(p.confidence)
    print("  predicted confidence by action (vs reference band):")
    for action, confs in sorted(by_action.items()):
        mean_conf = sum(confs) / len(confs)
        lo, hi = _REFERENCE_CONFIDENCE_BANDS.get(action, (0.0, 1.0))
        flag = "" if lo <= mean_conf <= hi else "  <-- outside reference band"
        print(f"    {action:7s}: n={len(confs):2d} mean={mean_conf:.2f} (reference {lo}-{hi}){flag}")
