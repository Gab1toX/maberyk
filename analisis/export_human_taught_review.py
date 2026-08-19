"""Standalone script: read-only export of flagged human_taught rows for
manual review -- companion to analisis/audit_english.py.

audit_english.py's flagged list stores normalize_text()'d, concatenated
question+answer text (accent-stripped, lowercased, glued together) since
that's what the model actually tokenizes -- useful for the ratio math, but
not something you can act on: you can't tell the original question from
the answer, or find the row to edit. This script re-applies the exact same
flagging rule (same stopword set, same >30% threshold, same normalized
text for the ratio) directly against `conversations WHERE source =
'human_taught'`, grouped into unique (question, answer) pairs the same way
_load_training_samples() dedupes them -- so the count here matches
audit_english.py's 52 rather than counting one taught pair once per
duplicate row -- and writes out the real row id(s), original question, and
original answer for each, so a fix can target a specific row by id (e.g.
rewrite the question, keep the answer) without guessing.

Never writes to episodic_memory.sqlite3 (opened read-only via a SQLite URI)
or corpus_agente.txt. The only file written is the output review file.

Usage:
    python -m analisis.export_human_taught_review
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analisis.audit_english import FLAG_THRESHOLD, _english_ratio, _read_only_connection
from brain.language_model import normalize_text

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "human_taught_english_review.txt"


def find_flagged_pairs(memory_path: Path) -> list[tuple[list[int], str, str, list[float], float]]:
    """Returns (ids, question, answer, timestamps, english_ratio) for every
    UNIQUE (question, answer) human_taught pair whose normalized text is
    flagged -- grouped exactly like _load_training_samples() dedupes pairs,
    so this matches audit_english.py's 52-flagged count rather than
    counting the same taught pair once per duplicate row."""
    connection = _read_only_connection(memory_path)
    try:
        rows = connection.execute(
            "SELECT id, question, answer, timestamp FROM conversations "
            "WHERE source = 'human_taught'"
        ).fetchall()
    finally:
        connection.close()

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row_id, question, answer, timestamp in rows:
        if not question or not question.strip() or not answer or not answer.strip():
            continue
        grouped.setdefault((question, answer), []).append((row_id, timestamp))

    flagged = []
    for (question, answer), occurrences in grouped.items():
        combined = f"{normalize_text(question)} {normalize_text(answer)}"
        ratio = _english_ratio(combined)
        if ratio > FLAG_THRESHOLD:
            ids = [row_id for row_id, _ in occurrences]
            timestamps = [timestamp for _, timestamp in occurrences]
            flagged.append((ids, question, answer, timestamps, ratio))

    flagged.sort(key=lambda row: row[4], reverse=True)
    return flagged


def write_review_file(
    flagged: list[tuple[list[int], str, str, list[float], float]], output_path: Path
) -> None:
    lines = []
    for ids, question, answer, timestamps, ratio in flagged:
        ids_display = ",".join(str(i) for i in ids)
        note = f" (taught {len(ids)}x)" if len(ids) > 1 else ""
        lines.append(f"ids={ids_display}\tenglish_ratio={ratio:.2f}{note}")
        lines.append(f"  Q: {question}")
        lines.append(f"  A: {answer}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    memory_path = repo_root / "episodic_memory.sqlite3"
    flagged = find_flagged_pairs(memory_path)
    write_review_file(flagged, DEFAULT_OUTPUT_PATH)
    print(f"Flagged human_taught rows: {len(flagged)}")
    print(f"Written to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
