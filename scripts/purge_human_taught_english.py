"""Standalone script: apply a human review verdict to the flagged
human_taught pairs listed in analisis/human_taught_english_review.txt
(produced by analisis/export_human_taught_review.py).

Two pairs are rescued by rewriting ONLY their question (their answer is
left untouched) -- see RESCUES below. Every other row id listed in the
review file is deleted outright, including every duplicate id in its
"ids=" list: these are grid-era English aphorisms that got taught hundreds
of times each and, through _load_training_samples()'s pair-oversampling,
dominate a training run out of proportion to how many genuinely distinct
pairs they represent.

Run scripts/backup.py first -- this script does not do that for you.

Usage:
    python scripts/purge_human_taught_english.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REVIEW_FILE = Path(__file__).resolve().parent.parent / "analisis" / "human_taught_english_review.txt"

# (id, new_question) -- answer is left exactly as stored.
RESCUES = {
    17408: "que es hooked on a feeling",
    15983: "quien es claude",
}

_IDS_LINE_RE = re.compile(r"^ids=([\d,]+)\s")


def parse_review_file(path: Path) -> list[int]:
    """Returns every row id listed across every 'ids=' line in the review
    file, in file order, with duplicates preserved (a repeated id would
    indicate a bug in the review file, not something to silently collapse)."""
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _IDS_LINE_RE.match(line)
        if match:
            ids.extend(int(part) for part in match.group(1).split(","))
    return ids


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=repo_root / "episodic_memory.sqlite3")
    parser.add_argument("--review-file", type=Path, default=REVIEW_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing anything.")
    args = parser.parse_args()

    all_ids = parse_review_file(args.review_file)
    if len(all_ids) != len(set(all_ids)):
        seen = set()
        dupes = sorted({i for i in all_ids if i in seen or seen.add(i)})
        raise ValueError(f"review file lists duplicate ids across groups: {dupes}")

    missing_rescues = [rid for rid in RESCUES if rid not in all_ids]
    if missing_rescues:
        raise ValueError(f"rescued id(s) not found in review file: {missing_rescues}")

    delete_ids = [row_id for row_id in all_ids if row_id not in RESCUES]

    print(f"Review file groups (unique pairs listed): "
          f"{sum(1 for line in args.review_file.read_text(encoding='utf-8').splitlines() if line.startswith('ids='))}")
    print(f"Total row ids listed: {len(all_ids)}")
    print(f"Rescued (question rewritten, answer untouched): {sorted(RESCUES)}")
    print(f"Row ids to delete: {len(delete_ids)}")

    connection = sqlite3.connect(args.database)
    try:
        before_total = connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE source = 'human_taught'"
        ).fetchone()[0]
        before_unique = connection.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT question, answer FROM conversations "
            "WHERE source = 'human_taught')"
        ).fetchone()[0]
        print(f"human_taught rows before: {before_total} ({before_unique} unique pairs)")

        if args.dry_run:
            for row_id, new_question in RESCUES.items():
                row = connection.execute(
                    "SELECT question, answer FROM conversations WHERE id = ?", (row_id,)
                ).fetchone()
                print(f"  would rewrite id={row_id}: {row[0]!r} -> {new_question!r} (answer unchanged: {row[1]!r})")
            print("Dry run -- no rows changed or deleted.")
            return

        for row_id, new_question in RESCUES.items():
            cursor = connection.execute(
                "UPDATE conversations SET question = ? WHERE id = ? AND source = 'human_taught'",
                (new_question, row_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"expected exactly 1 row updated for id={row_id}, got {cursor.rowcount}")

        placeholders = ",".join("?" for _ in delete_ids)
        cursor = connection.execute(
            f"DELETE FROM conversations WHERE id IN ({placeholders}) AND source = 'human_taught'",
            delete_ids,
        )
        rows_deleted = cursor.rowcount
        connection.commit()

        after_total = connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE source = 'human_taught'"
        ).fetchone()[0]
        after_unique = connection.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT question, answer FROM conversations "
            "WHERE source = 'human_taught')"
        ).fetchone()[0]

        print(f"Rows deleted: {rows_deleted}")
        print(f"human_taught rows after: {after_total} ({after_unique} unique pairs)")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
