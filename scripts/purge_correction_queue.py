"""Standalone script: purge stale 'rejected' entries from the batch tutor
correction queue (brain/correction_queue.py, table correction_queue inside
episodic_memory.sqlite3).

A rejected entry can mean two very different things: the tutor judged the
raw answer's quality and declined it (real signal), or the Groq API call
itself failed -- a deprecated model, a network error, or the reasoning-token
bug that let a hidden chain-of-thought consume the whole max_tokens budget
and leave content empty -- and got recorded as a rejection anyway (noise
that poisons queue stats). This script only ever deletes status='rejected'
rows older than a cutoff; 'pending' and 'approved' entries are never
touched, no matter how old.

Usage:
    python scripts/purge_correction_queue.py --before "2026-08-18" [--dry-run]
    python scripts/purge_correction_queue.py --before 1755500000 --database episodic_memory.sqlite3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.correction_queue import CorrectionQueue


def parse_cutoff(value: str) -> float:
    """Accepts either a unix timestamp or an ISO date/datetime string."""
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--before",
        required=True,
        help="Purge rejected entries created before this unix timestamp or ISO date/datetime.",
    )
    parser.add_argument(
        "--database",
        default="episodic_memory.sqlite3",
        type=Path,
        help="Path to the episodic memory SQLite file (default: episodic_memory.sqlite3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be purged without deleting anything.",
    )
    args = parser.parse_args()

    cutoff = parse_cutoff(args.before)
    cutoff_display = datetime.fromtimestamp(cutoff)

    queue = CorrectionQueue(args.database)
    try:
        if args.dry_run:
            count = queue.connection.execute(
                "SELECT COUNT(*) FROM correction_queue WHERE status = 'rejected' AND created_at < ?",
                (cutoff,),
            ).fetchone()[0]
            print(f"Would purge {count} rejected entries created before {cutoff_display}.")
        else:
            purged = queue.purge_rejected_before(cutoff)
            print(f"Purged {purged} rejected entries created before {cutoff_display}.")

        print("Current queue stats:", queue.stats())
    finally:
        queue.close()


if __name__ == "__main__":
    main()
