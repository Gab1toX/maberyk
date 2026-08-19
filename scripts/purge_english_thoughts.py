"""Standalone script: purge English LanguageEngine.express() template
thoughts from episodes.thought (brain/language.py, brain/memory.py).

These are the five hardcoded English sentences express() used to produce
before its English branch was removed (it now only emits Spanish). Rows
already stored in episodic_memory.sqlite3 before that fix still carry the
old English text, and brain/language_model.py's _load_training_samples()
reads this column directly (`WHERE thought != '' AND thought IS NOT NULL`),
so they keep re-entering every training run until cleared.

Only the `thought` TEXT column is cleared (set to '') on matching rows --
the rest of the episode (observation/action/outcome/surprise_level) is left
untouched. `thought` is write-only for every other consumer: neither
_row_to_episode() nor _episode_for_public_use() in brain/memory.py expose
it, so clearing it cannot affect curiosity/memory/similarity behavior. It
only removes these rows from _load_training_samples()'s selection.

Every matched (id, thought) pair is written to a backup file before the
UPDATE runs. Run scripts/backup.py first for a full-database snapshot --
this script does not do that for you.

Usage:
    python scripts/purge_english_thoughts.py [--dry-run]
    python scripts/purge_english_thoughts.py --database episodic_memory.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Matches exactly the five templates removed from LanguageEngine.express()
# in brain/language.py -- each SQL '%' stands in for the single filler word
# (object_name/color/emotion_word) the old template inserted at that spot.
_MATCH_CLAUSE = (
    "thought = 'I sense danger here' "
    "OR thought = 'I am moving and exploring' "
    "OR thought LIKE 'I touched % and something happened' "
    "OR thought LIKE 'I see a % % nearby' "
    "OR thought LIKE 'I feel % about this place'"
)

DEFAULT_BACKUP_PATH = (
    Path(__file__).resolve().parent.parent / "analisis" / "deleted_episode_thoughts_backup.txt"
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--database", type=Path, default=repo_root / "episodic_memory.sqlite3",
        help="Path to episodic_memory.sqlite3 (default: repo root).",
    )
    parser.add_argument(
        "--backup-file", type=Path, default=DEFAULT_BACKUP_PATH,
        help="Where to write the (id, thought) backup of matched rows.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report matches and write the backup file, but clear nothing.",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    try:
        before_total = connection.execute(
            "SELECT COUNT(*) FROM episodes WHERE thought != '' AND thought IS NOT NULL"
        ).fetchone()[0]

        matches = connection.execute(
            f"SELECT id, thought FROM episodes WHERE {_MATCH_CLAUSE}"
        ).fetchall()

        print(f"Non-empty episode thoughts before: {before_total}")
        print(f"Matching English-template thoughts: {len(matches)}")

        if not matches:
            print("Nothing to purge.")
            return

        args.backup_file.parent.mkdir(parents=True, exist_ok=True)
        with args.backup_file.open("w", encoding="utf-8") as handle:
            for row_id, thought in matches:
                handle.write(f"{row_id}\t{thought}\n")
        print(f"Backed up {len(matches)} matched rows to {args.backup_file}")

        if args.dry_run:
            print("Dry run -- no rows cleared.")
            return

        connection.execute(f"UPDATE episodes SET thought = '' WHERE {_MATCH_CLAUSE}")
        connection.commit()

        after_total = connection.execute(
            "SELECT COUNT(*) FROM episodes WHERE thought != '' AND thought IS NOT NULL"
        ).fetchone()[0]
        remaining_matches = connection.execute(
            f"SELECT COUNT(*) FROM episodes WHERE {_MATCH_CLAUSE}"
        ).fetchone()[0]
        print(f"Non-empty episode thoughts after: {after_total}")
        print(f"Remaining English-template matches: {remaining_matches}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
