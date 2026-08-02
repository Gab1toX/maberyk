"""Standalone script: snapshot Maberyk's full state to a timestamped backup folder.

Copies agent_state.pt, language_model.pt, and corpus_agente.txt directly, and
backs up episodic_memory.sqlite3 / permissions.sqlite3 through the SQLite
online backup API (sqlite3.Connection.backup) from a read-only source
connection, so the snapshot stays consistent even while the agent is
running and writing to those databases.

Usage:
    python scripts/backup.py [--root PATH] [--keep N] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

FILES_TO_COPY = ("agent_state.pt", "language_model.pt", "corpus_agente.txt")
SQLITE_FILES = ("episodic_memory.sqlite3", "permissions.sqlite3")
BACKUP_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")


def git_commit_hash(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_sqlite(source_path: Path, dest_path: Path) -> None:
    source_uri = source_path.resolve().as_uri() + "?mode=ro"
    source_conn = sqlite3.connect(source_uri, uri=True)
    try:
        dest_conn = sqlite3.connect(dest_path)
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        source_conn.close()


def integrity_check(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [row[0] for row in conn.execute("PRAGMA integrity_check").fetchall()]
    except sqlite3.DatabaseError as exc:
        return [str(exc)]
    finally:
        conn.close()


def conversation_stats(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversations'"
        ).fetchone()
        if not has_table:
            return {}
        totals_by_source = dict(
            conn.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source")
        )
        unique_pairs_by_source = dict(
            conn.execute(
                """
                SELECT source, COUNT(*) FROM (
                    SELECT DISTINCT source, question, answer FROM conversations
                )
                GROUP BY source
                """
            )
        )
        return {
            "conversation_totals_by_source": totals_by_source,
            "unique_pairs_by_source": unique_pairs_by_source,
        }
    finally:
        conn.close()


def prune_old_backups(out_root: Path, keep: int) -> None:
    candidates = sorted(
        p for p in out_root.iterdir() if p.is_dir() and BACKUP_DIR_PATTERN.match(p.name)
    )
    excess = len(candidates) - keep
    for old_dir in candidates[:max(excess, 0)]:
        shutil.rmtree(old_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot Maberyk's full state to a timestamped backup folder."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="Repository root containing agent_state.pt, episodic_memory.sqlite3, etc.",
    )
    parser.add_argument("--keep", type=int, default=10, help="Number of backups to retain.")
    parser.add_argument(
        "--out", type=Path, default=Path("backups"),
        help="Backup output directory (relative paths are resolved against --root).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_root = args.out if args.out.is_absolute() else root / args.out
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H%M")
    backup_dir = out_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "timestamp": timestamp,
        "git_commit": git_commit_hash(root),
        "files": {},
    }

    for name in FILES_TO_COPY:
        source = root / name
        if not source.exists():
            print(f"WARNING: {name} not found at {source}, skipping.", file=sys.stderr)
            continue
        dest = backup_dir / name
        shutil.copy2(source, dest)
        manifest["files"][name] = {
            "size_bytes": dest.stat().st_size,
            "sha256": sha256_of(dest),
        }

    for name in SQLITE_FILES:
        source = root / name
        if not source.exists():
            print(f"WARNING: {name} not found at {source}, skipping.", file=sys.stderr)
            continue
        dest = backup_dir / name
        backup_sqlite(source, dest)

        check_result = integrity_check(dest)
        if check_result != ["ok"]:
            print(
                f"ERROR: integrity check failed for {name}: {'; '.join(check_result)}",
                file=sys.stderr,
            )
            return 1

        entry = {
            "size_bytes": dest.stat().st_size,
            "sha256": sha256_of(dest),
        }
        entry.update(conversation_stats(dest))
        manifest["files"][name] = entry

    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    prune_old_backups(out_root, args.keep)

    print(f"Backup written to {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
