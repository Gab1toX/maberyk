"""Standalone script: read-only audit for English-language leakage across
every corpus the language model draws from.

Never writes to episodic_memory.sqlite3 (opened via a SQLite read-only URI,
so a write would raise rather than silently succeed) or to corpus_agente.txt
-- the only file this script writes is analisis/english_samples.txt, its own
report output.

Scope, and where it differs from brain/language_model.py's
LanguageModelTrainer._load_training_samples():

  - That method only ever queries `conversations` rows WHERE source =
    'human_taught' -- despite CLAUDE.md's source-rules table marking
    tutor_approved as training data too, the current code does not select
    it. This script surfaces that gap instead of hiding it: it scans
    human_taught, tutor_approved, agent_generated, and unknown (the four
    conversation sources named in the audit request) so all of them are
    visible, and the report says plainly which ones _load_training_samples()
    actually uses today. `retrieved` (a duplicate echo of a human_taught
    answer) and `voice` (the external LLM's own words, which must never
    reach training per CLAUDE.md) are deliberately excluded -- neither was
    named in the request and voice must never be treated as a candidate.

  - It also scans episode thoughts (brain/memory.py's `episodes.thought`)
    and corpus_agente.txt, matching _load_training_samples()'s two
    fluency sources.

  - Every source is deduplicated independently here (so the per-source
    breakdown stays interpretable). _load_training_samples() itself merges
    and dedupes thought+corpus samples together and caps thoughts at 2x the
    pair count before a training run -- this audit intentionally looks at
    the full, uncapped picture instead, since the point is to see every
    sample that could reach training, not just one run's sampled subset.

Each sample is flagged "likely-English" when more than 30% of its words
(whitespace/punctuation split, lowercased) are in a small hardcoded English
stopword set. This is a coarse lexical heuristic, not a language
classifier: short samples and genuinely mixed-language ones can be flagged
either way. Text is scanned through the same normalize_text() the word
tokenizer trains on (accent-stripped, lowercased) so the ratio matches what
the model actually sees.

Frequency ranking ("most common flagged samples") counts RAW occurrences
before per-source dedup -- episode thoughts in particular are known to
collapse from hundreds of thousands of rows to a much smaller unique set
(see CLAUDE.md's "measure signal, not volume"), and that raw count is what
shows which specific English strings were produced or taught most often,
information the deduped set alone can't show.

Usage:
    python -m analisis.audit_english
    python -m analisis.audit_english --memory episodic_memory.sqlite3 --top 30
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import normalize_text

ENGLISH_STOPWORDS = frozenset((
    "the", "of", "and", "to", "in", "that", "is", "it", "you", "for",
    "with", "this", "what", "why", "does", "happen", "are", "was", "i",
    "my", "me", "a", "an", "on", "at", "be", "have",
))

FLAG_THRESHOLD = 0.30
DEFAULT_TOP_N = 30

# Conversation sources this audit looks at -- deliberately excludes
# 'retrieved' (a duplicate echo of a human_taught answer, not new text) and
# 'voice' (the external LLM's own words, which CLAUDE.md says must never be
# treated as training data), since neither was asked for and 'voice' must
# never be implied as a training candidate.
CONVERSATION_SOURCES = ("human_taught", "tutor_approved", "agent_generated", "unknown")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

OUTPUT_PATH = Path(__file__).resolve().parent / "english_samples.txt"


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _english_ratio(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    return hits / len(words)


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise FileNotFoundError(f"no such database: {database_path}")
    uri = f"file:{database_path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


class SourceScan:
    """Raw rows for one source tag, formatted the same way training would
    see them, before this audit's own per-source dedup is applied."""

    def __init__(self, tag: str, raw_texts: list[str]) -> None:
        self.tag = tag
        self.raw_texts = raw_texts

    @property
    def unique_texts(self) -> list[str]:
        return list(dict.fromkeys(self.raw_texts))


def _scan_conversations(connection: sqlite3.Connection) -> list[SourceScan]:
    scans = []
    for source in CONVERSATION_SOURCES:
        rows = connection.execute(
            "SELECT question, answer FROM conversations WHERE source = ?", (source,)
        ).fetchall()
        texts = [
            f"{normalize_text(question)} {normalize_text(answer)}"
            for question, answer in rows
            if question and question.strip() and answer and answer.strip()
        ]
        scans.append(SourceScan(source, texts))
    return scans


def _scan_episode_thoughts(connection: sqlite3.Connection) -> SourceScan:
    rows = connection.execute(
        "SELECT thought FROM episodes WHERE thought != '' AND thought IS NOT NULL"
    ).fetchall()
    texts = [normalize_text(row[0].strip()) for row in rows if row[0] and row[0].strip()]
    return SourceScan("episode_thought", texts)


def _scan_corpus_file(corpus_path: Path) -> SourceScan:
    if not corpus_path.is_file():
        return SourceScan("corpus_agente", [])
    texts = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            texts.append(normalize_text(line))
    return SourceScan("corpus_agente", texts)


def run_audit(memory_path: Path, corpus_path: Path) -> dict:
    connection = _read_only_connection(memory_path)
    try:
        scans = _scan_conversations(connection)
        scans.append(_scan_episode_thoughts(connection))
    finally:
        connection.close()
    scans.append(_scan_corpus_file(corpus_path))

    # Raw (pre-dedup) frequency, across every source, for the "most common
    # flagged samples" ranking -- keyed by (tag, text) so identical text
    # from two different sources is still counted and reported separately.
    raw_counts: Counter[tuple[str, str]] = Counter()
    for scan in scans:
        for text in scan.raw_texts:
            raw_counts[(scan.tag, text)] += 1

    per_source: dict[str, dict] = {}
    total_scanned = 0
    total_flagged = 0
    flagged_rows: list[tuple[str, str, float, int]] = []  # tag, text, ratio, raw_count

    for scan in scans:
        unique_texts = scan.unique_texts
        flagged_count = 0
        for text in unique_texts:
            ratio = _english_ratio(text)
            if ratio > FLAG_THRESHOLD:
                flagged_count += 1
                flagged_rows.append((scan.tag, text, ratio, raw_counts[(scan.tag, text)]))
        per_source[scan.tag] = {
            "scanned": len(unique_texts),
            "flagged": flagged_count,
        }
        total_scanned += len(unique_texts)
        total_flagged += flagged_count

    flagged_rows.sort(key=lambda row: row[3], reverse=True)

    return {
        "total_scanned": total_scanned,
        "total_flagged": total_flagged,
        "per_source": per_source,
        "flagged_rows": flagged_rows,
    }


def _print_report(result: dict, memory_path: Path, corpus_path: Path, top_n: int) -> None:
    total_scanned = result["total_scanned"]
    total_flagged = result["total_flagged"]
    percentage = (100.0 * total_flagged / total_scanned) if total_scanned else 0.0

    print("=== English leakage audit ===")
    print(f"memory: {memory_path}")
    print(f"corpus: {corpus_path}")
    print(
        "NOTE: brain/language_model.py's _load_training_samples() only queries "
        "source='human_taught' today -- tutor_approved/agent_generated/unknown "
        "conversations are scanned here for visibility but are NOT currently fed "
        "to training."
    )
    print()
    print(f"Total samples scanned: {total_scanned}")
    print(f"Flagged likely-English (>{FLAG_THRESHOLD:.0%} stopwords): {total_flagged} ({percentage:.1f}%)")
    print()
    print("By source tag (unique, deduped within each source):")
    for tag, counts in result["per_source"].items():
        scanned = counts["scanned"]
        flagged = counts["flagged"]
        pct = (100.0 * flagged / scanned) if scanned else 0.0
        print(f"  {tag:<16} scanned={scanned:<8} flagged={flagged:<8} ({pct:.1f}%)")
    print()

    flagged_rows = result["flagged_rows"]
    print(f"Top {min(top_n, len(flagged_rows))} flagged samples by raw occurrence count:")
    for rank, (tag, text, ratio, raw_count) in enumerate(flagged_rows[:top_n], start=1):
        preview = text if len(text) <= 100 else text[:97] + "..."
        print(f"  {rank:>2}. [{raw_count}x] ({tag}, {ratio:.0%} english) {preview!r}")
    print()


def _write_flagged_list(result: dict, output_path: Path) -> None:
    lines = [
        f"raw_count={raw_count}\tsource={tag}\tenglish_ratio={ratio:.2f}\t{text}"
        for tag, text, ratio, raw_count in result["flagged_rows"]
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Full flagged list ({len(lines)} lines) written to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--memory", type=Path, default=repo_root / "episodic_memory.sqlite3",
        help="Path to episodic_memory.sqlite3 (default: repo root).",
    )
    parser.add_argument(
        "--corpus", type=Path, default=repo_root / "corpus_agente.txt",
        help="Path to corpus_agente.txt (default: repo root).",
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP_N,
        help=f"How many flagged samples to print inline (default {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Where to write the full flagged list (default: analisis/english_samples.txt).",
    )
    args = parser.parse_args()

    result = run_audit(args.memory, args.corpus)
    _print_report(result, args.memory, args.corpus, args.top)
    _write_flagged_list(result, args.output)


if __name__ == "__main__":
    main()
