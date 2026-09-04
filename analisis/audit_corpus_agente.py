"""Standalone script: read-only audit of corpus_agente.txt.

Never writes to corpus_agente.txt -- the only file this script writes is
analisis/corpus_agente_audit.txt, its own report output.

Locates corpus_agente.txt the same way brain/language_model.py's
LanguageModelTrainer._load_training_samples() does: as
`memory_path.parent / "corpus_agente.txt"`, where memory_path defaults to
repo_root / "episodic_memory.sqlite3" (see analisis/retrain_production_big.py
and analisis/audit_english.py, which resolve the same default). The sqlite
database itself is never opened here -- corpus_agente.txt's own content does
not depend on it, only its location does.

"Lines" throughout this audit means what _load_training_samples() actually
feeds to training: each raw line stripped, with blank lines and lines
starting with "#" (comments/section headers) excluded. Blank and comment line
counts are reported separately for transparency but excluded from every
other section.

The English-flagged section reuses audit_english.py's exact heuristic
(ENGLISH_STOPWORDS, 30% threshold, normalize_text() before scoring) so the
two reports stay comparable. Per corpus_agente.txt's own header comment
("Bilingüe español/inglés — igual que su voz natural"), some English content
here is by design, not necessarily contamination -- this script only reports
the flagged lines, it does not judge them.

Usage:
    python -m analisis.audit_corpus_agente
    python -m analisis.audit_corpus_agente --corpus corpus_agente.txt --top 40
"""

from __future__ import annotations

import argparse
import re
import statistics
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
ENGLISH_FLAG_THRESHOLD = 0.30

IDENTITY_MARKERS = ("maberyk", "gabito", "yo", "mi", "me", "soy")
GRID_WORLD_MARKERS = (
    "grid", "zona", "objeto", "lampara", "cristal", "portal", "piedra", "campana",
)

SAMPLE_SIZE = 40
SAMPLE_SEED = 42
TOP_PHRASES_N = 40

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "corpus_agente_audit.txt"


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _english_ratio(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    return hits / len(words)


def _load_lines(corpus_path: Path) -> tuple[list[str], int, int]:
    """Returns (content_lines, blank_count, comment_count), mirroring
    _load_training_samples()'s strip/blank/comment filter exactly."""
    raw_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    content_lines: list[str] = []
    blank_count = 0
    comment_count = 0
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            blank_count += 1
        elif line.startswith("#"):
            comment_count += 1
        else:
            content_lines.append(line)
    return content_lines, blank_count, comment_count


def _length_stats(content_lines: list[str]) -> dict:
    lengths = [len(line.split()) for line in content_lines]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
    }


def _qa_breakdown(content_lines: list[str]) -> dict:
    has_q = sum(1 for line in content_lines if "<q>" in line)
    has_a = sum(1 for line in content_lines if "<a>" in line)
    has_both = sum(1 for line in content_lines if "<q>" in line and "<a>" in line)
    standalone = sum(1 for line in content_lines if "<q>" not in line and "<a>" not in line)
    return {"has_q": has_q, "has_a": has_a, "has_both": has_both, "standalone": standalone}


def _top_opening_phrases(content_lines: list[str], top_n: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for line in content_lines:
        tokens = line.lower().split()[:3]
        if tokens:
            counter[" ".join(tokens)] += 1
    return counter.most_common(top_n)


def _marker_counts(content_lines: list[str], markers: tuple[str, ...]) -> tuple[dict, int]:
    """Word-boundary matches (via normalize_text + word tokenization, same
    as what the word-level tokenizer path actually sees) -- not substring
    matches, so e.g. "mi" won't fire on "camino"."""
    per_marker: dict[str, int] = {marker: 0 for marker in markers}
    any_marker_count = 0
    marker_set = set(markers)
    for line in content_lines:
        words = set(_words(normalize_text(line)))
        hits = marker_set & words
        for marker in hits:
            per_marker[marker] += 1
        if hits:
            any_marker_count += 1
    return per_marker, any_marker_count


def _english_flagged(content_lines: list[str]) -> list[tuple[str, float]]:
    flagged = []
    for line in content_lines:
        ratio = _english_ratio(normalize_text(line))
        if ratio > ENGLISH_FLAG_THRESHOLD:
            flagged.append((line, ratio))
    return flagged


def run_audit(corpus_path: Path, sample_size: int, sample_seed: int, top_n: int) -> dict:
    import random

    content_lines, blank_count, comment_count = _load_lines(corpus_path)
    unique_lines = list(dict.fromkeys(content_lines))
    duplicate_count = len(content_lines) - len(unique_lines)

    length_stats = _length_stats(content_lines)
    qa_breakdown = _qa_breakdown(content_lines)
    top_phrases = _top_opening_phrases(content_lines, top_n)
    identity_per_marker, identity_any = _marker_counts(content_lines, IDENTITY_MARKERS)
    grid_per_marker, grid_any = _marker_counts(content_lines, GRID_WORLD_MARKERS)

    sample_n = min(sample_size, len(content_lines))
    sample = random.Random(sample_seed).sample(content_lines, sample_n) if sample_n else []

    english_flagged = _english_flagged(content_lines)

    return {
        "raw_line_count": blank_count + comment_count + len(content_lines),
        "blank_count": blank_count,
        "comment_count": comment_count,
        "content_count": len(content_lines),
        "unique_count": len(unique_lines),
        "duplicate_count": duplicate_count,
        "length_stats": length_stats,
        "qa_breakdown": qa_breakdown,
        "top_phrases": top_phrases,
        "identity_per_marker": identity_per_marker,
        "identity_any": identity_any,
        "grid_per_marker": grid_per_marker,
        "grid_any": grid_any,
        "sample": sample,
        "sample_seed": sample_seed,
        "english_flagged": english_flagged,
    }


def _render_report(result: dict, corpus_path: Path) -> str:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit("=== corpus_agente.txt audit ===")
    emit(f"corpus: {corpus_path}")
    emit(
        "NOTE: this file's own header claims it is deliberately bilingual "
        "(\"Bilingüe español/inglés — igual que su voz natural\"), so English "
        "content below is not automatically contamination."
    )
    emit()

    emit("-- Line counts --")
    emit(f"raw lines in file:      {result['raw_line_count']}")
    emit(f"blank lines:             {result['blank_count']}")
    emit(f"comment lines (#):       {result['comment_count']}")
    emit(f"content lines (trained): {result['content_count']}")
    emit(f"unique content lines:    {result['unique_count']}")
    emit(f"exact-duplicate lines:   {result['duplicate_count']}")
    emit()

    stats = result["length_stats"]
    emit("-- Line length in words (content lines) --")
    emit(f"min:    {stats['min']}")
    emit(f"max:    {stats['max']}")
    emit(f"mean:   {stats['mean']:.2f}")
    emit(f"median: {stats['median']:.2f}")
    emit()

    qa = result["qa_breakdown"]
    emit("-- Q/A pairs vs standalone --")
    emit(f"lines containing <q>:      {qa['has_q']}")
    emit(f"lines containing <a>:      {qa['has_a']}")
    emit(f"lines containing both:     {qa['has_both']}")
    emit(f"standalone (neither):      {qa['standalone']}")
    emit()

    emit(f"-- Top {len(result['top_phrases'])} opening phrases (first 3 words) --")
    for rank, (phrase, count) in enumerate(result["top_phrases"], start=1):
        emit(f"  {rank:>2}. [{count:>4}x] {phrase!r}")
    emit()

    emit("-- Identity / first-person markers --")
    emit(f"any of {IDENTITY_MARKERS}: {result['identity_any']} lines")
    for marker in IDENTITY_MARKERS:
        emit(f"  {marker:<10} {result['identity_per_marker'][marker]}")
    emit()

    emit("-- Grid-world vocabulary --")
    emit(f"any of {GRID_WORLD_MARKERS}: {result['grid_any']} lines")
    for marker in GRID_WORLD_MARKERS:
        emit(f"  {marker:<10} {result['grid_per_marker'][marker]}")
    emit()

    emit(f"-- Random sample of {len(result['sample'])} lines (seed={result['sample_seed']}) --")
    for line in result["sample"]:
        emit(f"  {line}")
    emit()

    flagged = result["english_flagged"]
    emit(f"-- English-flagged lines (>{ENGLISH_FLAG_THRESHOLD:.0%} stopwords): {len(flagged)} --")
    for line, ratio in flagged:
        emit(f"  [{ratio:.0%}] {line}")
    emit()

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus", type=Path,
        default=REPO_ROOT / "episodic_memory.sqlite3",
        help=(
            "Path to episodic_memory.sqlite3, whose parent directory is used to "
            "locate corpus_agente.txt (same resolution as _load_training_samples()). "
            "Pass corpus_agente.txt directly instead if you want to point at a "
            "different file."
        ),
    )
    parser.add_argument(
        "--top", type=int, default=TOP_PHRASES_N,
        help=f"How many opening phrases to report (default {TOP_PHRASES_N}).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE,
        help=f"Random sample size (default {SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SAMPLE_SEED,
        help=f"Seed for the random sample (default {SAMPLE_SEED}).",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Where to write the report (default: analisis/corpus_agente_audit.txt).",
    )
    args = parser.parse_args()

    corpus_path = args.corpus
    if corpus_path.suffix == ".sqlite3":
        corpus_path = corpus_path.parent / "corpus_agente.txt"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus_agente.txt not found at {corpus_path}")

    result = run_audit(corpus_path, args.sample_size, args.seed, args.top)
    report = _render_report(result, corpus_path)

    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
