"""Standalone script: prune corpus_agente.txt in four ordered passes.

Run scripts/backup.py first -- this script does not do that for you (it was
run separately before this script, see backups/2026-08-21_1058).

Writes the untouched original to corpus_agente_original.txt (verbatim copy,
taken before any pass runs) and overwrites corpus_agente.txt with the pruned
result. Structural lines (blank lines and "#" comments/section headers) are
never touched by pruning and pass through unchanged, except the header's
bilingualism line, which is rewritten (see HEADER_OLD/HEADER_NEW below) --
MABERYK_VISION.md states Maberyk's primary language is Spanish, and English
was already purged from every other corpus source in a prior session.

Passes, applied strictly in this order (each pass only sees lines that
survived every earlier pass):

  1. English-flagged lines -- reuses analisis/audit_corpus_agente.py's exact
     heuristic (ENGLISH_STOPWORDS, >30% threshold, scored on normalize_text()
     output) so the two reports stay comparable.

  2. Grid-world lines -- any line containing a whole word from
     GRID_WORLD_MARKERS (word-boundary match via normalize_text(), same as
     the audit script), removed because Maberyk no longer lives in the grid
     world and these lines produce confused answers about objects that no
     longer exist for him.

  3. Template throttling -- recomputed on what's left after passes 1-2: the
     40 most frequent opening phrases (first 3 words, lowercased) each keep
     at most 10 lines. Which 10 survive is chosen by greedy farthest-first
     selection over Jaccard distance between lines' word sets (normalize_text
     + word tokenization), seeded from the first line in file order, so the
     kept set maximizes minimum pairwise content distance rather than just
     being "the first 10 in the file."

  4. Exact duplicates -- recomputed on what's left after passes 1-3; first
     occurrence in file order is kept, later exact repeats are dropped.

Usage:
    python scripts/prune_corpus_agente.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import normalize_text

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "corpus_agente.txt"
ORIGINAL_PATH = REPO_ROOT / "corpus_agente_original.txt"

ENGLISH_STOPWORDS = frozenset((
    "the", "of", "and", "to", "in", "that", "is", "it", "you", "for",
    "with", "this", "what", "why", "does", "happen", "are", "was", "i",
    "my", "me", "a", "an", "on", "at", "be", "have",
))
ENGLISH_FLAG_THRESHOLD = 0.30

GRID_WORLD_MARKERS = frozenset((
    "grid", "zona", "objeto", "lampara", "cristal", "portal", "piedra", "campana",
))

TOP_PHRASES_N = 40
KEEP_PER_PHRASE = 10
REPORT_TOP_N = 20

HEADER_OLD = "# Bilingüe español/inglés — igual que su voz natural"
HEADER_NEW = (
    "# Español unicamente -- bilinguismo removido el 2026-08-21 "
    "(MABERYK_VISION.md: el idioma principal de Maberyk es el espanol)"
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _word_set(line: str) -> set[str]:
    return set(_words(normalize_text(line)))


def _english_ratio(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    return hits / len(words)


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _select_diverse(items: list[tuple[int, str]], k: int) -> list[tuple[int, str]]:
    """Greedy farthest-first selection over Jaccard word-set distance.
    Seeded from items[0] (first in file order) for determinism, then
    repeatedly adds whichever remaining item maximizes the minimum distance
    to everything already selected. Returns the kept items in original file
    order, not selection order."""
    if len(items) <= k:
        return list(items)

    word_sets = [_word_set(text) for _, text in items]
    selected = [0]
    remaining = list(range(1, len(items)))
    while len(selected) < k and remaining:
        best_idx = None
        best_dist = -1.0
        for i in remaining:
            dist = min(_jaccard_distance(word_sets[i], word_sets[s]) for s in selected)
            if dist > best_dist:
                best_dist = dist
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    selected_sorted = sorted(selected)
    return [items[i] for i in selected_sorted]


def _opening_phrase(text: str) -> str:
    tokens = text.lower().split()[:3]
    return " ".join(tokens)


def _top_phrases(items: list[tuple[int, str]], top_n: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for _, text in items:
        phrase = _opening_phrase(text)
        if phrase:
            counter[phrase] += 1
    return counter.most_common(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing anything.")
    args = parser.parse_args()

    raw_lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()

    structural: dict[int, str] = {}
    content: list[tuple[int, str]] = []
    for idx, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            structural[idx] = raw_line
        else:
            content.append((idx, line))

    lines_before = len(content)

    # -- Pass 1: English-flagged --
    remaining: list[tuple[int, str]] = []
    english_removed: list[tuple[int, str]] = []
    for idx, text in content:
        if _english_ratio(normalize_text(text)) > ENGLISH_FLAG_THRESHOLD:
            english_removed.append((idx, text))
        else:
            remaining.append((idx, text))

    # -- Pass 2: grid-world --
    after_grid: list[tuple[int, str]] = []
    grid_removed: list[tuple[int, str]] = []
    for idx, text in remaining:
        if _word_set(text) & GRID_WORLD_MARKERS:
            grid_removed.append((idx, text))
        else:
            after_grid.append((idx, text))

    # -- Pass 3: template throttling --
    groups: dict[str, list[tuple[int, str]]] = {}
    for idx, text in after_grid:
        groups.setdefault(_opening_phrase(text), []).append((idx, text))

    top_phrases = _top_phrases(after_grid, TOP_PHRASES_N)
    throttled_ids: set[int] = set()
    throttle_report: list[tuple[str, int, int, int]] = []  # phrase, before, kept, removed
    for phrase, count_before in top_phrases:
        group = groups[phrase]
        kept = _select_diverse(group, KEEP_PER_PHRASE)
        kept_ids = {idx for idx, _ in kept}
        removed_count = len(group) - len(kept)
        for idx, _ in group:
            if idx not in kept_ids:
                throttled_ids.add(idx)
        throttle_report.append((phrase, count_before, len(kept), removed_count))

    after_throttle = [(idx, text) for idx, text in after_grid if idx not in throttled_ids]
    throttle_removed_total = sum(r[3] for r in throttle_report)

    # -- Pass 4: exact duplicates --
    seen: set[str] = set()
    after_dedup: list[tuple[int, str]] = []
    duplicate_removed: list[tuple[int, str]] = []
    for idx, text in after_throttle:
        if text in seen:
            duplicate_removed.append((idx, text))
        else:
            seen.add(text)
            after_dedup.append((idx, text))

    lines_after = len(after_dedup)
    survivor_text = {idx: text for idx, text in after_dedup}

    final_top20 = _top_phrases(after_dedup, REPORT_TOP_N)

    # -- Report --
    print("=== corpus_agente.txt prune report ===")
    print(f"lines before: {lines_before}")
    print()
    print("Removed per category (in application order):")
    print(f"  1. English-flagged:     {len(english_removed)}")
    print(f"  2. Grid-world:          {len(grid_removed)}")
    print(f"  3. Template throttling: {throttle_removed_total}")
    print(f"  4. Exact duplicates:    {len(duplicate_removed)}")
    total_removed = len(english_removed) + len(grid_removed) + throttle_removed_total + len(duplicate_removed)
    print(f"  total removed:          {total_removed}")
    print()
    print(f"lines after: {lines_after}")
    print()

    print(f"-- Pass 3 detail: template throttling, per phrase (top {TOP_PHRASES_N}) --")
    for phrase, count_before, kept, removed_count in throttle_report:
        print(f"  {phrase!r:<30} before={count_before:<4} kept={kept:<3} removed={removed_count}")
    print()

    print(f"-- New top {REPORT_TOP_N} opening phrases (post-prune) --")
    for rank, (phrase, count) in enumerate(final_top20, start=1):
        print(f"  {rank:>2}. [{count:>4}x] {phrase!r}")
    print()

    if args.dry_run:
        print("Dry run -- no files written.")
        return

    ORIGINAL_PATH.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"Original preserved at {ORIGINAL_PATH}")

    output_lines: list[str] = []
    for idx, raw_line in enumerate(raw_lines):
        if idx in structural:
            stored = structural[idx]
            if stored.strip() == HEADER_OLD:
                output_lines.append(HEADER_NEW)
            else:
                output_lines.append(stored)
        elif idx in survivor_text:
            output_lines.append(survivor_text[idx])
        # else: removed content line, dropped entirely

    CORPUS_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Pruned corpus written to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
