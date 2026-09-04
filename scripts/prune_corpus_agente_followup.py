"""Standalone script: follow-up pass on corpus_agente.txt, removing English
content that survived the previous prune (scripts/prune_corpus_agente.py).

Run scripts/backup.py first -- this script does not do that for you.

Two passes, applied in order (pass 2 only sees lines that survived pass 1):

  1. Strict English test -- flags a line if 3+ of its words are BOTH (a) a
     known English word AND (b) NOT a known Spanish word, per the offline
     word-frequency lists bundled with the `pyspellchecker` package (pure
     data, no model/embeddings/pretrained weights -- installed solely for
     this corpus-hygiene script, never imported by brain/). Words are
     compared through normalize_text() (lowercase, accent-stripped) so the
     check matches what the word-tokenizer path actually sees.

     The task that requested this pass suggested language_model.pt's own
     tokenizer_words as the Spanish reference instead. That was tried first
     and abandoned: language_model.pt was trained on the corpus BEFORE any
     of this cleanup, so its vocab already contains "crystal", "glows",
     "wanderer" and every other English word that ever appeared in a
     training line -- checking "not in that vocab" flagged nothing at all,
     since the contamination we're trying to remove is baked into the very
     vocab we'd be using to detect it. pyspellchecker's separate en/es
     dictionaries have no such circularity, which is what "or a wordlist if
     that is cleaner" (the task's own fallback) was for.

     A second problem showed up with the bare two-dictionary test: pyspell-
     checker's Spanish dictionary is missing very ordinary words -- "dice",
     "personas", "aves", "libero" (accent-stripped "libero") are all absent
     from it -- while its English dictionary is broad enough to contain
     common Latin American proper nouns and loanwords (karol, panama,
     bolivar, garcia, marquez, rodriguez, condor, reggae, rio, g, of...).
     Combined, that flagged genuinely Spanish sentences about real-world
     topics ("gabriel garcia marquez gano el premio nobel de literatura",
     "karol g es una cantante paisa de musica urbana") as predominantly
     English. SPANISH_FUNCTION_WORDS -- a hardcoded list of ~130 of the most
     common Spanish grammatical words (articles, prepositions, pronouns,
     conjunctions, ser/estar/haber/tener forms) -- gates the check: a line
     with 2+ of these is treated as Spanish outright and skips the strict
     test entirely, regardless of what its content words look like. Every
     false positive found in manual review was a sentence carrying several
     such function words; every genuine English line carried zero or one.

  2. Grid-world markers, English spellings -- removes any remaining line
     containing a whole-word match for crystal/lamp/bell/switch/stone/
     mirror/portal/wanderer/plant/zone/boundary, or a substring match for
     the compound phrases "chaos zone"/"rest zone"/"knowledge zone"/
     "danger zone" (each already implied by the standalone "zone" marker,
     checked anyway for literal completeness). Same rationale as the
     Spanish-spelling grid-world pass in scripts/prune_corpus_agente.py:
     Maberyk no longer lives in the grid world.

Every line removed by either pass is printed before deletion, tagged with
which pass and which word(s) triggered it.

Usage:
    python scripts/prune_corpus_agente_followup.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import normalize_text

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "corpus_agente.txt"

STRICT_ENGLISH_MIN_WORDS = 3
SPANISH_FUNCTION_GATE = 2

# Common Spanish grammatical words (articles, prepositions, conjunctions,
# pronouns, ser/estar/haber/tener forms) -- accent-stripped to match
# normalize_text() output. Presence of SPANISH_FUNCTION_GATE or more of
# these in a line is treated as decisive: the line is Spanish, regardless
# of what its content words look like to the en/es dictionaries.
SPANISH_FUNCTION_WORDS = frozenset("""
que de el la en y a los del se las por un para con no una su al lo como mas
pero sus le ya o este si porque esta entre cuando muy sin sobre tambien me
hasta hay donde quien desde todo nos durante todos uno les ni contra otros
ese eso ante ellos esto mi antes algunos unos yo otro otras otra tanto esa
estos mucho quienes nada muchos cual poco ella estar estas algunas algo
nosotros mis tu te ti tus ellas nosotras vosotros vosotras os soy eres es
somos son era fue fueron he has ha hemos han tengo tiene tienen habia
seria seran fuimos eramos estaba estan estoy esta este estos estas ese
esos aquel aquella aquellos aquellas cada cual cuales cuyo cuya cuyos cuyas
""".split())

SINGLE_WORD_MARKERS = frozenset((
    "crystal", "lamp", "bell", "switch", "stone", "mirror", "portal",
    "wanderer", "plant", "zone", "boundary",
))
COMPOUND_MARKERS = ("chaos zone", "rest zone", "knowledge zone", "danger zone")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _load_wordlists() -> tuple[set[str], set[str]]:
    from spellchecker import SpellChecker

    english_words = set(SpellChecker(language="en").word_frequency.dictionary.keys())
    spanish_words = set(SpellChecker(language="es").word_frequency.dictionary.keys())
    return english_words, spanish_words


def _strict_english_hits(text: str, english_words: set[str], spanish_words: set[str]) -> list[str]:
    normalized = normalize_text(text)
    tokens = _words(normalized)
    function_word_hits = sum(1 for word in tokens if word in SPANISH_FUNCTION_WORDS)
    if function_word_hits >= SPANISH_FUNCTION_GATE:
        return []  # gated: enough Spanish grammar present to call this Spanish outright
    return [word for word in tokens if word in english_words and word not in spanish_words]


def _grid_english_hits(text: str) -> list[str]:
    normalized = normalize_text(text)
    hits = [word for word in _words(normalized) if word in SINGLE_WORD_MARKERS]
    hits.extend(phrase for phrase in COMPOUND_MARKERS if phrase in normalized)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report and print, but don't write anything.")
    args = parser.parse_args()

    print("Loading English/Spanish wordlists from pyspellchecker...")
    english_words, spanish_words = _load_wordlists()
    print(f"  english: {len(english_words)} words, spanish: {len(spanish_words)} words")
    print()

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

    # -- Pass 1: strict English --
    remaining: list[tuple[int, str]] = []
    strict_removed: list[tuple[int, str, list[str]]] = []
    for idx, text in content:
        hits = _strict_english_hits(text, english_words, spanish_words)
        if len(hits) >= STRICT_ENGLISH_MIN_WORDS:
            strict_removed.append((idx, text, hits))
        else:
            remaining.append((idx, text))

    print(f"-- Pass 1: strict English (>={STRICT_ENGLISH_MIN_WORDS} flagged words) --")
    for idx, text, hits in strict_removed:
        print(f"  DELETE [strict-english, hits={hits}] {text}")
    print(f"Pass 1 removed: {len(strict_removed)}")
    print()

    # -- Pass 2: grid-world, English spellings --
    after_grid: list[tuple[int, str]] = []
    grid_removed: list[tuple[int, str, list[str]]] = []
    for idx, text in remaining:
        hits = _grid_english_hits(text)
        if hits:
            grid_removed.append((idx, text, hits))
        else:
            after_grid.append((idx, text))

    print("-- Pass 2: grid-world markers, English spellings --")
    for idx, text, hits in grid_removed:
        print(f"  DELETE [grid-english, hits={hits}] {text}")
    print(f"Pass 2 removed: {len(grid_removed)}")
    print()

    lines_after = len(after_grid)
    survivor_text = {idx: text for idx, text in after_grid}

    print("=== Summary ===")
    print(f"lines before: {lines_before}")
    print(f"  1. strict English removed: {len(strict_removed)}")
    print(f"  2. grid-world (English)  removed: {len(grid_removed)}")
    print(f"lines after: {lines_after}")

    if args.dry_run:
        print()
        print("Dry run -- no files written.")
        return

    output_lines: list[str] = []
    for idx, raw_line in enumerate(raw_lines):
        if idx in structural:
            output_lines.append(structural[idx])
        elif idx in survivor_text:
            output_lines.append(survivor_text[idx])
        # else: removed content line, dropped entirely

    CORPUS_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nPruned corpus written to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
