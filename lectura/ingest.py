"""Clean datos_lectura/raw/*.txt into datos_lectura/clean/*.txt.

Pure stdlib. Read-only against brain/ and the raw files — only ever writes
under datos_lectura/clean/.

CLI: python -m lectura.ingest
"""

import hashlib
import json
import re
import sys
from pathlib import Path

# Wikipedia/TeX text can contain symbols outside the Windows console's
# cp1252 codepage (e.g. U+2264 ≤); without this, printing an example
# containing one crashes the whole run instead of just mangling that glyph.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datos_lectura" / "raw"
CLEAN_DIR = PROJECT_ROOT / "datos_lectura" / "clean"
MANIFEST_PATH = CLEAN_DIR / "manifest.json"

MIN_LINE_LENGTH = 20
STOPWORD_FRACTION_THRESHOLD = 0.4

STOPWORDS = {
    "the", "of", "and", "to", "in", "that", "is", "you", "for", "with",
    "this", "work", "any", "or", "project", "gutenberg", "electronic", "terms",
}

# Stricter second pass: a much larger set of English function words, checked
# by absolute count rather than fraction-of-line. The original STOPWORDS/
# fraction rule above was tuned for whole lines of Gutenberg legal English
# and mostly misses shorter English fragments embedded inside otherwise-
# Spanish lines (e.g. Wikipedia prose that quotes an English film title or
# names English-language DJs/artists inline) — those never cross a 40%
# fraction because most of the line is still Spanish. Every word here was
# checked against common Spanish function words to avoid collisions — no
# "a", "no", "en", "es", "el", "la", "so", "may", etc., which would false-
# positive on ordinary Spanish text. Verified on this corpus: 0 hits on
# quiroga_cuentos.txt (already clean), 10/1790 lines on wikipedia_es.txt,
# all genuinely English-bearing on inspection.
STRICT_ENGLISH_WORDS = {
    "about", "after", "again", "against", "all", "also", "although", "always",
    "an", "and", "another", "any", "are", "at", "because", "been", "before",
    "being", "below", "between", "but", "by", "can", "could", "did", "do",
    "does", "done", "during", "each", "electronic", "every", "for", "from",
    "get", "got", "gutenberg", "had", "has", "have", "here", "how", "into",
    "is", "it", "its", "just", "like", "many", "more", "most", "much",
    "never", "not", "now", "of", "off", "often", "on", "once", "only", "or",
    "other", "our", "out", "over", "own", "project", "same", "she", "should",
    "since", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "though", "through", "terms",
    "to", "toward", "towards", "twice", "under", "until", "up", "upon", "us",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "within", "without", "work", "would", "yesterday", "you", "your",
    "yours",
}
STRICT_ENGLISH_MIN_MATCHES = 3

# Leading/trailing punctuation stripped from each word (step 4). Deliberately
# does not touch accents or ñ — only the characters listed in the task.
WORD_PUNCT_CHARS = ".,;:!?¿¡\"'()…"

GUTENBERG_START_RE = re.compile(
    r"\*{3}\s*START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK", re.IGNORECASE
)
GUTENBERG_END_RE = re.compile(
    r"\*{3}\s*END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK", re.IGNORECASE
)

WIKI_ARTIFACT_RE = re.compile(r"\[cita requerida\]|\[editar\]|\[\d+\]", re.IGNORECASE)

# LaTeX/math markup from Wikipedia. `{\displaystyle ...}` bodies can contain
# nested braces (e.g. `{\displaystyle \frac{a}{b}}`), so the block is matched
# by finding the opener with a regex and then walking brace depth by hand
# rather than trying to express nesting in the regex itself.
DISPLAYSTYLE_START_RE = re.compile(r"\{\\displaystyle\b")
# Any remaining bare TeX command token (\frac, \mathbb, \cdot, ...) that
# wasn't inside a {\displaystyle ...} wrapper.
STRAY_TEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")
# Real formulas are short. If braces never rebalance within this many chars
# of the opener, the source has a stray/unescaped brace (common in scraped
# Wikipedia markup) and this is NOT a genuine {\displaystyle ...} wrapper —
# treat it as unmatched rather than consuming to end-of-file.
DISPLAYSTYLE_MAX_SPAN = 2000

# Line-level table/formula-fragment filter: tokens made entirely of digits
# and the listed math symbols (with . and , as decimal/thousands separators).
NUMERIC_SYMBOL_TOKEN_RE = re.compile(r"^[0-9%=+\-−×.,]+$")
NUMERIC_SYMBOL_FRACTION_THRESHOLD = 0.3

QUOTE_DASH_TRANSLATION = str.maketrans({
    "«": '"', "»": '"',
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "—": "-", "–": "-",
})
DIALOGUE_MARKERS = ("--", "—", "–")


def strip_gutenberg_boilerplate(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and GUTENBERG_START_RE.search(line):
            start_idx = i
            continue
        if start_idx is not None and GUTENBERG_END_RE.search(line):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        return text, False
    return "\n".join(lines[start_idx + 1:end_idx]), True


def remove_wiki_artifacts(text: str) -> str:
    return WIKI_ARTIFACT_RE.sub("", text)


def _balanced_brace_end(text: str, open_idx: int, max_span: int = DISPLAYSTYLE_MAX_SPAN) -> int | None:
    """`open_idx` points at the '{' of the opening brace. Returns the index
    just past its matching '}', or None if the braces don't rebalance within
    `max_span` characters (treated as not a genuine wrapper — see
    DISPLAYSTYLE_MAX_SPAN — rather than silently consuming to end-of-file)."""
    depth = 0
    limit = min(len(text), open_idx + max_span)
    for i in range(open_idx, limit):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


EXAMPLE_CONTEXT_CHARS = 60
MAX_LATEX_EXAMPLES = 3


def strip_latex_markup(text: str) -> tuple[str, dict]:
    """Remove `{\\displaystyle ...}` blocks (contents included, brace-balanced)
    and any remaining stray TeX commands. Returns (cleaned_text, stats), where
    stats accounts for exactly how many chars each sub-step removed and
    includes up to MAX_LATEX_EXAMPLES samples of removed displaystyle spans
    with surrounding context, for eyeballing whether prose is being cut."""
    parts = []
    pos = 0
    displaystyle_blocks = 0
    displaystyle_chars_removed = 0
    unbalanced_skipped = 0
    examples = []
    for m in DISPLAYSTYLE_START_RE.finditer(text):
        if m.start() < pos:
            continue  # already consumed as part of a previous block
        end = _balanced_brace_end(text, m.start())
        if end is None:
            unbalanced_skipped += 1
            continue  # leave this occurrence untouched, don't consume anything
        parts.append(text[pos:m.start()])
        displaystyle_blocks += 1
        displaystyle_chars_removed += end - m.start()
        if len(examples) < MAX_LATEX_EXAMPLES:
            examples.append({
                "before": text[max(0, m.start() - EXAMPLE_CONTEXT_CHARS):m.start()],
                "removed": text[m.start():end],
                "after": text[end:end + EXAMPLE_CONTEXT_CHARS],
            })
        pos = end
    parts.append(text[pos:])
    text = "".join(parts)

    stray_commands = 0
    stray_chars_removed = 0

    def _count_and_drop(m: re.Match) -> str:
        nonlocal stray_commands, stray_chars_removed
        stray_commands += 1
        stray_chars_removed += len(m.group(0))
        return ""

    text = STRAY_TEX_COMMAND_RE.sub(_count_and_drop, text)

    return text, {
        "displaystyle_blocks": displaystyle_blocks,
        "displaystyle_chars_removed": displaystyle_chars_removed,
        "unbalanced_skipped": unbalanced_skipped,
        "stray_commands": stray_commands,
        "stray_chars_removed": stray_chars_removed,
        "examples": examples,
    }


def numeric_symbol_fraction(line: str) -> float:
    words = line.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if NUMERIC_SYMBOL_TOKEN_RE.match(w))
    return hits / len(words)


def collapse_whitespace_runs(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_line(raw_line: str) -> str:
    line = raw_line.strip()
    for marker in DIALOGUE_MARKERS:
        if line.startswith(marker):
            line = line[len(marker):].lstrip()
            break
    line = line.translate(QUOTE_DASH_TRANSLATION)
    return re.sub(r"\s+", " ", line).strip()


def strip_word_punctuation(line: str) -> str:
    words = [w.strip(WORD_PUNCT_CHARS) for w in line.split()]
    return " ".join(w for w in words if w)


def stopword_fraction(line: str) -> float:
    words = line.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in STOPWORDS)
    return hits / len(words)


def strict_english_match_count(line: str) -> int:
    words = line.split()
    return sum(1 for w in words if w.lower() in STRICT_ENGLISH_WORDS)


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_file(path: Path) -> tuple[str, dict]:
    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    chars_before = len(raw_text)

    # chars_removed accounts for every character that disappears between
    # chars_before and chars_after, one entry per filter step. clean_file
    # asserts chars_before - sum(chars_removed.values()) == chars_after
    # below, so a bug that silently over-deletes (like the displaystyle
    # regex runaway this replaced) shows up as a loud mismatch instead of
    # a quiet difference only visible by eyeballing chars_before/after.
    chars_removed = {}

    stage = raw_text
    stage, gutenberg_marker_found = strip_gutenberg_boilerplate(stage)
    chars_removed["gutenberg_boilerplate"] = chars_before - len(stage)

    prev_len = len(stage)
    stage = remove_wiki_artifacts(stage)
    chars_removed["wiki_artifacts"] = prev_len - len(stage)

    prev_len = len(stage)
    stage, latex_stats = strip_latex_markup(stage)
    chars_removed["latex_displaystyle"] = latex_stats["displaystyle_chars_removed"]
    chars_removed["latex_stray_commands"] = latex_stats["stray_chars_removed"]

    prev_len = len(stage)
    stage = collapse_whitespace_runs(stage)
    chars_removed["whitespace_collapse"] = prev_len - len(stage)

    lines = stage.split("\n")

    drop_counts = {
        "blank": 0, "punctuation_only": 0, "too_short": 0,
        "english_stopwords": 0, "english_strict": 0, "numeric_symbol_heavy": 0,
    }
    drop_chars = {name: 0 for name in drop_counts}
    normalization_shrink = 0
    kept_lines = []

    for raw_line in lines:
        line = normalize_line(raw_line)

        if not line:
            drop_counts["blank"] += 1
            drop_chars["blank"] += len(raw_line)
            continue

        if not re.search(r"\w", line, re.UNICODE):
            drop_counts["punctuation_only"] += 1
            drop_chars["punctuation_only"] += len(raw_line)
            continue

        line = strip_word_punctuation(line)
        if not line:
            drop_counts["punctuation_only"] += 1
            drop_chars["punctuation_only"] += len(raw_line)
            continue

        if len(line) < MIN_LINE_LENGTH:
            drop_counts["too_short"] += 1
            drop_chars["too_short"] += len(raw_line)
            continue

        if numeric_symbol_fraction(line) > NUMERIC_SYMBOL_FRACTION_THRESHOLD:
            drop_counts["numeric_symbol_heavy"] += 1
            drop_chars["numeric_symbol_heavy"] += len(raw_line)
            continue

        if stopword_fraction(line) > STOPWORD_FRACTION_THRESHOLD:
            drop_counts["english_stopwords"] += 1
            drop_chars["english_stopwords"] += len(raw_line)
            continue

        if strict_english_match_count(line) >= STRICT_ENGLISH_MIN_MATCHES:
            drop_counts["english_strict"] += 1
            drop_chars["english_strict"] += len(raw_line)
            continue

        normalization_shrink += len(raw_line) - len(line)
        kept_lines.append(line)

    clean_text = "\n".join(kept_lines)

    chars_removed["line_normalization"] = normalization_shrink
    for name, n in drop_chars.items():
        chars_removed[f"drop_{name}"] = n
    # split("\n")/"\n".join round-trip exactly, so the separator count is
    # (num_lines - 1) before and (num_kept - 1) after (0 if nothing kept).
    chars_removed["newline_separators"] = (len(lines) - 1) - max(len(kept_lines) - 1, 0)

    chars_after = len(clean_text)
    total_removed = sum(chars_removed.values())
    reconciled = (chars_before - total_removed) == chars_after
    if not reconciled:
        print(
            f"  WARNING: char accounting mismatch for {path.name}: "
            f"chars_before({chars_before}) - total_removed({total_removed}) = "
            f"{chars_before - total_removed}, but chars_after = {chars_after} "
            f"(off by {chars_before - total_removed - chars_after})"
        )

    stats = {
        "source_path": str(path),
        "sha256": sha256_of_bytes(raw_bytes),
        "chars_before": chars_before,
        "chars_after": chars_after,
        "chars_removed": chars_removed,
        "chars_reconciled": reconciled,
        "lines_kept": len(kept_lines),
        "lines_dropped": drop_counts,
        "gutenberg_marker_found": gutenberg_marker_found,
        "latex_removed": {k: v for k, v in latex_stats.items() if k != "examples"},
        "latex_examples": latex_stats["examples"],
    }
    return clean_text, stats


def main() -> None:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Raw text directory not found: {RAW_DIR}")

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No .txt files found under {RAW_DIR}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"files": {}}

    for path in raw_files:
        clean_text, stats = clean_file(path)

        dest = CLEAN_DIR / path.name
        dest.write_text(clean_text, encoding="utf-8")

        manifest["files"][path.name] = stats

        print(f"{path.name}:")
        if not stats["gutenberg_marker_found"]:
            print("  WARNING: no Project Gutenberg START/END marker found — file left unchanged by step 1")
        print(f"  chars: {stats['chars_before']} -> {stats['chars_after']}")
        print("  chars removed by step:")
        for step_name, n in stats["chars_removed"].items():
            if n:
                print(f"    {step_name}: {n}")
        if stats["chars_reconciled"]:
            print(f"  chars reconciled: OK ({sum(stats['chars_removed'].values())} accounted for)")
        # mismatch WARNING for this file was already printed by clean_file

        latex = stats["latex_removed"]
        print(
            f"  latex removed: {latex['displaystyle_blocks']} displaystyle block(s) "
            f"({latex['displaystyle_chars_removed']} chars), "
            f"{latex['stray_commands']} stray command(s) ({latex['stray_chars_removed']} chars)"
        )
        if latex["unbalanced_skipped"]:
            print(f"    unbalanced/skipped (left untouched): {latex['unbalanced_skipped']}")
        for i, ex in enumerate(stats["latex_examples"], 1):
            print(f"    example {i}: ...{ex['before']!r} [[{ex['removed']!r}]] {ex['after']!r}...")

        print(f"  lines kept: {stats['lines_kept']}")
        print(f"  lines dropped: {stats['lines_dropped']}")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[ingest] manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
