"""Clean datos_lectura/raw/*.txt into datos_lectura/clean/*.txt.

Pure stdlib. Read-only against brain/ and the raw files — only ever writes
under datos_lectura/clean/.

CLI: python -m lectura.ingest
"""

import hashlib
import json
import re
from pathlib import Path

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

    text, gutenberg_marker_found = strip_gutenberg_boilerplate(raw_text)
    text = remove_wiki_artifacts(text)
    text = collapse_whitespace_runs(text)

    drop_counts = {
        "blank": 0, "punctuation_only": 0, "too_short": 0,
        "english_stopwords": 0, "english_strict": 0,
    }
    kept_lines = []

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)

        if not line:
            drop_counts["blank"] += 1
            continue

        if not re.search(r"\w", line, re.UNICODE):
            drop_counts["punctuation_only"] += 1
            continue

        line = strip_word_punctuation(line)
        if not line:
            drop_counts["punctuation_only"] += 1
            continue

        if len(line) < MIN_LINE_LENGTH:
            drop_counts["too_short"] += 1
            continue

        if stopword_fraction(line) > STOPWORD_FRACTION_THRESHOLD:
            drop_counts["english_stopwords"] += 1
            continue

        if strict_english_match_count(line) >= STRICT_ENGLISH_MIN_MATCHES:
            drop_counts["english_strict"] += 1
            continue

        kept_lines.append(line)

    clean_text = "\n".join(kept_lines)

    stats = {
        "source_path": str(path),
        "sha256": sha256_of_bytes(raw_bytes),
        "chars_before": chars_before,
        "chars_after": len(clean_text),
        "lines_kept": len(kept_lines),
        "lines_dropped": drop_counts,
        "gutenberg_marker_found": gutenberg_marker_found,
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
        print(f"  lines kept: {stats['lines_kept']}")
        print(f"  lines dropped: {stats['lines_dropped']}")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[ingest] manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
