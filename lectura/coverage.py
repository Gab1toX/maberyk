"""Measure whether datos_lectura/clean/*.txt is usable with Maberyk's CURRENT
word-level vocabulary (the tokenizer baked into language_model.pt), before any
training run or tokenizer change. Read-only: never touches brain/.

Falls back to datos_lectura/raw/ (with a loud warning) if clean/ is missing
or empty, so this can still run before lectura/ingest.py has ever been used.

CLI: python -m lectura.coverage
"""

import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datos_lectura" / "clean"
FALLBACK_RAW_DIR = PROJECT_ROOT / "datos_lectura" / "raw"
CHECKPOINT_PATH = PROJECT_ROOT / "language_model.pt"
OUTPUT_PATH = PROJECT_ROOT / "datos_lectura" / "coverage_clean.txt"
FALLBACK_WARNING = (
    "WARNING: clean/ not found or empty — measuring RAW UNCLEANED text."
)

TOP_OOV_COUNT = 50
COVERAGE_THRESHOLDS = (0.95, 0.99)
PUNCT_CHARS = ".,;:!?¿¡\"'«»()—–…"
SPOT_CHECK_SAMPLE_SIZE = 20
SPOT_CHECK_SEED = 42

REPLICATED_NOTE = None

try:
    from brain.language_model import AgentTokenizer, LanguageModelTrainer, normalize_text
except ImportError as exc:
    REPLICATED_NOTE = (
        f"brain.language_model could not be imported ({exc}). Everything below "
        "that would normally reuse brain/language_model.py's own code — "
        "normalize_text(), the corpus_agente.txt ingestion block "
        "(_load_training_samples, lines 326-332), and the vocabulary split "
        "(_collect_vocabulary, lines 347-351) — is a hand-copied replica instead "
        "of the real module. If the two ever drift apart, this report is wrong."
    )

    import unicodedata

    def normalize_text(text: str) -> str:
        protected = str(text).replace("ñ", "\x00").replace("Ñ", "\x00")
        stripped = "".join(
            c for c in unicodedata.normalize("NFD", protected)
            if unicodedata.category(c) != "Mn"
        )
        return stripped.replace("\x00", "ñ").lower()

    def _collect_vocabulary_fallback(samples: list[str]) -> list[str]:
        # Replica of LanguageModelTrainer._collect_vocabulary (language_model.py:347-351).
        words: list[str] = []
        for sample in samples:
            words.extend(sample.lower().split())
        return words

    AgentTokenizer = None
    LanguageModelTrainer = None


def prepare_corpus_lines(text: str) -> list[str]:
    # Mirrors the corpus_agente.txt ingestion block inside
    # LanguageModelTrainer._load_training_samples (brain/language_model.py:326-332):
    # strip each line, skip blanks and '#'-comments, normalize_text() the rest.
    # That is the exact code path that turns a raw corpus string into a training
    # sample — generate_reply()'s inline `re.sub(r"[^\w]", "", word)` cleanup is a
    # different path used only for live user questions, not corpus ingestion.
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(normalize_text(line))
    return lines


def tokens_from_lines(lines: list[str]) -> list[str]:
    # Reuses LanguageModelTrainer._collect_vocabulary directly (brain/language_model.py:
    # 347-351). That method never touches `self`, so it's safe to call unbound —
    # this is the real split, not a reimplementation, whenever the import succeeded.
    if LanguageModelTrainer is not None:
        return LanguageModelTrainer._collect_vocabulary(None, lines)
    return _collect_vocabulary_fallback(lines)


def load_current_vocab() -> list[str]:
    if LanguageModelTrainer is None:
        raise RuntimeError(
            "Cannot load the current vocabulary: brain.language_model failed to "
            "import (see the replication note above), and the checkpoint format "
            "requires the real AgentTokenizer/torch to load."
        )
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    _model, tokenizer = LanguageModelTrainer.load_model(CHECKPOINT_PATH, map_location="cpu")
    return list(tokenizer.words)


def vocab_size_needed_for_coverage(counter: Counter, total_tokens: int, thresholds) -> dict:
    ranked = counter.most_common()
    thresholds_sorted = sorted(thresholds)
    result = {}
    cumulative = 0
    pending = list(thresholds_sorted)
    for rank, (_word, count) in enumerate(ranked, start=1):
        cumulative += count
        while pending and total_tokens and cumulative / total_tokens >= pending[0]:
            result[pending.pop(0)] = rank
        if not pending:
            break
    for threshold in thresholds_sorted:
        result.setdefault(threshold, len(ranked))
    return result


def compute_metrics(tokens: list[str], vocab: set[str]) -> dict:
    total_tokens = len(tokens)
    counter = Counter(tokens)
    unique_types = len(counter)

    covered_tokens = sum(count for word, count in counter.items() if word in vocab)
    token_coverage = (covered_tokens / total_tokens * 100) if total_tokens else 0.0
    oov_rate = 100.0 - token_coverage

    covered_types = sum(1 for word in counter if word in vocab)
    type_coverage = (covered_types / unique_types * 100) if unique_types else 0.0

    oov_counter = Counter({word: count for word, count in counter.items() if word not in vocab})
    oov_type_count = len(oov_counter)
    hapax_oov = sum(1 for count in oov_counter.values() if count == 1)
    hapax_share = (hapax_oov / oov_type_count * 100) if oov_type_count else 0.0

    needed = vocab_size_needed_for_coverage(counter, total_tokens, COVERAGE_THRESHOLDS)

    return {
        "total_tokens": total_tokens,
        "unique_types": unique_types,
        "token_coverage": token_coverage,
        "type_coverage": type_coverage,
        "oov_rate": oov_rate,
        "oov_counter": oov_counter,
        "hapax_share": hapax_share,
        "needed_95": needed[0.95],
        "needed_99": needed[0.99],
    }


def build_variants(raw_tokens: list[str], vocab: set[str]) -> dict:
    stripped_tokens = [t for t in (tok.strip(PUNCT_CHARS) for tok in raw_tokens) if t]
    numeric_tokens = [t for t in stripped_tokens if t.isdigit()]
    no_numeric_tokens = [t for t in stripped_tokens if not t.isdigit()]
    numeric_count = len(numeric_tokens)
    numeric_pct = (numeric_count / len(stripped_tokens) * 100) if stripped_tokens else 0.0

    return {
        "RAW": compute_metrics(raw_tokens, vocab),
        "STRIPPED": compute_metrics(stripped_tokens, vocab),
        "NO-NUMERIC": compute_metrics(no_numeric_tokens, vocab),
        "numeric_count": numeric_count,
        "numeric_pct": numeric_pct,
    }


def format_breakdown(title: str, metrics: dict) -> list[str]:
    lines = [
        f"--- {title} ---",
        f"total tokens: {metrics['total_tokens']}",
        f"unique word types: {metrics['unique_types']}",
        f"token-level coverage vs current vocab: {metrics['token_coverage']:.2f}%",
        f"type-level coverage vs current vocab: {metrics['type_coverage']:.2f}%",
        f"OOV rate (tokens not in current vocab): {metrics['oov_rate']:.2f}%",
        f"vocab size needed for 95% token coverage (own-text ranking): {metrics['needed_95']}",
        f"vocab size needed for 99% token coverage (own-text ranking): {metrics['needed_99']}",
        f"OOV types that are hapax (count == 1): {metrics['hapax_share']:.2f}%",
        f"top {TOP_OOV_COUNT} most frequent OOV words:",
    ]
    for word, count in metrics["oov_counter"].most_common(TOP_OOV_COUNT):
        lines.append(f"  {word}: {count}")
    lines.append("")
    return lines


def format_scope(title: str, raw_tokens: list[str], vocab: set[str]) -> list[str]:
    variants = build_variants(raw_tokens, vocab)
    lines = [f"=== {title} ==="]
    lines.append(
        f"numeric tokens (after stripping): {variants['numeric_count']} "
        f"({variants['numeric_pct']:.2f}% of STRIPPED total)"
    )
    lines.append("")
    for variant_name in ("RAW", "STRIPPED", "NO-NUMERIC"):
        lines.extend(
            format_breakdown(f"{title} [{variant_name}]", variants[variant_name])
        )
    return lines


def spot_check_vocab_punctuation(vocab_words: list[str]) -> list[str]:
    specials = set(AgentTokenizer.SPECIAL_TOKENS) if AgentTokenizer is not None else set()
    content_words = sorted(w for w in vocab_words if w not in specials)

    rng = random.Random(SPOT_CHECK_SEED)
    sample_size = min(SPOT_CHECK_SAMPLE_SIZE, len(content_words))
    sample = rng.sample(content_words, sample_size)
    sample_hits = [w for w in sample if any(c in PUNCT_CHARS for c in w)]
    full_hits = [w for w in content_words if any(c in PUNCT_CHARS for c in w)]

    lines = ["=== VOCAB PUNCTUATION SPOT-CHECK ==="]
    lines.append(f"{sample_size} random current-vocab tokens: {sample}")
    lines.append(
        f"{len(sample_hits)}/{sample_size} sampled tokens contain punctuation "
        f"characters ({PUNCT_CHARS!r})"
    )
    full_pct = (len(full_hits) / len(content_words) * 100) if content_words else 0.0
    lines.append(
        f"full-vocab scan (not just the sample above): {len(full_hits)}/"
        f"{len(content_words)} content words ({full_pct:.2f}%) contain punctuation"
    )
    if full_hits:
        lines.append(f"  examples from the full scan: {full_hits[:15]}")

    if full_hits:
        lines.append(
            "Authoritative measure: RAW. brain/language_model.py never strips "
            "punctuation anywhere in the corpus-to-vocab path — "
            "_load_training_samples' corpus_agente.txt block (lines 326-332) and "
            "_collect_vocabulary (lines 347-351) both just normalize_text() + "
            "str.lower().split(). The full-vocab scan above confirms this: "
            "punctuation-bearing tokens exist in the trained vocabulary itself, so "
            "training genuinely does not strip it — RAW is what the model sees "
            "today. STRIPPED and NO-NUMERIC are diagnostic only: they show the "
            "coverage a punctuation-aware preprocessing step would buy if one were "
            "ever added, not what happens now. (Punctuation-bearing tokens are a "
            f"small share of the vocab — {full_pct:.2f}% — so a "
            f"{sample_size}-token random sample can easily land on zero hits even "
            "though the full scan shows they exist; that is why this conclusion is "
            "based on the full scan, not the sample alone.)"
        )
    else:
        lines.append(
            "Authoritative measure: STRIPPED. No punctuation-bearing tokens exist "
            "anywhere in the current vocabulary, consistent with a pipeline that "
            "strips punctuation before the vocab is built. Caveat: "
            "brain/language_model.py's own split() calls (_collect_vocabulary, "
            "lines 347-351) do not strip punctuation, so an all-clean vocab is more "
            "likely an artifact of clean upstream data (short Q/A pairs, generated "
            "sentences) than of an actual stripping step — re-check the code path "
            "if this conclusion looks surprising."
        )
    lines.append("")
    return lines


def resolve_source_dir() -> tuple[Path, bool]:
    if RAW_DIR.is_dir() and any(RAW_DIR.glob("*.txt")):
        return RAW_DIR, False
    return FALLBACK_RAW_DIR, True


def main() -> None:
    source_dir, fell_back = resolve_source_dir()

    report_lines = []
    if fell_back:
        report_lines.append(FALLBACK_WARNING)
    report_lines.append(f"source directory: {source_dir}")
    if REPLICATED_NOTE:
        report_lines.append(f"WARNING: {REPLICATED_NOTE}")
    report_lines.append("")

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Text directory not found: {source_dir}")

    raw_files = sorted(source_dir.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No .txt files found under {source_dir}")

    vocab_words = load_current_vocab()
    vocab = set(vocab_words)
    report_lines.append(f"current vocabulary loaded from: {CHECKPOINT_PATH}")
    report_lines.append(f"current vocabulary size (incl. special tokens): {len(vocab)}")
    report_lines.append("")

    report_lines.extend(spot_check_vocab_punctuation(vocab_words))

    per_file_tokens = {}
    for path in raw_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        per_file_tokens[path.name] = tokens_from_lines(prepare_corpus_lines(text))

    all_tokens = [token for tokens in per_file_tokens.values() for token in tokens]

    report_lines.extend(format_scope("ALL FILES COMBINED", all_tokens, vocab))

    report_lines.append("=== PER SOURCE FILE ===")
    report_lines.append("")
    for name, tokens in per_file_tokens.items():
        report_lines.extend(format_scope(name, tokens, vocab))

    report_text = "\n".join(report_lines)
    print(report_text)

    OUTPUT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n[coverage] report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
