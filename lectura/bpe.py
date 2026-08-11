"""Byte-pair-encoding tokenizer, trained from scratch on Maberyk's own corpus.

Pure stdlib. No tokenizers/sentencepiece/transformers or any other external
tokenizer library — every merge rule here is learned from this project's own
text, nothing borrowed. Read-only against brain/ and episodic_memory.sqlite3
(SELECT only); writes only datos_lectura/tokenizer.json.

Word-level BPE (the classic algorithm, not GPT-2's byte-level variant):
each whitespace-delimited word gets a trailing "</w>" end-of-word marker,
is decomposed into its individual characters, and adjacent character pairs
are merged most-frequent-first until vocab_size is reached. Accents, ñ and ü
are ordinary characters here — nothing strips or decomposes them, unlike
brain/language_model.py's normalize_text() (which is a deliberately
different, lossy pipeline for the word-level model; this tokenizer is
lossless and must round-trip exactly).

CLI: python -m lectura.bpe --train --vocab-size 8000
"""

from __future__ import annotations

import argparse
import heapq
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "datos_lectura" / "clean"
CORPUS_AGENTE_PATH = PROJECT_ROOT / "corpus_agente.txt"
MEMORY_PATH = PROJECT_ROOT / "episodic_memory.sqlite3"
TOKENIZER_PATH = PROJECT_ROOT / "datos_lectura" / "tokenizer.json"

END_OF_WORD = "</w>"

PAD, UNK, Q, A, END = "<pad>", "<unk>", "<q>", "<a>", "<end>"
SPECIAL_ORDER = (PAD, UNK, Q, A, END)  # fixed ids 0-4, in this order
SPECIAL_ID = {tok: i for i, tok in enumerate(SPECIAL_ORDER)}
SPECIAL_SET = set(SPECIAL_ORDER)
# These four always represent a whole-word boundary when emitted by encode()
# (see encode()'s atomic-special handling), so decode() gives them the same
# auto-trailing-space treatment as a "</w>"-terminated subword token. <unk>
# is excluded: it doubles as the fallback for a single unrecognized
# character *inside* a word, where auto-spacing would corrupt the word.
BOUNDARY_SPECIALS = {PAD, Q, A, END}

SELF_TEST_STRINGS = [
    "el niño comió una manzana pequeña",
    "¿cómo estás, Vaenda?",
    "aprende: la curiosidad es mi motor",
    "raza razas estímulo estímulos",
]

# Held-out segmentation check: none of these should appear in the training
# corpus (verified at runtime, not just assumed) — they test whether the
# tokenizer generalizes to unseen words via reusable subword pieces, rather
# than only ever emitting whole words it happened to memorize.
HELD_OUT_WORDS = [
    "jirafas",
    "descentralización",
    "correteando",
    "microscopios",
    "ilegalmente",
    "reprogramaría",
]
IN_CORPUS_COMPARISON_WORDS = ["raza", "razas", "casa", "casas"]


# --------------------------------------------------------------------------
# Corpus gathering (read-only)
# --------------------------------------------------------------------------

def gather_book_texts() -> list[str]:
    """datos_lectura/clean/*.txt — the reading corpus."""
    if not CLEAN_DIR.is_dir():
        return []
    return [
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(CLEAN_DIR.glob("*.txt"))
    ]


def gather_conversational_texts() -> list[str]:
    """corpus_agente.txt + human_taught/tutor_approved pairs from the DB —
    the world Maberyk actually talks in, as opposed to the books he reads."""
    texts = []

    if CORPUS_AGENTE_PATH.is_file():
        lines = []
        for line in CORPUS_AGENTE_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
        if lines:
            texts.append("\n".join(lines))

    if MEMORY_PATH.is_file():
        connection = sqlite3.connect(MEMORY_PATH)
        try:
            rows = connection.execute(
                "SELECT question, answer FROM conversations "
                "WHERE source IN ('human_taught', 'tutor_approved')"
            ).fetchall()
        finally:
            connection.close()
        pair_lines = []
        for question, answer in rows:
            if question and question.strip():
                pair_lines.append(question.strip())
            if answer and answer.strip():
                pair_lines.append(answer.strip())
        if pair_lines:
            texts.append("\n".join(pair_lines))

    return texts


def word_frequencies(texts: list[str]) -> Counter:
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    return counter


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(texts: list[str], vocab_size: int = 8000) -> "BPETokenizer":
    word_freq = word_frequencies(texts)
    if not word_freq:
        raise ValueError("No words to train on — check the corpus sources.")

    splits: dict[str, list[str]] = {
        word: list(word) + [END_OF_WORD] for word in word_freq
    }

    vocab: dict[str, int] = dict(SPECIAL_ID)
    next_id = len(vocab)

    base_symbols = set()
    for symbols in splits.values():
        base_symbols.update(symbols)
    for symbol in sorted(base_symbols):
        if symbol not in vocab:
            vocab[symbol] = next_id
            next_id += 1

    merges: list[tuple[str, str]] = []

    pair_counts: Counter = Counter()
    pair_words: dict[tuple[str, str], set[str]] = defaultdict(set)

    def register_pairs(word: str) -> None:
        symbols = splits[word]
        freq = word_freq[word]
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freq
            pair_words[pair].add(word)

    for word in splits:
        register_pairs(word)

    heap = [(-count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)

    while len(vocab) < vocab_size and heap:
        neg_count, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg_count or pair_counts[pair] <= 0:
            continue  # stale heap entry (lazy deletion) — pair's count has changed

        a, b = pair
        merged = a + b

        if merged in SPECIAL_SET:
            # Never let a merge produce a reserved special string. Disqualify
            # this pair permanently and move on to the next-best candidate.
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
            continue

        merges.append(pair)
        vocab[merged] = next_id
        next_id += 1

        for word in list(pair_words.get(pair, ())):
            symbols = splits[word]
            freq = word_freq[word]

            for old_pair in zip(symbols, symbols[1:]):
                pair_counts[old_pair] -= freq

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            splits[word] = new_symbols

            for new_pair in zip(new_symbols, new_symbols[1:]):
                pair_counts[new_pair] += freq
                pair_words[new_pair].add(word)
                heapq.heappush(heap, (-pair_counts[new_pair], new_pair))

        pair_counts.pop(pair, None)
        pair_words.pop(pair, None)

    return BPETokenizer(vocab, merges, dict(SPECIAL_ID))


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]] | list[list[str]],
        special: dict[str, int],
    ) -> None:
        self.vocab = dict(vocab)
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.merges = [tuple(pair) for pair in merges]
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special = dict(special)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bpe_word(self, word: str) -> list[str]:
        symbols = list(word) + [END_OF_WORD]
        if len(symbols) == 1:
            return symbols

        while len(symbols) > 1:
            ranked_pairs = (
                (self.merge_rank[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.merge_rank
            )
            best = min(ranked_pairs, default=None, key=lambda item: item[0])
            if best is None:
                break
            _, (a, b) = best

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        unk_id = self.vocab[UNK]
        for word in text.split():
            if word in SPECIAL_SET:
                ids.append(self.vocab[word])
                continue
            for symbol in self._bpe_word(word):
                ids.append(self.vocab.get(symbol, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        parts = []
        for index in ids:
            token = self.id_to_token.get(index, UNK)
            if token in BOUNDARY_SPECIALS:
                parts.append(token + " ")
            else:
                parts.append(token.replace(END_OF_WORD, " "))
        return "".join(parts).rstrip(" ")

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "special": self.special,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], data["merges"], data["special"])


# --------------------------------------------------------------------------
# Self-test / reporting
# --------------------------------------------------------------------------

def run_self_test(tokenizer: BPETokenizer, book_texts: list[str], conversational_texts: list[str]) -> None:
    print("\n=== SELF-TEST: round-trip ===")
    all_passed = True
    for text in SELF_TEST_STRINGS:
        decoded = tokenizer.decode(tokenizer.encode(text))
        passed = decoded == text
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {text!r} -> {decoded!r}" if not passed else f"[{status}] {text!r}")

    print(f"\nfinal vocab size: {tokenizer.vocab_size}")
    print(f"number of merges: {len(tokenizer.merges)}")

    book_words = [word for text in book_texts for word in text.split()]
    if book_words:
        total_tokens = sum(len(tokenizer._bpe_word(w)) if w not in SPECIAL_SET else 1 for w in book_words)
        mean_tokens = total_tokens / len(book_words)
        print(f"mean tokens per word over the clean corpus: {mean_tokens:.3f} ({len(book_words)} word occurrences)")
    else:
        print("mean tokens per word over the clean corpus: n/a (no book text found)")

    unk_id = tokenizer.vocab[UNK]
    conv_token_total = 0
    conv_unk_total = 0
    for text in conversational_texts:
        ids = tokenizer.encode(text)
        conv_token_total += len(ids)
        conv_unk_total += sum(1 for i in ids if i == unk_id)
    if conv_token_total:
        unk_rate = conv_unk_total / conv_token_total * 100
        print(f"<unk> rate on conversational corpus: {unk_rate:.4f}% ({conv_unk_total}/{conv_token_total} tokens)")
    else:
        print("<unk> rate on conversational corpus: n/a (no conversational text found)")

    print("\n=== last test string, word by word ===")
    last_words = SELF_TEST_STRINGS[-1].split()
    for word in last_words:
        pieces = tokenizer._bpe_word(word)
        print(f"  {word!r} -> {pieces}")

    print(f"\noverall round-trip: {'PASS' if all_passed else 'FAIL'}")


def run_held_out_test(tokenizer: BPETokenizer, word_freq: Counter) -> None:
    print("\n=== SELF-TEST: held-out segmentation ===")
    print("held-out words (verified NOT in training corpus):")
    any_leaked = False
    for word in HELD_OUT_WORDS:
        count = word_freq.get(word, 0)
        pieces = tokenizer._bpe_word(word)
        if count:
            any_leaked = True
            print(f"  {word!r} -> {pieces}  ** WARNING: appears {count}x in training corpus, not actually held out **")
        else:
            print(f"  {word!r} -> {pieces}")
    print("(no held-out word found in training corpus)" if not any_leaked else "WARNING: some held-out words leaked into the training corpus — see above")

    print("\nin-corpus comparison words:")
    for word in IN_CORPUS_COMPARISON_WORDS:
        count = word_freq.get(word, 0)
        pieces = tokenizer._bpe_word(word)
        note = "" if count else "  (NOTE: not actually found in training corpus)"
        print(f"  {word!r} -> {pieces}{note}")

    whole_word = sum(
        1 for token in tokenizer.vocab if token not in SPECIAL_SET and token.endswith(END_OF_WORD)
    )
    subword = sum(
        1 for token in tokenizer.vocab if token not in SPECIAL_SET and not token.endswith(END_OF_WORD)
    )
    ratio = whole_word / subword if subword else float("inf")
    print(f"\nvocab entries ending in {END_OF_WORD!r} (whole words): {whole_word}")
    print(f"vocab entries not ending in {END_OF_WORD!r} (reusable subword pieces): {subword}")
    print(f"whole-word : subword ratio = {ratio:.3f} : 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and self-test the BPE tokenizer.")
    parser.add_argument("--train", action="store_true", help="Train a new tokenizer and save it.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=TOKENIZER_PATH)
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    book_texts = gather_book_texts()
    conversational_texts = gather_conversational_texts()
    print(
        f"[bpe] training on {len(book_texts)} book file(s) from {CLEAN_DIR} "
        f"and {len(conversational_texts)} conversational text block(s)"
    )

    tokenizer = train(book_texts + conversational_texts, vocab_size=args.vocab_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print(f"[bpe] tokenizer saved to {args.output}")

    run_self_test(tokenizer, book_texts, conversational_texts)

    word_freq = word_frequencies(book_texts + conversational_texts)
    run_held_out_test(tokenizer, word_freq)


if __name__ == "__main__":
    main()
