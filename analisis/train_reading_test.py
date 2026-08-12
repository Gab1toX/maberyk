"""Standalone script: test whether adding free Spanish reading text to
training improves the native Transformer's generalization.

Trains three models from random init, same architecture/hyperparameters/vocab
cap as production (brain/language_model.py), on the SAME held-out validation
set so val loss is directly comparable:

  A) CONTROL — exactly the production training data (human_taught pairs +
     episode thoughts + corpus_agente.txt), loaded the same way
     brain/language_model.py's own main() does it.
  B) READING — CONTROL's data plus datos_lectura/clean/*.txt, one sample per
     cleaned line (this codebase already treats "one line = one training
     sample" for corpus_agente.txt/episode thoughts, so reading text is held
     to the same convention rather than attempting grammatical sentence
     splitting on text that ingest.py has already stripped of
     sentence-ending punctuation).
  C) TWOPHASE — same combined data as READING, but split into two training
     phases instead of one mixed pass: phase 1 trains ONLY on reading text
     (from random init), then phase 2 continues from those weights training
     ONLY on the base pool at a lower learning rate (1e-4 vs 1e-3), like a
     pretrain-then-fine-tune schedule. Its vocabulary is built once, up
     front, with every current-production vocab word guaranteed a slot
     before any reading word fills the rest of the budget — unlike READING,
     where pure frequency ranking bumped ~15% of production words out
     (see language_model_reading.pt's run: 4343/5136 survived).
  D) BPE — CONTROL's exact data (base pool only, no reading text — this
     isolates the tokenizer change from the reading-text change) and exact
     hyperparameters/epochs, but tokenizer_type="bpe" instead of the
     word-level AgentTokenizer: brain/language_model.py loads
     datos_lectura/tokenizer.json (lectura.bpe) and encodes/decodes through
     it. Its val loss is NOT comparable to the other runs' — BPE tokens are
     sub-word pieces, not whole words, so cross-entropy is over a different
     vocabulary and different-length sequences per sample. The comparison
     that matters for this run is the probe generations, not the loss
     number (see the CAVEAT line in the written report).
  E) BPE_READING — BPE's tokenizer_type="bpe", but CONTROL's base pool PLUS
     the cleaned reading text (same combined data as READING). Isolates
     "does reading text help" for the BPE path the way READING does for the
     word path.

brain/language_model.py now skips normalize_text() entirely when
tokenizer_type == "bpe" (lowercasing/accent-stripping text the BPE
tokenizer was trained to preserve made no sense — see its own change
report). That means the base pool exists in TWO representations here: raw
(unnormalized, for D/E) and normalized (for A/B/C) — but they must still
share the exact same 730-row validation SPLIT, or "same held-out val set"
stops being true. _load_training_samples() also trims episode thoughts with
an unseeded random.sample() and reshuffles with the unseeded global random
module, so calling it twice independently (once per representation) would
silently hand word-mode and bpe-mode runs two different validation splits
even from identical DB content. So the loader runs exactly ONCE here, always
with tokenizer_type="bpe" (raw text, losslessly forward-transformable), and
the split happens once on that raw pool; the normalized pool used by
A/B/C is then derived by applying normalize_text() to each already-split
raw sample — same rows, same split, just re-cased. Reading samples get the
same raw-then-derived-normalized treatment. Reading text is only ever
appended to train sides, never val, so val is the same underlying rows in
every run, always.

Every method actually used (_load_training_samples, _split_train_val,
_oversample_pairs, _collect_vocabulary, _run_epoch, generate_reply, save,
load_model) is LanguageModelTrainer's own — none of it is reimplemented, and
brain/language_model.py is never modified. The one genuinely new piece of
logic is build_priority_tokenizer() for run C's vocabulary (AgentTokenizer
itself only supports pure-frequency ranking, which is exactly what run C
needs to NOT do — see its docstring).

Writes language_model_control.pt, language_model_reading.pt,
language_model_twophase.pt, language_model_bpe.pt, and
language_model_bpe_reading.pt (never language_model.pt or agent_state.pt)
plus a side-by-side summary at analisis/reading_test_results.txt.

Usage:
    python analisis/train_reading_test.py
"""

import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from brain.language_model import AgentLanguageModel, AgentTokenizer, LanguageModelTrainer, normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = PROJECT_ROOT / "episodic_memory.sqlite3"
CLEAN_DIR = PROJECT_ROOT / "datos_lectura" / "clean"
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "language_model.pt"
CONTROL_OUTPUT = PROJECT_ROOT / "language_model_control.pt"
READING_OUTPUT = PROJECT_ROOT / "language_model_reading.pt"
TWOPHASE_OUTPUT = PROJECT_ROOT / "language_model_twophase.pt"
BPE_OUTPUT = PROJECT_ROOT / "language_model_bpe.pt"
BPE_READING_OUTPUT = PROJECT_ROOT / "language_model_bpe_reading.pt"
BPE_VAL_LOSS_CAVEAT = (
    "CAVEAT: BPE/BPE_READING's val loss is NOT directly comparable to "
    "CONTROL/READING/TWOPHASE's. BPE tokens are sub-word pieces, not whole words, so "
    "cross-entropy is computed over a different vocabulary and different-length token "
    "sequences per sample than the word-level runs. The comparison that matters here is "
    "the probe generations below, not the loss numbers in the table above."
)
RESULTS_PATH = Path(__file__).resolve().parent / "reading_test_results.txt"

# Same architecture/hyperparameters/vocab cap production training uses
# (brain/language_model.py's own LanguageModelTrainer defaults / main() CLI
# defaults) — nothing here changes the model.
MAX_VOCAB_SIZE = 8192
EMBEDDING_DIM = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
LEARNING_RATE = 1e-3
PHASE2_LEARNING_RATE = 1e-4
DEVICE = "cpu"  # CPU only on this machine, per CLAUDE.md
EPOCHS = 10
BATCH_SIZE = 16
GENERATION_TEMPERATURE = 0.8

# Fixed probes: exact-topic paraphrases of real human_taught questions,
# worded differently from the corpus (verified in episodic_memory.sqlite3).
PROBE_QUESTIONS = [
    "que significa la humedad",
    "explicame que es una red neuronal",
    "que es el aprendizaje automatico",
    "como se hace un experimento cientifico",
    "en que momento naciste",
    "puedes sentir dolor",
    "que significa viajar",
    "explicame que es un eclipse",
    "que significa la contaminacion",
    "tienes una historia propia",
]


def build_reading_samples(normalize: bool) -> tuple[list[str], list[Path]]:
    """normalize=False (raw, matches lectura/bpe.py's own training text) for
    BPE-mode runs; normalize=True (lowercase + strip accents, matching the
    word tokenizer's own vocabulary) for word-mode runs. Deduping happens
    after the normalize/no-op choice, on whichever representation is being
    built, so each variant's sample count matches what that mode would
    naturally produce on its own.
    """
    if not CLEAN_DIR.is_dir():
        raise FileNotFoundError(
            f"{CLEAN_DIR} not found — run `python -m lectura.ingest` first."
        )
    clean_files = sorted(CLEAN_DIR.glob("*.txt"))
    if not clean_files:
        raise FileNotFoundError(f"No .txt files found under {CLEAN_DIR}")

    samples = []
    for path in clean_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                text = normalize_text(line) if normalize else line
                samples.append(f"{AgentTokenizer.A} {text} {AgentTokenizer.END}")
    return list(dict.fromkeys(samples)), clean_files


def load_base_split() -> tuple[LanguageModelTrainer, dict[str, list[str]], dict[str, list[str]]]:
    """Loads the base pool ONCE, always with tokenizer_type="bpe" so the text
    stays raw/unnormalized (see FIX 1 in brain/language_model.py), and splits
    it ONCE. That split is the single shared source of truth every run in
    this script reuses:
      - "raw" (unnormalized) feeds BPE-mode runs (D, E) directly.
      - "normalized" is derived by applying normalize_text() to each already-
        split raw sample — same rows, same train/val membership, just
        re-cased — and feeds word-mode runs (A, B, C).
    _load_training_samples() trims episode thoughts with an unseeded
    random.sample() and reshuffles with the unseeded global random module, so
    calling it a second time (even with identical DB content) could hand the
    two representations different validation splits. Deriving instead of
    re-loading guarantees "same held-out val set" is actually true.

    scratch itself is a real LanguageModelTrainer instance (via __new__,
    __init__ skipped) used purely to call its own bound methods — safe since
    _load_training_samples/_split_train_val/_oversample_pairs/
    _collect_vocabulary only ever touch self.memory_path and (now)
    self.tokenizer_type, both set below.
    """
    scratch = LanguageModelTrainer.__new__(LanguageModelTrainer)
    scratch.tokenizer_type = "bpe"
    scratch.memory_path = MEMORY_PATH

    base_samples_raw = scratch._load_training_samples()
    if not base_samples_raw:
        raise ValueError(f"No training samples found via {MEMORY_PATH}")

    pre_oversample_train_raw, val_samples_raw = scratch._split_train_val(base_samples_raw)

    raw = {
        "base_samples": base_samples_raw,
        "pre_oversample_train": pre_oversample_train_raw,
        "val_samples": val_samples_raw,
    }
    normalized = {key: [normalize_text(sample) for sample in samples] for key, samples in raw.items()}
    return scratch, raw, normalized


def build_trainer(
    scratch: LanguageModelTrainer,
    pool_samples: list[str],
    pre_oversample_train: list[str],
    val_samples: list[str],
) -> LanguageModelTrainer:
    train_samples = scratch._oversample_pairs(pre_oversample_train)
    vocabulary_words = scratch._collect_vocabulary(pool_samples)
    tokenizer = AgentTokenizer(vocabulary_words, max_vocab_size=MAX_VOCAB_SIZE)

    trainer = LanguageModelTrainer.__new__(LanguageModelTrainer)
    trainer.tokenizer_type = "word"  # required by save() — see build_bpe_trainer for the bpe counterpart
    trainer.memory_path = MEMORY_PATH
    trainer.device = torch.device(DEVICE)
    trainer.samples = pool_samples
    trainer.train_samples = train_samples
    trainer.val_samples = list(val_samples)
    trainer.tokenizer = tokenizer
    trainer.model = AgentLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_index=tokenizer.pad_index,
        tokenizer=tokenizer,
    ).to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=LEARNING_RATE)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    return trainer


def build_bpe_trainer(
    scratch: LanguageModelTrainer,
    base_samples: list[str],
    pre_oversample_train: list[str],
    val_samples: list[str],
) -> LanguageModelTrainer:
    """Run D: CONTROL's exact base-pool data (no reading text), but tokenized
    through datos_lectura/tokenizer.json instead of AgentTokenizer. Reuses
    LanguageModelTrainer._load_bpe_tokenizer — the same method
    brain/language_model.py's own __init__/load_model use for tokenizer_type
    "bpe" — so this is exactly what production BPE training would load.
    """
    train_samples = scratch._oversample_pairs(pre_oversample_train)
    tokenizer = LanguageModelTrainer._load_bpe_tokenizer()

    trainer = LanguageModelTrainer.__new__(LanguageModelTrainer)
    trainer.tokenizer_type = "bpe"
    trainer.memory_path = MEMORY_PATH
    trainer.device = torch.device(DEVICE)
    trainer.samples = base_samples
    trainer.train_samples = train_samples
    trainer.val_samples = list(val_samples)
    trainer.tokenizer = tokenizer
    trainer.model = AgentLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_index=tokenizer.pad_index,
        tokenizer=tokenizer,
    ).to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=LEARNING_RATE)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    return trainer


def build_priority_tokenizer(
    vocabulary_words: list[str], priority_words: list[str], max_vocab_size: int
) -> AgentTokenizer:
    """Same vocab format AgentTokenizer.__init__ builds, but two-tier ranked
    instead of pure-frequency ranked: every word in `priority_words` is
    guaranteed a slot first (production-vocab survival must be 100% for run
    C, which pure frequency ranking cannot promise — see run B, where it
    only hit 84.6%), then the remaining budget is filled with the most
    frequent words from `vocabulary_words` that aren't already in.

    AgentTokenizer.__init__ has no hook for custom ranking, so the handful of
    attribute-assignment lines below (words/word_to_index/index_to_word/the
    special-token indices) are copied verbatim from it — that part is
    intentionally identical to the real class, only the ranking that feeds
    it is new.
    """
    counter = Counter(
        word for word in (str(raw).lower().strip() for raw in vocabulary_words)
        if word and word not in AgentTokenizer.SPECIAL_TOKENS
    )
    budget = max_vocab_size - len(AgentTokenizer.SPECIAL_TOKENS)

    priority_unique = list(dict.fromkeys(
        word for word in (str(raw).lower().strip() for raw in priority_words)
        if word and word not in AgentTokenizer.SPECIAL_TOKENS
    ))
    if len(priority_unique) > budget:
        raise ValueError(
            f"priority vocabulary ({len(priority_unique)} words) exceeds the "
            f"max_vocab_size budget ({budget}) — cannot guarantee 100% survival"
        )
    # Deterministic order: rank priority words by their own frequency in the
    # combined pool too (0 if a production word never occurs in it).
    priority_unique.sort(key=lambda w: counter.get(w, 0), reverse=True)

    priority_set = set(priority_unique)
    remaining_budget = budget - len(priority_unique)
    filler = [
        word for word, _count in counter.most_common()
        if word not in priority_set
    ][:remaining_budget]

    collected = priority_unique + filler

    tokenizer = AgentTokenizer.__new__(AgentTokenizer)
    tokenizer.words = list(AgentTokenizer.SPECIAL_TOKENS) + collected
    tokenizer.word_to_index = {word: index for index, word in enumerate(tokenizer.words)}
    tokenizer.index_to_word = {index: word for word, index in tokenizer.word_to_index.items()}
    tokenizer.pad_index = tokenizer.word_to_index[AgentTokenizer.PAD]
    tokenizer.unk_index = tokenizer.word_to_index[AgentTokenizer.UNK]
    tokenizer.q_index = tokenizer.word_to_index[AgentTokenizer.Q]
    tokenizer.a_index = tokenizer.word_to_index[AgentTokenizer.A]
    tokenizer.end_index = tokenizer.word_to_index[AgentTokenizer.END]
    return tokenizer


def run_training(trainer: LanguageModelTrainer, label: str) -> tuple[list[float], list[float | None]]:
    # Mirrors LanguageModelTrainer.train() exactly (same _run_epoch calls,
    # same early-stop constants) but also keeps the per-epoch val losses so
    # they can go in the results file, not just stdout.
    train_losses: list[float] = []
    val_losses: list[float | None] = []
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(EPOCHS):
        random.shuffle(trainer.train_samples)
        train_loss = trainer._run_epoch(trainer.train_samples, BATCH_SIZE, train=True)
        train_losses.append(train_loss)

        if trainer.val_samples:
            val_loss = trainer._run_epoch(trainer.val_samples, BATCH_SIZE, train=False)
            val_losses.append(val_loss)
            print(
                f"[{label}] epoch {epoch + 1}/{EPOCHS} — "
                f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )
            if best_val_loss - val_loss > LanguageModelTrainer.EARLY_STOP_MIN_DELTA:
                best_val_loss = val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= LanguageModelTrainer.EARLY_STOP_PATIENCE:
                    print(f"[{label}] early stopping at epoch {epoch + 1}/{EPOCHS}")
                    break
        else:
            val_losses.append(None)
            print(f"[{label}] epoch {epoch + 1}/{EPOCHS} — train loss {train_loss:.4f} (no val split)")

    trainer.model.train()
    return train_losses, val_losses


def generate_probes(trainer: LanguageModelTrainer) -> dict[str, str]:
    # generate_reply() returns list[str] for the word tokenizer, str for BPE
    # (brain/language_model.py's generate_reply docstring) — join only applies
    # to the former; naively " ".join()-ing a BPE string would space out its
    # individual characters instead.
    replies = {}
    for question in PROBE_QUESTIONS:
        reply = trainer.model.generate_reply(question, temperature=GENERATION_TEMPERATURE)
        if isinstance(reply, str):
            text = reply
        else:
            text = " ".join(reply) if reply else ""
        replies[question] = text if text else "(empty)"
    return replies


def train_two_phase(
    scratch: LanguageModelTrainer,
    base_samples: list[str],
    pre_oversample_train: list[str],
    val_samples: list[str],
    reading_samples: list[str],
    production_vocab_words: list[str],
) -> dict:
    combined_pool = base_samples + reading_samples
    vocabulary_words = scratch._collect_vocabulary(combined_pool)
    tokenizer = build_priority_tokenizer(vocabulary_words, production_vocab_words, MAX_VOCAB_SIZE)

    survived_content = len(set(production_vocab_words) & set(tokenizer.words))
    if survived_content != len(set(production_vocab_words)):
        raise RuntimeError(
            f"build_priority_tokenizer failed to guarantee 100% production vocab "
            f"survival: {survived_content}/{len(set(production_vocab_words))}"
        )
    # Specials (PAD/UNK/Q/A/END) are always present in every AgentTokenizer by
    # construction, so they trivially survive too — add them back in for a
    # count that's directly comparable to CONTROL/READING's "survived" row,
    # which counts against the full production vocab (specials included).
    survived = survived_content + len(AgentTokenizer.SPECIAL_TOKENS)
    print(f"[TWOPHASE] vocab size {tokenizer.vocab_size}, production vocab survived {survived}/{survived}")

    device = torch.device(DEVICE)
    model = AgentLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pad_index=tokenizer.pad_index,
        tokenizer=tokenizer,
    ).to(device)

    trainer = LanguageModelTrainer.__new__(LanguageModelTrainer)
    trainer.tokenizer_type = "word"  # required by save()
    trainer.memory_path = MEMORY_PATH
    trainer.device = device
    trainer.tokenizer = tokenizer
    trainer.model = model
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    trainer.val_samples = list(val_samples)

    start = time.monotonic()

    print("\n=== training TWOPHASE (phase 1: reading text only, lr=1e-3) ===")
    trainer.samples = reading_samples
    trainer.train_samples = list(reading_samples)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    phase1_train_losses, phase1_val_losses = run_training(trainer, "TWOPHASE-P1")

    print("\n=== training TWOPHASE (phase 2: base pool only, lr=1e-4, continuing phase 1 weights) ===")
    trainer.samples = base_samples
    trainer.train_samples = scratch._oversample_pairs(pre_oversample_train)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LEARNING_RATE)
    phase2_train_losses, phase2_val_losses = run_training(trainer, "TWOPHASE-P2")

    elapsed = time.monotonic() - start
    print(f"[TWOPHASE] training finished in {elapsed:.1f}s")

    trainer.save(TWOPHASE_OUTPUT)
    print(f"[TWOPHASE] weights saved to {TWOPHASE_OUTPUT}")

    probes = generate_probes(trainer)

    return {
        "trainer": trainer,
        # Aliased to phase 2 (the final trained state) so the shared
        # CONTROL/READING/TWOPHASE summary table below can treat all three
        # runs uniformly; phase1_*/phase2_* stay available for the dedicated
        # per-phase rows.
        "train_losses": phase2_train_losses,
        "val_losses": phase2_val_losses,
        "phase1_train_losses": phase1_train_losses,
        "phase1_val_losses": phase1_val_losses,
        "phase2_train_losses": phase2_train_losses,
        "phase2_val_losses": phase2_val_losses,
        "vocab_size": tokenizer.vocab_size,
        "survived": survived,
        "probes": probes,
        "elapsed": elapsed,
    }


def main() -> None:
    print(f"[reading_test] loading production vocab from {PRODUCTION_CHECKPOINT} (read-only)")
    _prod_model, prod_tokenizer = LanguageModelTrainer.load_model(
        PRODUCTION_CHECKPOINT, map_location="cpu"
    )
    production_vocab = set(prod_tokenizer.words)

    print(f"[reading_test] loading base training pool from {MEMORY_PATH} (one read, shared by every run)")
    scratch, raw, normalized = load_base_split()
    base_samples, pre_oversample_train, val_samples = (
        normalized["base_samples"], normalized["pre_oversample_train"], normalized["val_samples"]
    )
    base_samples_raw, pre_oversample_train_raw, val_samples_raw = (
        raw["base_samples"], raw["pre_oversample_train"], raw["val_samples"]
    )
    print(
        f"[reading_test] base pool: {len(base_samples)} samples, "
        f"{len(pre_oversample_train)} train (pre-oversample) / {len(val_samples)} val "
        f"(same split, raw+normalized representations)"
    )

    reading_samples, clean_files = build_reading_samples(normalize=True)
    reading_samples_raw, _ = build_reading_samples(normalize=False)
    print(
        f"[reading_test] {len(reading_samples)} reading samples (normalized) / "
        f"{len(reading_samples_raw)} (raw) from {len(clean_files)} file(s) under {CLEAN_DIR}"
    )

    runs = {}
    for label, pool_samples, train_pre_oversample, output_path in (
        ("CONTROL", base_samples, pre_oversample_train, CONTROL_OUTPUT),
        ("READING", base_samples + reading_samples, pre_oversample_train + reading_samples, READING_OUTPUT),
    ):
        print(f"\n=== training {label} ===")
        start = time.monotonic()
        trainer = build_trainer(scratch, pool_samples, train_pre_oversample, val_samples)
        print(
            f"[{label}] vocab size {trainer.tokenizer.vocab_size}, "
            f"{len(trainer.train_samples)} train samples, {len(trainer.val_samples)} val samples"
        )
        train_losses, val_losses = run_training(trainer, label)
        elapsed = time.monotonic() - start
        print(f"[{label}] training finished in {elapsed:.1f}s")

        trainer.save(output_path)
        print(f"[{label}] weights saved to {output_path}")

        survived = len(production_vocab & set(trainer.tokenizer.words))
        probes = generate_probes(trainer)

        runs[label] = {
            "trainer": trainer,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "vocab_size": trainer.tokenizer.vocab_size,
            "survived": survived,
            "probes": probes,
            "elapsed": elapsed,
        }

    production_vocab_words = [w for w in prod_tokenizer.words if w not in AgentTokenizer.SPECIAL_TOKENS]
    runs["TWOPHASE"] = train_two_phase(
        scratch, base_samples, pre_oversample_train, val_samples, reading_samples, production_vocab_words
    )

    for label, pool_samples_raw, train_pre_oversample_raw, output_path in (
        ("BPE", base_samples_raw, pre_oversample_train_raw, BPE_OUTPUT),
        (
            "BPE_READING",
            base_samples_raw + reading_samples_raw,
            pre_oversample_train_raw + reading_samples_raw,
            BPE_READING_OUTPUT,
        ),
    ):
        print(f"\n=== training {label} ===")
        start = time.monotonic()
        bpe_trainer = build_bpe_trainer(scratch, pool_samples_raw, train_pre_oversample_raw, val_samples_raw)
        print(
            f"[{label}] vocab size {bpe_trainer.tokenizer.vocab_size} (datos_lectura/tokenizer.json), "
            f"{len(bpe_trainer.train_samples)} train samples, {len(bpe_trainer.val_samples)} val samples"
        )
        bpe_train_losses, bpe_val_losses = run_training(bpe_trainer, label)
        bpe_elapsed = time.monotonic() - start
        print(f"[{label}] training finished in {bpe_elapsed:.1f}s")

        bpe_trainer.save(output_path)
        print(f"[{label}] weights saved to {output_path}")

        bpe_probes = generate_probes(bpe_trainer)

        runs[label] = {
            "trainer": bpe_trainer,
            "train_losses": bpe_train_losses,
            "val_losses": bpe_val_losses,
            "vocab_size": bpe_trainer.tokenizer.vocab_size,
            "survived": None,  # not a word vocab — see BPE_VAL_LOSS_CAVEAT / write_results
            "probes": bpe_probes,
            "elapsed": bpe_elapsed,
        }

    write_results(production_vocab, base_samples, val_samples, reading_samples, clean_files, runs)


def write_results(production_vocab, base_samples, val_samples, reading_samples, clean_files, runs) -> None:
    lines = []
    lines.append("=== READING TEST RESULTS ===")
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"memory source: {MEMORY_PATH}")
    lines.append(f"reading source: {CLEAN_DIR} ({len(clean_files)} file(s), {len(reading_samples)} samples)")
    lines.append(
        f"epochs={EPOCHS} batch_size={BATCH_SIZE} max_vocab_size={MAX_VOCAB_SIZE} "
        f"embedding_dim={EMBEDDING_DIM} nhead={NHEAD} num_layers={NUM_LAYERS} "
        f"dim_feedforward={DIM_FEEDFORWARD} dropout={DROPOUT} lr={LEARNING_RATE}"
    )
    lines.append(f"production vocab (current {PRODUCTION_CHECKPOINT.name}): {len(production_vocab)} words")
    lines.append(f"shared base pool: {len(base_samples)} samples")
    lines.append(f"shared validation set (identical for both runs): {len(val_samples)} samples")
    lines.append("")

    run_labels = ("CONTROL", "READING", "TWOPHASE", "BPE", "BPE_READING")
    header = f"{'':32}" + "".join(f"{label:>18}" for label in run_labels)
    lines.append(header)
    lines.append("-" * len(header))

    def row(label, key_fn):
        values = "".join(f"{key_fn(runs[r]):>18}" for r in run_labels)
        lines.append(f"{label:32}{values}")

    row("final train loss", lambda r: f"{r['train_losses'][-1]:.4f}")
    row("final val loss", lambda r: f"{r['val_losses'][-1]:.4f}" if r['val_losses'] and r['val_losses'][-1] is not None else "n/a")
    row("epochs run", lambda r: str(len(r["train_losses"])))
    row("train samples", lambda r: str(len(r["trainer"].train_samples)))
    row("vocab size", lambda r: str(r["vocab_size"]))
    row(
        "production vocab survived",
        lambda r: (
            f"{r['survived']}/{len(production_vocab)} ({r['survived'] / len(production_vocab) * 100:.1f}%)"
            if r["survived"] is not None else "n/a (not word vocab)"
        ),
    )
    row("training time (s)", lambda r: f"{r['elapsed']:.1f}")
    lines.append("")
    lines.append(BPE_VAL_LOSS_CAVEAT)
    lines.append("")

    lines.append("--- TWOPHASE: val loss by phase (same held-out val set both times) ---")
    twophase = runs["TWOPHASE"]
    p1_val = twophase["phase1_val_losses"][-1] if twophase["phase1_val_losses"] and twophase["phase1_val_losses"][-1] is not None else None
    p2_val = twophase["phase2_val_losses"][-1] if twophase["phase2_val_losses"] and twophase["phase2_val_losses"][-1] is not None else None
    lines.append(f"after phase 1 (reading only, {len(twophase['phase1_train_losses'])} epochs): val loss {p1_val:.4f}" if p1_val is not None else "after phase 1: n/a")
    lines.append(f"after phase 2 (base pool only, lr=1e-4, {len(twophase['phase2_train_losses'])} epochs): val loss {p2_val:.4f}" if p2_val is not None else "after phase 2: n/a")
    lines.append("")

    lines.append("--- per-epoch losses ---")
    for label in ("CONTROL", "READING", "BPE", "BPE_READING"):
        lines.append(f"{label}:")
        run = runs[label]
        for i, (t_loss, v_loss) in enumerate(zip(run["train_losses"], run["val_losses"]), start=1):
            v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
            lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("TWOPHASE phase 1 (reading only):")
    for i, (t_loss, v_loss) in enumerate(zip(twophase["phase1_train_losses"], twophase["phase1_val_losses"]), start=1):
        v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
        lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("TWOPHASE phase 2 (base pool only, lr=1e-4):")
    for i, (t_loss, v_loss) in enumerate(zip(twophase["phase2_train_losses"], twophase["phase2_val_losses"]), start=1):
        v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
        lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("")

    lines.append("--- probe generations (paraphrases of real human_taught questions) ---")
    for question in PROBE_QUESTIONS:
        lines.append(f"Q: {question}")
        for label in run_labels:
            lines.append(f"  {label}: {runs[label]['probes'][question]}")
    lines.append("")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    RESULTS_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n[reading_test] results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
