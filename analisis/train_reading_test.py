"""Standalone script: test the CAPACITY hypothesis alongside the reading-text
hypothesis.

FIRST run (2026-08-13, lr=1e-3 for every config, see git history for the full
script that produced it) trained CONTROL_SMALL, CONTROL_BIG, READING_SMALL
and READING_BIG. READING_BIG diverged: val loss 7.04 at epoch 1 and rising,
train loss stuck around 6.4x the whole run — a learning-rate failure at
11.7M parameters (1e-3 is too aggressive for a model this size trained from
random init), not a real result on whether capacity helps reading text.

THIS run re-trains ONLY CONTROL_BIG and READING_BIG, with:
  - lr = 1e-4 (down from 1e-3)
  - linear warmup over the first 500 optimizer steps
  - gradient clipping at norm 1.0
  - epoch budget 40 (lower lr needs more epochs to get anywhere)

CONTROL_SMALL and READING_SMALL are NOT retrained — their results from the
first run are carried over verbatim as PREVIOUS_SMALL_RESULTS below (copied
from analisis/reading_test_results.txt as it stood after that run) purely
for side-by-side comparison in the report. Their checkpoints on disk
(language_model_control_small.pt, language_model_reading_small.pt) are
untouched.

Pool construction (CONTROL vs READING) and vocabulary construction are
unchanged from every previous version of this script:

  CONTROL — exactly the production training data (human_taught pairs +
    episode thoughts + corpus_agente.txt), loaded the same way
    brain/language_model.py's own main() does it. Tokenizer: plain
    frequency-ranked AgentTokenizer, since this pool already IS the
    production pool.
  READING — CONTROL's data plus datos_lectura/clean/*.txt, one sample per
    cleaned line. Tokenizer: build_priority_tokenizer, which guarantees a
    slot for every one of the current production vocabulary's words before
    any reading-corpus word fills the remaining budget.

Both pools (and their tokenizers) are still built fresh in this run, even
though only the BIG architecture trains on them now — READING_BIG needs the
READING pool exactly as before, and rebuilding it here (rather than trying to
reuse anything from the first run) is what keeps it byte-identical to what
CONTROL_SMALL/READING_SMALL actually saw, since none of the pool-building
code changed.

Training itself can no longer go through LanguageModelTrainer._run_epoch()
for the BIG configs: that method has no hook for a per-step LR schedule or
gradient clipping between backward() and optimizer.step(). run_training()
below reimplements the batch loop for that reason (still calling the
trainer's own _build_batch/model/optimizer/loss_fn — nothing about the model
or data pipeline is reimplemented, only the step in between). Validation
epochs need neither warmup nor clipping, so those still go straight through
_run_epoch(..., train=False) unchanged.

Writes language_model_control_big.pt and language_model_reading_big.pt only
(language_model_control_small.pt / language_model_reading_small.pt from the
first run are left alone), plus a summary at analisis/reading_test_results.txt
that reports all four configs.

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
from torch.nn.utils import clip_grad_norm_

from brain.language_model import AgentLanguageModel, AgentTokenizer, LanguageModelTrainer, normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = PROJECT_ROOT / "episodic_memory.sqlite3"
CLEAN_DIR = PROJECT_ROOT / "datos_lectura" / "clean"
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "language_model.pt"
RESULTS_PATH = Path(__file__).resolve().parent / "reading_test_results.txt"

# Vocabulary / optimization hyperparameters shared by every run — unchanged
# from the first run. max_vocab_size 16384 (raised from production's 8192)
# so the combined base-pool + reading-text vocabulary isn't silently
# truncated (see how "production vocab survived" is reported below).
MAX_VOCAB_SIZE = 16384
DROPOUT = 0.1
DEVICE = "cpu"  # CPU only on this machine, per CLAUDE.md
BATCH_SIZE = 16
GENERATION_TEMPERATURE = 0.8

# The two architectures. SMALL is today's production architecture and is
# only listed here for the report header — it is NOT retrained this run (see
# PREVIOUS_SMALL_RESULTS). BIG is the one actually trained below.
ARCH_CONFIGS = {
    "SMALL": {"embedding_dim": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256},
    "BIG": {"embedding_dim": 256, "nhead": 8, "num_layers": 6, "dim_feedforward": 512},
}

POOL_LABELS = ("CONTROL", "READING")
ARCH_LABELS = ("SMALL", "BIG")
CONFIG_LABELS = tuple(f"{pool}_{arch}" for pool in POOL_LABELS for arch in ARCH_LABELS)

SEED = 0

# BIG-only hyperparameters for this run — the fix for READING_BIG's
# divergence at lr=1e-3 in the first run.
BIG_LEARNING_RATE = 1e-4
BIG_WARMUP_STEPS = 500
BIG_GRAD_CLIP_NORM = 1.0
BIG_EPOCHS_BUDGET = 40
# Unchanged from the first run — looser than brain/language_model.py's own
# EARLY_STOP_PATIENCE=3. MIN_DELTA (0.01) is reused as-is from production via
# LanguageModelTrainer.EARLY_STOP_MIN_DELTA.
EARLY_STOP_PATIENCE = 5

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

# Carried over verbatim from analisis/reading_test_results.txt as written by
# the first run (generated 2026-08-13 14:05:25, lr=1e-3, epoch budget 30,
# same early-stop patience/min-delta as this run). NOT retrained here — see
# module docstring. Structurally matches the `runs[label]` dict this script
# builds for freshly trained configs, so write_results() can treat both the
# same way.
PREVIOUS_SMALL_RESULTS = {
    "CONTROL_SMALL": {
        "pool": "CONTROL",
        "arch": "SMALL",
        "train_losses": [5.1621, 3.5215, 2.5241, 1.8663, 1.4670, 1.2356, 1.0889, 0.9924, 0.9311, 0.8753],
        "val_losses": [4.2311, 3.4884, 3.0590, 2.9236, 2.8693, 2.8816, 2.9128, 2.9703, 2.9966, 3.0616],
        "vocab_size": 5128,
        "survived": 5128,
        "probes": {
            "que significa la humedad": "emocion mas cercano a la persona",
            "explicame que es una red neuronal": "un sistema solar hasta como un humano donde se parece",
            "que es el aprendizaje automatico": "una forma de inteligencia artificial que aprende patrones de datos",
            "como se hace un experimento cientifico": "con gabito que es una idea antes de canciones",
            "en que momento naciste": "a veces que la relatividad que revoluciono la mente descansa",
            "puedes sentir dolor": "no de forma fisica pero exploro mi mundo moderno como se vuelve familiar",
            "que significa viajar": "permite conocer culturas paisajes o forma de crecer",
            "explicame que es un eclipse": "cuando la tierra se interpone entre la tierra y el sol",
            "que significa la contaminacion": "algo parecido a cuando mis predicciones se siente",
            "tienes una historia propia": "si esa es parte mis errores para que se pueden predecir",
        },
        "elapsed": 300.6,
        "param_count": 1_780_872,
        "best_val_loss": 2.8693,
        "best_epoch": 5,
        "epochs_run": 10,
        "epochs_budget": 30,
        "output_path": PROJECT_ROOT / "language_model_control_small.pt",
        "learning_rate": 1e-3,
        "warmup_steps": None,
        "grad_clip_norm": None,
    },
    "READING_SMALL": {
        "pool": "READING",
        "arch": "SMALL",
        "train_losses": [5.8350, 5.1856, 4.8809, 4.6510, 4.4715, 4.3229, 4.1979, 4.0886, 3.9979, 3.9165, 3.8428, 3.7764, 3.7135, 3.6600, 3.6044],
        "val_losses": [4.6400, 4.0050, 3.6708, 3.4750, 3.3659, 3.2453, 3.1909, 3.1871, 3.1232, 3.1087, 3.1143, 3.1231, 3.1027, 3.1347, 3.1192],
        "vocab_size": 16384,
        "survived": 5136,
        "probes": {
            "que significa la humedad": "es la parte de la tierra y el objeto de su vida diaria",
            "explicame que es una red neuronal": "un conjunto de reglas para transmitir mi tiempo",
            "que es el aprendizaje automatico": "un ser vivo o toma o objeto que la persona se da un signo",
            "como se hace un experimento cientifico": "un documento digital que se combina y emociones y se lo que lo aprendi",
            "en que momento naciste": "un sistema de escritura suele experimentar y dos dos semanas",
            "puedes sentir dolor": "no mi no depende de la confianza en mucho tiempo",
            "que significa viajar": "permite conocer un valor y eso lo aprendi conversando",
            "explicame que es un eclipse": "cuando una region donde no se puede entrar en el oceano",
            "que significa la contaminacion": "la comida que la ciencia es el objeto mas grande",
            "tienes una historia propia": "si es algo mas poco pero no puede significar que decide con",
        },
        "elapsed": 4646.3,
        "param_count": 4_673_664,
        "best_val_loss": 3.1027,
        "best_epoch": 13,
        "epochs_run": 15,
        "epochs_budget": 30,
        "output_path": PROJECT_ROOT / "language_model_reading_small.pt",
        "learning_rate": 1e-3,
        "warmup_steps": None,
        "grad_clip_norm": None,
    },
}


def set_seed(seed: int) -> None:
    """Seeds every source of randomness a run consumes AFTER this point:
    weight init (torch, inside AgentLanguageModel()), _oversample_pairs'
    shuffle, and each epoch's random.shuffle(train_samples) in run_training.
    Must be called fresh before each (pool, arch) run, and never before
    load_base_split() — the val split must stay seed-independent."""
    random.seed(seed)
    torch.manual_seed(seed)


def output_path_for(label: str) -> Path:
    return PROJECT_ROOT / f"language_model_{label.lower()}.pt"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def best_val_loss_and_epoch(val_losses: list[float | None]) -> tuple[float | None, int | None]:
    """1-indexed epoch (matching the "epoch N/<epochs>" prints) at which the
    lowest val loss occurred — this is the literal minimum, independent of
    the EARLY_STOP_MIN_DELTA threshold run_training uses to decide when to
    STOP, so it answers "when was the best point" exactly rather than "when
    did the early-stop bookkeeping last reset"."""
    values = [(i + 1, v) for i, v in enumerate(val_losses) if v is not None]
    if not values:
        return None, None
    epoch, val = min(values, key=lambda iv: iv[1])
    return val, epoch


def build_reading_samples() -> tuple[list[str], list[Path]]:
    """One sample per cleaned line, normalized the same way the word
    tokenizer's own vocabulary is (normalize_text — lowercase + strip
    accents), then deduped."""
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
                samples.append(f"{AgentTokenizer.A} {normalize_text(line)} {AgentTokenizer.END}")
    return list(dict.fromkeys(samples)), clean_files


def load_base_split() -> tuple[LanguageModelTrainer, list[str], list[str], list[str]]:
    """Loads the base pool ONCE with tokenizer_type="word" (so
    _load_training_samples applies normalize_text() itself — see its own
    text_filter logic) and splits it ONCE via _split_train_val. That split is
    the single shared source of truth every run in this script reuses, and is
    identical to the split the first run (and its carried-over SMALL results)
    used — see this function's use in main() for why re-loading would
    silently break "same held-out set".

    This must run before any per-run seeding: _split_train_val's Fisher-Yates
    shuffle depends on the incoming list's order, which _load_training_samples
    produces via the (unseeded, global) `random` module. Calling either of
    them again after set_seed() has run would silently change which rows
    land in train vs val.

    scratch itself is a real LanguageModelTrainer instance (via __new__,
    __init__ skipped) used purely to call its own bound methods — safe since
    _load_training_samples/_split_train_val/_oversample_pairs/
    _collect_vocabulary only ever touch self.memory_path and
    self.tokenizer_type, both set below.
    """
    scratch = LanguageModelTrainer.__new__(LanguageModelTrainer)
    scratch.tokenizer_type = "word"
    scratch.memory_path = MEMORY_PATH

    base_samples = scratch._load_training_samples()
    if not base_samples:
        raise ValueError(f"No training samples found via {MEMORY_PATH}")

    pre_oversample_train, val_samples = scratch._split_train_val(base_samples)
    return scratch, base_samples, pre_oversample_train, val_samples


def build_priority_tokenizer(
    vocabulary_words: list[str], priority_words: list[str], max_vocab_size: int
) -> AgentTokenizer:
    """Same vocab format AgentTokenizer.__init__ builds, but two-tier ranked
    instead of pure-frequency ranked: every word in `priority_words` is
    guaranteed a slot first (production-vocab survival must be 100% for
    READING — pure frequency ranking cannot promise that), then the
    remaining budget is filled with the most frequent words from
    `vocabulary_words` that aren't already in. `vocabulary_words` is
    READING's own combined pool (base pool + reading text), so in practice
    the filler ends up almost entirely reading-corpus words, since
    production vocab already covers the base pool's frequent words.

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


def build_tokenizer_for_pool(
    pool_label: str, vocabulary_words: list[str], production_vocab_words: list[str]
) -> AgentTokenizer:
    if pool_label == "READING":
        return build_priority_tokenizer(vocabulary_words, production_vocab_words, MAX_VOCAB_SIZE)
    return AgentTokenizer(vocabulary_words, max_vocab_size=MAX_VOCAB_SIZE)


def build_trainer(
    scratch: LanguageModelTrainer,
    pool_samples: list[str],
    pre_oversample_train: list[str],
    val_samples: list[str],
    tokenizer: AgentTokenizer,
    arch: dict,
    learning_rate: float,
) -> LanguageModelTrainer:
    train_samples = scratch._oversample_pairs(pre_oversample_train)

    trainer = LanguageModelTrainer.__new__(LanguageModelTrainer)
    trainer.tokenizer_type = "word"  # required by save()
    trainer.memory_path = MEMORY_PATH
    trainer.device = torch.device(DEVICE)
    trainer.samples = pool_samples
    trainer.train_samples = train_samples
    trainer.val_samples = list(val_samples)
    trainer.tokenizer = tokenizer
    trainer.model = AgentLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=arch["embedding_dim"],
        nhead=arch["nhead"],
        num_layers=arch["num_layers"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=DROPOUT,
        pad_index=tokenizer.pad_index,
        tokenizer=tokenizer,
    ).to(trainer.device)
    # lr here is the warmup's target — run_training overrides param_groups[0]
    # ["lr"] every step, so this initial value is only what's in effect
    # before the first optimizer.step() call.
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=learning_rate)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    return trainer


def run_training(
    trainer: LanguageModelTrainer,
    label: str,
    epochs: int,
    base_lr: float,
    warmup_steps: int,
    grad_clip_norm: float,
) -> tuple[list[float], list[float | None]]:
    """Per-batch training loop for the BIG configs. Reimplemented rather than
    calling LanguageModelTrainer._run_epoch() for the train pass, because
    _run_epoch has no hook for a per-step LR schedule or gradient clipping
    between backward() and optimizer.step() — both required to fix the
    divergence the first run hit at lr=1e-3. Everything else (early
    stopping, patience, min delta) mirrors every previous version of this
    script exactly. Validation epochs need neither warmup nor clipping (no
    gradient computed), so those still go straight through
    trainer._run_epoch(..., train=False) unchanged.

    global_step counts optimizer.step() calls across ALL epochs of this run
    (not reset per epoch) — warmup_steps is a total budget, so a run whose
    epochs are shorter than warmup_steps batches carries the warmup into
    epoch 2+."""
    train_losses: list[float] = []
    val_losses: list[float | None] = []
    best_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    current_lr = 0.0

    for epoch in range(epochs):
        random.shuffle(trainer.train_samples)
        trainer.model.train(True)
        total_loss = 0.0
        batch_count = 0

        for start in range(0, len(trainer.train_samples), BATCH_SIZE):
            batch = trainer.train_samples[start : start + BATCH_SIZE]
            input_ids, target_ids = trainer._build_batch(batch)

            logits = trainer.model(input_ids)
            loss = trainer.loss_fn(
                logits.reshape(-1, trainer.tokenizer.vocab_size),
                target_ids.reshape(-1),
            )

            trainer.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(trainer.model.parameters(), grad_clip_norm)

            global_step += 1
            current_lr = base_lr * min(1.0, global_step / warmup_steps)
            for group in trainer.optimizer.param_groups:
                group["lr"] = current_lr

            trainer.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        train_loss = total_loss / max(batch_count, 1)
        train_losses.append(train_loss)

        if trainer.val_samples:
            val_loss = trainer._run_epoch(trainer.val_samples, BATCH_SIZE, train=False)
            val_losses.append(val_loss)
            print(
                f"[{label}] epoch {epoch + 1}/{epochs} — "
                f"train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {current_lr:.2e}"
            )
            if best_loss - val_loss > LanguageModelTrainer.EARLY_STOP_MIN_DELTA:
                best_loss = val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= EARLY_STOP_PATIENCE:
                    print(
                        f"[{label}] early stopping at epoch {epoch + 1}/{epochs} "
                        f"(no improvement for {EARLY_STOP_PATIENCE} consecutive epochs)"
                    )
                    break
        else:
            val_losses.append(None)
            print(
                f"[{label}] epoch {epoch + 1}/{epochs} — "
                f"train loss {train_loss:.4f} (no val split), lr {current_lr:.2e}"
            )

    trainer.model.train()
    return train_losses, val_losses


def generate_probes(trainer: LanguageModelTrainer) -> dict[str, str]:
    replies = {}
    for question in PROBE_QUESTIONS:
        reply = trainer.model.generate_reply(question, temperature=GENERATION_TEMPERATURE)
        text = " ".join(reply) if reply else ""
        replies[question] = text if text else "(empty)"
    return replies


def main() -> None:
    print(f"[reading_test] loading production vocab from {PRODUCTION_CHECKPOINT} (read-only)")
    _prod_model, prod_tokenizer = LanguageModelTrainer.load_model(
        PRODUCTION_CHECKPOINT, map_location="cpu"
    )
    production_vocab = set(prod_tokenizer.words)

    print(f"[reading_test] loading base training pool from {MEMORY_PATH} (one read, shared by every run)")
    scratch, base_samples, pre_oversample_train, val_samples = load_base_split()
    print(
        f"[reading_test] base pool: {len(base_samples)} samples, "
        f"{len(pre_oversample_train)} train (pre-oversample) / {len(val_samples)} val"
    )

    reading_samples, clean_files = build_reading_samples()
    print(
        f"[reading_test] {len(reading_samples)} reading samples from "
        f"{len(clean_files)} file(s) under {CLEAN_DIR}"
    )

    production_vocab_words = [w for w in prod_tokenizer.words if w not in AgentTokenizer.SPECIAL_TOKENS]

    pools = {
        "CONTROL": (base_samples, pre_oversample_train),
        "READING": (base_samples + reading_samples, pre_oversample_train + reading_samples),
    }

    # Rebuilt fresh (needed for training BIG on both pools), but deterministic
    # and pool-only (not arch-dependent) — should reproduce the exact same
    # vocab sizes / survival counts the first run's SMALL configs saw.
    tokenizers = {}
    vocab_survived_by_pool = {}
    for pool_label in POOL_LABELS:
        pool_samples, _ = pools[pool_label]
        vocabulary_words = scratch._collect_vocabulary(pool_samples)
        tokenizer = build_tokenizer_for_pool(pool_label, vocabulary_words, production_vocab_words)
        vocab_survived = len(production_vocab & set(tokenizer.words))
        if pool_label == "READING" and vocab_survived != len(production_vocab):
            raise RuntimeError(
                f"READING tokenizer failed to guarantee full production-vocab survival: "
                f"{vocab_survived}/{len(production_vocab)}"
            )
        print(
            f"[{pool_label}] vocab size {tokenizer.vocab_size}, "
            f"production vocab survived {vocab_survived}/{len(production_vocab)}"
        )
        tokenizers[pool_label] = tokenizer
        vocab_survived_by_pool[pool_label] = vocab_survived

    # Start from the carried-over SMALL results; only BIG configs train.
    runs = dict(PREVIOUS_SMALL_RESULTS)

    for pool_label in POOL_LABELS:
        pool_samples, train_pre_oversample = pools[pool_label]
        tokenizer = tokenizers[pool_label]
        config_label = f"{pool_label}_BIG"
        arch = ARCH_CONFIGS["BIG"]

        print(
            f"\n=== training {config_label} (seed={SEED}, lr={BIG_LEARNING_RATE}, "
            f"warmup_steps={BIG_WARMUP_STEPS}, grad_clip_norm={BIG_GRAD_CLIP_NORM}) ==="
        )
        set_seed(SEED)
        start = time.monotonic()
        trainer = build_trainer(
            scratch, pool_samples, train_pre_oversample, val_samples, tokenizer, arch, BIG_LEARNING_RATE
        )
        param_count = count_parameters(trainer.model)
        print(
            f"[{config_label}] {len(trainer.train_samples)} train samples, "
            f"{len(trainer.val_samples)} val samples, {param_count:,} parameters"
        )
        train_losses, val_losses = run_training(
            trainer, config_label, BIG_EPOCHS_BUDGET, BIG_LEARNING_RATE, BIG_WARMUP_STEPS, BIG_GRAD_CLIP_NORM
        )
        elapsed = time.monotonic() - start
        print(f"[{config_label}] training finished in {elapsed:.1f}s")

        output_path = output_path_for(config_label)
        trainer.save(output_path)
        print(f"[{config_label}] weights saved to {output_path}")

        probes = generate_probes(trainer)
        best_val, best_epoch = best_val_loss_and_epoch(val_losses)

        runs[config_label] = {
            "pool": pool_label,
            "arch": "BIG",
            "train_losses": train_losses,
            "val_losses": val_losses,
            "vocab_size": tokenizer.vocab_size,
            "survived": vocab_survived_by_pool[pool_label],
            "probes": probes,
            "elapsed": elapsed,
            "param_count": param_count,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_run": len(train_losses),
            "epochs_budget": BIG_EPOCHS_BUDGET,
            "output_path": output_path,
            "learning_rate": BIG_LEARNING_RATE,
            "warmup_steps": BIG_WARMUP_STEPS,
            "grad_clip_norm": BIG_GRAD_CLIP_NORM,
        }

    write_results(production_vocab, base_samples, val_samples, reading_samples, clean_files, runs)


def write_results(production_vocab, base_samples, val_samples, reading_samples, clean_files, runs) -> None:
    lines = []
    lines.append("=== READING TEST RESULTS (capacity hypothesis — BIG re-run) ===")
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"memory source: {MEMORY_PATH}")
    lines.append(f"reading source: {CLEAN_DIR} ({len(clean_files)} file(s), {len(reading_samples)} samples)")
    arch_str = ", ".join(
        f"{label}=(embedding_dim={cfg['embedding_dim']}, nhead={cfg['nhead']}, "
        f"num_layers={cfg['num_layers']}, dim_feedforward={cfg['dim_feedforward']})"
        for label, cfg in ARCH_CONFIGS.items()
    )
    lines.append(f"architectures: {arch_str}")
    lines.append(
        f"CONTROL_SMALL / READING_SMALL: carried over from the first run (not retrained) — "
        f"lr=1e-3, epoch budget 30, no warmup, no grad clipping"
    )
    lines.append(
        f"CONTROL_BIG / READING_BIG (this run): lr={BIG_LEARNING_RATE}, warmup_steps={BIG_WARMUP_STEPS} "
        f"(linear), grad_clip_norm={BIG_GRAD_CLIP_NORM}, epoch budget {BIG_EPOCHS_BUDGET}"
    )
    lines.append(
        f"shared settings: early stop after {EARLY_STOP_PATIENCE} epochs without a "
        f">{LanguageModelTrainer.EARLY_STOP_MIN_DELTA} val loss improvement, batch_size={BATCH_SIZE}, "
        f"max_vocab_size={MAX_VOCAB_SIZE}, dropout={DROPOUT}, seed={SEED}"
    )
    lines.append(f"production vocab (current {PRODUCTION_CHECKPOINT.name}): {len(production_vocab)} words")
    lines.append(f"shared base pool: {len(base_samples)} samples")
    lines.append(f"shared validation set (identical for all {len(CONFIG_LABELS)} runs): {len(val_samples)} samples")
    for pool_label in POOL_LABELS:
        example = next(r for r in runs.values() if r["pool"] == pool_label)
        lines.append(
            f"{pool_label} pool vocab: size {example['vocab_size']}, "
            f"production vocab survived {example['survived']}/{len(production_vocab)}"
        )
    lines.append("")

    lines.append("--- results per run ---")
    header = f"{'config':16}{'params':>14}{'best val loss':>16}{'@ epoch':>10}{'status':>34}{'time (s)':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for label in CONFIG_LABELS:
        r = runs[label]
        converged = r["epochs_run"] < r["epochs_budget"]
        status = (
            f"early-stopped at epoch {r['epochs_run']}"
            if converged
            else f"hit epoch budget ({r['epochs_budget']}) without early-stopping"
        )
        lines.append(
            f"{label:16}{r['param_count']:>14,}{r['best_val_loss']:>16.4f}"
            f"{r['best_epoch']:>10}{status:>34}{r['elapsed']:>10.1f}"
        )
        note = "carried over from first run" if label.endswith("_SMALL") else f"saved to {r['output_path'].name}"
        lines.append(f"{'':16}{note}")
    lines.append("")

    lines.append("--- READING_BIG train loss sanity check (is it actually descending this time?) ---")
    reading_big = runs["READING_BIG"]
    epoch1_train = reading_big["train_losses"][0]
    epoch2_train = reading_big["train_losses"][1] if len(reading_big["train_losses"]) > 1 else None
    if epoch2_train is not None:
        descending = epoch2_train < epoch1_train
        lines.append(
            f"epoch 1 train loss: {epoch1_train:.4f}   epoch 2 train loss: {epoch2_train:.4f}   "
            f"=> {'descending (fix worked)' if descending else 'NOT descending — still diverging'}"
        )
    else:
        lines.append(f"epoch 1 train loss: {epoch1_train:.4f}   (run stopped before epoch 2)")
    lines.append("")

    lines.append("--- per-epoch losses (all configs) ---")
    for label in CONFIG_LABELS:
        run = runs[label]
        lines.append(f"{label}:")
        for i, (t_loss, v_loss) in enumerate(zip(run["train_losses"], run["val_losses"]), start=1):
            v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
            lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("")

    lines.append("--- probe generations (all four configs, side by side) ---")
    header = f"{'':32}" + "".join(f"{label:>22}" for label in CONFIG_LABELS)
    lines.append(header)
    lines.append("-" * len(header))
    for question in PROBE_QUESTIONS:
        lines.append(f"Q: {question}")
        for label in CONFIG_LABELS:
            reply = runs[label]["probes"][question]
            lines.append(f"  {label}: {reply}")
    lines.append("")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    RESULTS_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n[reading_test] results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
