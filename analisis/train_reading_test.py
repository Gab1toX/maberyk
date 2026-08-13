"""Standalone script: test whether adding free Spanish reading text to
training improves the native Transformer's generalization.

Trains two configurations from random init, same architecture/hyperparameters
as production (brain/language_model.py), on the SAME held-out validation set
so val loss is directly comparable:

  CONTROL — exactly the production training data (human_taught pairs +
    episode thoughts + corpus_agente.txt), loaded the same way
    brain/language_model.py's own main() does it.
  READING — CONTROL's data plus datos_lectura/clean/*.txt, one sample per
    cleaned line (this codebase already treats "one line = one training
    sample" for corpus_agente.txt/episode thoughts, so reading text is held
    to the same convention rather than attempting grammatical sentence
    splitting on text that ingest.py has already stripped of
    sentence-ending punctuation).

Dropped: the TWOPHASE, BPE and BPE_READING configurations this script used
to also run. Their results were settled (see reading_test_results.txt run
history) — TWOPHASE needed a hand-built priority tokenizer just to reach
100% production-vocab survival, and BPE/BPE_READING's val loss was never
comparable to the word-level runs' in the first place (different vocabulary,
different-length sequences per sample). Keeping them here would only add
dead weight to a script now focused on a single question: does reading text
help, and is the effect bigger than seed noise.

Each configuration is trained with 3 random seeds (SEEDS below) so a
CONTROL-vs-READING difference can be judged against the spread the SAME
configuration produces just from weight init / shuffling order — see
write_results()'s verdict line, which only calls a difference real if it
exceeds the larger of the two configs' seed-to-seed standard deviations.

CONTROL trains for 10 epochs; READING trains for up to 30 — CONTROL already
overfits by epoch 5 on its own (small) pool, so more epochs would only push
it further into memorization, while READING's much larger combined pool may
need the extra budget to actually converge. Both configs share the same
early-stopping rule (5 consecutive epochs without a >EARLY_STOP_MIN_DELTA
improvement in val loss — see EARLY_STOP_PATIENCE, deliberately looser than
brain/language_model.py's own production default of 3, since this script
wants to see whether READING keeps improving past where production would
have already cut it off) and report.txt records the epoch each run's best
val loss actually occurred at, so a run that used its full epoch budget
without early-stopping (best epoch == last epoch) is visibly distinguishable
from one that converged and plateaued.

READING's tokenizer guarantees a slot for every one of the current
production vocabulary's 5136 words before any reading-corpus word fills the
remaining budget (build_priority_tokenizer) — pure frequency ranking cannot
promise that (see git history: an earlier run of this script lost ~15% of
production vocab that way), and losing production words mid-experiment would
confound "did reading text help" with "did we also cripple the model's
existing vocabulary". CONTROL keeps plain frequency-ranked AgentTokenizer
since its pool already IS the production pool.

The validation split is loaded and split exactly ONCE, before any per-seed
work starts, and reused by all 6 runs — see load_base_split()'s docstring
for why re-loading would silently break "same held-out set".

Every method actually used (_load_training_samples, _split_train_val,
_oversample_pairs, _collect_vocabulary, _run_epoch, generate_reply, save,
load_model) is LanguageModelTrainer's own — none of it is reimplemented, and
brain/language_model.py is never modified. build_priority_tokenizer is the
one genuinely new piece of logic (AgentTokenizer itself only supports pure-
frequency ranking).

Writes language_model_control_s{seed}.pt and language_model_reading_s{seed}.pt
for each seed in SEEDS (never language_model.pt or agent_state.pt), plus a
summary at analisis/reading_test_results.txt.

Usage:
    python analisis/train_reading_test.py
"""

import random
import statistics
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
RESULTS_PATH = Path(__file__).resolve().parent / "reading_test_results.txt"

# Same architecture/hyperparameters production training uses (brain/
# language_model.py's own LanguageModelTrainer defaults / main() CLI
# defaults) — nothing here changes the model. max_vocab_size is raised from
# production's 8192 to 16384 so the combined base-pool + reading-text
# vocabulary isn't silently truncated the way READING's single-seed run was
# before (see how "production vocab survived" is reported below).
MAX_VOCAB_SIZE = 16384
EMBEDDING_DIM = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
LEARNING_RATE = 1e-3
DEVICE = "cpu"  # CPU only on this machine, per CLAUDE.md
BATCH_SIZE = 16
GENERATION_TEMPERATURE = 0.8

# CONTROL already overfits by epoch 5 on its small pool (more epochs would
# only deepen memorization); READING's much larger combined pool gets a
# bigger budget to actually converge. Early stopping (below) can still cut
# either short of its cap.
EPOCHS_BY_CONFIG = {"CONTROL": 10, "READING": 30}
# Looser than brain/language_model.py's own EARLY_STOP_PATIENCE=3 — this
# script wants to see whether READING keeps improving past where production
# would already have cut it off. MIN_DELTA is reused as-is from production.
EARLY_STOP_PATIENCE = 5

CONFIG_LABELS = ("CONTROL", "READING")
SEEDS = (0, 1, 2)

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


def set_seed(seed: int) -> None:
    """Seeds every source of randomness a run consumes AFTER this point:
    weight init (torch, inside AgentLanguageModel()), _oversample_pairs'
    shuffle, and each epoch's random.shuffle(train_samples) in run_training.
    Must be called fresh before each (config, seed) run, and never before
    load_base_split() — the val split must stay seed-independent."""
    random.seed(seed)
    torch.manual_seed(seed)


def output_path_for(label: str, seed: int) -> Path:
    return PROJECT_ROOT / f"language_model_{label.lower()}_s{seed}.pt"


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
    the single shared source of truth every run in this script reuses — both
    configs, all 3 seeds each — so "same held-out validation set across all
    6 runs" is actually true.

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
    READING — pure frequency ranking cannot promise that, see the module
    docstring), then the remaining budget is filled with the most frequent
    words from `vocabulary_words` that aren't already in. `vocabulary_words`
    is READING's own combined pool (base pool + reading text), so in
    practice the filler ends up almost entirely reading-corpus words, since
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


def build_tokenizer_for_config(
    label: str, pool_samples: list[str], vocabulary_words: list[str], production_vocab_words: list[str]
) -> AgentTokenizer:
    if label == "READING":
        return build_priority_tokenizer(vocabulary_words, production_vocab_words, MAX_VOCAB_SIZE)
    return AgentTokenizer(vocabulary_words, max_vocab_size=MAX_VOCAB_SIZE)


def build_trainer(
    scratch: LanguageModelTrainer,
    pool_samples: list[str],
    pre_oversample_train: list[str],
    val_samples: list[str],
    tokenizer: AgentTokenizer,
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


def run_training(
    trainer: LanguageModelTrainer, label: str, epochs: int
) -> tuple[list[float], list[float | None]]:
    # Mirrors LanguageModelTrainer.train() (same _run_epoch calls, same
    # EARLY_STOP_MIN_DELTA), but with this script's own epoch budget and
    # EARLY_STOP_PATIENCE (5, looser than production's 3 — see module
    # docstring), and keeps the per-epoch val losses so they can go in the
    # results file, not just stdout.
    train_losses: list[float] = []
    val_losses: list[float | None] = []
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(epochs):
        random.shuffle(trainer.train_samples)
        train_loss = trainer._run_epoch(trainer.train_samples, BATCH_SIZE, train=True)
        train_losses.append(train_loss)

        if trainer.val_samples:
            val_loss = trainer._run_epoch(trainer.val_samples, BATCH_SIZE, train=False)
            val_losses.append(val_loss)
            print(
                f"[{label}] epoch {epoch + 1}/{epochs} — "
                f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
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
            print(f"[{label}] epoch {epoch + 1}/{epochs} — train loss {train_loss:.4f} (no val split)")

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

    configs = {
        "CONTROL": (base_samples, pre_oversample_train),
        "READING": (base_samples + reading_samples, pre_oversample_train + reading_samples),
    }

    runs = {}
    for label in CONFIG_LABELS:
        pool_samples, train_pre_oversample = configs[label]
        epochs = EPOCHS_BY_CONFIG[label]

        # Vocab is deterministic (frequency/priority ranking, no randomness)
        # and depends only on pool_samples, so it's built once per config and
        # shared by all 3 seeds — not rebuilt per seed.
        vocabulary_words = scratch._collect_vocabulary(pool_samples)
        tokenizer = build_tokenizer_for_config(label, pool_samples, vocabulary_words, production_vocab_words)
        vocab_survived = len(production_vocab & set(tokenizer.words))
        if label == "READING" and vocab_survived != len(production_vocab):
            raise RuntimeError(
                f"READING tokenizer failed to guarantee full production-vocab survival: "
                f"{vocab_survived}/{len(production_vocab)}"
            )
        print(
            f"[{label}] vocab size {tokenizer.vocab_size}, "
            f"production vocab survived {vocab_survived}/{len(production_vocab)}, "
            f"epoch budget {epochs}"
        )

        for seed in SEEDS:
            run_label = f"{label}_s{seed}"
            print(f"\n=== training {run_label} (seed={seed}) ===")
            set_seed(seed)
            start = time.monotonic()
            trainer = build_trainer(scratch, pool_samples, train_pre_oversample, val_samples, tokenizer)
            print(
                f"[{run_label}] {len(trainer.train_samples)} train samples, "
                f"{len(trainer.val_samples)} val samples"
            )
            train_losses, val_losses = run_training(trainer, run_label, epochs)
            elapsed = time.monotonic() - start
            print(f"[{run_label}] training finished in {elapsed:.1f}s")

            output_path = output_path_for(label, seed)
            trainer.save(output_path)
            print(f"[{run_label}] weights saved to {output_path}")

            probes = generate_probes(trainer)
            best_val, best_epoch = best_val_loss_and_epoch(val_losses)

            runs[(label, seed)] = {
                "trainer": trainer,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "vocab_size": tokenizer.vocab_size,
                "survived": vocab_survived,
                "probes": probes,
                "elapsed": elapsed,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "epochs_run": len(train_losses),
                "epochs_budget": epochs,
                "output_path": output_path,
            }

    write_results(production_vocab, base_samples, val_samples, reading_samples, clean_files, runs)


def write_results(production_vocab, base_samples, val_samples, reading_samples, clean_files, runs) -> None:
    lines = []
    lines.append("=== READING TEST RESULTS ===")
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"memory source: {MEMORY_PATH}")
    lines.append(f"reading source: {CLEAN_DIR} ({len(clean_files)} file(s), {len(reading_samples)} samples)")
    epochs_str = ", ".join(f"{label}={EPOCHS_BY_CONFIG[label]}" for label in CONFIG_LABELS)
    lines.append(
        f"epoch budgets: {epochs_str} (early stop after {EARLY_STOP_PATIENCE} epochs "
        f"without a >{LanguageModelTrainer.EARLY_STOP_MIN_DELTA} val loss improvement) "
        f"batch_size={BATCH_SIZE} max_vocab_size={MAX_VOCAB_SIZE} "
        f"embedding_dim={EMBEDDING_DIM} nhead={NHEAD} num_layers={NUM_LAYERS} "
        f"dim_feedforward={DIM_FEEDFORWARD} dropout={DROPOUT} lr={LEARNING_RATE} "
        f"seeds={list(SEEDS)}"
    )
    lines.append(f"production vocab (current {PRODUCTION_CHECKPOINT.name}): {len(production_vocab)} words")
    lines.append(f"shared base pool: {len(base_samples)} samples")
    lines.append(f"shared validation set (identical for all {len(CONFIG_LABELS) * len(SEEDS)} runs): {len(val_samples)} samples")
    lines.append("")

    # --- per-seed detail + per-config mean/std of best val loss ---
    config_stats = {}
    for label in CONFIG_LABELS:
        seed_runs = [runs[(label, seed)] for seed in SEEDS]
        best_vals = [r["best_val_loss"] for r in seed_runs]
        if any(v is None for v in best_vals):
            raise RuntimeError(f"{label}: at least one seed produced no val loss (empty val split?)")
        mean_best = statistics.mean(best_vals)
        std_best = statistics.stdev(best_vals) if len(best_vals) > 1 else 0.0
        best_seed_idx = min(range(len(SEEDS)), key=lambda i: best_vals[i])
        config_stats[label] = {
            "best_vals": best_vals,
            "mean": mean_best,
            "std": std_best,
            "best_seed": SEEDS[best_seed_idx],
        }

        lines.append(
            f"--- {label}: best val loss per seed "
            f"(epoch budget {EPOCHS_BY_CONFIG[label]}, vocab {seed_runs[0]['vocab_size']}, "
            f"production vocab survived {seed_runs[0]['survived']}/{len(production_vocab)}) ---"
        )
        for seed, val in zip(SEEDS, best_vals):
            r = runs[(label, seed)]
            converged = r["epochs_run"] < r["epochs_budget"]
            status = (
                f"early-stopped at epoch {r['epochs_run']}"
                if converged
                else f"hit epoch budget ({r['epochs_budget']}) without early-stopping"
            )
            lines.append(
                f"  seed {seed}: best val loss {val:.4f} at epoch {r['best_epoch']}/{r['epochs_run']} "
                f"({status}, {r['elapsed']:.1f}s, saved to {r['output_path'].name})"
            )
        lines.append(f"  mean = {mean_best:.4f}, std = {std_best:.4f}, best seed = {SEEDS[best_seed_idx]}")
        lines.append("")

    diff = config_stats["READING"]["mean"] - config_stats["CONTROL"]["mean"]
    spread = max(config_stats["CONTROL"]["std"], config_stats["READING"]["std"])
    significant = abs(diff) > spread
    lines.append("--- CONTROL vs READING verdict ---")
    lines.append(
        f"mean best val loss: CONTROL {config_stats['CONTROL']['mean']:.4f} "
        f"(std {config_stats['CONTROL']['std']:.4f}), "
        f"READING {config_stats['READING']['mean']:.4f} (std {config_stats['READING']['std']:.4f})"
    )
    lines.append(f"difference (READING - CONTROL): {diff:+.4f}")
    lines.append(f"spread (max of the two configs' seed-to-seed std): {spread:.4f}")
    lines.append(
        "=> "
        + (
            f"difference exceeds seed spread — {'READING' if diff < 0 else 'CONTROL'} looks genuinely better."
            if significant
            else "difference does NOT exceed seed spread — cannot conclude reading text helped or hurt; "
            "this is noise at this seed count."
        )
    )
    lines.append("")

    lines.append("--- per-epoch losses (all seeds) ---")
    for label in CONFIG_LABELS:
        for seed in SEEDS:
            run = runs[(label, seed)]
            lines.append(f"{label}_s{seed}:")
            for i, (t_loss, v_loss) in enumerate(zip(run["train_losses"], run["val_losses"]), start=1):
                v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
                lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("")

    lines.append("--- probe generations (best seed per configuration) ---")
    best_seed_labels = [f"{label} (s{config_stats[label]['best_seed']})" for label in CONFIG_LABELS]
    header = f"{'':32}" + "".join(f"{lbl:>22}" for lbl in best_seed_labels)
    lines.append(header)
    lines.append("-" * len(header))
    for question in PROBE_QUESTIONS:
        lines.append(f"Q: {question}")
        for label in CONFIG_LABELS:
            best_seed = config_stats[label]["best_seed"]
            reply = runs[(label, best_seed)]["probes"][question]
            lines.append(f"  {label} (s{best_seed}): {reply}")
    lines.append("")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    RESULTS_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n[reading_test] results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
