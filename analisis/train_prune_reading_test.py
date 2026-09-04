"""Standalone script: re-run the reading-text hypothesis, this time against
the PRUNED corpus as the base pool.

Why re-run it: analisis/reading_test_results.txt (2026-08-13) found reading
text hurt (READING_BIG val 2.8380 vs CONTROL_BIG val 2.6331) -- but that
CONTROL pool was built from corpus_agente.txt BEFORE the 2026-08-21 pruning
passes (scripts/prune_corpus_agente.py / _followup.py). analisis/
train_prune_comparison.py (this same week) then showed the pruned pool is
missing 2,442 lines that, when added back, act mostly as a volume/
regularizer effect on the BIG architecture rather than qualitatively
meaningful signal -- ORIGINAL_BIG's better val loss wasn't really about what
those lines said, just that there were more of them. That means the old
CONTROL pool the first reading test used was already "padded" with that same
kind of filler, so its verdict on reading text is confounded: reading text
was compared against a base pool that didn't need the volume reading text
would have supplied. This run removes that confound by using the pruned pool
(no filler) as the base for all three configs.

Three runs, seed=0, BIG architecture only (256d/8head/6layer/512ff -- current
production architecture), identical held-out validation set for all three:

  PRUNED_ONLY         -- pruned corpus_agente.txt + pairs. Today's production
                         baseline (see analisis/prune_comparison_results.txt's
                         PRUNED_BIG, val 3.8743; this run's own number may
                         differ slightly since the tokenizer construction
                         changes below, but should land in the same range).
  PRUNED_READING      -- PRUNED_ONLY's pool + every line of
                         datos_lectura/clean/*.txt as sentence-level samples.
  PRUNED_READING_HALF -- PRUNED_ONLY's pool + a random 50% subsample of the
                         same reading lines (seeded, deterministic), to see
                         whether any reading-text effect scales with volume
                         or a smaller dose does better.

Same held-out validation set by construction, same technique as
analisis/train_reading_test.py and analisis/train_prune_comparison.py before
it: the pruned base pool (pairs + episode thoughts + corpus_agente.txt,
loaded exactly as brain/language_model.py's own _load_training_samples()
does it) is loaded and split ONCE via _split_train_val(), before any reading
text is added and before any per-run seeding. That val_samples is reused
verbatim by all three runs; reading text (full or half) is added only to
train.

Vocabulary: two-tier priority tokenizer (build_priority_tokenizer, adapted
from train_reading_test.py) -- every word appearing in the human_taught/
tutor_approved pairs is guaranteed a slot first (pairs are the actual
train/val signal per CLAUDE.md's "measure signal, not volume"; corpus/thought/
reading words are not guaranteed), then remaining budget (max_vocab_size
16384) is filled by frequency across that run's full pool. Applied
identically to all three configs, including PRUNED_ONLY, where it is a no-op
in effect (total pool vocabulary is far under 16384, so nothing is dropped or
reordered by tier). Pairs-vocab survival is computed and reported per run --
guaranteed to be 100% by build_priority_tokenizer's own construction (it
raises rather than silently truncating if priority vocab ever exceeded the
budget; it does not here).

Writes language_model_prunetest_{pruned_only,pruned_reading,
pruned_reading_half}.pt plus a report at
analisis/prune_reading_comparison_results.txt, rewritten after each run
finishes (not only at the end) so a partial report survives if this
multi-hour run is interrupted.

Usage:
    python analisis/train_prune_reading_test.py
"""

from __future__ import annotations

import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from brain.language_model import (
    AgentLanguageModel,
    AgentTokenizer,
    LanguageModelTrainer,
    normalize_text,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = PROJECT_ROOT / "episodic_memory.sqlite3"
CLEAN_DIR = PROJECT_ROOT / "datos_lectura" / "clean"
RESULTS_PATH = Path(__file__).resolve().parent / "prune_reading_comparison_results.txt"

MAX_VOCAB_SIZE = 16384
DROPOUT = 0.1
DEVICE = "cpu"  # CPU only on this machine, per CLAUDE.md
BATCH_SIZE = 16
GENERATION_TEMPERATURE = 0.8
SEED = 0
READING_HALF_SEED = 0  # separate deterministic sample, does not touch the global `random` stream at call time

ARCH = {"embedding_dim": 256, "nhead": 8, "num_layers": 6, "dim_feedforward": 512}

LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
GRAD_CLIP_NORM = 1.0
EPOCHS_BUDGET = 40
EARLY_STOP_PATIENCE = 5

RUN_LABELS = ["PRUNED_ONLY", "PRUNED_READING", "PRUNED_READING_HALF"]

# Same fixed paraphrase probes used in the two prior comparison scripts.
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

# Same diversity-check methodology as analisis/train_language_model.py.
DIVERSITY_QUESTIONS = [
    "que es una casa",
    "que ves",
    "como te sientes",
    "que es el tiempo",
    "por que te mueves",
]
GENERATIONS_PER_QUESTION = 20


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def output_path_for(label: str) -> Path:
    return PROJECT_ROOT / f"language_model_prunetest_{label.lower()}.pt"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def best_val_loss_and_epoch(val_losses: list[float | None]) -> tuple[float | None, int | None]:
    values = [(i + 1, v) for i, v in enumerate(val_losses) if v is not None]
    if not values:
        return None, None
    epoch, val = min(values, key=lambda iv: iv[1])
    return val, epoch


def load_pruned_base_split() -> tuple[LanguageModelTrainer, list[str], list[str], list[str]]:
    """Loads the pruned pool (pairs + episode thoughts + current
    corpus_agente.txt) via the exact production code path and splits it ONCE.
    This split's val_samples is the single shared held-out set every run in
    this script reuses. Must run before any set_seed() call -- see
    analisis/train_prune_comparison.py's identical function for why."""
    scratch = LanguageModelTrainer.__new__(LanguageModelTrainer)
    scratch.tokenizer_type = "word"
    scratch.memory_path = MEMORY_PATH

    base_samples = scratch._load_training_samples()
    if not base_samples:
        raise ValueError(f"No training samples found via {MEMORY_PATH}")

    pre_oversample_train, val_samples = scratch._split_train_val(base_samples)
    return scratch, base_samples, pre_oversample_train, val_samples


def load_pair_samples() -> list[str]:
    """Replicates _load_training_samples()'s pair-loading exactly (source in
    human_taught/tutor_approved, normalize_text, dict.fromkeys dedup,
    <q>...<a>...<end> wrap) to isolate JUST the pairs, since
    _load_training_samples() itself returns pairs already mixed with
    thoughts/corpus with no hook to pull pairs out alone. This is only used
    to build the priority-tokenizer's guaranteed vocabulary -- it is not a
    second copy of training data."""
    connection = sqlite3.connect(MEMORY_PATH)
    try:
        pair_rows = connection.execute(
            "SELECT question, answer FROM conversations "
            "WHERE source IN ('human_taught', 'tutor_approved')"
        ).fetchall()
    finally:
        connection.close()

    return list(dict.fromkeys(
        f"{AgentTokenizer.Q} {normalize_text(question)} {AgentTokenizer.A} "
        f"{normalize_text(answer)} {AgentTokenizer.END}"
        for question, answer in pair_rows
        if question and question.strip() and answer and answer.strip()
    ))


def build_reading_samples() -> tuple[list[str], list[Path]]:
    """One sample per cleaned line, normalized the same way the word
    tokenizer's own vocabulary is -- unchanged from
    analisis/train_reading_test.py's build_reading_samples()."""
    if not CLEAN_DIR.is_dir():
        raise FileNotFoundError(f"{CLEAN_DIR} not found -- run `python -m lectura.ingest` first.")
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


def build_priority_tokenizer(
    vocabulary_words: list[str], priority_words: list[str], max_vocab_size: int
) -> AgentTokenizer:
    """Two-tier ranked tokenizer: every word in `priority_words` (here, the
    pairs vocabulary) is guaranteed a slot first, then the remaining budget
    is filled with the most frequent words from `vocabulary_words` (that
    run's full pool) that aren't already in. Adapted verbatim from
    analisis/train_reading_test.py's function of the same name -- only the
    meaning of `priority_words` changed (pairs vocabulary here, vs. that
    script's production-checkpoint vocabulary)."""
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
            f"max_vocab_size budget ({budget}) -- cannot guarantee 100% survival"
        )
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
    return tokenizer, priority_set


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
        embedding_dim=ARCH["embedding_dim"],
        nhead=ARCH["nhead"],
        num_layers=ARCH["num_layers"],
        dim_feedforward=ARCH["dim_feedforward"],
        dropout=DROPOUT,
        pad_index=tokenizer.pad_index,
        tokenizer=tokenizer,
    ).to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=LEARNING_RATE)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    trainer.base_lr = LEARNING_RATE
    trainer.warmup_steps = WARMUP_STEPS
    trainer.grad_clip_norm = GRAD_CLIP_NORM
    trainer.global_step = 0
    trainer.EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE
    return trainer


def generate_probes(trainer: LanguageModelTrainer) -> dict[str, str]:
    replies = {}
    for question in PROBE_QUESTIONS:
        reply = trainer.model.generate_reply(question, temperature=GENERATION_TEMPERATURE)
        text = reply if isinstance(reply, str) else " ".join(reply)
        replies[question] = text if text else "(empty)"
    return replies


def run_diversity_check(model: AgentLanguageModel) -> dict:
    first_word_counts: Counter[str] = Counter()
    per_question = []

    for question in DIVERSITY_QUESTIONS:
        replies = []
        for _ in range(GENERATIONS_PER_QUESTION):
            reply = model.generate_reply(question, temperature=GENERATION_TEMPERATURE)
            text = reply if isinstance(reply, str) else " ".join(reply)
            replies.append(text)
            words = text.split()
            first_word_counts[words[0].lower() if words else "<empty>"] += 1

        reply_counts = Counter(replies)
        most_common_text, most_common_n = reply_counts.most_common(1)[0]
        per_question.append(
            {
                "question": question,
                "unique_count": len(reply_counts),
                "most_common_text": most_common_text,
                "most_common_n": most_common_n,
            }
        )

    total = GENERATIONS_PER_QUESTION * len(DIVERSITY_QUESTIONS)
    top_word, top_count = first_word_counts.most_common(1)[0]
    return {
        "per_question": per_question,
        "total": total,
        "top_word": top_word,
        "top_count": top_count,
        "top_pct": 100.0 * top_count / total,
        "top20": first_word_counts.most_common(20),
    }


def write_results(
    base_samples: list[str],
    reading_samples: list[str],
    reading_samples_half: list[str],
    clean_files: list[Path],
    val_samples: list[str],
    pair_vocab_total: int,
    results: dict,
    in_progress: bool,
) -> None:
    lines = []
    lines.append("=== PRUNE + READING COMPARISON RESULTS (reading text re-tested against the pruned base pool) ===")
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}" + ("  [IN PROGRESS]" if in_progress else ""))
    lines.append(f"memory source: {MEMORY_PATH}")
    lines.append(f"base corpus: corpus_agente.txt (pruned)")
    lines.append(f"reading source: {CLEAN_DIR} ({len(clean_files)} file(s), {len(reading_samples)} samples, "
                 f"{len(reading_samples_half)} in the 50% subsample)")
    lines.append(f"architecture (all 3 runs): embedding_dim={ARCH['embedding_dim']}, nhead={ARCH['nhead']}, "
                 f"num_layers={ARCH['num_layers']}, dim_feedforward={ARCH['dim_feedforward']}")
    lines.append(f"shared settings: lr={LEARNING_RATE}, warmup_steps={WARMUP_STEPS}, grad_clip_norm={GRAD_CLIP_NORM}, "
                 f"epoch_budget={EPOCHS_BUDGET}, early_stop_patience={EARLY_STOP_PATIENCE}, "
                 f"batch_size={BATCH_SIZE}, max_vocab_size={MAX_VOCAB_SIZE}, dropout={DROPOUT}, seed={SEED}")
    lines.append(f"shared pruned base pool: {len(base_samples)} samples")
    lines.append(f"shared validation set (identical for all 3 runs): {len(val_samples)} samples")
    lines.append(f"pairs vocabulary (priority tier, guaranteed a slot in every run): {pair_vocab_total} unique words")
    lines.append("")

    lines.append("--- results per run ---")
    header = (f"{'config':22}{'params':>14}{'vocab':>8}{'pairs surv.':>12}{'best val loss':>16}{'@ epoch':>10}"
              f"{'train@best':>12}{'gap':>8}{'status':>34}{'time (s)':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for label in RUN_LABELS:
        if label not in results:
            lines.append(f"{label:22}  (not yet run)")
            continue
        r = results[label]
        converged = r["epochs_run"] < r["epochs_budget"]
        status = (
            f"early-stopped at epoch {r['epochs_run']}"
            if converged
            else f"hit epoch budget ({r['epochs_budget']}) without early-stopping"
        )
        train_at_best_str = f"{r['train_at_best']:.4f}" if r["train_at_best"] is not None else "n/a"
        gap_str = f"{r['gap']:.4f}" if r["gap"] is not None else "n/a"
        surv_str = f"{r['pair_vocab_survived']}/{pair_vocab_total} ({r['pair_vocab_survival_pct']:.1f}%)"
        lines.append(
            f"{label:22}{r['param_count']:>14,}{r['vocab_size']:>8}{surv_str:>12}{r['best_val_loss']:>16.4f}"
            f"{r['best_epoch']:>10}{train_at_best_str:>12}{gap_str:>8}{status:>34}{r['elapsed']:>10.1f}"
        )
        lines.append(
            f"{'':22}train={r['train_samples_n']}, val={r['val_samples_n']}, saved to {r['output_path'].name}"
        )
    lines.append("")

    lines.append("--- per-epoch losses ---")
    for label in RUN_LABELS:
        if label not in results:
            continue
        r = results[label]
        lines.append(f"{label}:")
        for i, (t_loss, v_loss) in enumerate(zip(r["train_losses"], r["val_losses"]), start=1):
            v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
            lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("")

    lines.append("--- diversity check (5 questions x 20 generations, temperature=0.8) ---")
    for label in RUN_LABELS:
        if label not in results:
            continue
        d = results[label]["diversity"]
        lines.append(f"{label}:")
        for pq in d["per_question"]:
            lines.append(
                f"  '{pq['question']}': {pq['unique_count']}/{GENERATIONS_PER_QUESTION} unique, "
                f"most frequent ({pq['most_common_n']}x): {pq['most_common_text']!r}"
            )
        collapse_note = " -- MODE COLLAPSE" if d["top_pct"] > 50.0 else ""
        lines.append(
            f"  top first word overall: {d['top_word']!r} starts {d['top_count']}/{d['total']} "
            f"generations ({d['top_pct']:.1f}%){collapse_note}"
        )
        lines.append(f"  top 20 first words: {d['top20']}")
    lines.append("")

    lines.append("--- probe generations (10 fixed paraphrase questions, side by side) ---")
    run_labels_done = [label for label in RUN_LABELS if label in results]
    header = f"{'':32}" + "".join(f"{label:>24}" for label in run_labels_done)
    lines.append(header)
    lines.append("-" * len(header))
    for question in PROBE_QUESTIONS:
        lines.append(f"Q: {question}")
        for label in run_labels_done:
            reply = results[label]["probes"][question]
            lines.append(f"  {label}: {reply}")
    lines.append("")

    report_text = "\n".join(lines)
    RESULTS_PATH.write_text(report_text + "\n", encoding="utf-8")


def main() -> None:
    print("[prune_reading_test] loading pruned base pool + shared val split (loaded once, before any seeding)")
    scratch, base_samples, pre_oversample_train, val_samples = load_pruned_base_split()
    print(
        f"[prune_reading_test] pruned base pool: {len(base_samples)} samples, "
        f"{len(pre_oversample_train)} train (pre-oversample) / {len(val_samples)} val"
    )

    pair_samples = load_pair_samples()
    priority_words = scratch._collect_vocabulary(pair_samples)
    pair_vocab_set_preview = set(
        w for w in (str(x).lower().strip() for x in priority_words)
        if w and w not in AgentTokenizer.SPECIAL_TOKENS
    )
    print(f"[prune_reading_test] {len(pair_samples)} pair samples, {len(pair_vocab_set_preview)} unique pairs-vocab words (priority tier)")

    reading_samples, clean_files = build_reading_samples()
    print(f"[prune_reading_test] {len(reading_samples)} reading samples from {len(clean_files)} file(s) under {CLEAN_DIR}")

    reading_samples_half = random.Random(READING_HALF_SEED).sample(reading_samples, len(reading_samples) // 2)
    print(f"[prune_reading_test] {len(reading_samples_half)} reading samples in the 50% subsample (seed={READING_HALF_SEED})")

    pools = {
        "PRUNED_ONLY": (base_samples, pre_oversample_train),
        "PRUNED_READING": (base_samples + reading_samples, pre_oversample_train + reading_samples),
        "PRUNED_READING_HALF": (base_samples + reading_samples_half, pre_oversample_train + reading_samples_half),
    }

    results: dict = {}
    for label in RUN_LABELS:
        pool_samples, pool_train_pre = pools[label]

        print(f"\n=== training {label} (seed={SEED}) ===")
        set_seed(SEED)
        vocabulary_words = scratch._collect_vocabulary(pool_samples)
        tokenizer, priority_set = build_priority_tokenizer(vocabulary_words, priority_words, MAX_VOCAB_SIZE)
        pair_vocab_survived = len(priority_set & set(tokenizer.words))

        start = time.monotonic()
        trainer = build_trainer(scratch, pool_samples, pool_train_pre, val_samples, tokenizer)
        param_count = count_parameters(trainer.model)
        print(
            f"[{label}] {len(trainer.train_samples)} train samples (post-oversample), "
            f"{len(trainer.val_samples)} val samples, vocab {trainer.tokenizer.vocab_size}, "
            f"pairs-vocab survival {pair_vocab_survived}/{len(pair_vocab_set_preview)}, "
            f"{param_count:,} parameters"
        )

        train_losses = trainer.train(epochs=EPOCHS_BUDGET, batch_size=BATCH_SIZE)
        val_losses = trainer.val_losses
        elapsed = time.monotonic() - start
        print(f"[{label}] training finished in {elapsed:.1f}s")

        output_path = output_path_for(label)
        trainer.save(output_path)
        print(f"[{label}] weights saved to {output_path}")

        probes = generate_probes(trainer)
        diversity = run_diversity_check(trainer.model)
        best_val, best_epoch = best_val_loss_and_epoch(val_losses)
        train_at_best = train_losses[best_epoch - 1] if best_epoch else None
        gap = (best_val - train_at_best) if (best_val is not None and train_at_best is not None) else None

        results[label] = dict(
            train_losses=train_losses,
            val_losses=val_losses,
            vocab_size=trainer.tokenizer.vocab_size,
            pair_vocab_survived=pair_vocab_survived,
            pair_vocab_survival_pct=100.0 * pair_vocab_survived / len(pair_vocab_set_preview),
            param_count=param_count,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            train_at_best=train_at_best,
            gap=gap,
            epochs_run=len(train_losses),
            epochs_budget=EPOCHS_BUDGET,
            elapsed=elapsed,
            output_path=output_path,
            probes=probes,
            diversity=diversity,
            train_samples_n=len(trainer.train_samples),
            val_samples_n=len(trainer.val_samples),
        )

        write_results(
            base_samples, reading_samples, reading_samples_half, clean_files,
            val_samples, len(pair_vocab_set_preview), results, in_progress=True,
        )
        print(f"[{label}] partial report written to {RESULTS_PATH}")

        del trainer

    write_results(
        base_samples, reading_samples, reading_samples_half, clean_files,
        val_samples, len(pair_vocab_set_preview), results, in_progress=False,
    )
    print(f"\n[prune_reading_test] final results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
