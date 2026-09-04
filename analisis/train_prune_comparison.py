"""Standalone script: controlled comparison to answer one question --
did pruning corpus_agente.txt (5,946 -> 3,479 content lines, see
scripts/prune_corpus_agente.py / _followup.py) hurt because the corpus is now
too small for the BIG (5.9M-param) architecture, or because the removed
content was load-bearing?

Three runs, seed=0, identical held-out validation set:
  PRUNED_BIG    -- current corpus_agente.txt (pruned) + pairs, BIG arch
  PRUNED_SMALL  -- same pool as PRUNED_BIG,                    SMALL arch
  ORIGINAL_BIG  -- corpus_agente_original.txt (unpruned) + pairs, BIG arch

If PRUNED_SMALL beats PRUNED_BIG on val loss, the pruned corpus is too small
for BIG's capacity (overfitting) -- a size problem. If ORIGINAL_BIG beats
PRUNED_BIG by a wide margin, the pruned content was load-bearing signal, not
noise -- a content problem. Both effects can be present at once.

Same held-out validation set across all three runs, by construction: the
pruned pool (human_taught/tutor_approved pairs + episode thoughts +
corpus_agente.txt, loaded exactly as brain/language_model.py's own
_load_training_samples() does it) is loaded and split ONCE via
_split_train_val(). That split's val_samples is reused verbatim by every run.
For ORIGINAL_BIG, the lines that exist in corpus_agente_original.txt but were
removed by pruning (2,446 of them -- diffed at the raw wrapped-sample level,
so exact matches already surviving in the pruned pool, e.g. duplicates the
dedup pass collapsed, are correctly excluded) are added ONLY to that run's
train split, never to val. This is the same technique
analisis/train_reading_test.py used to add reading-text samples on top of the
shared CONTROL pool -- see that file's load_base_split()/pools dict for the
precedent.

Architectures match analisis/reading_test_results.txt exactly:
  BIG   = embedding_dim=256, nhead=8, num_layers=6, dim_feedforward=512
  SMALL = embedding_dim=128, nhead=4, num_layers=3, dim_feedforward=256

BIG hyperparameters (lr=1e-4, warmup_steps=500, grad_clip_norm=1.0, epoch
budget 40) are carried over from analisis/retrain_production_big.py /
reading_test_results.txt's CONTROL_BIG, which is what made BIG train instead
of diverging. SMALL uses lr=1e-3 (its own stable regime) with a short
warmup_steps=100 -- brain/language_model.py's current _run_epoch() always
applies warmup+clipping (added after the historical CONTROL_SMALL numbers in
reading_test_results.txt were produced, which predate that code and are not
reused here), so SMALL gets the same mechanism with parameters suited to its
size rather than a special-cased training loop. Both use grad_clip_norm=1.0,
early_stop_patience=5, batch_size=16, dropout=0.1, max_vocab_size=8192
(production's own default -- confirmed by a pre-flight vocab check to
comfortably cover both pools' unique words).

Writes language_model_prunetest_{pruned_big,pruned_small,original_big}.pt
(distinct prefix so nothing collides with existing experiment checkpoints
like language_model_control_big.pt) plus a report at
analisis/prune_comparison_results.txt, rewritten after each run finishes (not
only at the end) so a partial report survives if this ~2h run is interrupted.

Usage:
    python analisis/train_prune_comparison.py
"""

from __future__ import annotations

import random
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
ORIGINAL_CORPUS_PATH = PROJECT_ROOT / "corpus_agente_original.txt"
RESULTS_PATH = Path(__file__).resolve().parent / "prune_comparison_results.txt"

MAX_VOCAB_SIZE = 8192
DROPOUT = 0.1
DEVICE = "cpu"  # CPU only on this machine, per CLAUDE.md
BATCH_SIZE = 16
GENERATION_TEMPERATURE = 0.8
SEED = 0

ARCH_CONFIGS = {
    "BIG": {"embedding_dim": 256, "nhead": 8, "num_layers": 6, "dim_feedforward": 512},
    "SMALL": {"embedding_dim": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256},
}

# label, pool, arch, lr, warmup_steps, grad_clip_norm, epoch_budget, early_stop_patience
RUN_SPECS = [
    ("PRUNED_BIG", "PRUNED", "BIG", 1e-4, 500, 1.0, 40, 5),
    ("PRUNED_SMALL", "PRUNED", "SMALL", 1e-3, 100, 1.0, 30, 5),
    ("ORIGINAL_BIG", "ORIGINAL", "BIG", 1e-4, 500, 1.0, 40, 5),
]

# Same fixed paraphrase probes used in analisis/train_reading_test.py, for
# consistency with the project's established probe set.
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
    """Must be called fresh before each run's tokenizer/model/training, and
    never before load_pruned_base_split() -- the val split must stay
    seed-independent (see that function's docstring)."""
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
    corpus_agente.txt) via the exact production code path, and splits it ONCE.
    This split's val_samples is the single shared held-out set every run in
    this script reuses -- see load_extra_original_lines() for how ORIGINAL
    adds to train only, never touching this val set.

    Must run before any set_seed() call: _load_training_samples()'s thought
    cap uses the unseeded global `random` module, and _split_train_val()'s
    shuffle depends on the incoming list order that produces. Calling either
    again after seeding would silently change which rows land in train vs
    val.

    scratch is a real LanguageModelTrainer instance (via __new__, __init__
    skipped) used only to call its own bound methods -- safe since
    _load_training_samples/_split_train_val/_oversample_pairs/
    _collect_vocabulary only touch self.memory_path and self.tokenizer_type,
    both set below.
    """
    scratch = LanguageModelTrainer.__new__(LanguageModelTrainer)
    scratch.tokenizer_type = "word"
    scratch.memory_path = MEMORY_PATH

    base_samples = scratch._load_training_samples()
    if not base_samples:
        raise ValueError(f"No training samples found via {MEMORY_PATH}")

    pre_oversample_train, val_samples = scratch._split_train_val(base_samples)
    return scratch, base_samples, pre_oversample_train, val_samples


def load_extra_original_lines(pruned_base_samples: list[str]) -> list[str]:
    """Lines present in corpus_agente_original.txt but not in the pruned
    pool, wrapped exactly the way _load_training_samples() wraps corpus
    lines (`<a> {normalize_text(line)} <end>`). Diffed against the full
    assembled pruned_base_samples (not just corpus_agente.txt's own lines) so
    a removed original line that happens to duplicate an episode-thought
    sample already in the pruned pool is correctly excluded -- it would add
    no new signal and risks nothing, but there is no reason to double it up
    either. Deterministic: no randomness involved, safe to add straight to
    train without touching the shared val split at all.
    """
    if not ORIGINAL_CORPUS_PATH.is_file():
        raise FileNotFoundError(f"{ORIGINAL_CORPUS_PATH} not found")

    original_lines = []
    for raw_line in ORIGINAL_CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            original_lines.append(line)

    wrapped = [
        f"{AgentTokenizer.A} {normalize_text(line)} {AgentTokenizer.END}"
        for line in original_lines
    ]
    base_set = set(pruned_base_samples)
    extra = [sample for sample in dict.fromkeys(wrapped) if sample not in base_set]
    return extra


def build_trainer(
    scratch: LanguageModelTrainer,
    pool_samples: list[str],
    pre_oversample_train: list[str],
    val_samples: list[str],
    arch: dict,
    lr: float,
    warmup_steps: int,
    grad_clip_norm: float,
    patience: int,
) -> LanguageModelTrainer:
    train_samples = scratch._oversample_pairs(pre_oversample_train)
    vocabulary_words = scratch._collect_vocabulary(pool_samples)
    tokenizer = AgentTokenizer(vocabulary_words, max_vocab_size=MAX_VOCAB_SIZE)

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
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
    trainer.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    trainer.base_lr = lr
    trainer.warmup_steps = warmup_steps
    trainer.grad_clip_norm = grad_clip_norm
    trainer.global_step = 0
    trainer.EARLY_STOP_PATIENCE = patience
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
    pruned_base_samples: list[str],
    extra_original: list[str],
    val_samples: list[str],
    results: dict,
    in_progress: bool,
) -> None:
    lines = []
    lines.append("=== PRUNE COMPARISON RESULTS (pruned corpus vs original corpus, BIG vs SMALL) ===")
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}" + ("  [IN PROGRESS]" if in_progress else ""))
    lines.append(f"memory source: {MEMORY_PATH}")
    lines.append(f"pruned corpus: corpus_agente.txt")
    lines.append(f"original corpus: {ORIGINAL_CORPUS_PATH.name}")
    arch_str = ", ".join(
        f"{label}=(embedding_dim={cfg['embedding_dim']}, nhead={cfg['nhead']}, "
        f"num_layers={cfg['num_layers']}, dim_feedforward={cfg['dim_feedforward']})"
        for label, cfg in ARCH_CONFIGS.items()
    )
    lines.append(f"architectures: {arch_str}")
    lines.append(f"shared settings: batch_size={BATCH_SIZE}, max_vocab_size={MAX_VOCAB_SIZE}, dropout={DROPOUT}, seed={SEED}")
    lines.append(f"shared pruned base pool: {len(pruned_base_samples)} samples")
    lines.append(f"original-only extra lines added to ORIGINAL's train split: {len(extra_original)}")
    lines.append(f"shared validation set (identical for all 3 runs): {len(val_samples)} samples")
    lines.append("")

    lines.append("--- results per run ---")
    header = f"{'config':16}{'pool':>10}{'arch':>7}{'params':>14}{'best val loss':>16}{'@ epoch':>10}{'train@best':>12}{'gap':>8}{'status':>34}{'time (s)':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for label, pool_label, arch_label, lr, warmup, clip, epochs_budget, patience in RUN_SPECS:
        if label not in results:
            lines.append(f"{label:16}  (not yet run)")
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
        lines.append(
            f"{label:16}{pool_label:>10}{arch_label:>7}{r['param_count']:>14,}{r['best_val_loss']:>16.4f}"
            f"{r['best_epoch']:>10}{train_at_best_str:>12}{gap_str:>8}{status:>34}{r['elapsed']:>10.1f}"
        )
        lines.append(
            f"{'':16}lr={lr}, warmup={warmup}, clip={clip}, vocab={r['vocab_size']}, "
            f"train={r['train_samples_n']}, val={r['val_samples_n']}, saved to {r['output_path'].name}"
        )
    lines.append("")

    lines.append("--- per-epoch losses ---")
    for label, *_ in RUN_SPECS:
        if label not in results:
            continue
        r = results[label]
        lines.append(f"{label}:")
        for i, (t_loss, v_loss) in enumerate(zip(r["train_losses"], r["val_losses"]), start=1):
            v_str = f"{v_loss:.4f}" if v_loss is not None else "n/a"
            lines.append(f"  epoch {i}: train {t_loss:.4f}  val {v_str}")
    lines.append("")

    lines.append("--- diversity check (5 questions x 20 generations, temperature=0.8) ---")
    for label, *_ in RUN_SPECS:
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
    run_labels_done = [label for label, *_ in RUN_SPECS if label in results]
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
    print("[prune_test] loading pruned base pool + shared val split (loaded once, before any seeding)")
    scratch, pruned_base_samples, pre_oversample_train, val_samples = load_pruned_base_split()
    print(
        f"[prune_test] pruned base pool: {len(pruned_base_samples)} samples, "
        f"{len(pre_oversample_train)} train (pre-oversample) / {len(val_samples)} val"
    )

    extra_original = load_extra_original_lines(pruned_base_samples)
    print(f"[prune_test] {len(extra_original)} original-only corpus lines -> ORIGINAL pool's train split only")

    pools = {
        "PRUNED": (pruned_base_samples, pre_oversample_train),
        "ORIGINAL": (pruned_base_samples + extra_original, pre_oversample_train + extra_original),
    }

    results: dict = {}
    for label, pool_label, arch_label, lr, warmup, clip, epochs_budget, patience in RUN_SPECS:
        pool_samples, pool_train_pre = pools[pool_label]
        arch = ARCH_CONFIGS[arch_label]

        print(
            f"\n=== training {label} (pool={pool_label}, arch={arch_label}, seed={SEED}, "
            f"lr={lr}, warmup_steps={warmup}, grad_clip_norm={clip}, "
            f"epoch_budget={epochs_budget}, patience={patience}) ==="
        )
        set_seed(SEED)
        start = time.monotonic()
        trainer = build_trainer(
            scratch, pool_samples, pool_train_pre, val_samples, arch, lr, warmup, clip, patience
        )
        param_count = count_parameters(trainer.model)
        print(
            f"[{label}] {len(trainer.train_samples)} train samples (post-oversample), "
            f"{len(trainer.val_samples)} val samples, vocab {trainer.tokenizer.vocab_size}, "
            f"{param_count:,} parameters"
        )

        train_losses = trainer.train(epochs=epochs_budget, batch_size=BATCH_SIZE)
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
            pool=pool_label,
            arch=arch_label,
            train_losses=train_losses,
            val_losses=val_losses,
            vocab_size=trainer.tokenizer.vocab_size,
            param_count=param_count,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            train_at_best=train_at_best,
            gap=gap,
            epochs_run=len(train_losses),
            epochs_budget=epochs_budget,
            elapsed=elapsed,
            output_path=output_path,
            probes=probes,
            diversity=diversity,
            train_samples_n=len(trainer.train_samples),
            val_samples_n=len(trainer.val_samples),
        )

        # Rewrite the report after every run so a partial result survives
        # if this multi-hour script is interrupted.
        write_results(pruned_base_samples, extra_original, val_samples, results, in_progress=True)
        print(f"[{label}] partial report written to {RESULTS_PATH}")

        del trainer

    write_results(pruned_base_samples, extra_original, val_samples, results, in_progress=False)
    print(f"\n[prune_test] final results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
