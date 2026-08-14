"""One-off: migrate the production language_model.pt to the BIG architecture.

brain/language_model.py's defaults (embedding_dim, nhead, num_layers,
dim_feedforward, learning_rate, warmup_steps, grad_clip_norm) were changed to
the BIG configuration that analisis/reading_test_results.txt's CONTROL_BIG run
validated (best val loss 2.6331, see that file) — lr=1e-4, linear warmup over
500 steps, gradient clipping at norm 1.0. lr=1e-3 with neither of those
diverges at this size (CONTROL_BIG's own first attempt: val loss 7.04 rising).

This script just calls LanguageModelTrainer with those now-default
hyperparameters, on the ordinary production pool — human_taught pairs +
episode thoughts + corpus_agente.txt from episodic_memory.sqlite3, no reading
text — which is exactly what CONTROL (as opposed to READING) meant in that
report. early_stop_patience is set to 5 to match CONTROL_BIG's own run
(looser than the trainer's class default of 3).

language_model.pt (the previous 128d checkpoint) must already be copied to
language_model_128d_final.pt before this runs — this script overwrites
language_model.pt on save().

Usage:
    python analisis/retrain_production_big.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import LanguageModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = PROJECT_ROOT / "episodic_memory.sqlite3"
OUTPUT_PATH = PROJECT_ROOT / "language_model.pt"

EPOCHS_BUDGET = 40
EARLY_STOP_PATIENCE = 5

# Subset of the paraphrase probes used in analisis/reading_test_results.txt,
# for direct comparison against CONTROL_BIG's own answers to the same
# questions.
PROBE_QUESTIONS = [
    "que significa la humedad",
    "explicame que es una red neuronal",
    "que es el aprendizaje automatico",
    "que significa viajar",
    "explicame que es un eclipse",
]


def main() -> None:
    trainer = LanguageModelTrainer(MEMORY_PATH, device="cpu")
    trainer.EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE

    trainer.train(epochs=EPOCHS_BUDGET, batch_size=16)

    trainer.save(OUTPUT_PATH)
    print(f"\n[retrain_production_big] weights saved to {OUTPUT_PATH}")

    if trainer.best_val_loss is not None:
        print(f"[retrain_production_big] best val loss: {trainer.best_val_loss:.4f}")
    else:
        print("[retrain_production_big] no validation split was available")

    print("[retrain_production_big] probe answers:")
    for question in PROBE_QUESTIONS:
        reply = trainer.model.generate_reply(question, temperature=0.8)
        text = reply if isinstance(reply, str) else " ".join(reply)
        print(f"  Q: {question}")
        print(f"  A: {text if text else '(empty)'}")


if __name__ == "__main__":
    main()
