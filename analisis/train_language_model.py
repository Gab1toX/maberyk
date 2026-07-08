import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import LanguageModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the scaffold AgentLanguageModel on agent_generated conversations."
    )
    parser.add_argument(
        "--memory", required=True, type=Path, help="Path to episodic_memory.sqlite3"
    )
    parser.add_argument("--output", type=Path, default=Path("language_model.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_language_model] using device: {device}")

    trainer = LanguageModelTrainer(args.memory, device=device)
    epoch_losses = trainer.train(epochs=20, batch_size=32)
    trainer.save(args.output)

    final_loss = epoch_losses[-1] if epoch_losses else float("nan")
    print(f"[train_language_model] final loss: {final_loss:.4f}")
    print(f"[train_language_model] weights saved to {args.output}")

    print("[train_language_model] sample generations (prompt='curious', temperature=0.8):")
    for _ in range(5):
        words = trainer.model.generate(
            prompt_words=["curious"],
            max_new_tokens=8,
            temperature=0.8,
        )
        print(f"  - {' '.join(words)}")


if __name__ == "__main__":
    main()
