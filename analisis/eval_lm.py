"""Standalone script: qualitative eval of a trained language_model.pt checkpoint.

Prints generate_reply() output for a fixed set of questions grouped by how far
they are from the training distribution — in-distribution (verbatim training
questions), paraphrase (same intent, new wording), and novel (unseen intent).

Usage:
    python analisis/eval_lm.py --model language_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import LanguageModelTrainer

QUESTION_GROUPS: list[tuple[str, list[str]]] = [
    (
        "IN-DISTRIBUTION",
        [
            "que es una casa",
            "que es un restaurante",
            "colombia tiene ballenas",
            "que funcion de activacion usas",
            "que es el tiempo en fisica",
        ],
    ),
    (
        "PARAPHRASE",
        [
            "cuentame que es una casa",
            "sabes si hay ballenas en colombia",
        ],
    ),
    (
        "NOVEL",
        [
            "que hay en tu mundo",
            "te gusta aprender",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print generate_reply() output for a fixed eval question set."
    )
    parser.add_argument("--model", type=Path, default=Path("language_model.pt"))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=2, help="Generations per question")
    args = parser.parse_args()

    model, _tokenizer = LanguageModelTrainer.load_model(args.model, map_location="cpu")

    for group_name, questions in QUESTION_GROUPS:
        print(f"\n=== {group_name} ===")
        for question in questions:
            for _ in range(args.samples):
                words = model.generate_reply(question, temperature=args.temperature)
                print(f"[{group_name}] {question} -> {' '.join(words)}")


if __name__ == "__main__":
    main()
