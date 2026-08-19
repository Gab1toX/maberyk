"""Standalone script: runs brain/language_model.py's training entry point
unchanged, then reloads the freshly saved checkpoint and runs an
output-diversity check on top of it.

The check generates GENERATIONS_PER_QUESTION replies to each of
DIVERSITY_QUESTIONS (5 fixed questions x 20 = 100 generations total) and
reports, per question, how many replies were distinct and what the single
most frequent one was, plus the 20 most common first words across all 100
generations -- a single word dominating that distribution is mode collapse,
reported as a count and a percentage, not an impression.

Does not touch training: brain.language_model.main() runs exactly as it
always has (same args, same train()/save() calls, same 5-sample printout);
this only adds a post-training reload-and-generate pass in this file.

Usage:
    python analisis/train_language_model.py --memory episodic_memory.sqlite3
    (accepts every flag brain/language_model.py's main() does -- see its
    --help -- plus reads --output/--temperature for the diversity check.)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.language_model import LanguageModelTrainer, main

GENERATIONS_PER_QUESTION = 20
DIVERSITY_QUESTIONS = [
    "que es una casa",
    "que ves",
    "como te sientes",
    "que es el tiempo",
    "por que te mueves",
]


def _output_path_and_temperature() -> tuple[Path, float]:
    """Reads just --output/--temperature from sys.argv, mirroring the
    defaults brain.language_model.main()'s own parser uses for the same two
    flags, without re-declaring (or risking drift from) every other
    training flag main() already owns. parse_known_args() + add_help=False
    means this never intercepts -h/--help or errors on flags meant for
    main()'s parser -- main() still does the real, authoritative parsing."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=Path("language_model.pt"))
    parser.add_argument("--temperature", type=float, default=0.8)
    args, _unknown = parser.parse_known_args()
    return args.output, args.temperature


def run_diversity_check(model_path: Path, temperature: float) -> None:
    model, _tokenizer = LanguageModelTrainer.load_model(model_path, map_location="cpu")

    total_generations = GENERATIONS_PER_QUESTION * len(DIVERSITY_QUESTIONS)
    print()
    print(
        f"[diversity] {GENERATIONS_PER_QUESTION} generations x {len(DIVERSITY_QUESTIONS)} "
        f"questions ({total_generations} total), temperature={temperature}"
    )

    first_word_counts: Counter[str] = Counter()

    for question in DIVERSITY_QUESTIONS:
        replies = []
        for _ in range(GENERATIONS_PER_QUESTION):
            reply = model.generate_reply(question, temperature=temperature)
            text = reply if isinstance(reply, str) else " ".join(reply)
            replies.append(text)
            words = text.split()
            first_word_counts[words[0].lower() if words else "<empty>"] += 1

        reply_counts = Counter(replies)
        unique_count = len(reply_counts)
        most_common_text, most_common_n = reply_counts.most_common(1)[0]
        print(
            f"[diversity] '{question}': {unique_count}/{GENERATIONS_PER_QUESTION} unique, "
            f"most frequent ({most_common_n}x): {most_common_text!r}"
        )

    top_word, top_count = first_word_counts.most_common(1)[0]
    top_pct = 100.0 * top_count / total_generations
    print()
    print(
        f"[diversity] top first word overall: {top_word!r} starts {top_count}/{total_generations} "
        f"generations ({top_pct:.1f}%)"
        + (" -- MODE COLLAPSE" if top_pct > 50.0 else "")
    )
    print(f"[diversity] top 20 first words across all {total_generations} generations:")
    for word, count in first_word_counts.most_common(20):
        pct = 100.0 * count / total_generations
        print(f"  {word!r}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    output_path, temperature = _output_path_and_temperature()
    main()
    run_diversity_check(output_path, temperature)
