# STATUS: SCAFFOLD ONLY — not connected to agent
# Activate when agent_generated conversation count >= 500 pure entries

from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class AgentTokenizer:
    """Word-level tokenizer built from the agent's own vocabulary. No pretrained embeddings."""

    PAD = "<pad>"
    UNK = "<unk>"
    SPECIAL_TOKENS = (PAD, UNK)

    def __init__(self, words: Iterable[str], max_vocab_size: int = 1024) -> None:
        collected: list[str] = []
        seen: set[str] = set()
        budget = max_vocab_size - len(self.SPECIAL_TOKENS)
        for word in words:
            word = str(word).lower().strip()
            if word and word not in seen:
                seen.add(word)
                collected.append(word)
            if len(collected) >= budget:
                break

        self.words = list(self.SPECIAL_TOKENS) + collected
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.pad_index = self.word_to_index[self.PAD]
        self.unk_index = self.word_to_index[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    def encode(self, text: str, max_len: int = 16) -> list[int]:
        tokens = [
            self.word_to_index.get(word, self.unk_index)
            for word in str(text).lower().split()
        ]
        tokens = tokens[:max_len]
        tokens += [self.pad_index] * (max_len - len(tokens))
        return tokens

    def decode(self, indices: Iterable[int]) -> list[str]:
        return [
            self.index_to_word.get(int(index), self.UNK)
            for index in indices
            if int(index) != self.pad_index
        ]


class AgentLanguageModel(nn.Module):
    """Minimal word-level autoregressive LSTM. Every weight starts at random init
    and is shaped only by the agent's own generated conversations — no attention,
    no positional encoding, no pretrained embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_index: int = 0,
        tokenizer: "AgentTokenizer | None" = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.tokenizer = tokenizer

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input_ids)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.output_layer(output)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        prompt_words: list[str],
        max_new_tokens: int = 8,
        temperature: float = 1.0,
    ) -> list[str]:
        if self.tokenizer is None:
            raise ValueError("AgentLanguageModel.generate requires a tokenizer.")

        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        indices = [
            self.tokenizer.word_to_index.get(word.lower().strip(), self.tokenizer.unk_index)
            for word in prompt_words
            if word.strip()
        ] or [self.tokenizer.unk_index]

        # Structural-artifact guard: ResponseEngine's fixed SVO pattern makes
        # the prompt word itself (or <unk>) the dominant next token in the
        # corpus, so left unchecked the model mostly echoes the prompt back.
        # The first generated token is resampled away from the prompt's own
        # token ids instead of just accepting whatever scores highest.
        prompt_index_set = set(indices)

        generated = list(indices)
        input_tensor = torch.tensor([indices], dtype=torch.long, device=device)
        logits, hidden = self.forward(input_tensor)

        tokens_used = 0
        first_token_resolved = False
        while tokens_used < max_new_tokens:
            next_logits = logits[0, -1] / max(temperature, 1e-6)
            probabilities = torch.softmax(next_logits, dim=-1)
            next_index = int(torch.multinomial(probabilities, 1).item())
            tokens_used += 1

            if next_index == self.tokenizer.pad_index:
                break

            if not first_token_resolved and next_index in prompt_index_set:
                # Skip: resample from the same distribution (hidden state
                # has not advanced) until a non-prompt word is produced or
                # the max_new_tokens budget runs out.
                continue

            first_token_resolved = True
            generated.append(next_index)
            input_tensor = torch.tensor([[next_index]], dtype=torch.long, device=device)
            logits, hidden = self.forward(input_tensor, hidden)

        if was_training:
            self.train()
        return self.tokenizer.decode(generated)


class LanguageModelTrainer:
    """Loads agent_generated conversations from episodic_memory.sqlite3 and trains
    AgentLanguageModel from scratch. Weights are saved separately from agent_state.pt."""

    MAX_LEN = 16

    def __init__(
        self,
        memory_path: str | Path,
        max_vocab_size: int = 1024,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        self.memory_path = Path(memory_path)
        self.device = torch.device(device)

        self.sentences = self._load_agent_generated_sentences()
        if not self.sentences:
            raise ValueError(
                f"No 'agent_generated' conversations found in {self.memory_path}. "
                "This scaffold trains exclusively on the agent's own generated "
                "language — human_taught and unknown sources are excluded."
            )

        vocabulary_words = self._collect_vocabulary(self.sentences)
        self.tokenizer = AgentTokenizer(vocabulary_words, max_vocab_size=max_vocab_size)

        self.model = AgentLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_index=self.tokenizer.pad_index,
            tokenizer=self.tokenizer,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)

        self._print_startup_summary()

    def _print_startup_summary(self) -> None:
        sample_size = min(5, len(self.sentences))
        sample = random.sample(self.sentences, sample_size)
        print(
            f"[language_model] loaded {len(self.sentences)} agent_generated sentences, "
            f"vocab size {self.tokenizer.vocab_size}"
        )
        print("[language_model] sample sentences:")
        for sentence in sample:
            print(f"  - {sentence}")

    def _load_agent_generated_sentences(self) -> list[str]:
        connection = sqlite3.connect(self.memory_path)
        try:
            rows = connection.execute(
                "SELECT answer FROM conversations WHERE source = 'agent_generated'"
            ).fetchall()
        finally:
            connection.close()
        return [
            self._strip_structural_prefix(row[0])
            for row in rows
            if row[0] and row[0].strip()
        ]

    def _strip_structural_prefix(self, sentence: str) -> str:
        # ResponseEngine emits a fixed SVO pattern, so nearly every sentence
        # opens with the same two words (e.g. "curious see"). Dropping them
        # removes that structural artifact from the training corpus without
        # losing semantic content — but only when enough content remains
        # after stripping, so short sentences are left intact.
        words = sentence.split()
        remainder = words[2:]
        if len(remainder) > 4:
            return " ".join(remainder)
        return sentence

    def _collect_vocabulary(self, sentences: list[str]) -> list[str]:
        words: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in seen:
                    seen.add(word)
                    words.append(word)
        return words

    def _build_batch(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_batch = []
        target_batch = []
        for sentence in sentences:
            encoded = self.tokenizer.encode(sentence, max_len=self.MAX_LEN + 1)
            input_batch.append(encoded[:-1])
            target_batch.append(encoded[1:])
        return (
            torch.tensor(input_batch, dtype=torch.long, device=self.device),
            torch.tensor(target_batch, dtype=torch.long, device=self.device),
        )

    def train(self, epochs: int = 10, batch_size: int = 16) -> list[float]:
        self.model.train()
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            for start in range(0, len(self.sentences), batch_size):
                batch_sentences = self.sentences[start : start + batch_size]
                input_ids, target_ids = self._build_batch(batch_sentences)

                logits, _ = self.model(input_ids)
                loss = self.loss_fn(
                    logits.reshape(-1, self.tokenizer.vocab_size),
                    target_ids.reshape(-1),
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            average_loss = total_loss / max(batch_count, 1)
            epoch_losses.append(average_loss)
            print(f"[language_model] epoch {epoch + 1}/{epochs} — loss {average_loss:.4f}")

        return epoch_losses

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "vocab_size": self.tokenizer.vocab_size,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "pad_index": self.tokenizer.pad_index,
                "tokenizer_words": self.tokenizer.words,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load_model(
        cls, path: str | Path, map_location: str | torch.device | None = None
    ) -> tuple[AgentLanguageModel, AgentTokenizer]:
        checkpoint = torch.load(path, map_location=map_location)
        tokenizer = AgentTokenizer.__new__(AgentTokenizer)
        tokenizer.words = checkpoint["tokenizer_words"]
        tokenizer.word_to_index = {word: index for index, word in enumerate(tokenizer.words)}
        tokenizer.index_to_word = {index: word for word, index in tokenizer.word_to_index.items()}
        tokenizer.pad_index = checkpoint["pad_index"]
        tokenizer.unk_index = tokenizer.word_to_index[AgentTokenizer.UNK]

        model = AgentLanguageModel(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            pad_index=checkpoint["pad_index"],
            tokenizer=tokenizer,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the scaffold AgentLanguageModel on agent_generated conversations."
    )
    parser.add_argument(
        "--memory", required=True, type=Path, help="Path to episodic_memory.sqlite3"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("language_model.pt"))
    args = parser.parse_args()

    trainer = LanguageModelTrainer(args.memory)
    print(
        f"[language_model] training on {len(trainer.sentences)} agent_generated "
        f"sentences, vocab size {trainer.tokenizer.vocab_size}"
    )
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    trainer.save(args.output)
    print(f"[language_model] weights saved to {args.output}")


if __name__ == "__main__":
    main()
