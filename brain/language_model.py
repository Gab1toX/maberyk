# STATUS: SCAFFOLD ONLY — not connected to agent
# Activate when agent_generated conversation count >= 500 pure entries

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


def normalize_text(text: str) -> str:
    """Lowercase and strip accents, but preserve ñ (protect it around the NFD pass)."""
    protected = str(text).replace("ñ", "\x00").replace("Ñ", "\x00")
    stripped = ''.join(
        c for c in unicodedata.normalize('NFD', protected)
        if unicodedata.category(c) != 'Mn'
    )
    return stripped.replace("\x00", "ñ").lower()


class AgentTokenizer:
    """Word-level tokenizer built from the agent's own vocabulary. No pretrained embeddings."""

    PAD = "<pad>"
    UNK = "<unk>"
    Q = "<q>"
    A = "<a>"
    END = "<end>"
    SPECIAL_TOKENS = (PAD, UNK, Q, A, END)

    def __init__(self, words: Iterable[str], max_vocab_size: int = 8192) -> None:
        # Vocabulary is ranked by frequency (not first-seen order) so common
        # content words win the fixed budget over one-off rarities.
        counter = Counter(
            word
            for word in (str(raw).lower().strip() for raw in words)
            if word and word not in self.SPECIAL_TOKENS
        )
        budget = max_vocab_size - len(self.SPECIAL_TOKENS)
        collected = [word for word, _count in counter.most_common(budget)]

        self.words = list(self.SPECIAL_TOKENS) + collected
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.pad_index = self.word_to_index[self.PAD]
        self.unk_index = self.word_to_index[self.UNK]
        self.q_index = self.word_to_index[self.Q]
        self.a_index = self.word_to_index[self.A]
        self.end_index = self.word_to_index[self.END]

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


class _BPETokenizerBridge:
    """Adapts lectura.bpe.BPETokenizer to the attribute surface
    AgentLanguageModel/LanguageModelTrainer expect from a tokenizer
    (pad_index/unk_index/q_index/a_index/end_index, vocab_size, and a
    padded/truncated encode() for fixed-length training batches), so the
    rest of this module can treat word-level and BPE tokenizers the same
    way almost everywhere. <pad>/<unk>/<q>/<a>/<end> map onto the BPE
    tokenizer's own reserved ids 0/1/2/3/4 (see lectura/bpe.py's
    SPECIAL_ID) — no duplicate special tokens are created.

    The wrapped tokenizer is reachable via .bpe for the few call sites
    (generate_reply) that need BPE's own unpadded encode()/string decode()
    instead of the padded/list-returning contract this bridge presents.
    """

    def __init__(self, bpe_tokenizer: "BPETokenizer") -> None:
        self.bpe = bpe_tokenizer
        self.pad_index = bpe_tokenizer.special[AgentTokenizer.PAD]
        self.unk_index = bpe_tokenizer.special[AgentTokenizer.UNK]
        self.q_index = bpe_tokenizer.special[AgentTokenizer.Q]
        self.a_index = bpe_tokenizer.special[AgentTokenizer.A]
        self.end_index = bpe_tokenizer.special[AgentTokenizer.END]

    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size

    def encode(self, text: str, max_len: int = 16) -> list[int]:
        tokens = self.bpe.encode(str(text))[:max_len]
        tokens += [self.pad_index] * (max_len - len(tokens))
        return tokens


class AgentLanguageModel(nn.Module):
    """Word-level conditional Q->A Transformer encoder (causal self-attention).
    Every weight starts at random init and is shaped only by the agent's own
    experience — learned positional encoding, no pretrained embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_index: int = 0,
        tokenizer: "AgentTokenizer | _BPETokenizerBridge | None" = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_index = pad_index
        self.tokenizer = tokenizer

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.pos_encoding = nn.Embedding(512, embedding_dim)  # learned positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_encoding(positions)
        pad_mask = input_ids == self.pad_index
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True)
        return self.output_layer(x)

    @torch.no_grad()
    def generate_reply(
        self,
        question: str,
        max_new_tokens: int = 14,
        temperature: float = 1.0,
    ) -> list[str] | str:
        """Conditional Q->A generation: encode '<q> question <a>' and sample
        tokens until <pad>/<q>/<end> (or the model runs out of budget). <unk>
        is banned from being sampled so replies never surface it directly.

        Word tokenizer: returns a list[str] of words (unchanged behavior).
        BPE tokenizer: returns the decoded reply as a single str — BPE is
        lossless/case-and-accent-preserving by design, so the word path's
        normalize_text()+regex word cleanup is skipped entirely; the raw
        question goes straight through the BPE tokenizer's own encode().
        """
        if self.tokenizer is None:
            raise ValueError("AgentLanguageModel.generate_reply requires a tokenizer.")

        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        is_word_tokenizer = isinstance(self.tokenizer, AgentTokenizer)

        if is_word_tokenizer:
            question_words = []
            for word in normalize_text(question).split():
                clean = re.sub(r"[^\w]", "", word)
                if clean:
                    question_words.append(clean)
            prompt_ids = [
                self.tokenizer.word_to_index.get(word, self.tokenizer.unk_index)
                for word in question_words
            ]
        else:
            prompt_ids = self.tokenizer.bpe.encode(question)

        indices = [self.tokenizer.q_index, *prompt_ids, self.tokenizer.a_index]

        generated = list(indices)
        stop_indices = {
            self.tokenizer.pad_index,
            self.tokenizer.q_index,
            self.tokenizer.a_index,
            self.tokenizer.end_index,
        }

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.forward(input_tensor)
            next_logits = logits[0, -1] / max(temperature, 1e-6)
            next_logits[self.tokenizer.unk_index] = float("-inf")
            probabilities = torch.softmax(next_logits, dim=-1)
            next_index = int(torch.multinomial(probabilities, 1).item())

            if next_index in stop_indices:
                break

            generated.append(next_index)

        if was_training:
            self.train()

        generated_ids = generated[len(indices):]
        if is_word_tokenizer:
            return self.tokenizer.decode(generated_ids)
        return self.tokenizer.bpe.decode(generated_ids)


class LanguageModelTrainer:
    """Loads human_taught Q/A pairs (primary) and corpus/thought sentences
    (secondary, for fluency) from episodic_memory.sqlite3 and trains a
    conditional AgentLanguageModel from scratch. Weights are saved separately
    from agent_state.pt. agent_generated conversations are excluded — they are
    the model's own past output, not ground truth to imitate."""

    MAX_LEN = 32
    VAL_FRACTION = 0.1
    VAL_SEED = 1234
    EARLY_STOP_PATIENCE = 3
    EARLY_STOP_MIN_DELTA = 0.01

    def __init__(
        self,
        memory_path: str | Path,
        max_vocab_size: int = 8192,
        embedding_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
        tokenizer_type: str = "word",
    ) -> None:
        if tokenizer_type not in ("word", "bpe"):
            raise ValueError(f"tokenizer_type must be 'word' or 'bpe', got {tokenizer_type!r}")
        self.tokenizer_type = tokenizer_type
        self.memory_path = Path(memory_path)
        self.device = torch.device(device)

        # self.samples stays the UNIQUE deduped pool — the train/val split is
        # built from it first so val can never contain a duplicate of a
        # sample that also landed in train. Oversampling only ever touches
        # train_samples afterwards.
        self.samples = self._load_training_samples()
        if not self.samples:
            raise ValueError(
                f"No training samples found in {self.memory_path}. "
                "This scaffold trains primarily on human_taught question/answer "
                "pairs, with corpus_agente.txt and episode thoughts mixed in for "
                "fluency — agent_generated and unknown sources are excluded."
            )

        self.train_samples, self.val_samples = self._split_train_val(self.samples)
        self.train_samples = self._oversample_pairs(self.train_samples)

        if self.tokenizer_type == "bpe":
            self.tokenizer = self._load_bpe_tokenizer()
        else:
            vocabulary_words = self._collect_vocabulary(self.samples)
            self.tokenizer = AgentTokenizer(vocabulary_words, max_vocab_size=max_vocab_size)
            self._print_vocab_coverage(vocabulary_words)

        self.model = AgentLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_index=self.tokenizer.pad_index,
            tokenizer=self.tokenizer,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_index)

        self._print_startup_summary()

    def _print_startup_summary(self) -> None:
        sample_size = min(5, len(self.samples))
        sample = random.sample(self.samples, sample_size)
        print(
            f"[language_model] loaded {len(self.samples)} unique samples "
            f"({len(self.train_samples)} train after pair oversampling / "
            f"{len(self.val_samples)} val), vocab size {self.tokenizer.vocab_size}"
        )
        print("[language_model] sample training rows:")
        for row in sample:
            print(f"  - {row}")

    @staticmethod
    def _load_bpe_tokenizer() -> _BPETokenizerBridge:
        from lectura.bpe import BPETokenizer  # lazy: keep the word-level path free of this dependency

        tokenizer_path = Path(__file__).resolve().parent.parent / "datos_lectura" / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"BPE tokenizer not found at {tokenizer_path}. Train it first with "
                "`python -m lectura.bpe --train`."
            )
        return _BPETokenizerBridge(BPETokenizer.load(tokenizer_path))

    def _print_vocab_coverage(self, vocabulary_words: list[str]) -> None:
        unique_words = {
            word for word in (str(raw).lower().strip() for raw in vocabulary_words) if word
        }
        dropped = sorted(unique_words - set(self.tokenizer.word_to_index))
        covered = len(unique_words) - len(dropped)
        print(
            f"[language_model] vocab coverage: {covered}/{len(unique_words)} unique corpus "
            f"words fit in vocab size {self.tokenizer.vocab_size}"
        )
        if dropped:
            print(
                f"[language_model] WARNING: {len(dropped)} unique word(s) dropped from the "
                "vocabulary and will map to <unk> during training — raise max_vocab_size to "
                f"reach zero <unk>. sample dropped words: {dropped[:20]}"
            )

    def _split_train_val(self, samples: list[str]) -> tuple[list[str], list[str]]:
        shuffled = list(samples)
        random.Random(self.VAL_SEED).shuffle(shuffled)
        val_size = int(len(shuffled) * self.VAL_FRACTION) if len(shuffled) >= 10 else 0
        val_samples = shuffled[:val_size]
        train_samples = shuffled[val_size:]
        if not train_samples:
            train_samples, val_samples = shuffled, []
        return train_samples, val_samples

    def _oversample_pairs(self, samples: list[str]) -> list[str]:
        # Q/A pairs start with "<q> "; repeating them 3x inside the already
        # dedup-safe train split (never val) pushes Q->A signal from ~18% to
        # ~40% of training batches without risking a train/val leak.
        q_prefix = f"{AgentTokenizer.Q} "
        pairs = [sample for sample in samples if sample.startswith(q_prefix)]
        plain = [sample for sample in samples if not sample.startswith(q_prefix)]
        oversampled = plain + pairs * 3
        random.shuffle(oversampled)
        return oversampled

    def _load_training_samples(self) -> list[str]:
        connection = sqlite3.connect(self.memory_path)
        try:
            pair_rows = connection.execute(
                "SELECT question, answer FROM conversations WHERE source = 'human_taught'"
            ).fetchall()
            thought_rows = connection.execute(
                "SELECT thought FROM episodes WHERE thought != '' AND thought IS NOT NULL"
            ).fetchall()
        finally:
            connection.close()

        # BPE (lectura/bpe.py) is lossless and was trained on original-case,
        # accented text — normalize_text()'s lowercasing/accent-stripping
        # would feed it text unlike anything it actually learned from. The
        # word tokenizer's own vocabulary IS built from normalize_text()
        # output (_collect_vocabulary), so that path is untouched.
        text_filter = str if self.tokenizer_type == "bpe" else normalize_text

        pair_samples = list(dict.fromkeys(
            f"{AgentTokenizer.Q} {text_filter(question)} {AgentTokenizer.A} "
            f"{text_filter(answer)} {AgentTokenizer.END}"
            for question, answer in pair_rows
            if question and question.strip() and answer and answer.strip()
        ))

        # Episode thoughts run ~170k rows but very few are unique — dedupe
        # first, then cap at 2x the pair count so fluency data can't drown
        # out the Q/A signal the model actually needs to learn.
        thought_sentences = list(dict.fromkeys(
            text_filter(row[0].strip())
            for row in thought_rows
            if row[0] and row[0].strip()
        ))
        thought_cap = len(pair_samples) * 2
        if len(thought_sentences) > thought_cap:
            thought_sentences = random.sample(thought_sentences, thought_cap)
        thought_samples = [
            f"{AgentTokenizer.A} {sentence} {AgentTokenizer.END}" for sentence in thought_sentences
        ]

        corpus_path = self.memory_path.parent / "corpus_agente.txt"
        corpus_samples = []
        if corpus_path.is_file():
            for line in corpus_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    corpus_samples.append(f"{AgentTokenizer.A} {text_filter(line)} {AgentTokenizer.END}")

        plain_samples = list(dict.fromkeys(thought_samples + corpus_samples))
        all_samples = pair_samples + plain_samples
        random.shuffle(all_samples)

        print(
            f"[language_model] {len(pair_samples)} human_taught q/a pairs, "
            f"{len(thought_samples)} deduped episode thoughts (capped at {thought_cap}), "
            f"{len(corpus_samples)} from corpus_agente.txt, "
            f"{len(all_samples)} total samples"
        )

        return all_samples

    def _collect_vocabulary(self, samples: list[str]) -> list[str]:
        words: list[str] = []
        for sample in samples:
            words.extend(sample.lower().split())
        return words

    def _build_batch(self, samples: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_batch = []
        target_batch = []
        for sample in samples:
            encoded = self.tokenizer.encode(sample, max_len=self.MAX_LEN + 1)
            target = encoded[1:]
            # Loss is computed only on tokens after <a> — mask everything up
            # to and including the <a> marker (the prompt) with pad_index so
            # the model is never trained to reproduce the question back. The
            # trailing <end> token sits well after a_pos (it's part of the
            # answer/content, not the prompt) so it is never masked here and
            # is trained like any other answer token.
            if self.tokenizer.a_index in encoded:
                a_pos = encoded.index(self.tokenizer.a_index)
                for i in range(min(a_pos, len(target))):
                    target[i] = self.tokenizer.pad_index
            input_batch.append(encoded[:-1])
            target_batch.append(target)
        return (
            torch.tensor(input_batch, dtype=torch.long, device=self.device),
            torch.tensor(target_batch, dtype=torch.long, device=self.device),
        )

    def _run_epoch(self, samples: list[str], batch_size: int, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        batch_count = 0
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            input_ids, target_ids = self._build_batch(batch)

            with torch.set_grad_enabled(train):
                logits = self.model(input_ids)
                loss = self.loss_fn(
                    logits.reshape(-1, self.tokenizer.vocab_size),
                    target_ids.reshape(-1),
                )

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / max(batch_count, 1)

    def train(self, epochs: int = 10, batch_size: int = 16) -> list[float]:
        train_losses = []
        best_val_loss = float("inf")
        stale_epochs = 0
        for epoch in range(epochs):
            random.shuffle(self.train_samples)
            train_loss = self._run_epoch(self.train_samples, batch_size, train=True)
            train_losses.append(train_loss)

            if self.val_samples:
                val_loss = self._run_epoch(self.val_samples, batch_size, train=False)
                print(
                    f"[language_model] epoch {epoch + 1}/{epochs} — "
                    f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )
                if best_val_loss - val_loss > self.EARLY_STOP_MIN_DELTA:
                    best_val_loss = val_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.EARLY_STOP_PATIENCE:
                        print(
                            f"[language_model] early stopping at epoch {epoch + 1}/{epochs} "
                            f"— val loss did not improve by more than {self.EARLY_STOP_MIN_DELTA} "
                            f"for {self.EARLY_STOP_PATIENCE} consecutive epochs"
                        )
                        break
            else:
                print(f"[language_model] epoch {epoch + 1}/{epochs} — train loss {train_loss:.4f} (no val split)")

        self.model.train()
        return train_losses

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab_size": self.tokenizer.vocab_size,
            "embedding_dim": self.model.embedding_dim,
            "nhead": self.model.transformer.layers[0].self_attn.num_heads,
            "num_layers": len(self.model.transformer.layers),
            "dim_feedforward": self.model.transformer.layers[0].linear1.out_features,
            "pad_index": self.tokenizer.pad_index,
            "tokenizer_type": self.tokenizer_type,
            "state_dict": self.model.state_dict(),
        }
        if self.tokenizer_type == "word":
            payload["tokenizer_words"] = self.tokenizer.words
        # bpe mode saves no tokenizer data of its own — load_model() re-reads
        # datos_lectura/tokenizer.json fresh, same as training did (see
        # _load_bpe_tokenizer), so the checkpoint always matches the tokenizer
        # currently on disk rather than a frozen copy.
        torch.save(payload, path)

    @classmethod
    def load_model(
        cls, path: str | Path, map_location: str | torch.device | None = None
    ) -> tuple[AgentLanguageModel, AgentTokenizer | _BPETokenizerBridge]:
        checkpoint = torch.load(path, map_location=map_location)
        # Existing checkpoints predate tokenizer_type entirely — absence means
        # "word", so old language_model.pt files load exactly as before.
        tokenizer_type = checkpoint.get("tokenizer_type", "word")

        if tokenizer_type == "bpe":
            tokenizer = cls._load_bpe_tokenizer()
        else:
            tokenizer = AgentTokenizer.__new__(AgentTokenizer)
            tokenizer.words = checkpoint["tokenizer_words"]
            tokenizer.word_to_index = {word: index for index, word in enumerate(tokenizer.words)}
            tokenizer.index_to_word = {index: word for word, index in tokenizer.word_to_index.items()}
            tokenizer.pad_index = checkpoint["pad_index"]
            tokenizer.unk_index = tokenizer.word_to_index[AgentTokenizer.UNK]

            required_tokens = (AgentTokenizer.Q, AgentTokenizer.A, AgentTokenizer.END)
            missing_tokens = [token for token in required_tokens if token not in tokenizer.word_to_index]
            if missing_tokens:
                raise ValueError(
                    f"Checkpoint at {path} predates the current conditional Q->A model "
                    f"and is missing special token(s) {missing_tokens} from its vocabulary. "
                    "Retrain with the current brain/language_model.py to produce a "
                    "compatible checkpoint."
                )
            tokenizer.q_index = tokenizer.word_to_index[AgentTokenizer.Q]
            tokenizer.a_index = tokenizer.word_to_index[AgentTokenizer.A]
            tokenizer.end_index = tokenizer.word_to_index[AgentTokenizer.END]

        model = AgentLanguageModel(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dim_feedforward=checkpoint["dim_feedforward"],
            pad_index=checkpoint["pad_index"],
            tokenizer=tokenizer,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the scaffold AgentLanguageModel on human_taught Q/A pairs."
    )
    parser.add_argument(
        "--memory", required=True, type=Path, help="Path to episodic_memory.sqlite3"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("language_model.pt"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--sample-question", type=str, default="que ves",
        help="Question used for the post-training sample generations.",
    )
    parser.add_argument(
        "--tokenizer-type", choices=("word", "bpe"), default="word",
        help="'word' (default, unchanged) or 'bpe' (datos_lectura/tokenizer.json).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[language_model] using device: {device}")

    trainer = LanguageModelTrainer(args.memory, device=device, tokenizer_type=args.tokenizer_type)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    trainer.save(args.output)
    print(f"[language_model] weights saved to {args.output}")

    print(
        f"[language_model] sample generations (question='{args.sample_question}', "
        f"temperature={args.temperature}):"
    )
    for _ in range(5):
        reply = trainer.model.generate_reply(
            args.sample_question,
            temperature=args.temperature,
        )
        text = reply if isinstance(reply, str) else " ".join(reply)
        print(f"  - {text}")


if __name__ == "__main__":
    main()
