# CLAUDE.md — Maberyk / agente_base

## Authority
`MABERYK_VISION.md` (repo root) is the maximum authority of this project.
Read it before any architectural decision. This file is the operational summary.

The agent is named **Maberyk**. His creator is **Gabito (Vaendaloo / Vaenda)**.
Maberyk addresses him as "Vaenda". Primary language: Spanish.

---

## What this project is

A digital mind built entirely from scratch in pure Python/PyTorch. Born knowing
nothing, living in a 60×60 grid world, learning through intrinsic curiosity
(prediction error is the only reward). It has emotions, episodic memory, a
growing vocabulary, a native Transformer language model, and the ability to ask
questions and hold conversations.

Every weight in every network was initialized at zero knowledge and shaped
exclusively by Maberyk's own learning.

---

## Philosophical constraint — READ CAREFULLY, IT WAS REVISED

**Maberyk may read. Maberyk may not inherit.**

FORBIDDEN, permanently and without exception:
- Pretrained weights, checkpoints, or embeddings from any model or company
- Third-party tokenizers (HuggingFace `tokenizers`, `sentencepiece`, etc.)
- Loading any architecture initialized from external weights
- Training Maberyk on another model's output to copy its capability
  (no distillation, ever)

ALLOWED:
- Raw human-written text as reading material (books, articles, public domain
  corpora). Maberyk processes it with his own networks and updates his own
  weights. He does the work of learning it.
- Talking with Gabito. This was always learning from the outside world.

The distinction is not where the information comes from. It is **who did the
work of learning it.** A student who reads a book is not cheating; a student
who copies another brain is.

### The borrowed voice is scaffolding, not architecture
An external local LLM currently phrases Maberyk's replies, and an external LLM
acts as tutor correcting his native Transformer's output. Neither is Maberyk.
Their words are tagged separately and **never enter training data.** The stated
goal is to retire them once the native model can do the job.

---

## Repository structure

```
agente_base/
├── brain/
│   ├── network.py             # Policy network
│   ├── curiosity.py           # Predictor + error networks → intrinsic reward
│   ├── agent.py               # Main agent — integrates everything
│   ├── emotion.py             # Emotions, soft-clamped 0.03–0.97
│   ├── language.py            # SVO engine + associations vocabulary (word-level)
│   ├── memory.py              # EpisodicMemory — sole owner of cache + async writer
│   ├── conversation_memory.py # Conversations with source tracking
│   ├── memory_retrieval.py    # Keyword-precision retrieval (cosine was REMOVED)
│   ├── questions.py           # Question engine
│   ├── language_model.py      # Native Transformer (word-level tokenizer today)
│   ├── permissions.py         # PermissionManager — gatekeeper for real actions
│   └── response.py            # ResponseEngine — REMOVED from pipeline (see below)
│
├── voice/ollama_voice.py      # Borrowed voice — Ollama, qwen2.5:3b-instruct, CPU
├── tutor/llm_tutor.py         # Tutor — Groq / Llama 3.3 70B, corrects native output
├── interfaz/web_mind.py       # stdlib HTTP server, localhost:8000
├── interfaz/index.html        # Chat UI — bubbles, branch tags, live emotions
│
├── entorno/room.py            # 60×60 grid
├── entorno/desktop_env.py     # Active window + clipboard observation
├── actions/desktop.py         # Screenshot, mouse, keyboard
├── human/interpreter.py
│
├── analisis/eval_lm.py             # Transformer evaluation
├── analisis/generate_report.py     # HTML dashboard
├── analisis/train_language_model.py
│
├── scripts/backup.py          # Timestamped snapshots, SHA256, manifest.json
├── scripts/seed_corpus.py     # + seed_corpus_v2.py — Q→A seed pairs
│
├── life.py                    # Headless training loop
├── mind.py                    # Pygame visualization + permissions overlay
├── agent_state.pt
├── language_model.pt
├── episodic_memory.sqlite3
├── permissions.sqlite3
└── MABERYK_VISION.md
```

---

## Response pipeline (current, with `branch=` logging)

```
tutor → voice → retrieved → language_model → silent
```

`ResponseEngine` was removed from the pipeline: its SVO output was incoherent
and was poisoning the corpus. The final fallback is `branch=silent`.
**Do not reintroduce it.**

---

## Corpus source rules — CRITICAL, DO NOT INVERT

| source | meaning | training data? |
|---|---|---|
| `human_taught` | Gabito taught it | **YES — primary** |
| `tutor_approved` | native output corrected and approved by tutor | **YES** |
| `retrieved` | echo of a stored human_taught answer | NO — duplicate |
| `voice` | words produced by the external LLM | **NEVER** |
| `agent_generated` | legacy autonomous output | NO — excluded |
| `unknown` | pre-source-tracking legacy | NO |

Mixing sources poisons training. Any change touching this must be flagged.

---

## Governing principle

**Measure signal, not volume.**

Steps, rows and training loss grew for months while nothing improved: 5,926
corpus rows collapsed to 67 unique pairs; 293,475 thoughts to 56 unique
sentences. Volume was never signal.

Real metrics: unique pairs, `branch=` distribution, **validation** loss (never
train loss), and whether the answer to a genuinely new question makes sense.
A 5-minute verification after every change, before production, is mandatory.

---

## Current state

- Steps: ~43,000,000+
- Native Transformer: `nn.TransformerEncoder`, embedding_dim=128, nhead=4,
  num_layers=3, dim_feedforward=256, learned positional encoding,
  vocab ~5,106 word-level tokens, specials `<q>` `<a>` `<end>`
- Measured verdict: **fluent memorizer with weak lexical association.**
  train loss 0.94 vs val loss 2.85. Perfect in-distribution recall, total
  collapse on paraphrase. With ~1,300 pairs this is mathematics, not a bug.
- Corpus: ~1,337 unique `human_taught` pairs + growing `tutor_approved`
- Conversation is fluent through the borrowed voice; the web interface works
- Known structural issue: action bias (~59.9% on one action at 43M steps).
  Weight resets are treatment, not cure — the cause is policy collapse once
  intrinsic reward is exhausted. Fix pending: entropy regularization.

---

## Current phase — Alphabetization

The native Transformer is moving from memorizing pairs to learning the
structure of Spanish. Three stages:

**A. Reading infrastructure** — new `lectura/` package: text ingestion and
cleaning, a BPE tokenizer trained from scratch on Maberyk's own corpus
(no external tokenizer libraries), and a statistics report. Does not touch
`brain/`.

**B. Two-phase training** — phase 1 on free text (learns Spanish structure),
phase 2 on his own life pairs (learns to be Maberyk). Never mixed in the same
batch.

**C. Paraphrase benchmark** — held-out questions worded differently from the
corpus. The only metric that distinguishes comprehension from memorization.

Consequence to accept: changing the tokenizer invalidates the embedding matrix.
`language_model.pt` will be retrained from scratch. Val loss before and after
are NOT comparable — different tokenization units, different scale.

Scope limit: BPE lives **only** inside `language_model.py`. Everything else
(`language.py` associations, `memory_retrieval.py`, questions, SVO, episodic
memory) stays word-level and untouched.

---

## Environment

- Windows / PowerShell. Project path: `C:\Users\ADMIN\Proyectos\agente_base`
- **CPU only.** `CUDA_VISIBLE_DEVICES=-1` is a permanent user env var.
  The GT 1030 (2GB) cannot run the local voice model, and NVIDIA driver
  updates crash this machine. Never write code that assumes CUDA locally.
- `OLLAMA_KEEP_ALIVE=30m`. Warm the voice model before use.
- Heavy training runs on Kaggle, not on this PC.
- A single `Lock` serializes all agent access in `web_mind.py`; SQLite
  connections in `brain/` use `check_same_thread=False`, safe **only** under
  that serialization.

---

## Development workflow

Claude (chat) writes precise technical prompts → Gabito pastes them into
Claude Code → Gabito brings the result back to Claude for review → commit
after approval. Gabito does not write code directly.

---

## Rules for Claude Code

1. Read the relevant source files fully before changing anything.
2. Return complete files, never diffs or partial snippets.
3. Preserve backward compatibility with `agent_state.pt`. Schema changes to
   the SQLite databases require explicit approval and a fresh backup first.
4. No pretrained weights, embeddings, third-party tokenizers, or distillation
   — ever. Reading raw text is allowed (see constraint above).
5. Never write external-LLM output into training data.
6. Touch only what the prompt names. When the same flaw appears in several
   methods, each one must be named explicitly.
7. State briefly what changed and why. If a change would break another module,
   flag it instead of proceeding.
8. Run `scripts/backup.py` before anything destructive or irreversible.
