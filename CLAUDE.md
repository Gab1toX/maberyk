# CLAUDE.md — proyecto5 / agente_base

## Documento de visión
Este proyecto tiene un documento de visión completo: MABERYK_VISION.md
Léelo antes de cualquier decisión arquitectónica. Es la autoridad máxima.
El agente se llama Maberyk. Su creador es Gabito (Vaendaloo).

## What this project is

An artificial intelligence agent built entirely from scratch in pure Python/PyTorch.
No pretrained models. No fine-tuning. No external weights of any kind.
Every weight in every network was initialized at zero knowledge and shaped
exclusively by the agent's own lived experience inside its environment.

This is not a chatbot wrapper. It is an attempt to answer a question that no
one has answered definitively at scale:

> Can genuine linguistic intelligence emerge from pure experience, without
> internet text, without human supervision, without borrowed cognition?

The agent was born knowing nothing. It lives in a 60×60 grid world with 30
objects across 4 zones. It learns through intrinsic curiosity — prediction
error is its only reward. It develops emotions, episodic memory, a growing
vocabulary, and the ability to ask questions and generate responses. Every
capability it has, it earned.

The long-term trajectory: general digital mind → multiple domains → hardware.

---

## Absolute philosophical constraint

**No pretrained model may ever be connected to this system, for any reason.**
Not for language. Not for embeddings. Not for feature extraction. Not even
temporarily. Any capability introduced must be trainable from scratch using
only the agent's own generated data. This constraint is non-negotiable and
defines the entire value of the project.

---

## Repository structure

```
agente_base/
├── brain/
│   ├── network.py            # Policy network (PyTorch, configurable layers, ReLU)
│   ├── curiosity.py          # Intrinsic curiosity module (predictor + error networks)
│   ├── agent.py              # Main agent — integrates all subsystems
│   ├── emotion.py            # Emotional state (curiosity, fear, confidence, confusion)
│   │                           soft-clamped between 0.03 and 0.97 — never saturates
│   ├── language.py           # SVO bilingual language engine (en/es, no SQLite per step)
│   ├── memory.py             # Episodic memory — SQLite-backed, in-memory cache
│   ├── conversation_memory.py# Conversational memory — source tracking (human_taught /
│   │                           agent_generated / unknown)
│   ├── questions.py          # Question engine — statistical surprise detection + timer
│   └── response.py           # Response engine — generates directed replies to human
│
├── entorno/
│   └── room.py               # 60×60 room, 30 objects, 4 zones, bidirectional text channel
│
├── human/
│   └── interpreter.py        # Converts free text from human into environment modifications
│
├── analisis/
│   └── generate_report.py    # Standalone HTML analytics dashboard (Chart.js, read-only)
│
├── life.py                   # Headless training loop
├── mind.py                   # Pygame visualization + conversation panel
└── migrate_checkpoint.py     # Weight migration script for OBSERVATION_SIZE changes
```

---

## Architecture

### Neural networks (3 total, all trained from scratch)
- **Policy network** (`network.py`): selects actions from observations
- **Predictor network** (`curiosity.py`): predicts the next observation
- **Error network** (`curiosity.py`): measures prediction error → intrinsic reward

### Core parameters
```python
BEHAVIORS = ('noise','move_randomly','change_color','open_door','nothing',
             'reflect','grow','teleport','glow')
COLORS    = ('gray','gold','blue','red','green','yellow','purple','silver','cyan')
OBSERVATION_SIZE = 2 + 1 + 8*(1 + 1 + len(BEHAVIORS) + len(COLORS))  # = 163
ACTION_SIZE      = 5   # up, down, left, right, touch
hidden_layers    = (128, 128)
learning_rate    = 1e-3
exploration_rate = 0.1
```

### Per-step learning loop
```
observe → predict next_obs → act → measure prediction error (= intrinsic reward)
→ backprop → update emotions (soft-clamped) → generate SVO thought
→ if human_message in obs → ResponseEngine generates reply → store as agent_generated
→ if confusion high OR timer > 2000 → formulate question (if not already answered)
```

### Conversation memory sources
- `human_taught` — human answered a question the agent asked
- `agent_generated` — agent generated the response autonomously (ResponseEngine)
- `unknown` — legacy data from before source tracking was implemented

Only `agent_generated` conversations are valid training data for the future
language model. `human_taught` and `unknown` must be filtered out.

---

## Current agent state
- **Steps:** ~8,725,000+ across two Kaggle accounts (vaenda, vaenda2)
- **Vocabulary:** 722 words
- **Conversations:** 2,075 total (source breakdown unknown until next report)
- **Episodes:** ~98,613
- **Dominant emotion:** confidence (~0.75–0.97, soft-clamped, no longer at 1.0)
- **Known issue:** action `right` dominates ~80% of all actions — root cause
  not yet identified; suspected: initialization bias in `_emotion_adjusted_scores()`

---

## Training infrastructure
- **Platform:** Kaggle GPU T4 (accounts `vaenda` and `vaenda2`, sequential never
  simultaneous — both write to the same Google Drive folder `agente_base/`)
- **Persistence:** Google Drive via Service Account (`GOOGLE_CREDENTIALS` Kaggle Secret)
- **Checkpoint files:** `agent_state.pt`, `episodic_memory.sqlite3`
- **Saves:** every 2,000 steps, auto-uploaded to Drive

---

## Development workflow

Claude Code reads this file and the full repository before acting.
Tasks are specified below with exact file targets and constraints.
No architectural context needs to be re-explained — read the source.

**Code quality bar:** production-grade. Every change must be backward-compatible
with existing `agent_state.pt` checkpoints and `episodic_memory.sqlite3` schema.
No schema migrations. No breaking changes to public method signatures.

---

## IMMEDIATE TASKS — execute in order

### TASK 1 — Refactor `brain/memory.py`: single source of truth

`EpisodicMemory` must become the sole owner of the episode cache, async
persistence, and rolling statistics. Currently `agent.py` duplicates all of
this logic and accesses `memory._episodes_by_recency` directly (private
attribute). That architectural flaw must be eliminated here.

Requirements:

1. Add an async write queue (`threading.Queue` + daemon writer thread) inside
   `EpisodicMemory`. `store()` must update the in-memory cache immediately and
   enqueue the SQLite INSERT — never block the caller on disk I/O. Add `flush()`
   to drain the queue completely (call before save/close). Add `close()` that
   calls `flush()` then closes the connection.

2. Cap `_episodes_by_recency` at **10,000 entries** (most recent). On overflow,
   evict the oldest. On initial load, fetch only `ORDER BY id DESC LIMIT 10000`.
   Update the count-verification check to compare against `min(db_total, 10000)`.

3. Replace the O(n) `_rebuild_summary_cache_from_memory()` with O(1) rolling stats:
   - Maintain `_rolling_surprise_sum: float` and `_rolling_count: int`
   - `store()`: add new surprise to sum, increment count
   - Eviction: subtract evicted episode's surprise, decrement count
   - `average_surprise = _rolling_surprise_sum / _rolling_count` (or 0.0)
   - Action stats (`_action_counts_from_memory`, `_surprise_by_action_from_memory`)
     still scan `_episodes_by_recency[:500]` — do not change that logic

4. Add public method:
   ```python
   def memory_feature_snapshot(self, observation: Any) -> dict[str, float]:
   ```
   Returns keys: `memory.familiar`, `memory.strange`, `memory.surprising`,
   `memory.repeated`. Logic: `recall_similar(observation, limit=3)` for
   familiarity scores; rolling `average_surprise` for surprising;
   `min(total/100, 1.0)` for repeated.

5. All existing public signatures unchanged:
   `store`, `recall_similar`, `recall_by_surprise`, `summarize`, `close`

Return: complete `brain/memory.py`.

---

### TASK 2 — Simplify `brain/agent.py`: consume EpisodicMemory's public API

With TASK 1 complete, `agent.py` no longer needs its own cache, writer, or
stats. Remove all duplication.

Remove entirely:
- `_episode_cache_by_recency`, `_episode_cache_by_id`, `_episode_summary_cache`
- `_next_episode_id`, `_episode_write_queue`, `_episode_writer_stop`
- `_episode_writer_closed`, `_episode_writer_thread`
- `_load_episode_cache()`, `_cache_episode()`, `_episode_writer_loop()`
- `_persist_episode_batch()`, `_rebuild_episode_summary_cache()`
- `_action_counts_from_cache()`, `_surprise_by_action_from_cache()`
- `_patterns_from_cache()`, `_episode_for_public_use()`, `flush_memory_writes()`
- `atexit.register(self.close)` — close must be called explicitly by the notebook

Replace all call sites with `EpisodicMemory`'s public API:
- `_is_familiar(obs)` → `self.memory.recall_similar(self._plain_value(obs), limit=1)`
- `memory_feature_snapshot(obs)` → `self.memory.memory_feature_snapshot(obs)`
- `recall_similar(...)` → `self.memory.recall_similar(...)`
- `summarize()` → `self.memory.summarize()`
- `_store_episode(...)` → `self.memory.store(episode)` (async handled inside memory)
- `save()` → replace `flush_memory_writes()` with `self.memory.flush()`
- `close()` → call `self.memory.flush()` then `self.memory.close()`

Do not touch: `act()`, `learn()`, `save()`, `load()`, emotional state, curiosity,
policy, language, question_engine, response_engine, `_respond_to_human_message()`,
`_emotion_adjusted_scores()`, all tensor utilities.

Return: complete `brain/agent.py`.

---

### TASK 3 — Diagnose and fix action `right` dominance in `brain/agent.py`

The agent selects action `right` (index 3) approximately 80% of the time.
This is not an exploration artifact — it persists across millions of steps and
is getting worse, not converging. The corpus quality of the agent depends on
experiential diversity, so this must be fixed before further training.

Step 1 — Diagnose:
Read `_emotion_adjusted_scores()` in `agent.py` and `action_counts` initialization.
Identify whether the bias originates from:
a) Score initialization asymmetry in the policy network
b) Asymmetric adjustment in `_emotion_adjusted_scores()` (curiosity bonus,
   fear penalty, or `unseen_bonus` calculation)
c) Accumulated `action_counts` state making `unseen_bonus` always favor `right`
d) A combination

Step 2 — Fix:
Apply the minimal targeted fix. Do not redesign the exploration strategy.
Preserve the existing curiosity/fear adjustment logic — only eliminate the
structural bias. If the cause is (c), add count normalization. If (b), correct
the asymmetry. Document the root cause in a comment.

Step 3 — Verify:
Add an assertion or log line (active only when `step_count % 10000 == 0`) that
prints the percentage of each action in the last 1000 steps from `action_counts`,
so bias can be monitored during Kaggle training without stopping the loop.

Return: complete `brain/agent.py` (incorporating changes from TASK 2 as base).

---

### TASK 4 — Expand stopwords in `brain/response.py`

Add the following Spanish functional words to the existing STOPWORDS list
(skip any already present — check first):

```
debes, puedes, debe, puede, para, con, sin, sobre, entre, desde, hasta,
durante, mediante, según, contra, hacia, ante, bajo, tras, mismo, misma,
mismos, mismas, cada, otro, otra, otros, otras, mucho, mucha, muchos, muchas,
poco, poca, pocos, pocas, todo, toda, todos, todas, algo, alguien, nadie,
nada, este, esta, estos, estas, ese, esa, esos, esas, aquel, aquella, cual,
cuales, quien, quienes, cuyo, cuya
```

No other changes. Return: complete `brain/response.py`.

---

### TASK 5 — Prepare `brain/language_model.py` scaffold (architecture only)

Do not train yet. Do not connect to the agent yet.
This task creates the file and defines the architecture for future use.

Design a minimal autoregressive language model in pure PyTorch that:
- Takes the agent's existing vocabulary (up to 1024 words) as its token space
- Uses a character-free, word-level tokenizer built from `LanguageEngine.vocabulary`
- Architecture: 2-layer LSTM with embedding dim=64, hidden_dim=128, dropout=0.2
- Input: sequence of word indices (max_len=16)
- Output: probability distribution over vocabulary (next word prediction)
- Training signal: `agent_generated` conversations from `conversation_memory`
- Loss: cross-entropy over next-token prediction
- No attention, no positional encoding, no pretrained embeddings — pure learned
  embeddings from scratch initialized with `nn.Embedding`

The file must include:
1. `AgentLanguageModel(nn.Module)` — the network
2. `AgentTokenizer` — builds vocab from a list of words, encodes/decodes sequences
3. `LanguageModelTrainer` — loads `agent_generated` conversations from SQLite,
   trains for N epochs, saves weights to a separate file `language_model.pt`
   (never inside `agent_state.pt` — keep them decoupled)
4. A `generate(prompt_words, max_new_tokens, temperature)` method on the model
5. A `__main__` block that trains when run directly:
   `python brain/language_model.py --memory episodic_memory.sqlite3 --epochs 10`

The model must NOT be imported or used anywhere else in the codebase yet.
It lives in isolation until the corpus is large enough and the architecture
is validated. Mark the top of the file with:
```python
# STATUS: SCAFFOLD ONLY — not connected to agent
# Activate when agent_generated conversation count >= 500 pure entries
```

Return: complete `brain/language_model.py`.

---

## Known bugs and technical debt

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Action `right` dominance ~80% | HIGH | → TASK 3 |
| 2 | agent.py duplicates memory.py cache and writer | HIGH | → TASK 1+2 |
| 3 | response.py missing Spanish stopwords | MEDIUM | → TASK 4 |
| 4 | Tokenization inconsistent across agent.py / conversation_memory.py / response.py | LOW | deferred |
| 5 | Room.receive_message() only holds one pending message (overwrites on multiple) | LOW | deferred |
| 6 | recall() uses LIKE substring matching with false positive risk | LOW | deferred |
| 7 | has_answered() is O(n) over all conversations | LOW | irrelevant at current scale |
| 8 | 864+ legacy conversations marked "unknown" — unusable for language model training | INFO | accepted, not fixable retroactively |

---

## Roadmap (do not implement — context only)

### Phase 1 — Foundations (current)
Fix action bias, unify architecture, clean corpus, build language model scaffold.
Every future phase depends on the corpus being experientially diverse and clean.

### Phase 2 — Causal environment
Objects accumulate history. Zones evolve based on agent behavior. Actions have
consequences that unfold over time, not just the next step. This transforms
prediction error into causal understanding — the difference between reaction
and comprehension.

### Phase 3 — Language model activation
Once `agent_generated` corpus >= 500 clean entries and action distribution is
balanced, activate `language_model.py`. Train exclusively on agent-generated
data. The goal: every word the agent produces traceable to a lived experience
that produced it. No word borrowed from internet text. Ever.

### Phase 4 — Metacognition
A separate uncertainty network that predicts when the policy network will be
wrong, and uses that prediction to direct curiosity strategically instead of
reactively. The closest approximation to self-awareness buildable from scratch
in this paradigm.

### Phase 5 — Domain transfer
The same emotional/curiosity/memory core operating across distinct environments
(music, code, physics) with genuine abstract knowledge transfer between them.
Not multitask. Transfer: "danger" learned in the grid world reused to understand
"compilation error" in a code environment.

### Phase 6 — Hardware
Policy network outputs controlling physical actuators. Same curiosity engine
that explored the virtual world exploring the physical one.

---

## Rules for Claude Code

1. Read the full relevant source files before making any change.
2. Return complete files, never diffs or partial snippets.
3. Preserve backward compatibility with existing `agent_state.pt` checkpoints.
4. No schema changes to `episodic_memory.sqlite3`.
5. No pretrained models, embeddings, or external weights — ever.
6. Keep prompts and responses concise. No re-explaining architecture already
   visible in source.
7. After completing each task, state briefly what was changed and why.
8. If a task would break another module, flag it before proceeding.
