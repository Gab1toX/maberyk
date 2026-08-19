from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import threading
import time
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.agent import _LANGUAGE_MODEL_RETRY_SECONDS
from brain.correction_queue import CorrectionQueue
from brain.language_model import AgentTokenizer, normalize_text
from brain.questions import QuestionEngine
from entorno.desktop_env import DesktopEnv
from mind import STATE_PATH, create_agent, desktop_step, encode_desktop_observation

HOST = "localhost"
PORT = 8000
INDEX_PATH = Path(__file__).parent / "index.html"

# How often the background thread advances the agent when nothing else is
# driving it. This is unrelated to STEPS_PER_FRAME in mind.py (there is no
# pygame frame here) — it just keeps the desktop-mode agent alive between
# chat messages, exactly like leaving mind.py --desktop running.
IDLE_STEP_INTERVAL = 0.2
SAVE_EVERY_STEPS = 500
INNER_VOICE_MAXLEN = 5

# Batch correction queue worker: seconds between Groq calls, so an enqueued
# backlog never hammers the tutor API the way a burst of live chat messages
# could.
QUEUE_WORKER_INTERVAL = 3.0
DEFAULT_QUEUE_PENDING_LIMIT = 20
DEFAULT_QUEUE_SUGGEST_LIMIT = 20

# _respond_to_human_message() in brain/agent.py prints "[response] branch=X"
# for every reply. That line is the only place the branch is exposed —
# capturing it from stdout lets web_mind.py report it without touching
# brain/ logic.
_BRANCH_RE = re.compile(r"\[response\] branch=(\w+)")

# Fixed probe question for GET /debug/lm -- fixed so consecutive calls are
# comparable and the trace below always encodes the same prompt tokens.
_DEBUG_LM_QUESTION = "que es la curiosidad"


def _trace_language_model_generation(
    model: torch.nn.Module, tokenizer, question: str, max_new_tokens: int = 14, temperature: float = 0.8
) -> dict:
    """Reimplements AgentLanguageModel.generate_reply()'s own loop (brain/
    language_model.py) step by step, on the exact live model/tokenizer
    objects passed in, so /debug/lm can report the raw token ids alongside
    the decoded string -- generate_reply() itself only returns the final
    decoded reply, not the ids that produced it.
    """
    is_word_tokenizer = isinstance(tokenizer, AgentTokenizer)
    device = next(model.parameters()).device

    if is_word_tokenizer:
        question_words = []
        for word in normalize_text(question).split():
            clean = re.sub(r"[^\w]", "", word)
            if clean:
                question_words.append(clean)
        prompt_ids = [
            tokenizer.word_to_index.get(word, tokenizer.unk_index) for word in question_words
        ]
    else:
        prompt_ids = tokenizer.bpe.encode(question)

    indices = [tokenizer.q_index, *prompt_ids, tokenizer.a_index]
    generated = list(indices)
    stop_indices = {tokenizer.pad_index, tokenizer.q_index, tokenizer.a_index, tokenizer.end_index}

    def word_of(index: int) -> str:
        if is_word_tokenizer:
            return tokenizer.index_to_word.get(index, tokenizer.UNK)
        return tokenizer.bpe.decode([index])

    was_training = model.training
    model.eval()
    steps = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(input_tensor)
            next_logits = logits[0, -1] / max(temperature, 1e-6)
            next_logits[tokenizer.unk_index] = float("-inf")
            probabilities = torch.softmax(next_logits, dim=-1)
            next_index = int(torch.multinomial(probabilities, 1).item())
            is_stop = next_index in stop_indices
            steps.append({"id": next_index, "word": word_of(next_index), "stop": is_stop})
            if is_stop:
                break
            generated.append(next_index)
    if was_training:
        model.train()

    generated_ids = generated[len(indices):]
    decoded = tokenizer.decode(generated_ids) if is_word_tokenizer else tokenizer.bpe.decode(generated_ids)

    return {
        "prompt_ids": indices,
        "prompt_tokens": [word_of(i) for i in indices],
        "steps": steps,
        "generated_ids": generated_ids,
        "decoded": decoded if isinstance(decoded, str) else " ".join(decoded),
    }


class AgentSession:
    """Owns the single Agent + DesktopEnv pair and all mutable loop state.

    One lock guards every call into the agent so the background stepping
    thread and the HTTP handler threads (ThreadingHTTPServer spawns one per
    request) never touch it concurrently.
    """

    def __init__(self) -> None:
        self.agent = create_agent()
        self.agent.enable_desktop()
        self.agent.enable_voice()
        self.agent.enable_tutor()

        self.desktop_env = DesktopEnv()
        observation = self.desktop_env.get_observation()
        self.encoded_observation = encode_desktop_observation(observation)

        self.lock = threading.Lock()
        self.step = 0
        self._last_save_step = 0
        self.inner_voice_log: list[str] = []
        self._last_inner_voice = ""
        self.running = True

        # Batch correction queue: additive, separate from agent.pending_correction.
        # correction_queue persists generated corrections (SQLite, survives
        # restart); _question_backlog is the in-memory list of not-yet-processed
        # questions the worker still has to run through the tutor -- losing an
        # unprocessed backlog on restart is acceptable (no Groq call was spent
        # on it yet), unlike losing an already-generated pending correction.
        self.correction_queue = CorrectionQueue(self.agent.memory_path)
        self._question_backlog: deque[str] = deque()
        self._backlog_lock = threading.Lock()
        # Best-effort courtesy flag so the worker backs off while a live chat
        # request or a queue/correction resolution is in flight -- self.lock
        # is what actually guarantees safety, this just reduces contention.
        self._user_request_in_flight = False

    def _step_once_locked(self) -> str | None:
        """Advance the agent by exactly one observe/act/learn cycle.

        Returns the response branch parsed from the agent's own debug
        print, or None if it produced no reply this step. Caller must
        already hold self.lock.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _, encoded_next_observation, _ = desktop_step(
                self.agent, self.desktop_env, self.encoded_observation
            )
        captured = buffer.getvalue()
        if captured:
            print(captured, end="")

        self.encoded_observation = encoded_next_observation
        self.step += 1

        inner_voice = self.agent.get_inner_voice()
        if inner_voice and inner_voice != self._last_inner_voice:
            self.inner_voice_log.append(inner_voice)
            self.inner_voice_log = self.inner_voice_log[-INNER_VOICE_MAXLEN:]
            self._last_inner_voice = inner_voice

        if self.step - self._last_save_step >= SAVE_EVERY_STEPS:
            self.agent.save(STATE_PATH)
            self._last_save_step = self.step
            print(f"[web_mind] saved at step {self.step}")

        match = _BRANCH_RE.search(captured)
        return match.group(1) if match else None

    def send_message(self, text: str) -> dict[str, str | None]:
        """Route a chat message exactly like mind.py's Enter-key handler:
        if a question is pending, answer it; otherwise inject the text as a
        desktop human_message and step the agent forward once so the reply
        is produced synchronously, in this request.
        """
        self._user_request_in_flight = True
        try:
            with self.lock:
                pending_question = self.agent.get_question()
                if pending_question:
                    self.agent.receive_answer(text)
                    return {"reply": "Gracias, aprendi algo nuevo.", "branch": "answer"}

                self.desktop_env.receive_message(text)
                branch = self._step_once_locked()
                reply = self.agent.last_response
                self.agent.last_response = ""
                return {"reply": reply or "", "branch": branch}
        finally:
            self._user_request_in_flight = False

    def state_snapshot(self) -> dict:
        with self.lock:
            pending = self.agent.pending_correction
            correction = None
            if pending is not None:
                correction = {
                    "raw": pending["raw"],
                    "corrected": pending["corrected"],
                    "new_words": pending["new_words"],
                }
            return {
                "steps": self.step,
                "emotions": self.agent.emotional_state.values(),
                "dominant_emotion": self.agent.emotional_state.dominant(),
                "inner_voice": list(self.inner_voice_log[-INNER_VOICE_MAXLEN:]),
                "pending_question": self.agent.get_question() or None,
                "correction": correction,
                "language_model_unavailable": self.agent.language_model_unavailable,
            }

    def debug_language_model(self) -> dict:
        """Diagnostic snapshot of the language model state on THIS session's
        live Agent -- the same object every chat request and the queue
        worker use, not a freshly constructed one. Also actually calls the
        live generation path right now (self.agent._generate_language_model_
        response), which means this can itself trigger the same lazy
        load/retry a real chat message would -- the fields below reflect
        state AFTER that call, so a stuck latch shows up here directly
        instead of being inferred from a flag that might be lying.
        """
        with self.lock:
            agent = self.agent
            path = agent._checkpoint_dir / "language_model.pt"
            last_failure = agent._language_model_last_failure
            seconds_since_last_failure = (
                time.monotonic() - last_failure if last_failure is not None else None
            )

            info: dict = {
                "checkpoint_path": str(path),
                "checkpoint_exists": path.exists(),
                "model_loaded_before_call": agent._language_model is not None,
                "unavailable_before_call": agent._language_model_unavailable,
                "seconds_since_last_failure": seconds_since_last_failure,
                "retry_backoff_seconds": _LANGUAGE_MODEL_RETRY_SECONDS,
                "retry_due": agent._language_model_retry_due(),
            }

            live_reply = agent._generate_language_model_response(_DEBUG_LM_QUESTION)

            info["model_loaded_after_call"] = agent._language_model is not None
            info["unavailable_after_call"] = agent._language_model_unavailable
            info["live_reply"] = live_reply

            if agent._language_model is not None:
                model = agent._language_model
                tokenizer = agent._language_model_tokenizer
                info["vocab_size"] = tokenizer.vocab_size
                info["param_count"] = sum(p.numel() for p in model.parameters())
                info["generation_trace"] = _trace_language_model_generation(
                    model, tokenizer, _DEBUG_LM_QUESTION
                )
            else:
                info["vocab_size"] = None
                info["param_count"] = None
                info["generation_trace"] = None

            print(
                f"[debug/lm] loaded(before/after)={info['model_loaded_before_call']}/"
                f"{info['model_loaded_after_call']} unavailable(before/after)="
                f"{info['unavailable_before_call']}/{info['unavailable_after_call']} "
                f"live_reply={live_reply!r}"
            )
            if info["generation_trace"] is not None:
                print(
                    f"[debug/lm] prompt_ids={info['generation_trace']['prompt_ids']} "
                    f"generated_ids={info['generation_trace']['generated_ids']} "
                    f"decoded={info['generation_trace']['decoded']!r}"
                )

            return info

    def resolve_correction(self, action: str) -> bool:
        self._user_request_in_flight = True
        try:
            with self.lock:
                if action == "approve":
                    return self.agent.approve_correction()
                if action == "reject":
                    return self.agent.reject_correction()
                raise ValueError(f"unknown action: {action}")
        finally:
            self._user_request_in_flight = False

    def background_loop(self) -> None:
        while self.running:
            with self.lock:
                self._step_once_locked()
            time.sleep(IDLE_STEP_INTERVAL)

    # ------------------------------------------------------------------
    # Batch correction queue -- additive, never touches
    # agent.pending_correction or the single-correction approve/reject flow.
    # ------------------------------------------------------------------

    def enqueue_questions(self, questions: list[str]) -> int:
        """Appends non-empty questions to the in-memory backlog the worker
        thread drains. Returns how many were actually added."""
        added = 0
        with self._backlog_lock:
            for question in questions:
                cleaned = str(question).strip()
                if cleaned:
                    self._question_backlog.append(cleaned)
                    added += 1
        return added

    def queue_pending(self, limit: int = DEFAULT_QUEUE_PENDING_LIMIT) -> list[dict]:
        return self.correction_queue.pending(limit=limit)

    def queue_stats(self) -> dict[str, int]:
        return self.correction_queue.stats()

    def suggest_questions(self, n: int, episode_pool: int = 200) -> list[str]:
        """Asks the question engine what it would ask right now, without
        enqueuing anything and without disturbing the live question engine.

        QuestionEngine.formulate() is deterministic given (emotional_state,
        observation, vocabulary) -- calling the live engine's formulate()
        n times with the agent's current emotional state would just produce
        the same question n times, and would also reset its real cooldown/
        confusion counters as a side effect of merely inspecting it. Instead
        this runs formulate() on a disposable clone (same vocabulary/
        thresholds/confusion state as the live engine right now) against the
        agent's most recent stored episodes (brain/memory.py's
        recent_episodes(), newest first, in-memory only) -- so each call
        reflects the same "current state" the live engine would ask from,
        applied to different real observations Maberyk actually just had.
        Duplicate question strings are dropped, keeping the newest-first
        order, until n unique ones are collected or the pool runs out.
        """
        with self.lock:
            agent = self.agent
            live_engine = agent.question_engine
            clone = QuestionEngine(
                vocabulary=list(live_engine.vocabulary),
                confusion_threshold=live_engine.confusion_threshold,
                required_steps=live_engine.required_steps,
            )
            clone.confusion_steps = live_engine.confusion_steps
            clone.last_confusion = live_engine.last_confusion
            clone.last_surprise = live_engine.last_surprise
            clone._steps_since_last_question = live_engine._steps_since_last_question

            emotional_state = agent.emotional_state
            memory = agent.language
            episodes = agent.memory.recent_episodes(limit=episode_pool)

            seen: set[str] = set()
            suggestions: list[str] = []
            for episode in episodes:
                if len(suggestions) >= n:
                    break
                observation = episode.get("outcome") or episode.get("observation")
                if not isinstance(observation, dict):
                    continue
                question = clone.formulate(emotional_state, observation, memory)
                if question in seen:
                    continue
                seen.add(question)
                suggestions.append(question)
            return suggestions

    def _apply_correction_via_agent(
        self, question: str, corrected: str, new_words: list[str]
    ) -> bool:
        """Writes a queue entry's correction to the corpus through the exact
        same code path agent.approve_correction() uses (conversation_memory
        .store(..., source="tutor_approved") + language.add_word per new
        word) without duplicating that logic here. Does this by temporarily
        swapping in a synthetic pending_correction matching this queue entry,
        calling the real approve_correction(), then restoring whatever was
        there before -- so the live single-correction slot is left exactly as
        it was for every caller except this brief, lock-held swap. Caller
        must already hold self.lock.
        """
        previous_pending = self.agent.pending_correction
        self.agent.pending_correction = {
            "question": question,
            "raw": None,
            "corrected": corrected,
            "new_words": new_words,
        }
        try:
            return self.agent.approve_correction()
        finally:
            self.agent.pending_correction = previous_pending

    def _novel_words(self, text: str) -> list[str]:
        """Words in `text` not already in the agent's vocabulary -- used only
        for the 'edit' action, since a human-edited correction wasn't scored
        by the tutor's own new_words computation."""
        vocabulary = set(self.agent.language.vocabulary)
        words: list[str] = []
        seen: set[str] = set()
        for raw_word in text.split():
            clean = re.sub(r"[^\w]", "", raw_word).lower()
            if clean and clean not in vocabulary and clean not in seen:
                seen.add(clean)
                words.append(clean)
        return words

    def resolve_queue_entry(
        self, entry_id: int, action: str, submitted_text: str | None
    ) -> dict:
        if action not in ("approve", "reject", "edit", "save"):
            raise ValueError(f"unknown action: {action}")

        self._user_request_in_flight = True
        try:
            with self.lock:
                entry = self.correction_queue.get(entry_id)
                if entry is None:
                    return {"ok": False, "error": "not found"}
                if entry["status"] != "pending":
                    return {"ok": False, "error": f"already {entry['status']}"}

                if action == "reject":
                    self.correction_queue.set_status(entry_id, "rejected")
                    return {"ok": True, "id": entry_id, "status": "rejected"}

                if action == "save":
                    return self._save_queue_entry(entry_id, entry, submitted_text)

                if action == "edit":
                    if not submitted_text or not submitted_text.strip():
                        return {"ok": False, "error": "edit requires non-empty 'corrected'"}
                    corrected = submitted_text.strip()
                    new_words = self._novel_words(corrected)
                else:  # approve
                    corrected = entry["corrected_answer"]
                    if not corrected:
                        return {"ok": False, "error": "no corrected_answer to approve"}
                    new_words = entry["new_words"]

                applied = self._apply_correction_via_agent(entry["question"], corrected, new_words)
                if not applied:
                    return {"ok": False, "error": "agent.approve_correction() failed"}

                self.correction_queue.set_status(
                    entry_id, "approved", corrected_answer=corrected, new_words=new_words
                )
                return {"ok": True, "id": entry_id, "status": "approved"}
        finally:
            self._user_request_in_flight = False

    def _save_queue_entry(
        self, entry_id: int, entry: dict, submitted_text: str | None
    ) -> dict:
        """Batch-review 'save': compares the submitted text against the
        tutor's own stored corrected_answer -- server-side, never a
        client-sent flag -- to decide the source tag.

        Unchanged text IS the tutor's correction, so it is written through
        agent.approve_correction() exactly like 'approve' (source=
        tutor_approved, same reused code path, same new_words admission).
        Any edit means a human rewrote it, so it goes in as human_taught
        via conversation_memory.store() directly -- agent.approve_correction()
        cannot produce that source tag (it hardcodes tutor_approved), and
        agent._store_taught_answer() was deliberately avoided here: it also
        overwrites agent._last_taught, which the live chat's "olvida eso"
        undo phrase reads, and a batch-review save must never redirect that
        to the wrong entry. Caller must already hold self.lock.
        """
        if submitted_text is None or not submitted_text.strip():
            return {"ok": False, "error": "save requires non-empty 'text'"}

        text = submitted_text.strip()
        stored_corrected = (entry["corrected_answer"] or "").strip()

        if text == stored_corrected:
            source = "tutor_approved"
            new_words = entry["new_words"]
            applied = self._apply_correction_via_agent(entry["question"], text, new_words)
            if not applied:
                return {"ok": False, "error": "agent.approve_correction() failed"}
        else:
            source = "human_taught"
            new_words = []
            self.agent.conversation_memory.store(entry["question"], text, source="human_taught")

        self.correction_queue.set_status(
            entry_id, "approved", corrected_answer=text, new_words=new_words
        )
        return {"ok": True, "id": entry_id, "status": "approved", "source": source}

    def _process_backlog_question(self, question: str) -> None:
        """Runs one backlog question through the existing tutor path (same
        raw-reply generation and same LLMTutor.correct() call
        _generate_tutor_response() uses) and files the result in the
        correction queue. Never touches agent.pending_correction -- a tutor
        acceptance always lands as a new status='pending' queue row awaiting
        a human decision (never auto-approved), mirroring how a tutor
        rejection in the live flow never becomes a pending_correction either.

        agent._generate_language_model_response() is called unconditionally,
        before the tutor-availability check -- same as GET /debug/lm. It used
        to run after that check, so whenever tutor_enabled was False (or
        _tutor was None) the worker returned before ever reaching the LM
        call, and the model's lazy-load/retry never got a chance to run from
        this path -- even though /debug/lm, calling the same method with no
        such gate, worked fine. The LM call and the tutor call are
        independent concerns; only the tutor step needs tutor availability.
        """
        with self.lock:
            lm_reply = self.agent._generate_language_model_response(question)
            if not lm_reply:
                print(f"[queue] skipped (no language-model reply): {question}")
                return

            if not self.agent.tutor_enabled or self.agent._tutor is None:
                print(f"[queue] skipped (tutor unavailable): {question}")
                return

            result = self.agent._tutor.correct(lm_reply, set(self.agent.language.vocabulary))
            if result is None:
                print(f"[queue] skipped (tutor API failure): {question}")
                return

            if result["status"] == "rejected":
                self.correction_queue.enqueue_result(
                    question, lm_reply, None, result.get("new_words", []), status="rejected"
                )
                print(f"[queue] tutor rejected ({result.get('reason')}): {question}")
            else:
                self.correction_queue.enqueue_result(
                    question, lm_reply, result["corrected"], result["new_words"], status="pending"
                )
                print(f"[queue] queued for review: {question}")

    def queue_worker_loop(self) -> None:
        while self.running:
            if self._user_request_in_flight:
                time.sleep(QUEUE_WORKER_INTERVAL)
                continue

            with self._backlog_lock:
                question = self._question_backlog.popleft() if self._question_backlog else None

            if question is None:
                time.sleep(QUEUE_WORKER_INTERVAL)
                continue

            self._process_backlog_question(question)
            time.sleep(QUEUE_WORKER_INTERVAL)

    def shutdown(self) -> None:
        self.running = False
        with self.lock:
            self.agent.save(STATE_PATH)
            self.agent.close()
        self.correction_queue.close()


class ChatRequestHandler(BaseHTTPRequestHandler):
    session: AgentSession

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_index(self) -> None:
        body = INDEX_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/":
            self._send_index()
        elif parsed.path == "/state":
            self._send_json(200, self.session.state_snapshot())
        elif parsed.path == "/debug/lm":
            self._send_json(200, self.session.debug_language_model())
        elif parsed.path == "/queue/pending":
            params = urllib.parse.parse_qs(parsed.query)
            limit = DEFAULT_QUEUE_PENDING_LIMIT
            if "limit" in params:
                try:
                    limit = int(params["limit"][0])
                except ValueError:
                    self._send_json(400, {"error": "limit must be an integer"})
                    return
            self._send_json(200, {"pending": self.session.queue_pending(limit=limit)})
        elif parsed.path == "/queue/stats":
            self._send_json(200, self.session.queue_stats())
        elif parsed.path == "/queue/suggest":
            params = urllib.parse.parse_qs(parsed.query)
            n = DEFAULT_QUEUE_SUGGEST_LIMIT
            if "n" in params:
                try:
                    n = int(params["n"][0])
                except ValueError:
                    self._send_json(400, {"error": "n must be an integer"})
                    return
            self._send_json(200, {"questions": self.session.suggest_questions(n)})
        else:
            self._send_json(404, {"error": "not found"})

    def _read_json_body(self) -> dict | None:
        """Read and parse the request body as JSON. On malformed JSON, sends
        the 400 response itself and returns None -- callers just bail out.
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return None

    def do_POST(self) -> None:
        if self.path == "/message":
            data = self._read_json_body()
            if data is None:
                return

            text = str(data.get("text", "")).strip()
            if not text:
                self._send_json(400, {"error": "empty text"})
                return

            self._send_json(200, self.session.send_message(text))
        elif self.path == "/correction":
            data = self._read_json_body()
            if data is None:
                return

            action = str(data.get("action", ""))
            if action not in ("approve", "reject"):
                self._send_json(400, {"error": "invalid action"})
                return

            self._send_json(200, {"ok": self.session.resolve_correction(action)})
        elif self.path == "/queue/enqueue":
            data = self._read_json_body()
            if data is None:
                return

            questions = data.get("questions")
            if not isinstance(questions, list) or not questions:
                self._send_json(400, {"error": "'questions' must be a non-empty list"})
                return

            added = self.session.enqueue_questions(questions)
            self._send_json(200, {"ok": True, "added": added})
        elif self.path == "/queue/resolve":
            data = self._read_json_body()
            if data is None:
                return

            entry_id = data.get("id")
            action = str(data.get("action", ""))
            # "save" (batch review UI) sends 'text'; "edit" (legacy
            # single-review UI) sends 'corrected' -- both are just the
            # submitted replacement text, only the field name differs.
            submitted_text = data.get("text") if action == "save" else data.get("corrected")

            if not isinstance(entry_id, int):
                self._send_json(400, {"error": "'id' must be an integer"})
                return
            if action not in ("approve", "reject", "edit", "save"):
                self._send_json(400, {"error": "invalid action"})
                return
            if submitted_text is not None and not isinstance(submitted_text, str):
                self._send_json(400, {"error": "submitted text must be a string"})
                return

            result = self.session.resolve_queue_entry(entry_id, action, submitted_text)
            self._send_json(200 if result.get("ok") else 400, result)
        else:
            self._send_json(404, {"error": "not found"})


def main() -> None:
    session = AgentSession()
    ChatRequestHandler.session = session

    background = threading.Thread(target=session.background_loop, daemon=True)
    background.start()

    queue_worker = threading.Thread(target=session.queue_worker_loop, daemon=True)
    queue_worker.start()

    server = ThreadingHTTPServer((HOST, PORT), ChatRequestHandler)
    print(f"Maberyk web mind escuchando en http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.shutdown()


if __name__ == "__main__":
    main()
