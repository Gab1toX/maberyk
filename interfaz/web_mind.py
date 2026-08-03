from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# _respond_to_human_message() in brain/agent.py prints "[response] branch=X"
# for every reply. That line is the only place the branch is exposed —
# capturing it from stdout lets web_mind.py report it without touching
# brain/ logic.
_BRANCH_RE = re.compile(r"\[response\] branch=(\w+)")


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
            }

    def resolve_correction(self, action: str) -> bool:
        with self.lock:
            if action == "approve":
                return self.agent.approve_correction()
            if action == "reject":
                return self.agent.reject_correction()
            raise ValueError(f"unknown action: {action}")

    def background_loop(self) -> None:
        while self.running:
            with self.lock:
                self._step_once_locked()
            time.sleep(IDLE_STEP_INTERVAL)

    def shutdown(self) -> None:
        self.running = False
        with self.lock:
            self.agent.save(STATE_PATH)
            self.agent.close()


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
        if self.path == "/":
            self._send_index()
        elif self.path == "/state":
            self._send_json(200, self.session.state_snapshot())
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
        else:
            self._send_json(404, {"error": "not found"})


def main() -> None:
    session = AgentSession()
    ChatRequestHandler.session = session

    background = threading.Thread(target=session.background_loop, daemon=True)
    background.start()

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
