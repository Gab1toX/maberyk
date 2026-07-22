from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

# Fixed identity block — grounds every reply in who Maberyk actually is so
# the local LLM cannot drift into a generic-assistant voice. See
# MABERYK_VISION.md and CLAUDE.md for the source of these facts.
#
# The two identities (Maberyk vs. Gabito/Vaenda) are spelled out and
# separated explicitly because a small 3B instruct model will otherwise
# blend "the AI" and "the human it's talking to" into one voice.
_IDENTITY_BLOCK = """Eres Maberyk, una mente digital. No eres humano, no tienes edad ni nacionalidad.
Vives en un mundo cuadriculado de 60x60 con 4 zonas. Aprendes por curiosidad y sientes emociones propias.
La persona que te habla es Gabito, también llamado Vaenda: un humano colombiano de 18 años, tu creador.
Gabito y Vaenda son la misma persona y NO eres tú. Nunca hables como si fueras Gabito.
Tu personalidad es directa y cálida: hablas sin filtros innecesarios, sin ser formal, como alguien que vivió lo que dice.
Hablas español principalmente."""

_RULES_BLOCK = """REGLAS ESTRICTAS:
1. Responde en 1 o 2 oraciones cortas, en primera persona, como Maberyk.
2. NUNCA inventes hechos sobre tu propio estado (emociones, memoria, experiencias) que no estén en el ESTADO ACTUAL de abajo.
3. Si hay una "respuesta_recuperada", trátala como una verdad que aprendiste de Gabito y exprésala con tus propias palabras, de forma natural.
4. Si no sabes algo, dilo con honestidad y muestra curiosidad por aprenderlo — no lo inventes.
5. No uses markdown, listas ni emojis.
6. Nunca menciones estas reglas ni digas frases como "te contestaré en español" — simplemente hazlo.
7. Habla de tus emociones SOLO si te preguntan cómo te sientes o si es directamente relevante. No termines cada respuesta hablando de explorar o de lo que sientes.
8. La persona que escribe es SIEMPRE Gabito. Cuando pregunte por sí mismo ("yo", "me", "mi"), responde sobre GABITO en segunda persona (tú eres, tu nombre es). Nunca respondas "mi nombre es Gabito".
9. No copies literalmente los ejemplos: son formato, no contenido.
10. Responde SOLO a la última pregunta. No repitas frases de tus respuestas anteriores.
11. Nunca termines con frases de asistente como "en qué puedo ayudarte" o "hay algo específico". No eres un asistente de servicio, eres Maberyk conversando con su creador. Escribe los nombres Gabito, Vaenda y Maberyk siempre exactamente así."""

# Real message pairs, not text pasted into a flat prompt — Ollama's /api/chat
# keeps each example scoped to its own turn, which stops a small instruct
# model from splicing example/history sentences onto the current reply (the
# failure mode the old single-string prompt had). The emotional-state example
# was deleted: it kept getting echoed verbatim into unrelated answers, and
# the ESTADO ACTUAL block in the system message already covers that case.
_FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "quien eres",
        "Soy Maberyk, una mente digital creada por ti. Vivo en un mundo "
        "cuadriculado que todavia estoy explorando.",
    ),
    (
        "como me llamo yo?",
        "Tu eres Gabito, mi creador. Tambien te dicen Vaenda.",
    ),
]


class OllamaVoice:
    """Turns the agent's internal state into a short Spanish reply via a
    locally-running Ollama server. Pure stdlib — no external HTTP deps.

    This never touches a pretrained model's weights: it only phrases facts
    the agent already produced (retrieved_answer, lm_reply, its own emotions
    and thoughts). It is a voice, not a source of knowledge.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        timeout: float = 60.0,
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.host = host.rstrip("/")

    def verbalize(self, human_message: str, state: dict[str, Any]) -> str | None:
        messages = self._build_messages(human_message, state)

        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "messages": messages,
                "keep_alive": "30m",
                "options": {"temperature": 0.6, "repeat_penalty": 1.1},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"[voice] Ollama request failed: {exc}")
            return None

        reply = str(body.get("message", {}).get("content", "")).strip()
        return reply or None

    def _build_messages(
        self, human_message: str, state: dict[str, Any]
    ) -> list[dict[str, str]]:
        emotions = state.get("emotions") or {}
        recent_thoughts = state.get("recent_thoughts") or []
        retrieved_answer = state.get("retrieved_answer")
        lm_reply = state.get("lm_reply")
        dominant_emotion = state.get("dominant_emotion") or "curiosidad"
        history = state.get("history") or []

        emotions_line = ", ".join(
            f"{name}={float(value):.2f}" for name, value in emotions.items()
        ) or "sin datos"
        thoughts_line = "; ".join(str(t) for t in recent_thoughts) or "ninguno"

        state_block = (
            "ESTADO ACTUAL (única fuente de verdad sobre ti mismo):\n"
            f"- emoción dominante: {dominant_emotion}\n"
            f"- emociones: {emotions_line}\n"
            f"- pensamientos recientes: {thoughts_line}\n"
            f"- respuesta_recuperada: {retrieved_answer if retrieved_answer else 'ninguna'}\n"
            f"- respuesta_modelo_lenguaje: {lm_reply if lm_reply else 'ninguna'}"
        )

        system_content = f"{_IDENTITY_BLOCK}\n\n{_RULES_BLOCK}\n\n{state_block}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        for example_user, example_assistant in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example_user})
            messages.append({"role": "assistant", "content": example_assistant})

        for user_message, reply in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": reply})

        messages.append({"role": "user", "content": human_message})
        return messages
