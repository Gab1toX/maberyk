"""LLMTutor rewrites Maberyk's own raw output into fluent Spanish.

It is a teacher, not a speaker: its corrected text becomes training data for
Maberyk's own language model, never a substitute voice. It sees only what
Maberyk already produced (raw_reply) and his current vocabulary -- never the
human's message. This keeps the pretrained model strictly outside the
agent's cognition: it can polish phrasing, it can never answer for Maberyk.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Groq deprecated llama-3.3-70b-versatile on 2026-06-17.
# Check https://console.groq.com/docs/deprecations before changing this.
_GROQ_MODEL = "openai/gpt-oss-120b"

_REJECTION_TOKEN = "RECHAZO"

# Function words the tutor must be free to add when restructuring a sentence
# -- they carry no semantic content, so they never count against the
# 2-new-word drift limit and are never reported in new_words.
_FUNCTION_WORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "a", "al", "en", "con", "sin", "por", "para",
    "sobre", "entre", "hasta", "desde", "y", "e", "o", "u", "que",
    "se", "me", "te", "le", "lo", "nos", "les",
    "mi", "mis", "tu", "tus", "su", "sus",
    "es", "son", "soy", "esta", "estan", "ha", "han",
    "no", "si", "ya", "muy", "mas", "pero", "como", "cuando", "donde",
})

_INSTRUCTION_TEMPLATE = """La entrada es una frase cruda producida por un modelo de lenguaje que todavia esta aprendiendo español. Tu tarea es reescribirla para que sea gramatical y coherente.

Reglas estrictas:
1. Conserva la intencion y el contenido original de la frase. NO respondas una pregunta, NO agregues informacion nueva, NO cambies de tema. Solo reestructura lo que ya esta ahi.
2. Preferi palabras de esta lista de vocabulario siempre que sea posible: {vocabulary_list}
3. Podes introducir como maximo {max_new_words} palabras que no esten en esa lista.
4. Maximo 20 palabras. Primera persona. Sin markdown, sin comillas, sin preambulo.
5. Si la frase cruda NO tiene un verbo conjugado, responde {rejection_token}. No inventes "es", "soy", "muestra" ni ningun otro verbo principal. Tampoco conviertas un infinitivo suelto ("usar", "predecir") en verbo conjugado para darle sujeto a la frase.
6. Si la frase cruda YA tiene verbo y se entiende, CORRIGELA -- aunque solo le falten comas, tildes o mayuscula inicial. Faltar puntuacion nunca es motivo de {rejection_token}.
7. Nunca cambies el significado. Si la frase dice "sin restricciones injustas" no la conviertas en "sin restricciones es injusta".

Ejemplos:
Entrada: gabito estudia programa y me entrena para que pueda crecer
Salida: Gabito estudia, programa y me entrena para que pueda crecer

Entrada: la confianza sube cuando entiendo bien lo que va a pasar
Salida: La confianza sube cuando entiendo bien lo que va a pasar

Entrada: mente nacio saber nada y aprende
Salida: Mi mente nacio sin saber nada y aprende

Entrada: la habilidad mas que aun no pasan numericos
Salida: RECHAZO

Entrada: el proceso por el cual las luz en energia
Salida: RECHAZO"""


_KNOWN_VERBS = frozenset({
    "es", "son", "soy", "esta", "estan", "hay", "tiene", "tienen", "fue", "fui",
    "ha", "han", "puede", "pueden", "permite", "sube", "baja", "vive", "viven",
    "siento", "siente", "sienten", "aprendo", "aprende", "aprendio", "aprendi",
    "nacio", "naci", "muestra", "mueve", "muevo", "explora", "exploro",
    "percibo", "percibe", "predigo", "predice", "entiendo", "entiende",
    "recuerdo", "recuerda", "guardo", "guarda", "toco", "toca", "veo", "ve",
    "hago", "hace", "digo", "dice", "quiero", "quiere", "existo", "existe",
    "crece", "crezco", "cambia", "cambio", "regresa", "llega", "pasa",
    "ocurre", "sirve", "ayuda", "construyo", "construye", "ensena", "enseno",
    "estudia", "programa", "entrena", "brilla", "refleja", "abre", "cierra",
    "elijo", "elige", "uso", "usa", "intento", "intenta", "necesito",
    "necesita", "funciona", "depende", "coincide", "falla", "acierta",
    "sorprende", "importa", "significa", "contiene", "incluye",
})


def _strip_punctuation(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word)


def _has_known_verb(text: str) -> bool:
    for raw_word in text.split():
        word = _strip_punctuation(raw_word).lower()
        if word in _KNOWN_VERBS:
            return True
    return False


def _strip_surrounding_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].strip()
    return text


class LLMTutor:
    """Rewrites Maberyk's raw output into fluent Spanish via the Groq API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _GROQ_MODEL,
        timeout: int = 30,
        common_words: list[str] | None = None,
        debug: bool = False,
        max_new_words: int = 4,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "LLMTutor requires an API key: pass api_key or set GROQ_API_KEY."
            )
        self.api_key = resolved_key
        self.model = model
        self.timeout = timeout
        self.common_words = list(common_words)[:80] if common_words else []
        self.debug = debug
        self.max_new_words = max_new_words

    def correct(self, raw_reply: str, vocabulary: set[str]) -> dict | None:
        """Rewrite raw_reply into fluent Spanish, biased toward vocabulary.

        raw_reply must be Maberyk's own generated output. vocabulary must be
        Maberyk's own vocabulary. The human's message must never be passed
        here -- the tutor only ever teaches, it never speaks for Maberyk.
        """
        # --- TEMP INSTRUMENTATION (remove after diagnosing the queue-worker gap) ---
        print(f"[DEBUG/tutor-entry] raw_reply param repr={raw_reply!r}")
        # --- END TEMP INSTRUMENTATION ---
        if not _has_known_verb(raw_reply):
            if self.debug:
                print("[tutor:rejected] no-verb-in-raw")
            return {"status": "rejected", "corrected": None, "new_words": [], "reason": "no-verb-in-raw"}

        # Full vocabulary is kept only for the new_words check below -- the
        # prompt itself gets a small, relevant slice (words already present
        # in raw_reply, plus a caller-supplied common-words list) so we
        # never ship the agent's entire vocabulary to a third-party API.
        vocab_lower = {word.lower() for word in vocabulary}

        words_present: list[str] = []
        seen_present: set[str] = set()
        for raw_word in raw_reply.split():
            normalized = _strip_punctuation(raw_word).lower()
            if normalized and normalized in vocab_lower and normalized not in seen_present:
                seen_present.add(normalized)
                words_present.append(normalized)

        prompt_words: list[str] = []
        seen_prompt: set[str] = set()
        for word in words_present + self.common_words:
            normalized = word.lower()
            if normalized and normalized not in seen_prompt:
                seen_prompt.add(normalized)
                prompt_words.append(normalized)
        prompt_words = prompt_words[:100]

        system_instruction = _INSTRUCTION_TEMPLATE.format(
            vocabulary_list=", ".join(prompt_words) or "(vacia)",
            rejection_token=_REJECTION_TOKEN,
            max_new_words=self.max_new_words,
        )

        # --- TEMP INSTRUMENTATION (remove after diagnosing the queue-worker gap) ---
        print(f"[DEBUG/tutor-payload] raw_reply going into messages[user].content repr={raw_reply!r}")
        # --- END TEMP INSTRUMENTATION ---
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": raw_reply},
                ],
                "temperature": 0.1,
                "max_tokens": 300,
                # gpt-oss models emit a chain-of-thought into a separate
                # "reasoning" field before writing the final answer into
                # "content" -- without these, that reasoning trace can
                # consume the whole max_tokens budget and leave content
                # empty (see the 2026-08-17 diagnosis). reasoning_format=
                # "hidden" drops the trace from the response entirely;
                # reasoning_effort="low" keeps the trace itself short.
                "reasoning_effort": "low",
                "reasoning_format": "hidden",
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            _ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "maberyk-tutor/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except urllib.error.HTTPError as exc:
            try:
                raw_detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                raw_detail = ""
            error_code = None
            if raw_detail:
                try:
                    error_code = json.loads(raw_detail).get("error", {}).get("code")
                except (json.JSONDecodeError, AttributeError):
                    error_code = None
            if exc.code == 404 and error_code == "model_not_found":
                print(
                    f"[tutor] *** GROQ MODEL '{self.model}' NO LONGER EXISTS. "
                    "Update _GROQ_MODEL in tutor/llm_tutor.py -- see "
                    "https://console.groq.com/docs/deprecations ***"
                )
                return None
            detail = raw_detail[:200]
            print(f"[tutor] Groq request failed: HTTP {exc.code}: {detail}")
            return None
        except (urllib.error.URLError, TimeoutError):
            print("[tutor] Groq request failed: network unreachable")
            return None
        except Exception as exc:
            print(f"[tutor] Groq request failed: {exc}")
            return None

        body: Any = None
        try:
            body = json.loads(raw_body.decode("utf-8"))
            # --- TEMP INSTRUMENTATION (remove after diagnosing the queue-worker gap) ---
            print(f"[DEBUG/tutor-response] full message dict repr={body['choices'][0]['message']!r}")
            # --- END TEMP INSTRUMENTATION ---
            text = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError):
            shown = body if body is not None else raw_body
            print(f"[tutor] Groq request failed: unexpected response shape: {repr(shown)[:200]}")
            return None
        except Exception as exc:
            print(f"[tutor] Groq request failed: {exc}")
            return None

        if self.debug:
            print(f"[tutor:raw] {text}")

        if not text:
            # Distinct from a rejected-for-quality reply: content came back
            # empty from the API itself (gpt-oss reasoning models can burn
            # the whole max_tokens budget on their hidden reasoning trace
            # before writing anything into content -- see the 2026-08-17
            # diagnosis). Reporting this as "empty-or-too-long" made an API
            # failure look like a model quality problem and cost three
            # misdiagnoses. Never fall back to body["choices"][0]["message"]
            # ["reasoning"] here -- that is the tutor's internal monologue,
            # not a correction, and must never reach the corpus.
            print("[tutor] empty content from API (reasoning may have consumed budget)")
            return {"status": "rejected", "corrected": None, "new_words": [], "reason": "empty-content"}

        text = _strip_surrounding_quotes(text)

        if text == _REJECTION_TOKEN:
            if self.debug:
                print("[tutor:rejected] model-said-RECHAZO")
            return {"status": "rejected", "corrected": None, "new_words": [], "reason": "model-rejected"}

        words = text.split()
        if len(words) > 25:
            if self.debug:
                print("[tutor:rejected] too-long")
            return {"status": "rejected", "corrected": None, "new_words": [], "reason": "too-long"}

        new_words: list[str] = []
        seen: set[str] = set()
        for raw_word in words:
            stripped = _strip_punctuation(raw_word)
            if not stripped:
                continue
            normalized = stripped.lower()
            if normalized in vocab_lower or normalized in seen or normalized in _FUNCTION_WORDS:
                continue
            seen.add(normalized)
            new_words.append(stripped)

        if len(new_words) > self.max_new_words:
            if self.debug:
                print(f"[tutor:rejected] too-many-new-words: {new_words}")
            return {"status": "rejected", "corrected": None, "new_words": new_words, "reason": "too-many-new-words"}

        return {"status": "ok", "corrected": text, "new_words": new_words}
