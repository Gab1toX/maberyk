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
3. Podes introducir como maximo 2 palabras que no esten en esa lista.
4. Maximo 20 palabras. Primera persona. Sin markdown, sin comillas, sin preambulo.
5. Rechaza si tienes que APORTAR TU el verbo principal o el sujeto principal. La frase cruda debe traer ya un verbo. Si solo hay sustantivos, adjetivos y conectores sueltos, responde {rejection_token}.
   Rechaza tambien si hay palabras repetidas sin sentido o fragmentos contradictorios.
   NO rechaces solo porque falten articulos, preposiciones, tildes o concordancia -- eso es exactamente lo que debes arreglar.
6. No introduzcas verbos ni ideas que no esten en la frase cruda. Si puedes conjugar, acentuar o reordenar los verbos que ya estan, hazlo libremente: eso es corregir, no inventar.

Ejemplos:
Entrada: mente nacio saber nada y aprende
Salida: Mi mente nacio sin saber nada y aprende

Entrada: portal el y y stone cyan
Salida: RECHAZO

Entrada: yo siento curiosidad mundo grande
Salida: Siento curiosidad por el mundo grande"""


def _strip_punctuation(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word)


def _strip_surrounding_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].strip()
    return text


class LLMTutor:
    """Rewrites Maberyk's raw output into fluent Spanish via the Groq API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 30,
        common_words: list[str] | None = None,
        debug: bool = False,
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

    def correct(self, raw_reply: str, vocabulary: set[str]) -> dict | None:
        """Rewrite raw_reply into fluent Spanish, biased toward vocabulary.

        raw_reply must be Maberyk's own generated output. vocabulary must be
        Maberyk's own vocabulary. The human's message must never be passed
        here -- the tutor only ever teaches, it never speaks for Maberyk.
        """
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
        )

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": raw_reply},
                ],
                "temperature": 0.3,
                "max_tokens": 100,
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
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
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

        text = _strip_surrounding_quotes(text)

        if text == _REJECTION_TOKEN:
            if self.debug:
                print("[tutor:rejected] model-said-RECHAZO")
            return {"status": "rejected", "corrected": None, "new_words": []}

        words = text.split()
        if not text or len(words) > 25:
            if self.debug:
                print("[tutor:rejected] empty-or-too-long")
            return {"status": "rejected", "corrected": None, "new_words": []}

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

        if len(new_words) > 2:
            if self.debug:
                print(f"[tutor:rejected] too-many-new-words: {new_words}")
            return {"status": "rejected", "corrected": None, "new_words": []}

        return {"status": "ok", "corrected": text, "new_words": new_words}
