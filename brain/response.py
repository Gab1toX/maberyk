from __future__ import annotations

from typing import Any


class ResponseEngine:
    """Generates short vocabulary-bound responses to human messages."""

    EMOTION_WORDS = {
        "assertion": ("confidence", "safe", "familiar", "learning"),
        "question": ("confused", "unknown", "strange", "curious"),
        "warning": ("danger", "fear", "dark", "unknown"),
        "exploration": ("explore", "curious", "interesting", "learning"),
    }
    SUBJECT_WORDS = (
        "i",
        "agent",
        "mind",
        "curious",
        "confused",
    )
    VERB_WORDS = (
        "see",
        "feel",
        "sense",
        "know",
        "learn",
        "learning",
        "explore",
        "move",
        "touch",
        "fear",
        "ask",
    )
    STOPWORDS = {
        "soy",
        "eres",
        "the",
        "una",
        "uno",
        "que",
        "you",
        "are",
        "was",
        "has",
        "had",
        "its",
        "this",
        "that",
        "with",
        "yo",
        "tu",
        "el",
        "ella",
        "and",
        "not",
        "for",
        "pero",
        "como",
        "esto",
        "esta",
        "este",
        "ser",
        "hay",
        "desde",
        "hacia",
        "debes",
        "debe",
        "puedes",
        "puede",
        "tener",
        "tiene",
        "hacer",
        "haces",
        "quiero",
        "quieres",
        "vamos",
        "voy",
        "para",
        "por",
        "con",
        "sin",
        "sobre",
        "bajo",
        "ante",
        "tras",
        "unas",
        "unos",
        "los",
        "las",
        "del",
        "ale",
        "pues",
        "bien",
        "mal",
        "muy",
        "mas",
        "tan",
        "asi",
        "ahi",
        "aca",
        "alla",
        "entre",
        "hasta",
        "durante",
        "mediante",
        "según",
        "contra",
        "mismo",
        "misma",
        "mismos",
        "mismas",
        "cada",
        "otro",
        "otra",
        "otros",
        "otras",
        "mucho",
        "mucha",
        "muchos",
        "muchas",
        "poco",
        "poca",
        "pocos",
        "pocas",
        "todo",
        "toda",
        "todos",
        "todas",
        "algo",
        "alguien",
        "nadie",
        "nada",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "aquel",
        "aquella",
        "cual",
        "cuales",
        "quien",
        "quienes",
        "cuyo",
        "cuya",
    }

    def __init__(self, vocabulary: list[str]) -> None:
        self.update_vocabulary(vocabulary)

    def update_vocabulary(self, vocabulary: list[str]) -> None:
        seen = set()
        self.vocabulary = []
        for word in vocabulary:
            word = str(word).lower().strip()
            if word and word not in seen:
                seen.add(word)
                self.vocabulary.append(word)
        self._vocabulary_set = set(self.vocabulary)

    def generate(
        self,
        human_message: str,
        emotional_state: dict,
        memory_context: list[dict],
    ) -> str:
        if not self.vocabulary:
            return ""

        mode = self._response_mode(emotional_state)
        words = self._svo_words(mode, human_message, memory_context)
        if not words:
            words = self._fallback_words(mode)
        return " ".join(words[:7])

    def _response_mode(self, emotional_state: dict) -> str:
        values = emotional_state if isinstance(emotional_state, dict) else {}
        scored = [
            ("assertion", float(values.get("confidence", 0.0))),
            ("question", float(values.get("confusion", 0.0))),
            ("warning", float(values.get("fear", 0.0))),
            ("exploration", float(values.get("curiosity", 0.0))),
        ]

        thresholded = [
            item for item in scored
            if (item[0] == "assertion" and item[1] > 0.6)
            or (item[0] == "question" and item[1] > 0.4)
            or (item[0] == "warning" and item[1] > 0.4)
            or (item[0] == "exploration" and item[1] > 0.5)
        ]

        if thresholded:
            return max(thresholded, key=lambda item: item[1])[0]

        best = max(scored, key=lambda item: item[1])
        return best[0] if best[1] > 0.0 else "exploration"

    def _svo_words(
        self,
        mode: str,
        human_message: str,
        memory_context: list[dict],
    ) -> list[str]:
        subject = self._first_known(self.SUBJECT_WORDS)
        verb = self._verb_for_mode(mode)
        object_word = self._object_word(human_message, memory_context, mode)

        words = [word for word in (subject, verb, object_word) if word]
        extra = self._extra_context_words(human_message, words)
        words.extend(extra)
        return words[:7]

    def _verb_for_mode(self, mode: str) -> str | None:
        preferred = {
            "assertion": ("know", "see", "sense", "feel", "learning"),
            "question": ("ask", "explore", "see", "touch", "learning"),
            "warning": ("fear", "sense", "see", "touch"),
            "exploration": ("explore", "move", "touch", "learning"),
        }
        return self._first_known(preferred.get(mode, self.VERB_WORDS))

    def _object_word(
        self,
        human_message: str,
        memory_context: list[dict],
        mode: str,
    ) -> str | None:
        candidates = []
        for index, word in enumerate(self._message_words(human_message)):
            if word in self._vocabulary_set and word not in self.STOPWORDS:
                score = (1 if len(word) > 4 else 0, len(word), index)
                candidates.append((score, word))

        if candidates:
            return max(candidates, key=lambda item: (item[0][0], item[0][1], item[0][2]))[1]

        for memory in memory_context:
            if not isinstance(memory, dict):
                continue
            for value in memory.values():
                for word in self._message_words(str(value)):
                    if word in self._vocabulary_set and word not in self.STOPWORDS:
                        return word

        return self._first_known(self.EMOTION_WORDS.get(mode, ()))

    def _extra_context_words(self, human_message: str, existing: list[str]) -> list[str]:
        context_words = []
        for word in self._message_words(human_message):
            if (
                word in self._vocabulary_set
                and word not in self.STOPWORDS
                and word not in existing
                and word not in context_words
            ):
                context_words.append(word)
                if len(context_words) + len(existing) >= 7:
                    break
        return context_words

    def _fallback_words(self, mode: str) -> list[str]:
        return [
            word
            for word in self.EMOTION_WORDS.get(mode, ())
            if word in self._vocabulary_set
        ] or self.vocabulary[:1]

    def _first_known(self, words: tuple[str, ...]) -> str | None:
        return next((word for word in words if word in self._vocabulary_set), None)

    def _message_words(self, text: Any) -> list[str]:
        return [
            word.strip(".,!?;:()[]{}\"'").lower()
            for word in str(text or "").split()
            if word.strip(".,!?;:()[]{}\"'")
        ]
